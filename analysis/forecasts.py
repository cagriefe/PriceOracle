"""The forecast table: one out-of-sample forecast per candle from the validation period onwards.

Every downstream view (evaluation, backtests, dashboard, live forecast) reads from this table, so they all
see exactly the same numbers. All values are log returns over the next ``horizon`` candles unless noted.

Columns:

* ``split`` (val/test/gap/live), ``target_time``, ``close`` (last known close), ``actual`` (realised log
  return, NaN for live rows) and ``next_return`` (simple return of the next candle, for backtests);
* ``q5`` ... ``q95``: the network's raw quantiles; ``p_up_raw`` is the implied probability of a rise and
  ``p_up`` the same probability recalibrated on the validation period;
* ``lo90``/``hi90`` etc.: conformal intervals (rolling, falling back to static) and ``*_static`` variants;
* ``q5_hist`` ... and ``q5_ewma`` ...: quantiles of the historical-simulation and EWMA baselines.
"""

import numpy as np
import pandas as pd

from analysis.baselines import ewma_quantiles, historical_quantiles
from analysis.conformal import apply_adjustment, conformity_scores, rolling_adjustment, static_adjustment
from data.preprocess import candle_interval, prepare_frame
from models.predictor import PricePredictor
from utils.helpers import get_device, resolve_path

BASELINES = ("hist", "ewma")


def load_forecasts(model_path, data_path=None, device="auto"):
    """Load a checkpoint and its dataset, and build the forecast table: ``(predictor, frame, table)``."""
    predictor = PricePredictor.load(resolve_path(model_path), get_device(device))
    frame = prepare_frame(resolve_path(data_path or predictor.config["data_path"]), horizon=predictor.horizon)
    if len(frame) < predictor.seq_length:
        raise ValueError(f"Need at least {predictor.seq_length} candles after indicator warm-up, got {len(frame)}.")
    return predictor, frame, build_forecasts(predictor, frame)


def quantile_column(q: float, suffix: str = "") -> str:
    return f"q{q * 100:g}" + (f"_{suffix}" if suffix else "")


def level_name(level: float) -> str:
    return f"{level * 100:g}"


def interval_levels(quantiles) -> list:
    """``(level, lower_q, upper_q)`` for every symmetric quantile pair, widest first, e.g. ``(0.9, 0.05, 0.95)``."""
    rounded = [round(float(q), 6) for q in quantiles]
    return [(round(1 - 2 * q, 6), q, round(1 - q, 6)) for q in rounded if q < 0.5 and round(1 - q, 6) in rounded]


def live_forecast(table: pd.DataFrame, quantiles) -> dict:
    """The newest forecast (last row of the table) converted to prices, narrowest interval first."""
    row = table.iloc[-1]
    close = float(row["close"])
    median = float(row[quantile_column(0.5)])
    return {
        "last_candle": table.index[-1],
        "last_close": close,
        "target_candle": pd.Timestamp(row["target_time"]),
        "predicted_close": close * float(np.exp(median)),
        "predicted_change_pct": float(np.expm1(median)) * 100,
        "probability_up": float(row["p_up"]),
        "intervals": {
            level_name(level): (
                close * float(np.exp(row[f"lo{level_name(level)}"])),
                close * float(np.exp(row[f"hi{level_name(level)}"])),
            )
            for level, _, _ in reversed(interval_levels(quantiles))
        },
    }


def probability_up(quantile_values, quantiles) -> np.ndarray:
    """P(log return > 0) from the predictive CDF interpolated through the quantiles (clamped to their range)."""
    taus = np.asarray(quantiles, dtype=np.float64)
    return np.array([1.0 - np.interp(0.0, row, taus) for row in np.asarray(quantile_values, dtype=np.float64)])


def shrinkage_factor(probabilities, outcomes) -> float:
    """Brier-optimal factor ``k`` in ``0.5 + k * (p - 0.5)``, clipped to [0, 1].

    Quantile-implied probabilities tend to be overconfident when the directional signal is weak; fitting
    ``k`` on the validation period pulls them towards 50% just enough.
    """
    centered = np.asarray(probabilities, dtype=np.float64) - 0.5
    denominator = np.sum(centered**2)
    if denominator == 0:
        return 0.0
    k = np.sum(centered * (np.asarray(outcomes, dtype=np.float64) - 0.5)) / denominator
    return float(np.clip(k, 0.0, 1.0))


def build_forecasts(predictor, frame: pd.DataFrame, conformal_window: int | None = None) -> pd.DataFrame:
    window = int(conformal_window or predictor.config.get("conformal_window", 720))
    quantiles, horizon = predictor.quantiles, predictor.horizon
    splits = predictor.split_indices(frame)
    start = int(splits["val"][0])
    end_indices = np.arange(start, len(frame))

    actual = frame["target"].to_numpy()[end_indices]
    labels = np.full(len(end_indices), "gap", dtype=object)
    for name in ("val", "test"):
        labels[splits[name] - start] = name
    labels[np.isnan(actual)] = "live"

    target_positions = end_indices + horizon
    known = target_positions < len(frame)
    target_time = frame.index[end_indices] + horizon * candle_interval(frame.index)
    target_time = target_time.where(~known, frame.index[np.minimum(target_positions, len(frame) - 1)])

    close = frame["close"].to_numpy()
    table = pd.DataFrame(
        {
            "split": labels,
            "target_time": target_time,
            "close": close[end_indices],
            "actual": actual,
            "next_return": np.append(close[1:] / close[:-1] - 1, np.nan)[end_indices],
        },
        index=frame.index[end_indices],
    )

    is_val = labels == "val"
    predicted = predictor.predict_quantiles(frame, end_indices)
    for q, values in zip(quantiles, predicted.T, strict=True):
        table[quantile_column(q)] = values
    raw_probability = probability_up(predicted, quantiles)
    shrink = shrinkage_factor(raw_probability[is_val], actual[is_val] > 0)
    table["p_up_raw"] = raw_probability
    table["p_up"] = 0.5 + shrink * (raw_probability - 0.5)

    returns = frame["log_return"]
    baselines = {
        "hist": historical_quantiles(returns, quantiles, window, horizon),
        "ewma": ewma_quantiles(returns, quantiles, horizon=horizon),
    }
    for name, values in baselines.items():
        for q, column in zip(quantiles, values[end_indices].T, strict=True):
            table[quantile_column(q, name)] = column

    for level, lower_q, upper_q in interval_levels(quantiles):
        name = level_name(level)
        lower = table[quantile_column(lower_q)].to_numpy()
        upper = table[quantile_column(upper_q)].to_numpy()
        scores = conformity_scores(actual, lower, upper)
        static = static_adjustment(scores[is_val], 1 - level)
        rolling = rolling_adjustment(scores, 1 - level, window, delay=horizon)
        table[f"lo{name}_static"], table[f"hi{name}_static"] = apply_adjustment(lower, upper, static)
        table[f"lo{name}"], table[f"hi{name}"] = apply_adjustment(
            lower, upper, np.where(np.isnan(rolling), static, rolling)
        )
    return table
