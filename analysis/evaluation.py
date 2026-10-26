"""Scores for one split of the forecast table: point accuracy, direction, calibration and interval sharpness."""

import numpy as np
import pandas as pd

from analysis.forecasts import BASELINES, interval_levels, level_name, quantile_column
from utils.metrics import brier_score, interval_metrics, pinball_score, price_metrics, up_move_ratio

INTERVAL_METHODS = ("raw", "static", "rolling", *BASELINES)


def interval_bounds(rows: pd.DataFrame, method: str, level: float, lower_q: float, upper_q: float):
    name = level_name(level)
    if method == "raw":
        return rows[quantile_column(lower_q)], rows[quantile_column(upper_q)]
    if method == "static":
        return rows[f"lo{name}_static"], rows[f"hi{name}_static"]
    if method == "rolling":
        return rows[f"lo{name}"], rows[f"hi{name}"]
    return rows[quantile_column(lower_q, method)], rows[quantile_column(upper_q, method)]


def evaluate_forecasts(table: pd.DataFrame, quantiles, split: str = "test") -> dict:
    rows = table[table["split"] == split]
    if rows.empty:
        raise ValueError(f"The {split!r} split is empty for this dataset.")
    actual = rows["actual"].to_numpy()
    reference = rows["close"].to_numpy()
    actual_close = reference * np.exp(actual)
    predicted_close = reference * np.exp(rows[quantile_column(0.5)].to_numpy())

    naive = price_metrics(actual_close, reference, reference)
    naive.pop("directional_accuracy")  # a flat forecast never predicts a direction
    model = price_metrics(actual_close, predicted_close, reference)

    pinball = {"model": pinball_score(actual, rows[[quantile_column(q) for q in quantiles]], quantiles)}
    for name in BASELINES:
        values = rows[[quantile_column(q, name) for q in quantiles]].to_numpy()
        valid = np.isfinite(values).all(axis=1)
        pinball[name] = pinball_score(actual[valid], values[valid], quantiles) if valid.any() else float("nan")

    intervals = {}
    for level, lower_q, upper_q in interval_levels(quantiles):
        intervals[level_name(level)] = {}
        for method in INTERVAL_METHODS:
            lower, upper = (bound.to_numpy() for bound in interval_bounds(rows, method, level, lower_q, upper_q))
            valid = np.isfinite(lower) & np.isfinite(upper)
            intervals[level_name(level)][method] = (
                interval_metrics(actual[valid], lower[valid], upper[valid], level) if valid.any() else None
            )

    return {
        "split": split,
        "start": f"{rows['target_time'].iloc[0]:%Y-%m-%d %H:%M}",
        "end": f"{rows['target_time'].iloc[-1]:%Y-%m-%d %H:%M}",
        "samples": int(len(rows)),
        "point": {"model": model, "naive": naive},
        "direction": {
            "accuracy": model["directional_accuracy"],
            "always_up": up_move_ratio(actual_close, reference),
            "brier": brier_score(rows["p_up"], actual),
            "brier_coin_flip": 0.25,
        },
        "pinball": pinball,
        "intervals": intervals,
    }
