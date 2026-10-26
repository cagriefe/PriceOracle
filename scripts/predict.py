"""Forecast the next close with calibrated prediction intervals, optionally exporting every forecast."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd

from analysis.forecasts import interval_levels, level_name, live_forecast, load_forecasts, quantile_column
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, load_config


def forecast_history(table: pd.DataFrame, quantiles) -> pd.DataFrame:
    """Every out-of-sample forecast in prices, tagged by split (val, test, gap or live)."""
    close = table["close"]
    history = pd.DataFrame(
        {
            "timestamp": table.index,
            "target_time": table["target_time"].to_numpy(),
            "split": table["split"].to_numpy(),
            "close": close.to_numpy(),
            "predicted_close": (close * np.exp(table[quantile_column(0.5)])).to_numpy(),
            "actual_close": (close * np.exp(table["actual"])).to_numpy(),
            "probability_up": table["p_up"].to_numpy(),
        }
    )
    for level, _, _ in interval_levels(quantiles):
        name = level_name(level)
        history[f"lower_{name}"] = (close * np.exp(table[f"lo{name}"])).to_numpy()
        history[f"upper_{name}"] = (close * np.exp(table[f"hi{name}"])).to_numpy()
    return history


def format_forecast(result: dict) -> str:
    lines = [
        f"Latest candle: {result['last_candle']:%Y-%m-%d %H:%M} UTC, close {result['last_close']:,.2f} USDT",
        f"Forecast for the {result['target_candle']:%Y-%m-%d %H:%M} UTC candle close:",
        f"  Median:     {result['predicted_close']:,.2f} USDT ({result['predicted_change_pct']:+.3f}%)",
    ]
    for name, (lower, upper) in result["intervals"].items():
        lines.append(f"  {name}% range:  {lower:,.2f} - {upper:,.2f} USDT")
    lines.append(f"  Chance of a higher close: {result['probability_up'] * 100:.1f}%")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to use (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--history-out", help="Also write every out-of-sample forecast to this CSV")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    model_path = cli_path(args.model_path) or load_config(DEFAULT_CONFIG_PATH)["model_path"]
    predictor, _, table = load_forecasts(model_path, cli_path(args.data_path), args.device)
    print(format_forecast(live_forecast(table, predictor.quantiles)))

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        forecast_history(table, predictor.quantiles).to_csv(args.history_out, index=False)
        print(f"Wrote forecast history to {args.history_out}")


if __name__ == "__main__":
    main()
