"""Forecast the next close from the latest candles, optionally exporting every historical prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd

from data.preprocess import prepare_frame
from models.predictor import PricePredictor
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, get_device, load_config, resolve_path


def load_predictor_and_frame(model_path, data_path=None, device="auto"):
    predictor = PricePredictor.load(resolve_path(model_path), get_device(device))
    frame = prepare_frame(resolve_path(data_path or predictor.config["data_path"]), horizon=predictor.horizon)
    if len(frame) < predictor.seq_length:
        raise ValueError(f"Need at least {predictor.seq_length} candles after indicator warm-up, got {len(frame)}.")
    return predictor, frame


def candle_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most common spacing between candles (robust to occasional exchange-outage gaps)."""
    return pd.Series(index).diff().mode().iloc[0]


def forecast(predictor: PricePredictor, frame: pd.DataFrame) -> dict:
    """Forecast the close ``horizon`` candles after the latest candle in ``frame``."""
    log_return = float(predictor.predict_log_returns(frame, [len(frame) - 1])[0])
    last_close = float(frame["close"].iloc[-1])
    return {
        "last_candle": frame.index[-1],
        "last_close": last_close,
        "target_candle": frame.index[-1] + predictor.horizon * candle_interval(frame.index),
        "predicted_close": last_close * float(np.exp(log_return)),
        "predicted_change_pct": float(np.expm1(log_return)) * 100,
    }


def prediction_history(predictor: PricePredictor, frame: pd.DataFrame) -> pd.DataFrame:
    """Predictions for every window, labelled by split. Only ``test`` (and ``forecast``) rows are out-of-sample."""
    end_indices = np.arange(predictor.seq_length - 1, len(frame))
    reference = frame["close"].to_numpy()[end_indices]
    split = pd.Series("gap", index=end_indices, dtype=object)
    for name, indices in predictor.split_indices(frame).items():
        split.loc[indices] = name
    split.loc[frame["target"].isna().to_numpy()[end_indices]] = "forecast"

    return pd.DataFrame(
        {
            "timestamp": frame.index[end_indices],
            "close": reference,
            "predicted_close": reference * np.exp(predictor.predict_log_returns(frame, end_indices)),
            "actual_close": reference * np.exp(frame["target"].to_numpy()[end_indices]),
            "split": split.to_numpy(),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to use (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--history-out", help="Also write predictions for every historical window to this CSV")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    model_path = cli_path(args.model_path) or load_config(DEFAULT_CONFIG_PATH)["model_path"]
    predictor, frame = load_predictor_and_frame(model_path, cli_path(args.data_path), args.device)

    result = forecast(predictor, frame)
    print(f"Latest candle: {result['last_candle']:%Y-%m-%d %H:%M} UTC, close {result['last_close']:,.2f} USDT")
    print(
        f"Forecast for the {result['target_candle']:%Y-%m-%d %H:%M} UTC candle: "
        f"close {result['predicted_close']:,.2f} USDT ({result['predicted_change_pct']:+.3f}%)"
    )

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        prediction_history(predictor, frame).to_csv(args.history_out, index=False)
        print(f"Wrote prediction history to {args.history_out}")


if __name__ == "__main__":
    main()
