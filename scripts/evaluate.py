"""Evaluate a checkpoint on its held-out period and compare it with naive baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from data.preprocess import prepare_frame
from models.predictor import PricePredictor
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, get_device, load_config, resolve_path
from utils.metrics import price_metrics, up_move_ratio


def evaluate(model_path, data_path=None, split="test", device="auto", plot_path=None) -> dict:
    predictor = PricePredictor.load(resolve_path(model_path), get_device(device))
    frame = prepare_frame(resolve_path(data_path or predictor.config["data_path"]), horizon=predictor.horizon)
    end_indices = predictor.split_indices(frame)[split]
    if len(end_indices) == 0:
        raise ValueError(f"The {split!r} split is empty for this dataset.")

    reference = frame["close"].to_numpy()[end_indices]
    actual = reference * np.exp(frame["target"].to_numpy()[end_indices])
    predicted = reference * np.exp(predictor.predict_log_returns(frame, end_indices))
    # Timestamps of the forecasted candles, `horizon` rows after each window's last candle.
    target_times = frame.index[end_indices + predictor.horizon]
    naive = price_metrics(actual, reference, reference)
    naive.pop("directional_accuracy")  # a flat forecast never predicts a direction

    results = {
        "split": split,
        "model_type": predictor.config["model_type"],
        "horizon": predictor.horizon,
        "start": str(target_times[0]),
        "end": str(target_times[-1]),
        "samples": int(len(end_indices)),
        "model": price_metrics(actual, predicted, reference),
        "naive": naive,
        "always_up_accuracy": up_move_ratio(actual, reference),
    }
    if plot_path:
        plot_predictions(target_times, actual, predicted, plot_path, predictor.horizon)
    return results


def format_report(results: dict) -> str:
    model, naive = results["model"], results["naive"]
    name = results["model_type"].upper()
    horizon = results["horizon"]
    lines = [
        f"{results['split'].capitalize()} period: {results['start']} -> {results['end']} "
        f"({results['samples']:,} forecasts, {horizon} candle{'s' if horizon > 1 else ''} ahead)",
        "",
        f"{'':<20}{'MAE ($)':>12}{'RMSE ($)':>12}{'MAPE':>10}",
        f"{name:<20}{model['mae']:>12,.2f}{model['rmse']:>12,.2f}{model['mape']:>9.3f}%",
        f"{'Naive (last close)':<20}{naive['mae']:>12,.2f}{naive['rmse']:>12,.2f}{naive['mape']:>9.3f}%",
        "",
        f"RMSE vs naive: {(1 - model['rmse'] / naive['rmse']) * 100:+.2f}% (positive = better than naive)",
        f"Directional accuracy: {model['directional_accuracy'] * 100:.2f}% "
        f"(always-up baseline: {results['always_up_accuracy'] * 100:.2f}%)",
    ]
    return "\n".join(lines)


def plot_predictions(timestamps, actual, predicted, path, horizon=1, zoom=168) -> None:
    """Save actual vs predicted closes for the whole period plus a zoom on the last ``zoom`` forecasts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 8))
    ax_full.plot(timestamps, actual, label="Actual close", linewidth=1)
    ax_full.plot(timestamps, predicted, label="Predicted close", linewidth=1, alpha=0.8)
    ax_full.set_title(f"Out-of-sample forecasts, {horizon} candle{'s' if horizon > 1 else ''} ahead")

    recent = slice(-min(zoom, len(timestamps)), None)
    ax_zoom.plot(timestamps[recent], actual[recent], label="Actual close", marker=".", linewidth=1)
    ax_zoom.plot(timestamps[recent], predicted[recent], label="Predicted close", marker=".", linewidth=1, alpha=0.8)
    ax_zoom.set_title(f"Last {len(timestamps[recent])} forecasts")

    for ax in (ax_full, ax_zoom):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Price (USDT)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to evaluate (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--plot", help="Save an actual-vs-predicted chart to this PNG")
    parser.add_argument("--metrics-out", help="Write the metrics to this JSON file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    model_path = cli_path(args.model_path) or load_config(DEFAULT_CONFIG_PATH)["model_path"]
    results = evaluate(model_path, cli_path(args.data_path), args.split, args.device, cli_path(args.plot))
    print(format_report(results))
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
