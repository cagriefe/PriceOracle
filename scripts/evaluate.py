"""Evaluate a checkpoint on its held-out period: point accuracy, direction and calibrated risk intervals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from analysis.evaluation import evaluate_forecasts
from analysis.forecasts import interval_levels, level_name, load_forecasts, quantile_column
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, load_config

METHOD_LABELS = {
    "raw": "Model, raw quantiles",
    "static": "Model + static conformal",
    "rolling": "Model + rolling conformal",
    "hist": "Historical simulation",
    "ewma": "EWMA volatility",
}


def evaluate(model_path, data_path=None, split="test", device="auto", plot_path=None) -> dict:
    predictor, _, table = load_forecasts(model_path, data_path, device)
    results = evaluate_forecasts(table, predictor.quantiles, split)
    results["model_type"] = predictor.config["model_type"]
    results["horizon"] = predictor.horizon
    if plot_path:
        plot_forecasts(table[table["split"] == split], predictor.quantiles, plot_path, predictor.horizon)
    return results


def format_report(results: dict) -> str:
    model, naive = results["point"]["model"], results["point"]["naive"]
    direction, pinball = results["direction"], results["pinball"]
    horizon = results["horizon"]
    lines = [
        f"{results['split'].capitalize()} period: {results['start']} -> {results['end']} "
        f"({results['samples']:,} forecasts, {horizon} candle{'s' if horizon > 1 else ''} ahead)",
        "",
        f"{'Point forecast (median)':<28}{'MAE ($)':>12}{'RMSE ($)':>12}{'MAPE':>10}",
        f"{results['model_type'].upper():<28}{model['mae']:>12,.2f}{model['rmse']:>12,.2f}{model['mape']:>9.3f}%",
        f"{'Naive (last close)':<28}{naive['mae']:>12,.2f}{naive['rmse']:>12,.2f}{naive['mape']:>9.3f}%",
        "",
        f"Directional accuracy: {direction['accuracy'] * 100:.2f}% (always-up baseline: "
        f"{direction['always_up'] * 100:.2f}%)",
        f"Brier score of P(up): {direction['brier']:.4f} (coin flip: {direction['brier_coin_flip']:.4f})",
        "",
        f"{'Risk intervals':<28}{'Coverage':>10}{'Width':>10}{'Score':>10}",
    ]
    for level, intervals in results["intervals"].items():
        lines.append(f"{level}% interval")
        for method, metrics in intervals.items():
            if metrics:
                lines.append(
                    f"  {METHOD_LABELS[method]:<26}{metrics['coverage'] * 100:>9.2f}%"
                    f"{metrics['width'] * 100:>9.3f}%{metrics['interval_score'] * 100:>10.3f}"
                )
    lines += [
        "",
        "Pinball loss (bps, lower is better): "
        + " | ".join(f"{METHOD_LABELS.get(name, 'Model')} {value * 1e4:.3f}" for name, value in pinball.items()),
    ]
    return "\n".join(lines)


def plot_forecasts(rows, quantiles, path, horizon=1, zoom=168) -> None:
    """Save actual closes with the median forecast and calibrated bands: whole period plus a recent zoom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    times = rows["target_time"].to_numpy()
    close = rows["close"].to_numpy()
    actual = close * np.exp(rows["actual"].to_numpy())
    median = close * np.exp(rows[quantile_column(0.5)].to_numpy())
    bands = [
        (level, close * np.exp(rows[f"lo{level_name(level)}"]), close * np.exp(rows[f"hi{level_name(level)}"]))
        for level, _, _ in interval_levels(quantiles)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    recent = slice(-min(zoom, len(rows)), None)
    for ax, part in zip(axes, (slice(None), recent), strict=True):
        for i, (level, lower, upper) in enumerate(bands):
            ax.fill_between(
                times[part],
                lower[part],
                upper[part],
                color="tab:orange",
                alpha=0.18 + 0.17 * i,
                linewidth=0,
                label=f"{level:.0%} interval",
            )
        ax.plot(times[part], actual[part], color="tab:blue", linewidth=1, label="Actual close")
        ax.plot(times[part], median[part], color="tab:red", linewidth=1, linestyle="--", label="Median forecast")
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Price (USDT)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_title(
        f"Out-of-sample forecasts with conformal intervals, {horizon} candle{'s' if horizon > 1 else ''} ahead"
    )
    axes[1].set_title(f"Last {len(rows.iloc[recent])} forecasts")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to evaluate (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--plot", help="Save a chart of the forecasts and intervals to this PNG")
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
