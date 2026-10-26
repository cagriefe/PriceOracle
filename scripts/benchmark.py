"""Train every architecture on identical splits and rank them, together with classic baselines, in a leaderboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

from analysis.evaluation import evaluate_forecasts
from analysis.forecasts import build_forecasts, interval_levels, level_name
from data.preprocess import prepare_frame
from models import MODEL_TYPES, count_parameters
from scripts.train import train
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, load_config, resolve_path
from utils.logger import get_logger

DEFAULT_MODELS = ["linear", "gru", "lstm", "tcn", "transformer"]
# A 4-block TCN sees 61 candles, enough to cover the default 48-candle window.
MODEL_OVERRIDES = {"tcn": {"num_layers": 4}}
BASELINE_NAMES = {"hist": "Historical simulation", "ewma": "EWMA volatility"}

logger = get_logger(__name__)


def interval_summary(evaluation: dict, method: str) -> dict:
    level = level_name(max(float(name) / 100 for name in evaluation["intervals"]))
    metrics = evaluation["intervals"][level][method]
    return {
        "level": level,
        "coverage": metrics["coverage"],
        "width": metrics["width"],
        "interval_score": metrics["interval_score"],
    }


def run_benchmark(config: dict, model_types, device: str = "auto", output_dir=None) -> list:
    output_dir = Path(output_dir or resolve_path("reports/benchmark"))
    rows, evaluation = [], None
    for model_type in model_types:
        run_config = {
            **config,
            **MODEL_OVERRIDES.get(model_type, {}),
            "model_type": model_type,
            "model_path": str(output_dir / f"{model_type}.pt"),
        }
        started = time.perf_counter()
        predictor, history = train(run_config, device)
        seconds = time.perf_counter() - started
        frame = prepare_frame(resolve_path(run_config["data_path"]), horizon=predictor.horizon)
        evaluation = evaluate_forecasts(build_forecasts(predictor, frame), predictor.quantiles)
        point = evaluation["point"]
        rows.append(
            {
                "name": model_type.upper() if model_type != "linear" else "Linear",
                "parameters": count_parameters(predictor.model),
                "epochs": len(history),
                "train_seconds": round(seconds, 1),
                "pinball": evaluation["pinball"]["model"],
                **interval_summary(evaluation, "rolling"),
                "mae_skill": 1 - point["model"]["mae"] / point["naive"]["mae"],
                "directional_accuracy": evaluation["direction"]["accuracy"],
                "brier": evaluation["direction"]["brier"],
            }
        )
    for method, name in BASELINE_NAMES.items():
        rows.append({"name": name, "pinball": evaluation["pinball"][method], **interval_summary(evaluation, method)})
    return rows


def format_leaderboard(rows: list, quantiles) -> str:
    level = level_name(interval_levels(quantiles)[0][0])

    def cell(row, key, template):
        return template.format(row[key]) if row.get(key) is not None else "–"

    lines = [
        f"| Model | Parameters | Pinball (bps) | {level}% coverage | {level}% width | Interval score | "
        "MAE skill vs naive | Direction | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["pinball"]):
        lines.append(
            f"| {row['name']} | {cell(row, 'parameters', '{:,}')} | {row['pinball'] * 1e4:.3f} | "
            f"{row['coverage']:.1%} | {row['width'] * 100:.3f}% | {row['interval_score'] * 100:.3f} | "
            f"{cell(row, 'mae_skill', '{:+.2%}')} | {cell(row, 'directional_accuracy', '{:.2%}')} | "
            f"{cell(row, 'brier', '{:.4f}')} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_TYPES), default=DEFAULT_MODELS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON config file")
    parser.add_argument("--data-path", help="OHLCV CSV (overrides the config)")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs per model")
    parser.add_argument(
        "--output-dir", help="Where to write checkpoints and the leaderboard (default: reports/benchmark)"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    config = load_config(args.config, data_path=cli_path(args.data_path), num_epochs=args.epochs)
    output_dir = Path(cli_path(args.output_dir) or resolve_path("reports/benchmark"))
    rows = run_benchmark(config, args.models, args.device, output_dir)
    leaderboard = format_leaderboard(rows, config["quantiles"])
    print(leaderboard)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.md").write_text(leaderboard + "\n")
    (output_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Wrote %s", output_dir / "leaderboard.md")


if __name__ == "__main__":
    main()
