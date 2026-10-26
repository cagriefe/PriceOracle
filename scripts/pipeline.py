"""Run everything in one go: refresh candles, train (or reuse) the model, and build the dashboard and metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from models.predictor import PricePredictor
from scripts.fetch_data import update_csv
from scripts.report import build_report
from scripts.train import train
from utils.helpers import (
    DEFAULT_CONFIG_PATH,
    cli_path,
    github_annotation,
    github_step_summary,
    load_config,
    resolve_path,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def refresh_data(data_path) -> None:
    try:
        count = update_csv(data_path)
        logger.info("Fetched %d new or refreshed candles", count)
    except Exception as error:  # network problems should not stop the rest of the pipeline
        message = f"Could not refresh candles ({type(error).__name__}: {error}); using the existing dataset."
        logger.warning(message)
        github_annotation("warning", message)


def reusable(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    try:
        PricePredictor.load(model_path)
    except (ValueError, RuntimeError, KeyError) as error:
        logger.info("Retraining: the existing checkpoint cannot be reused (%s)", error)
        return False
    return True


def summary_markdown(summary: dict) -> str:
    live, evaluation = summary["live"], summary["evaluation"]
    intervals = "".join(
        f"| {name}% range | {lower:,.2f} – {upper:,.2f} USDT |\n" for name, (lower, upper) in live["intervals"].items()
    )
    level = next(iter(evaluation["intervals"]))
    rolling = evaluation["intervals"][level]["rolling"]
    return (
        f"### PriceOracle forecast for the {live['target_candle']} UTC candle\n\n"
        "| | |\n| --- | --- |\n"
        f"| Last close | {live['last_close']:,.2f} USDT |\n"
        f"| Median forecast | {live['predicted_close']:,.2f} USDT ({live['predicted_change_pct']:+.3f}%) |\n"
        f"{intervals}"
        f"| Chance of a higher close | {live['probability_up']:.1%} |\n"
        f"| Test {level}% coverage | {rolling['coverage']:.2%} |\n"
        f"| Test directional accuracy | {evaluation['direction']['accuracy']:.2%} |\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON config file")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not download new candles")
    parser.add_argument("--reuse-model", action="store_true", help="Skip training if a usable checkpoint exists")
    parser.add_argument("--output-dir", help="Directory for index.html and metrics.json (default: reports)")
    parser.add_argument("--offline", action="store_true", help="Embed plotly.js in the dashboard")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not args.skip_fetch:
        refresh_data(resolve_path(config["data_path"]))

    model_path = resolve_path(config["model_path"])
    if args.reuse_model and reusable(model_path):
        logger.info("Reusing %s", config["model_path"])
    else:
        train(config, device=args.device)

    output_dir = Path(cli_path(args.output_dir) or resolve_path("reports"))
    summary = build_report(model_path, output=output_dir / "index.html", device=args.device, offline=args.offline)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    logger.info("Wrote %s and %s", output_dir / "index.html", output_dir / "metrics.json")

    live = summary["live"]
    github_annotation(
        "notice",
        f"Forecast for {live['target_candle']}: {live['predicted_close']:,.2f} USDT "
        f"({live['predicted_change_pct']:+.3f}%), P(up) {live['probability_up']:.1%}",
    )
    github_step_summary(summary_markdown(summary))


if __name__ == "__main__":
    main()
