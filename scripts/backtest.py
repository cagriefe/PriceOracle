"""Backtest trading rules on the out-of-sample forecasts, net of exchange fees and slippage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from analysis.backtest import Costs, run_backtests
from analysis.forecasts import load_forecasts
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, load_config


def backtest(model_path, data_path=None, device="auto", fee_rate=None, slippage=None, plot_path=None):
    """Return ``(summary, backtests)``; ``backtests`` holds the per-candle results of every strategy."""
    predictor, _, table = load_forecasts(model_path, data_path, device)
    costs = Costs(
        fee_rate=predictor.config.get("fee_rate", 0.001) if fee_rate is None else fee_rate,
        slippage=predictor.config.get("slippage", 0.0002) if slippage is None else slippage,
    )
    backtests = run_backtests(table, predictor.quantiles, costs)
    first = next(iter(backtests.values()))["result"]
    summary = {
        "start": f"{first.index[0]:%Y-%m-%d %H:%M}",
        "end": f"{first.index[-1]:%Y-%m-%d %H:%M}",
        "fee_rate": costs.fee_rate,
        "slippage": costs.slippage,
        "strategies": {name: backtest["metrics"] for name, backtest in backtests.items()},
    }
    if plot_path:
        plot_equity(backtests, plot_path)
    return summary, backtests


def format_summary(summary: dict) -> str:
    lines = [
        f"Backtest {summary['start']} -> {summary['end']}, costs per trade: fee {summary['fee_rate'] * 100:.2f}% "
        f"+ slippage {summary['slippage'] * 100:.2f}%",
        "",
        f"{'Strategy':<28}{'Return':>9}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}{'Max DD':>9}"
        f"{'Exposure':>10}{'Trades':>8}{'Fees':>8}",
    ]
    for name, m in summary["strategies"].items():
        lines.append(
            f"{name:<28}{m['total_return'] * 100:>8.1f}%{m['cagr'] * 100:>8.1f}%{m['sharpe']:>8.2f}"
            f"{m['sortino']:>9.2f}{m['max_drawdown'] * 100:>8.1f}%{m['exposure'] * 100:>9.1f}%{m['trades']:>8,}"
            f"{m['costs_paid'] * 100:>7.1f}%"
        )
    lines.append("")
    lines.append("Fees: fees and slippage paid, as a share of the starting capital.")
    return "\n".join(lines)


def plot_equity(backtests: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for name, backtest in backtests.items():
        equity = backtest["result"]["equity"]
        drawdown = equity / equity.clip(lower=1.0).cummax() - 1
        ax_equity.plot(equity.index, equity.to_numpy(), linewidth=1.2, label=name)
        ax_drawdown.plot(drawdown.index, drawdown.to_numpy() * 100, linewidth=1)
    ax_equity.set_yscale("log")
    ax_equity.set_ylabel("Growth of 1 USDT (log scale)")
    ax_equity.set_title("Out-of-sample backtest, net of fees and slippage")
    ax_equity.legend(loc="upper left")
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.yaxis.set_major_formatter(PercentFormatter())
    locator = mdates.AutoDateLocator()
    ax_drawdown.xaxis.set_major_locator(locator)
    ax_drawdown.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in (ax_equity, ax_drawdown):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to use (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--fee", type=float, help="Fee per side as a fraction, e.g. 0.001 for 0.1%%")
    parser.add_argument("--slippage", type=float, help="Slippage per side as a fraction")
    parser.add_argument("--plot", help="Save the equity and drawdown curves to this PNG")
    parser.add_argument("--metrics-out", help="Write the strategy metrics to this JSON file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    model_path = cli_path(args.model_path) or load_config(DEFAULT_CONFIG_PATH)["model_path"]
    summary, _ = backtest(
        model_path, cli_path(args.data_path), args.device, args.fee, args.slippage, cli_path(args.plot)
    )
    print(format_summary(summary))
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
