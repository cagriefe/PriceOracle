"""Build a self-contained interactive HTML dashboard: live forecast, calibration, backtests and feature importance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import html
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from analysis.backtest import Costs, run_backtests
from analysis.evaluation import INTERVAL_METHODS, evaluate_forecasts, interval_bounds
from analysis.forecasts import interval_levels, level_name, live_forecast, load_forecasts, quantile_column
from analysis.importance import permutation_importance
from models import count_parameters
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, load_config, resolve_path

ORANGE = "#f7931a"
INK = "#1e293b"
METHOD_STYLE = {
    "rolling": ("Model + rolling conformal", ORANGE),
    "static": ("Model + static conformal", "#fb7185"),
    "raw": ("Model, raw quantiles", "#a78bfa"),
    "ewma": ("EWMA volatility", "#0ea5e9"),
    "hist": ("Historical simulation", "#64748b"),
}
STRATEGY_COLORS = ["#334155", "#0ea5e9", "#a78bfa", ORANGE, "#10b981", "#ef4444"]
PLOT_CONFIG = {"displaylogo": False, "responsive": True}
MAX_POINTS = 1200

CSS = """
:root { --bg:#f5f6fa; --card:#fff; --ink:#1e293b; --muted:#64748b; --accent:#f7931a; --line:#e2e8f0;
        --good:#15803d; --bad:#b91c1c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
header { background:#0f172a; color:#f8fafc; padding:28px 24px 76px; }
.wrap { max-width:1200px; margin:0 auto; }
header h1 { margin:0; font-size:28px; letter-spacing:-.02em; }
header h1 span { color:var(--accent); }
header p { margin:6px 0 0; color:#94a3b8; font-size:14px; }
main { max-width:1200px; margin:-52px auto 32px; padding:0 24px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:16px; }
.kpi, .card { background:var(--card); border-radius:14px;
              box-shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06); }
.kpi { padding:18px 20px; }
.kpi .label { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
.kpi .value { font-size:26px; font-weight:700; margin-top:2px; font-variant-numeric:tabular-nums; }
.kpi .sub { font-size:13px; color:var(--muted); }
.up { color:var(--good); } .down { color:var(--bad); }
.card { padding:22px; margin-top:20px; }
.card h2 { margin:0 0 4px; font-size:18px; }
.lead { margin:0 0 14px; color:var(--muted); font-size:14px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr)); gap:20px; }
table { width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }
tr.highlight td { background:#fff7ed; font-weight:600; }
.table-scroll { overflow-x:auto; }
ul { margin:8px 0 0; padding-left:20px; }
li { margin:4px 0; }
footer { max-width:1200px; margin:0 auto 40px; padding:0 24px; color:var(--muted); font-size:13px; }
"""


def downsample(frame):
    return frame.iloc[:: max(1, len(frame) // MAX_POINTS)]


def style(fig, height):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal"),
        hovermode="x unified",
        font=dict(family="-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", color=INK),
    )
    return fig


def fan_chart(table, quantiles, hours=168):
    rows = table[table["split"].isin(["test", "live"])].tail(hours + 1)
    known = rows[rows["split"] != "live"]
    x, close = rows["target_time"], rows["close"]
    fig = go.Figure()
    for i, (level, _, _) in enumerate(interval_levels(quantiles)):
        name = level_name(level)
        lower, upper = close * np.exp(rows[f"lo{name}"]), close * np.exp(rows[f"hi{name}"])
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(247,147,26,{0.16 + 0.16 * i:.2f})",
                name=f"{name}% interval",
                customdata=upper,
                hovertemplate=f"{name}%: %{{y:,.0f}} – %{{customdata:,.0f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=close * np.exp(rows[quantile_column(0.5)]),
            mode="lines",
            name="Median forecast",
            line=dict(color=ORANGE, dash="dash", width=1.6),
            hovertemplate="Median: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=known["target_time"],
            y=known["close"] * np.exp(known["actual"]),
            mode="lines+markers",
            name="Actual close",
            line=dict(color=INK, width=1.6),
            marker=dict(size=4),
            hovertemplate="Actual: %{y:,.2f}<extra></extra>",
        )
    )
    live = live_forecast(table, quantiles)
    lower, upper = next(reversed(live["intervals"].values()))
    fig.add_trace(
        go.Scatter(
            x=[live["target_candle"]],
            y=[live["predicted_close"]],
            mode="markers",
            name="Next forecast",
            marker=dict(size=13, color=ORANGE, symbol="diamond", line=dict(color="white", width=2)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[upper - live["predicted_close"]],
                arrayminus=[live["predicted_close"] - lower],
                color=ORANGE,
                thickness=2,
                width=8,
            ),
            hovertemplate="Next: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=",.0f", title="USDT")
    return style(fig, 470)


def calibration_chart(table, quantiles, window=720):
    level, lower_q, upper_q = interval_levels(quantiles)[0]
    rows = table[table["split"] == "test"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    for method in INTERVAL_METHODS:
        label, color = METHOD_STYLE[method]
        lower, upper = interval_bounds(rows, method, level, lower_q, upper_q)
        hits = ((rows["actual"] >= lower) & (rows["actual"] <= upper)).astype(float)
        coverage = downsample(hits.rolling(window, min_periods=window // 4).mean().to_frame("value"))
        fig.add_trace(
            go.Scatter(
                x=rows.loc[coverage.index, "target_time"],
                y=coverage["value"],
                name=label,
                legendgroup=method,
                line=dict(color=color, width=2.2 if method == "rolling" else 1.3),
                hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if method in ("rolling", "ewma"):
            width = downsample(((np.exp(upper) - np.exp(lower)) * 100).rolling(24).mean().to_frame("value"))
            fig.add_trace(
                go.Scatter(
                    x=rows.loc[width.index, "target_time"],
                    y=width["value"],
                    name=label,
                    legendgroup=method,
                    showlegend=False,
                    line=dict(color=color, width=1.3),
                    hovertemplate=f"{label} width: %{{y:.2f}}%<extra></extra>",
                ),
                row=2,
                col=1,
            )
    fig.add_hline(y=level, line=dict(color="#16a34a", dash="dot"), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title=f"{level:.0%} coverage, 30 days", row=1, col=1)
    fig.update_yaxes(title="Width, % of price", row=2, col=1)
    return style(fig, 520)


def log_ticks(low, high):
    candidates = [base * 10.0**exponent for exponent in range(-4, 3) for base in (1, 2, 5)]
    inside = [value for value in candidates if low * 0.9 <= value <= high * 1.1]
    return inside, [f"{value:g}" for value in inside]


def equity_chart(backtests):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    lowest = min(backtest["result"]["equity"].min() for backtest in backtests.values())
    highest = max(backtest["result"]["equity"].max() for backtest in backtests.values())
    for color, (name, backtest) in zip(STRATEGY_COLORS, backtests.items(), strict=False):
        result = downsample(backtest["result"])
        equity = result["equity"]
        drawdown = backtest["result"]["equity"] / backtest["result"]["equity"].clip(lower=1.0).cummax() - 1
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                name=name,
                legendgroup=name,
                line=dict(color=color, width=1.6),
                hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=drawdown.loc[equity.index],
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=color, width=1),
                hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    tick_values, tick_text = log_ticks(lowest, highest)
    fig.update_yaxes(type="log", title="Growth of 1 USDT", tickvals=tick_values, ticktext=tick_text, row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title="Drawdown", row=2, col=1)
    return style(fig, 520)


def importance_chart(importance):
    names = list(importance)[::-1]
    values = [importance[name] for name in names]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[ORANGE if v > 0 else "#cbd5e1" for v in values],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="Increase in pinball loss when shuffled (%)")
    fig.update_layout(hovermode="closest")
    return style(fig, 340)


def figure_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def table_html(headers, rows, highlight=None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        attributes = ' class="highlight"' if i == highlight else ""
        body += f"<tr{attributes}>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def kpi(label, value, sub="", css_class="") -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div><div class="value {css_class}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def build_report(
    model_path, data_path=None, output="reports/index.html", device="auto", offline=False, importance=True
) -> dict:
    """Write the dashboard to ``output`` and return the numbers it shows."""
    predictor, frame, table = load_forecasts(model_path, data_path, device)
    quantiles = predictor.quantiles
    evaluation = evaluate_forecasts(table, quantiles, "test")
    costs = Costs(predictor.config.get("fee_rate", 0.001), predictor.config.get("slippage", 0.0002))
    backtests = run_backtests(table, quantiles, costs)
    live = live_forecast(table, quantiles)
    feature_importance = (
        permutation_importance(predictor, frame, predictor.split_indices(frame)["val"]) if importance else {}
    )

    point, direction = evaluation["point"], evaluation["direction"]
    widest, narrowest = level_name(interval_levels(quantiles)[0][0]), level_name(interval_levels(quantiles)[-1][0])
    change_class = "up" if live["predicted_change_pct"] >= 0 else "down"
    kpis = "".join(
        [
            kpi("Last close", f"{live['last_close']:,.2f}", f"{live['last_candle']:%Y-%m-%d %H:%M} UTC candle"),
            kpi(
                "Median forecast",
                f"{live['predicted_close']:,.2f}",
                f'<span class="{change_class}">{live["predicted_change_pct"]:+.3f}%</span> · '
                f"{live['target_candle']:%Y-%m-%d %H:%M} UTC candle",
            ),
            kpi(
                f"{widest}% range",
                "{:,.0f} – {:,.0f}".format(*live["intervals"][widest]),
                "{}% range: {:,.0f} – {:,.0f}".format(narrowest, *live["intervals"][narrowest]),
            ),
            kpi(
                "Chance of a higher close",
                f"{live['probability_up'] * 100:.1f}%",
                "from the forecast quantiles, recalibrated on validation data",
            ),
        ]
    )

    risk_rows = []
    for method in INTERVAL_METHODS:
        row = [METHOD_STYLE[method][0]]
        for name in evaluation["intervals"]:
            metrics = evaluation["intervals"][name][method]
            row += (
                [
                    f"{metrics['coverage']:.1%}",
                    f"{metrics['width'] * 100:.3f}%",
                    f"{metrics['interval_score'] * 100:.3f}",
                ]
                if metrics
                else ["–", "–", "–"]
            )
        risk_rows.append(row)
    risk_headers = ["Method"] + [
        f"{name}% {column}" for name in evaluation["intervals"] for column in ("cover", "width", "score")
    ]
    pinball = evaluation["pinball"]
    point_rows = [
        [
            predictor.config["model_type"].upper() + " median",
            f"{point['model']['mae']:,.2f}",
            f"{point['model']['rmse']:,.2f}",
            f"{point['model']['mape']:.3f}%",
        ],
        [
            "Naive (last close)",
            f"{point['naive']['mae']:,.2f}",
            f"{point['naive']['rmse']:,.2f}",
            f"{point['naive']['mape']:.3f}%",
        ],
    ]
    backtest_rows = [
        [
            html.escape(name),
            f"{m['total_return']:.1%}",
            f"{m['cagr']:.1%}",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['max_drawdown']:.1%}",
            f"{m['exposure']:.0%}",
            f"{m['trades']:,}",
            f"{m['costs_paid']:.1%}",
        ]
        for name, m in ((name, backtest["metrics"]) for name, backtest in backtests.items())
    ]

    boundaries = predictor.boundaries
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    if offline:
        plotly_js = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_js = (
            f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>'
        )

    sections = [
        '<section class="card"><h2>Forecast with calibrated intervals</h2>'
        '<p class="lead">Each point is an out-of-sample forecast made one candle earlier. Shaded bands are '
        "conformal prediction intervals; the diamond is the forecast for the next candle.</p>"
        + figure_html(fan_chart(table, quantiles))
        + "</section>",
        '<section class="card"><h2>Risk intervals vs. classic baselines</h2>'
        f'<p class="lead">Test period {evaluation["start"]} → {evaluation["end"]} UTC '
        f"({evaluation['samples']:,} forecasts the model never saw). Coverage should match the nominal level; "
        "among calibrated methods, narrower is better. The interval score rewards both (lower is better). "
        f"Pinball loss: model <b>{pinball['model'] * 1e4:.3f}</b> bps, EWMA {pinball['ewma'] * 1e4:.3f}, "
        f"historical {pinball['hist'] * 1e4:.3f}.</p>"
        + table_html(risk_headers, risk_rows, highlight=INTERVAL_METHODS.index("rolling"))
        + "</section>",
        '<section class="card"><h2>Calibration over time</h2>'
        f'<p class="lead">Rolling 30-day coverage of the {widest}% interval (target: dotted line) and the '
        "interval width. Rolling conformal calibration adapts as volatility regimes change.</p>"
        + figure_html(calibration_chart(table, quantiles))
        + "</section>",
        '<div class="grid2">'
        '<section class="card"><h2>Point forecast and direction</h2>'
        '<p class="lead">The median forecast against simply repeating the last close.</p>'
        + table_html(["Point forecast", "MAE", "RMSE", "MAPE"], point_rows)
        + f'<p class="lead" style="margin-top:14px">Direction: <b>{direction["accuracy"]:.2%}</b> correct '
        f"(always-up {direction['always_up']:.2%}). Brier score of P(up): <b>{direction['brier']:.4f}</b> "
        "(coin flip 0.2500, lower is better).</p></section>"
        + (
            '<section class="card"><h2>What drives the forecast?</h2>'
            '<p class="lead">Permutation importance on the validation period: how much the quantile loss grows '
            "when one feature is shuffled.</p>" + figure_html(importance_chart(feature_importance)) + "</section>"
            if feature_importance
            else ""
        )
        + "</div>",
        '<section class="card"><h2>Backtest, net of costs</h2>'
        f'<p class="lead">Test period only. Every trade pays a {costs.fee_rate:.2%} fee plus {costs.slippage:.2%} '
        "slippage. Positions are set at each close from that hour's forecast and held for one candle. "
        "Fees: total fees and slippage paid, as a share of the starting capital.</p>"
        + figure_html(equity_chart(backtests))
        + table_html(
            ["Strategy", "Return", "CAGR", "Sharpe", "Sortino", "Max DD", "Exposure", "Trades", "Fees"], backtest_rows
        )
        + "</section>",
    ]
    sections.append(
        '<section class="card"><h2>How to read this</h2><ul>'
        "<li>Hourly crypto returns are close to a random walk: the median forecast is about as accurate as "
        "repeating the last close, and the directional edge is small.</li>"
        "<li>Volatility, however, is predictable. The calibrated intervals are the useful output: they say how far "
        "the price is likely to move.</li>"
        "<li>Strategies that trade on the direction lose to fees; risk-based position sizing is the more realistic "
        "use of these forecasts.</li>"
        f"<li>Model: {predictor.config['model_type'].upper()} ({count_parameters(predictor.model):,} parameters), "
        f"{predictor.seq_length}-candle window, trained {predictor.trained_at} on data up to "
        f"{boundaries['train_end']:%Y-%m-%d}, validated through {boundaries['val_end']:%Y-%m-%d}.</li>"
        "</ul></section>"
    )

    body = "".join(sections)
    data_through = f"{frame.index[-1]:%Y-%m-%d %H:%M} UTC"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PriceOracle · BTC/USDT forecast</title>
<style>{CSS}</style>
{plotly_js}
</head>
<body>
<header><div class="wrap"><h1>Price<span>Oracle</span></h1>
<p>Probabilistic hourly BTC/USDT forecasts · generated {generated} · data through {data_through}</p>
</div></header>
<main>
<div class="kpis">{kpis}</div>
{body}
</main>
<footer>Research project, not financial advice. Forecasts are generated automatically and can be wrong.</footer>
</body>
</html>
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {
        "generated": generated,
        "live": {key: str(value) if isinstance(value, pd.Timestamp) else value for key, value in live.items()},
        "evaluation": evaluation,
        "backtest": {name: backtest["metrics"] for name, backtest in backtests.items()},
        "importance": feature_importance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Checkpoint to use (defaults to model_path in the config)")
    parser.add_argument("--data-path", help="OHLCV CSV (defaults to the path stored in the checkpoint)")
    parser.add_argument("--output", help="HTML file to write (default: reports/index.html)")
    parser.add_argument("--offline", action="store_true", help="Embed plotly.js so the page works without internet")
    parser.add_argument("--no-importance", action="store_true", help="Skip the permutation importance (faster)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args(argv)

    model_path = cli_path(args.model_path) or load_config(DEFAULT_CONFIG_PATH)["model_path"]
    output = cli_path(args.output) or resolve_path("reports/index.html")
    build_report(model_path, cli_path(args.data_path), output, args.device, args.offline, not args.no_importance)
    print(f"Wrote dashboard to {output}")


if __name__ == "__main__":
    main()
