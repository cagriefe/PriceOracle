"""Cost-aware backtests of simple trading rules driven by the out-of-sample forecasts.

Positions are decided at the close of candle ``t`` using only the forecast made then, held until the
next close, and every change in position pays exchange fees plus slippage on the traded amount.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from analysis.forecasts import interval_levels, level_name, quantile_column

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class Costs:
    fee_rate: float = 0.001  # Binance spot taker fee per side
    slippage: float = 0.0002

    @property
    def per_unit_turnover(self) -> float:
        return self.fee_rate + self.slippage


def simulate(positions, returns, costs: Costs) -> pd.DataFrame:
    """Hold ``positions[t]`` over the next candle, earning the simple return ``returns[t]``, net of costs."""
    positions = np.asarray(positions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    turnover = np.abs(np.diff(positions, prepend=0.0))
    cost = turnover * costs.per_unit_turnover
    net = positions * returns - cost
    return pd.DataFrame(
        {"position": positions, "turnover": turnover, "cost": cost, "net_return": net, "equity": np.cumprod(1 + net)}
    )


def performance(result: pd.DataFrame, periods_per_year: float = HOURS_PER_YEAR) -> dict:
    net = result["net_return"].to_numpy()
    equity = result["equity"].to_numpy()
    equity_before = np.concatenate([[1.0], equity[:-1]])
    years = len(net) / periods_per_year
    std = net.std()
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
    max_drawdown = float((equity / np.maximum.accumulate(np.maximum(equity, 1.0)) - 1).min())
    cagr = float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 else -1.0
    return {
        "total_return": float(equity[-1] - 1),
        "cagr": cagr,
        "volatility": float(std * np.sqrt(periods_per_year)),
        "sharpe": float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "sortino": float(net.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else float("nan"),
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "exposure": float(np.mean(np.abs(result["position"].to_numpy()) > 1e-12)),
        "average_position": float(result["position"].mean()),
        "trades": int(np.sum(result["turnover"].to_numpy() > 1e-12)),
        # Fees and slippage actually paid, as a fraction of the starting capital.
        "costs_paid": float(np.sum(result["cost"].to_numpy() * equity_before)),
    }


def implied_volatility(lower, upper, level: float) -> np.ndarray:
    """Gaussian-equivalent volatility of a central interval with coverage ``level``."""
    return (np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)) / (
        2 * NormalDist().inv_cdf(0.5 + level / 2)
    )


def directional_positions(median) -> np.ndarray:
    """Long whenever the median forecast is positive, flat otherwise."""
    return np.where(np.asarray(median) > 0, 1.0, 0.0)


def cost_aware_positions(median, threshold: float) -> np.ndarray:
    """Go long only when the expected move beats ``threshold`` (the round-trip cost); exit when it turns negative."""
    positions = np.zeros(len(median))
    current = 0.0
    for i, expected in enumerate(np.asarray(median)):
        if expected > threshold:
            current = 1.0
        elif expected < 0:
            current = 0.0
        positions[i] = current
    return positions


def volatility_target_positions(
    volatility, target, max_position: float = 1.0, rebalance_every: int = 24, band: float = 0.1
):
    """Scale exposure inversely to forecast volatility (Moreira & Muir style volatility management).

    ``target`` may be a scalar or one value per candle. The volatility forecast is averaged over the last
    ``rebalance_every`` candles and the position is only revisited once per ``rebalance_every`` candles, and
    only if it changes by more than ``band``, to keep trading costs low.
    """
    smoothed = pd.Series(np.asarray(volatility, dtype=np.float64)).rolling(rebalance_every, min_periods=1).mean()
    desired = np.clip(np.asarray(target, dtype=np.float64) / smoothed.to_numpy(), 0.0, max_position)
    positions = np.zeros(len(desired))
    current = 0.0
    for i, wanted in enumerate(desired):
        if i % rebalance_every == 0 and abs(wanted - current) > band:
            current = wanted
        positions[i] = current
    return positions


def strategy_positions(table: pd.DataFrame, quantiles, costs: Costs, split: str = "test", window: int = 720):
    """Positions of every built-in strategy over ``split``.

    The volatility target is the median forecast volatility of the trailing ``window`` candles, so it adapts
    to the current regime while only using forecasts available at the time.
    """
    level, _, _ = interval_levels(quantiles)[0]
    name = level_name(level)
    volatility = pd.Series(implied_volatility(table[f"lo{name}"], table[f"hi{name}"], level), index=table.index)
    target = volatility.rolling(window, min_periods=24).median()
    in_split = (table["split"] == split).to_numpy()
    median = table.loc[in_split, quantile_column(0.5)].to_numpy()
    return {
        "Buy & hold": np.ones(int(in_split.sum())),
        "Directional (long/flat)": directional_positions(median),
        "Cost-aware directional": cost_aware_positions(median, threshold=2 * costs.per_unit_turnover),
        "Volatility-targeted hold": volatility_target_positions(volatility[in_split], target[in_split]),
    }


def run_backtests(table: pd.DataFrame, quantiles, costs: Costs | None = None, split: str = "test") -> dict:
    """Simulate every strategy on ``split`` and return ``{name: {"result": DataFrame, "metrics": dict}}``."""
    costs = costs or Costs()
    rows = table[table["split"] == split]
    backtests = {}
    for name, positions in strategy_positions(table, quantiles, costs, split).items():
        result = simulate(positions, rows["next_return"].to_numpy(), costs)
        result.index = rows["target_time"].to_numpy()
        backtests[name] = {"result": result, "metrics": performance(result)}
    return backtests
