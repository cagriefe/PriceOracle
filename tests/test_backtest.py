import numpy as np
import pytest

from analysis.backtest import (
    Costs,
    cost_aware_positions,
    directional_positions,
    implied_volatility,
    performance,
    run_backtests,
    simulate,
    volatility_target_positions,
)
from analysis.forecasts import build_forecasts
from tests.synthetic import make_predictor

FREE = Costs(fee_rate=0.0, slippage=0.0)


def test_buy_and_hold_without_costs_follows_the_price():
    returns = np.array([0.1, -0.2, 0.05])
    result = simulate(np.ones(3), returns, FREE)
    np.testing.assert_allclose(result["equity"].iloc[-1], np.prod(1 + returns))
    assert result["turnover"].tolist() == [1.0, 0.0, 0.0]


def test_costs_are_charged_on_every_change_in_position():
    costs = Costs(fee_rate=0.001, slippage=0.0005)
    result = simulate([1.0, 0.0, 1.0, 1.0], np.zeros(4), costs)
    np.testing.assert_allclose(result["cost"], [0.0015, 0.0015, 0.0015, 0.0])
    assert performance(result)["trades"] == 3


def test_flat_positions_never_trade():
    result = simulate(np.zeros(5), np.full(5, 0.01), Costs())
    metrics = performance(result)
    assert result["equity"].tolist() == [1.0] * 5
    assert metrics["trades"] == 0 and metrics["exposure"] == 0.0
    assert np.isnan(metrics["sharpe"])


def test_performance_metrics_on_a_known_path():
    result = simulate(np.ones(4), np.array([0.1, -0.2, 0.1, 0.0]), FREE)
    metrics = performance(result, periods_per_year=4)
    assert metrics["total_return"] == pytest.approx(1.1 * 0.8 * 1.1 - 1)
    assert metrics["max_drawdown"] == pytest.approx(-0.2)
    assert metrics["cagr"] == pytest.approx(metrics["total_return"])


def test_fees_paid_are_measured_against_the_starting_capital():
    costs = Costs(fee_rate=0.01, slippage=0.0)
    result = simulate([1.0, 0.0], [-0.5, 0.0], costs)
    # Entering costs 1% of 1.0; exiting costs 1% of the equity left after the first candle.
    assert performance(result)["costs_paid"] == pytest.approx(0.01 + 0.01 * (1 - 0.5 - 0.01))


def test_directional_and_cost_aware_rules():
    median = np.array([0.001, -0.001, 0.004, 0.001, -0.0001, 0.001])
    assert directional_positions(median).tolist() == [1, 0, 1, 1, 0, 1]
    assert cost_aware_positions(median, threshold=0.003).tolist() == [0, 0, 1, 1, 0, 0]


def test_volatility_targeting_caps_exposure_and_rebalances_rarely():
    volatility = np.concatenate([np.full(48, 0.01), np.full(48, 0.04)])
    positions = volatility_target_positions(volatility, target=0.02, rebalance_every=24, band=0.1)
    assert positions[:24].tolist() == [1.0] * 24
    assert positions.max() <= 1.0
    assert len(np.unique(positions)) <= 4
    assert positions[-1] == pytest.approx(0.5)


def test_implied_volatility_inverts_a_gaussian_interval():
    assert implied_volatility([-1.6448536], [1.6448536], 0.9)[0] == pytest.approx(1.0, rel=1e-6)


def test_run_backtests_returns_every_strategy(ohlcv):
    predictor, frame, _ = make_predictor(ohlcv)
    table = build_forecasts(predictor, frame)
    backtests = run_backtests(table, predictor.quantiles)
    assert list(backtests) == [
        "Buy & hold",
        "Directional (long/flat)",
        "Cost-aware directional",
        "Volatility-targeted hold",
    ]
    test_rows = (table["split"] == "test").sum()
    for backtest in backtests.values():
        assert len(backtest["result"]) == test_rows
        assert np.isfinite(backtest["metrics"]["total_return"])
