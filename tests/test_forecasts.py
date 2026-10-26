import numpy as np
import pandas as pd
import pytest

from analysis.evaluation import INTERVAL_METHODS, evaluate_forecasts
from analysis.forecasts import (
    build_forecasts,
    interval_levels,
    live_forecast,
    probability_up,
    quantile_column,
    shrinkage_factor,
)
from tests.synthetic import make_predictor

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@pytest.fixture
def forecasts(ohlcv):
    predictor, frame, data = make_predictor(ohlcv)
    return predictor, frame, data, build_forecasts(predictor, frame)


def test_interval_levels_pair_symmetric_quantiles():
    assert interval_levels(QUANTILES) == [(0.9, 0.05, 0.95), (0.5, 0.25, 0.75)]
    assert interval_levels([0.1, 0.5]) == []


def test_probability_up_interpolates_the_predictive_cdf():
    rows = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], [-0.5, -0.25, -0.1, -0.05, -0.01]])
    np.testing.assert_allclose(probability_up(rows, QUANTILES), [0.5, 0.95, 0.05])


def test_shrinkage_factor_pulls_overconfident_probabilities_to_half():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.3, 0.7, 5000)
    outcomes = rng.random(5000) < 0.5 + 0.5 * (probabilities - 0.5)  # true signal is half as strong
    assert shrinkage_factor(probabilities, outcomes) == pytest.approx(0.5, abs=0.15)
    assert shrinkage_factor(np.full(10, 0.5), np.ones(10)) == 0.0


def test_table_covers_validation_test_and_live_rows(forecasts):
    predictor, frame, data, table = forecasts
    assert table.index[0] == frame.index[data.splits["val"][0]]
    assert table.index[-1] == frame.index[-1]
    assert (table["split"] == "val").sum() == len(data.splits["val"])
    assert (table["split"] == "test").sum() == len(data.splits["test"])
    assert table["split"].iloc[-1] == "live"
    assert table.loc[table["split"] != "live", "actual"].notna().all()


def test_target_times_point_horizon_candles_ahead(ohlcv):
    predictor, frame, _ = make_predictor(ohlcv, horizon=3)
    table = build_forecasts(predictor, frame)
    known = table[table["split"] != "live"]
    positions = frame.index.get_indexer(known.index) + 3
    np.testing.assert_array_equal(known["target_time"].to_numpy(), frame.index[positions].to_numpy())
    assert (table["split"] == "live").sum() == 3
    assert table["target_time"].iloc[-1] == frame.index[-1] + pd.Timedelta(hours=3)


def test_quantiles_and_intervals_are_ordered(forecasts):
    *_, table = forecasts
    quantile_values = table[[quantile_column(q) for q in QUANTILES]].to_numpy()
    assert np.all(np.diff(quantile_values, axis=1) > 0)
    for name in ("90", "50"):
        assert (table[f"lo{name}"] < table[f"hi{name}"]).all()
        assert (table[f"lo{name}_static"] < table[f"hi{name}_static"]).all()
    assert table["p_up"].between(0.05, 0.95).all()


def test_static_conformal_intervals_cover_the_validation_period(forecasts):
    *_, table = forecasts
    validation = table[table["split"] == "val"]
    for name, level in (("90", 0.9), ("50", 0.5)):
        inside = (validation["actual"] >= validation[f"lo{name}_static"]) & (
            validation["actual"] <= validation[f"hi{name}_static"]
        )
        # Calibrated on these very rows, so coverage reaches the level (up to one boundary sample of rounding).
        assert inside.mean() >= level - 1 / len(validation)


def test_evaluation_reports_every_method(forecasts):
    *_, table = forecasts
    results = evaluate_forecasts(table, QUANTILES)
    assert results["samples"] == (table["split"] == "test").sum()
    assert set(results["intervals"]) == {"90", "50"}
    assert set(results["intervals"]["90"]) == set(INTERVAL_METHODS)
    assert set(results["pinball"]) == {"model", "hist", "ewma"}
    assert 0 <= results["direction"]["brier"] <= 1
    with pytest.raises(ValueError):
        evaluate_forecasts(table, QUANTILES, split="train")


def test_live_forecast_is_in_prices(forecasts):
    *_, table = forecasts
    live = live_forecast(table, QUANTILES)
    assert list(live["intervals"]) == ["50", "90"]
    lower90, upper90 = live["intervals"]["90"]
    assert 0 < lower90 < upper90
    assert live["predicted_close"] == pytest.approx(live["last_close"] * np.exp(table[quantile_column(0.5)].iloc[-1]))
