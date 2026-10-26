import numpy as np
import pytest

from utils.metrics import brier_score, interval_metrics, pinball_score, price_metrics, up_move_ratio


def test_price_metrics_and_directional_accuracy():
    reference = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 99.0, 102.0, 100.0])
    predicted = np.array([100.5, 99.5, 99.0, 100.0])
    metrics = price_metrics(actual, predicted, reference)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(9.5 / 4))
    assert metrics["mape"] == pytest.approx(np.mean([0.5 / 101, 0.5 / 99, 3 / 102, 0]) * 100)
    # The flat last candle is ignored; the third forecast points the wrong way.
    assert metrics["directional_accuracy"] == pytest.approx(2 / 3)
    assert up_move_ratio(actual, reference) == pytest.approx(2 / 3)


def test_naive_forecast_has_zero_error_on_a_flat_market():
    flat = np.full(5, 100.0)
    metrics = price_metrics(flat, flat, flat)
    assert metrics["mae"] == metrics["rmse"] == 0.0
    assert np.isnan(metrics["directional_accuracy"])


def test_interval_metrics_penalize_misses():
    actual = np.array([0.0, 2.0, -3.0, 0.5])
    lower, upper = np.full(4, -1.0), np.full(4, 1.0)
    metrics = interval_metrics(actual, lower, upper, level=0.5)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["width"] == pytest.approx(2.0)
    # Width 2 everywhere, plus 2 / alpha = 4 times the miss distances (1 and 2).
    assert metrics["interval_score"] == pytest.approx(2.0 + 4 * (1 + 2) / 4)


def test_pinball_score_is_zero_for_perfect_quantiles_and_symmetric_for_the_median():
    actual = np.array([1.0, 2.0])
    assert pinball_score(actual, np.column_stack([actual, actual]), [0.1, 0.9]) == 0.0
    assert pinball_score(actual, np.array([[2.0], [1.0]]), [0.5]) == pytest.approx(0.5)


def test_brier_score():
    assert brier_score([0.5, 0.5], [0.01, -0.01]) == pytest.approx(0.25)
    assert brier_score([1.0, 0.0], [0.01, -0.01]) == 0.0
