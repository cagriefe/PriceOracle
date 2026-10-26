import numpy as np
import pytest

from analysis.conformal import (
    MIN_ADJUSTMENT,
    apply_adjustment,
    conformal_level,
    conformity_scores,
    rolling_adjustment,
    static_adjustment,
)


def test_conformity_scores_are_signed_and_width_normalized():
    scores = conformity_scores(actual=[0.0, 2.0, -3.0], lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(scores, [-0.5, 0.5, 1.0])


def test_conformal_level_has_finite_sample_correction():
    assert conformal_level(9, alpha=0.1) == 1.0
    assert conformal_level(99, alpha=0.1) == pytest.approx(90 / 99)


def test_static_adjustment_restores_coverage_of_too_narrow_intervals():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 2.0, 4000)
    actual = rng.normal(0, sigma)
    lower, upper = -0.5 * sigma, 0.5 * sigma  # far too narrow: about 38% coverage
    calibration, test = slice(0, 2000), slice(2000, None)
    adjustment = static_adjustment(conformity_scores(actual[calibration], lower[calibration], upper[calibration]), 0.1)
    new_lower, new_upper = apply_adjustment(lower[test], upper[test], adjustment)
    coverage = np.mean((actual[test] >= new_lower) & (actual[test] <= new_upper))
    assert adjustment > 0
    assert coverage == pytest.approx(0.9, abs=0.02)


def test_adjustments_never_invert_intervals():
    scores = np.full(100, -0.5)  # intervals ten times too wide
    assert static_adjustment(scores, 0.1) == MIN_ADJUSTMENT
    lower, upper = apply_adjustment(np.array([-1.0]), np.array([1.0]), MIN_ADJUSTMENT)
    assert lower[0] < upper[0]


def test_static_adjustment_ignores_unknown_outcomes():
    assert static_adjustment([np.nan, np.nan], 0.1) == 0.0
    assert static_adjustment([np.nan, 0.2, 0.2], 0.5) == pytest.approx(0.2)


def test_rolling_adjustment_only_uses_scores_known_at_the_time():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=200)
    base = rolling_adjustment(scores, alpha=0.1, window=20, delay=3)
    changed_scores = scores.copy()
    changed_scores[100] = 50.0
    changed = rolling_adjustment(changed_scores, alpha=0.1, window=20, delay=3)
    np.testing.assert_array_equal(base[:103], changed[:103])
    assert changed[103] != base[103]
    assert np.isnan(base[: 20 + 3 - 1]).all() and not np.isnan(base[20 + 3 - 1])
