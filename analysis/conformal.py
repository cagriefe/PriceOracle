"""Conformalized quantile regression (CQR) with width-normalized scores.

The network's raw quantiles are rarely perfectly calibrated, especially when volatility regimes shift.
CQR measures how far realised returns fall outside an interval, relative to the interval's width, and
widens or shrinks future intervals by the score quantile that gives the desired coverage
(Romano et al., 2019). Normalizing by width keeps the correction proportional across calm and wild markets.

Two variants are provided:

* static: scores from the validation period, fixed once (split conformal);
* rolling: scores from the most recent ``window`` forecasts whose outcomes are already known, which adapts
  to regime changes and only ever uses past information.
"""

import numpy as np
import pandas as pd

# Never shrink an interval by more than 90% of its width, so it cannot invert.
MIN_ADJUSTMENT = -0.45


def conformity_scores(actual, lower, upper) -> np.ndarray:
    """Signed distance outside the interval in units of its width (negative = inside). NaN where unknown."""
    actual, lower, upper = (np.asarray(values, dtype=np.float64) for values in (actual, lower, upper))
    return np.maximum(lower - actual, actual - upper) / (upper - lower)


def conformal_level(n_scores: int, alpha: float) -> float:
    """Finite-sample corrected quantile level that guarantees ``1 - alpha`` coverage."""
    return min(1.0, np.ceil((n_scores + 1) * (1 - alpha)) / n_scores)


def static_adjustment(scores, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return 0.0
    return max(MIN_ADJUSTMENT, float(np.quantile(scores, conformal_level(len(scores), alpha), method="higher")))


def rolling_adjustment(scores, alpha: float, window: int, delay: int) -> np.ndarray:
    """Adjustment for each forecast from the previous ``window`` scores known ``delay`` steps earlier.

    The forecast made at step ``t`` only learns its outcome at ``t + delay``, so position ``t`` uses the
    scores of forecasts ``t - delay - window + 1 ... t - delay``. Positions without a full window are NaN.
    """
    rolled = (
        pd.Series(np.asarray(scores, dtype=np.float64))
        .rolling(window, min_periods=window)
        .quantile(conformal_level(window, alpha), interpolation="higher")
    )
    return np.maximum(rolled.shift(delay).to_numpy(), MIN_ADJUSTMENT)


def apply_adjustment(lower, upper, adjustment):
    width = np.asarray(upper) - np.asarray(lower)
    return np.asarray(lower) - adjustment * width, np.asarray(upper) + adjustment * width
