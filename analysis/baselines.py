"""Classic risk-forecasting baselines that the neural quantiles must beat.

Both only use returns up to the forecast time:

* historical simulation: empirical quantiles of the last ``window`` returns;
* EWMA (RiskMetrics-style): zero-mean Gaussian with exponentially weighted volatility.

Multi-candle horizons scale the one-candle quantiles by ``sqrt(horizon)``.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd


def historical_quantiles(log_returns: pd.Series, quantiles, window: int = 720, horizon: int = 1) -> np.ndarray:
    rolling = log_returns.rolling(window, min_periods=window)
    return np.column_stack([rolling.quantile(q).to_numpy() for q in quantiles]) * np.sqrt(horizon)


def ewma_volatility(log_returns: pd.Series, halflife: float = 24.0) -> np.ndarray:
    return np.sqrt((log_returns**2).ewm(halflife=halflife, adjust=False).mean().to_numpy())


def ewma_quantiles(log_returns: pd.Series, quantiles, halflife: float = 24.0, horizon: int = 1) -> np.ndarray:
    z_scores = np.array([NormalDist().inv_cdf(q) for q in quantiles])
    return ewma_volatility(log_returns, halflife)[:, None] * z_scores[None, :] * np.sqrt(horizon)
