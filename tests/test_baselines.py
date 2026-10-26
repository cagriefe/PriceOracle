from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from analysis.baselines import ewma_quantiles, ewma_volatility, historical_quantiles

QUANTILES = [0.05, 0.5, 0.95]


@pytest.fixture
def returns():
    return pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))


def test_ewma_volatility_matches_the_recursion(returns):
    halflife = 10
    decay = 0.5 ** (1 / halflife)
    variance = returns.iloc[0] ** 2
    expected = [np.sqrt(variance)]
    for value in returns.iloc[1:]:
        variance = decay * variance + (1 - decay) * value**2
        expected.append(np.sqrt(variance))
    np.testing.assert_allclose(ewma_volatility(returns, halflife), expected)


def test_ewma_quantiles_are_gaussian_and_scale_with_horizon(returns):
    one = ewma_quantiles(returns, QUANTILES, halflife=10)
    four = ewma_quantiles(returns, QUANTILES, halflife=10, horizon=4)
    sigma = ewma_volatility(returns, 10)
    np.testing.assert_allclose(one[:, 2], sigma * NormalDist().inv_cdf(0.95))
    np.testing.assert_allclose(one[:, 1], 0.0)
    np.testing.assert_allclose(four, one * 2)


def test_historical_quantiles_use_only_the_trailing_window(returns):
    result = historical_quantiles(returns, QUANTILES, window=50)
    assert np.isnan(result[:49]).all()
    np.testing.assert_allclose(result[120], np.quantile(returns.iloc[71:121], QUANTILES))
    future = returns.copy()
    future.iloc[121:] *= 100
    np.testing.assert_allclose(historical_quantiles(future, QUANTILES, window=50)[:121], result[:121])
