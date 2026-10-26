import numpy as np

from analysis.importance import permutation_importance
from data.preprocess import FEATURE_COLUMNS
from tests.synthetic import make_predictor


def test_permutation_importance_covers_every_feature(ohlcv):
    predictor, frame, data = make_predictor(ohlcv)
    importance = permutation_importance(predictor, frame, data.splits["val"], repeats=2)
    assert set(importance) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(value) for value in importance.values())
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


def test_shuffling_a_constant_feature_changes_nothing(ohlcv):
    predictor, frame, data = make_predictor(ohlcv)
    frame = frame.copy()
    frame["rsi"] = 0.5
    importance = permutation_importance(predictor, frame, data.splits["val"], repeats=1)
    assert importance["rsi"] == 0.0
