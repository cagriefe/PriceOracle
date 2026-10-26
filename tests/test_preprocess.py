import numpy as np
import pandas as pd
import pytest

from data.preprocess import FEATURE_COLUMNS, build_frame, compute_features, load_ohlcv
from tests.synthetic import write_ohlcv_csv


def test_load_ohlcv_sorts_and_deduplicates(tmp_path, ohlcv):
    messy = pd.concat([ohlcv.iloc[::-1], ohlcv.iloc[[5]]])
    loaded = load_ohlcv(write_ohlcv_csv(messy, tmp_path / "messy.csv"))
    assert loaded.index.is_monotonic_increasing
    assert not loaded.index.duplicated().any()
    assert len(loaded) == len(ohlcv)


def test_features_never_look_ahead(ohlcv):
    cut = 300
    shocked = ohlcv.copy()
    shocked.iloc[cut:] *= 1.5
    pd.testing.assert_frame_equal(compute_features(ohlcv).iloc[:cut], compute_features(shocked).iloc[:cut])


def test_warmup_rows_are_dropped_not_backfilled(ohlcv):
    frame = build_frame(ohlcv)
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert frame.index[0] > ohlcv.index[0]
    assert frame.index[-1] == ohlcv.index[-1]


@pytest.mark.parametrize("horizon", [1, 3])
def test_target_is_the_future_log_return(ohlcv, horizon):
    frame = build_frame(ohlcv, horizon=horizon)
    close = ohlcv["close"]
    expected = np.log(close.shift(-horizon) / close).loc[frame.index]
    np.testing.assert_allclose(frame["target"], expected)
    assert frame["target"].iloc[-horizon:].isna().all()
    assert frame["target"].iloc[:-horizon].notna().all()


def test_invalid_horizon_is_rejected(ohlcv):
    with pytest.raises(ValueError):
        build_frame(ohlcv, horizon=0)
