import numpy as np
import pytest

from data.transforms import StandardScaler


def test_round_trip_and_state_dict():
    values = np.random.default_rng(0).normal(5, 2, size=(100, 3))
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), values)
    np.testing.assert_allclose(StandardScaler.from_state_dict(scaler.state_dict()).transform(values), scaled)


def test_constant_columns_and_1d_targets():
    assert np.all(StandardScaler().fit(np.ones((10, 2))).scale == 1)
    target_scaler = StandardScaler().fit(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(target_scaler.transform(np.array([2.0])), [0.0])


def test_clip_is_applied_and_persisted():
    scaler = StandardScaler(clip=2.0).fit(np.array([[0.0], [1.0], [2.0]]))
    assert scaler.transform(np.array([[100.0]]))[0, 0] == 2.0
    assert StandardScaler.from_state_dict(scaler.state_dict()).clip == 2.0


def test_unfitted_scaler_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform([1.0])
