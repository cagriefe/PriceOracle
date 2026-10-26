import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import (
    FEATURE_CLIP,
    SequenceDataset,
    chronological_split,
    labelled_end_indices,
    prepare_training_data,
    split_by_boundaries,
)
from data.preprocess import FEATURE_COLUMNS, build_frame
from tests.synthetic import make_ohlcv

SEQ_LENGTH = 16


def prepare(ohlcv, horizon=1):
    frame = build_frame(ohlcv, horizon=horizon)
    return frame, prepare_training_data(frame, SEQ_LENGTH, val_ratio=0.15, test_ratio=0.15, horizon=horizon)


def test_sequence_dataset_window_ends_at_the_target_row():
    features = np.arange(20, dtype=float).reshape(10, 2)
    targets = np.arange(10, dtype=float) * 10
    dataset = SequenceDataset(features, targets, end_indices=[2, 5, 9], seq_length=3)
    window, target = dataset[1]
    assert len(dataset) == 3
    torch.testing.assert_close(window, torch.tensor(features[3:6], dtype=torch.float32))
    assert target.item() == 50.0


def test_sequence_dataset_requires_full_history():
    with pytest.raises(ValueError):
        SequenceDataset(np.zeros((10, 2)), np.zeros(10), end_indices=[1], seq_length=3)


def test_chronological_split_is_ordered_and_disjoint():
    train, val, test = chronological_split(np.arange(100), val_ratio=0.15, test_ratio=0.15)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert train.max() < val.min() and val.max() < test.min()


def test_chronological_split_purges_overlapping_targets():
    train, val, test = chronological_split(np.arange(100), val_ratio=0.15, test_ratio=0.15, gap=3)
    assert val.min() - train.max() == 4
    assert test.min() - val.max() == 4


def test_chronological_split_needs_enough_samples():
    with pytest.raises(ValueError):
        chronological_split(np.arange(5), val_ratio=0.15, test_ratio=0.15)


def test_labelled_end_indices_need_full_window_and_known_target(ohlcv):
    frame = build_frame(ohlcv, horizon=2)
    indices = labelled_end_indices(frame, seq_length=10)
    assert indices[0] == 9
    assert indices[-1] == len(frame) - 3


def test_scalers_are_fitted_on_training_rows_only(ohlcv):
    frame, data = prepare(ohlcv)
    train_end = data.splits["train"][-1]
    np.testing.assert_allclose(
        data.feature_scaler.mean, frame[FEATURE_COLUMNS].to_numpy()[: train_end + 1].mean(axis=0)
    )
    np.testing.assert_allclose(data.target_scaler.mean, frame["target"].to_numpy()[data.splits["train"]].mean())

    # Rewriting the held-out period must not change anything the model sees in training.
    shocked = ohlcv.copy()
    shocked.iloc[-80:] *= 3
    _, shocked_data = prepare(shocked)
    np.testing.assert_allclose(shocked_data.feature_scaler.mean, data.feature_scaler.mean)
    np.testing.assert_allclose(shocked_data.target_scaler.scale, data.target_scaler.scale)
    np.testing.assert_allclose(shocked_data.features[: train_end + 1], data.features[: train_end + 1])


def test_scaled_features_are_clipped(ohlcv):
    shocked = ohlcv.copy()
    shocked.iloc[-50:] *= 5
    _, data = prepare(shocked)
    assert np.abs(data.features).max() <= FEATURE_CLIP


def test_boundaries_reproduce_the_training_split(ohlcv):
    frame, data = prepare(ohlcv, horizon=3)
    splits = split_by_boundaries(frame, labelled_end_indices(frame, SEQ_LENGTH), data.boundaries)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(splits[name], data.splits[name])


def test_new_candles_extend_only_the_test_split(ohlcv):
    frame, data = prepare(ohlcv)
    extra = make_ohlcv(n_rows=100, seed=1, start=ohlcv.index[-1] + pd.Timedelta(hours=1))
    longer = build_frame(pd.concat([ohlcv, extra]))
    splits = split_by_boundaries(longer, labelled_end_indices(longer, SEQ_LENGTH), data.boundaries)
    np.testing.assert_array_equal(splits["train"], data.splits["train"])
    np.testing.assert_array_equal(splits["val"], data.splits["val"])
    assert len(splits["test"]) == len(data.splits["test"]) + 100
