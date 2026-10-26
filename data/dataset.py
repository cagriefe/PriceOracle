"""Sliding windows, chronological splits and leakage-free scaling for the sequence models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.preprocess import FEATURE_COLUMNS
from data.transforms import StandardScaler

SPLITS = ("train", "val", "test")
FEATURE_CLIP = 5.0


class SequenceDataset(Dataset):
    """Windows of ``seq_length`` rows ending at each index in ``end_indices``, sliced lazily.

    Item ``i`` is ``(features[end - seq_length + 1 : end + 1], targets[end])`` with ``end = end_indices[i]``,
    so the target belongs to the window's last row and every window only looks backwards in time.
    """

    def __init__(self, features, targets, end_indices, seq_length: int):
        end_indices = np.asarray(end_indices, dtype=np.int64)
        if len(end_indices) and end_indices.min() < seq_length - 1:
            raise ValueError("Every window needs seq_length rows of history before its end index.")
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        self.end_indices = end_indices
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx):
        end = int(self.end_indices[idx]) + 1
        return self.features[end - self.seq_length : end], self.targets[end - 1]


def labelled_end_indices(frame: pd.DataFrame, seq_length: int) -> np.ndarray:
    """Rows that can end a full window and whose target (future return) is already known."""
    indices = np.arange(len(frame))
    return indices[(indices >= seq_length - 1) & frame["target"].notna().to_numpy()]


def chronological_split(end_indices, val_ratio: float, test_ratio: float, gap: int = 0):
    """Split window end indices into consecutive train/val/test blocks. Nothing is shuffled across time.

    The last ``gap`` samples before each boundary are purged so multi-step targets of one block never
    overlap the price path of the next block.
    """
    end_indices = np.asarray(end_indices)
    n_samples = len(end_indices)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test
    if min(n_train - gap, n_val - gap, n_test) <= 0:
        raise ValueError(f"Not enough samples ({n_samples}) for the requested split.")
    train = end_indices[: n_train - gap]
    val = end_indices[n_train : n_train + n_val - gap]
    test = end_indices[n_train + n_val :]
    return train, val, test


def split_boundaries(frame: pd.DataFrame, train, val, test) -> dict:
    """Timestamps delimiting each block, stored in checkpoints so evaluation reuses the exact split."""
    index = frame.index
    return {
        "train_end": index[train[-1]],
        "val_start": index[val[0]],
        "val_end": index[val[-1]],
        "test_start": index[test[0]],
    }


def split_by_boundaries(frame: pd.DataFrame, end_indices, boundaries: dict) -> dict:
    """Assign window end indices to splits using stored timestamps.

    Candles appended after training simply extend the test block, giving a true out-of-sample check.
    """
    end_indices = np.asarray(end_indices)
    timestamps = frame.index[end_indices]
    return {
        "train": end_indices[timestamps <= boundaries["train_end"]],
        "val": end_indices[(timestamps >= boundaries["val_start"]) & (timestamps <= boundaries["val_end"])],
        "test": end_indices[timestamps >= boundaries["test_start"]],
    }


@dataclass
class PreparedData:
    features: np.ndarray
    targets: np.ndarray
    splits: dict
    boundaries: dict
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    seq_length: int

    def dataset(self, split: str) -> SequenceDataset:
        return SequenceDataset(self.features, self.targets, self.splits[split], self.seq_length)


def prepare_training_data(
    frame: pd.DataFrame, seq_length: int, val_ratio: float, test_ratio: float, horizon: int = 1
) -> PreparedData:
    """Split chronologically, then fit both scalers on the training block only."""
    end_indices = labelled_end_indices(frame, seq_length)
    train, val, test = chronological_split(end_indices, val_ratio, test_ratio, gap=horizon - 1)

    raw_features = frame[FEATURE_COLUMNS].to_numpy()
    raw_targets = frame["target"].to_numpy()
    feature_scaler = StandardScaler(clip=FEATURE_CLIP).fit(raw_features[: train[-1] + 1])
    target_scaler = StandardScaler().fit(raw_targets[train])

    return PreparedData(
        features=feature_scaler.transform(raw_features),
        targets=target_scaler.transform(raw_targets),
        splits=dict(zip(SPLITS, (train, val, test), strict=True)),
        boundaries=split_boundaries(frame, train, val, test),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        seq_length=seq_length,
    )
