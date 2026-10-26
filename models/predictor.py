"""Self-contained checkpoints: weights plus everything needed to reproduce predictions.

A checkpoint stores the config, feature list, fitted scalers and split boundaries next to the weights,
so evaluation and forecasting never refit anything (which would leak) or depend on a stale CSV.
Only plain Python types and tensors are saved, so loading works with ``weights_only=True``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset, labelled_end_indices, split_by_boundaries
from data.transforms import StandardScaler
from models import build_model

CHECKPOINT_VERSION = 2


class PricePredictor:
    def __init__(self, model, config, feature_columns, feature_scaler, target_scaler, boundaries, trained_at=None):
        self.model = model
        # Round-trip through JSON so the stored config only contains plain types.
        self.config = json.loads(json.dumps(config, default=str))
        self.feature_columns = list(feature_columns)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.boundaries = {name: pd.Timestamp(value) for name, value in boundaries.items()}
        self.trained_at = trained_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    @property
    def seq_length(self) -> int:
        return int(self.config["seq_length"])

    @property
    def horizon(self) -> int:
        return int(self.config.get("horizon", 1))

    @property
    def quantiles(self) -> list:
        return [float(q) for q in self.config["quantiles"]]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def save(self, path, history=None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "model_state": self.model.state_dict(),
                "config": self.config,
                "feature_columns": self.feature_columns,
                "feature_scaler": self.feature_scaler.state_dict(),
                "target_scaler": self.target_scaler.state_dict(),
                "boundaries": {name: value.isoformat() for name, value in self.boundaries.items()},
                "trained_at": self.trained_at,
                "history": history or [],
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu") -> "PricePredictor":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("version", 1) < CHECKPOINT_VERSION:
            raise ValueError(f"{Path(path).name} was saved by an older version; retrain it with scripts/train.py.")
        config = checkpoint["config"]
        model = build_model(
            config["model_type"],
            input_size=len(checkpoint["feature_columns"]),
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            output_size=len(config["quantiles"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        return cls(
            model,
            config,
            checkpoint["feature_columns"],
            StandardScaler.from_state_dict(checkpoint["feature_scaler"]),
            StandardScaler.from_state_dict(checkpoint["target_scaler"]),
            checkpoint["boundaries"],
            checkpoint.get("trained_at"),
        )

    def split_indices(self, frame: pd.DataFrame) -> dict:
        """Window end indices per split, using the boundaries recorded at training time."""
        return split_by_boundaries(frame, labelled_end_indices(frame, self.seq_length), self.boundaries)

    def scale_features(self, frame: pd.DataFrame) -> np.ndarray:
        return self.feature_scaler.transform(frame[self.feature_columns].to_numpy())

    @torch.no_grad()
    def predict_scaled(self, features: np.ndarray, end_indices, batch_size: int = 1024) -> np.ndarray:
        """Raw network outputs (standardized quantiles), shape ``(len(end_indices), len(quantiles))``."""
        dataset = SequenceDataset(features, np.zeros(len(features)), end_indices, self.seq_length)
        self.model.eval()
        outputs = [
            self.model(windows.to(self.device)).reshape(len(windows), -1).cpu().numpy()
            for windows, _ in DataLoader(dataset, batch_size=batch_size)
        ]
        return np.concatenate(outputs) if outputs else np.empty((0, len(self.quantiles)))

    def predict_quantiles(self, frame: pd.DataFrame, end_indices, batch_size: int = 1024) -> np.ndarray:
        """Predicted log-return quantiles over the next ``horizon`` candles for each window end index."""
        scaled = self.predict_scaled(self.scale_features(frame), end_indices, batch_size)
        return self.target_scaler.inverse_transform(scaled)

    def predict_log_returns(self, frame: pd.DataFrame, end_indices, batch_size: int = 1024) -> np.ndarray:
        """Median forecast of the log return over the next ``horizon`` candles."""
        median = self.quantiles.index(0.5)
        return self.predict_quantiles(frame, end_indices, batch_size)[:, median]
