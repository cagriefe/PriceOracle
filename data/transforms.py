import numpy as np


class StandardScaler:
    """Z-score scaler with the same API as scikit-learn's.

    ``clip`` optionally winsorises transformed values to ``±clip`` standard deviations, which keeps rare
    crash candles from saturating the network (``inverse_transform`` cannot undo the clipping).
    Its state is stored as plain Python types, so it can live inside a checkpoint that is loaded with
    ``torch.load(..., weights_only=True)``.
    """

    def __init__(self, mean=None, scale=None, clip=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.clip = clip

    def fit(self, values) -> "StandardScaler":
        values = np.asarray(values, dtype=np.float64)
        self.mean = np.atleast_1d(values.mean(axis=0))
        scale = np.atleast_1d(values.std(axis=0))
        self.scale = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, values) -> np.ndarray:
        self._check_fitted()
        scaled = (np.asarray(values, dtype=np.float64) - self.mean) / self.scale
        return scaled if self.clip is None else np.clip(scaled, -self.clip, self.clip)

    def inverse_transform(self, values) -> np.ndarray:
        self._check_fitted()
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def state_dict(self) -> dict:
        self._check_fitted()
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScaler":
        return cls(state["mean"], state["scale"], state.get("clip"))

    def _check_fitted(self) -> None:
        if self.mean is None or self.scale is None:
            raise RuntimeError("StandardScaler must be fitted before use.")
