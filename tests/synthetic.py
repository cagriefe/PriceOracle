import numpy as np
import pandas as pd

from data.dataset import prepare_training_data
from data.preprocess import FEATURE_COLUMNS, build_frame
from models import build_model
from models.predictor import PricePredictor
from utils.helpers import load_config


def make_ohlcv(n_rows: int = 400, seed: int = 0, start="2024-01-01") -> pd.DataFrame:
    """Synthetic hourly candles following a geometric random walk."""
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.005, n_rows)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n_rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n_rows))
    volume = rng.uniform(100, 1_000, n_rows)
    index = pd.date_range(start, periods=n_rows, freq="h", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def write_ohlcv_csv(frame: pd.DataFrame, path):
    frame.to_csv(path, date_format="%Y-%m-%d %H:%M:%S")
    return path


def make_predictor(ohlcv: pd.DataFrame, model_type: str = "gru", seq_length: int = 16, horizon: int = 1, **overrides):
    """An untrained predictor with properly fitted scalers: enough to exercise every inference path."""
    frame = build_frame(ohlcv, horizon=horizon)
    data = prepare_training_data(frame, seq_length, val_ratio=0.15, test_ratio=0.15, horizon=horizon)
    config = load_config(
        model_type=model_type,
        seq_length=seq_length,
        horizon=horizon,
        hidden_size=8,
        num_layers=1,
        conformal_window=24,
        **overrides,
    )
    model = build_model(model_type, len(FEATURE_COLUMNS), 8, 1, output_size=len(config["quantiles"]))
    predictor = PricePredictor(model, config, FEATURE_COLUMNS, data.feature_scaler, data.target_scaler, data.boundaries)
    return predictor, frame, data
