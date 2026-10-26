"""Feature engineering.

Every feature at time ``t`` is computed only from candles up to and including ``t`` (all indicators
are causal), so the frame can be built once over the whole history without leaking future data.
Features are scale-free (returns, ratios, oscillators) so they stay comparable between a $4k and a
$100k market, unlike raw prices.
"""

import numpy as np
import pandas as pd
import ta

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "log_return",
    "hl_range",
    "close_sma_ratio",
    "close_ema_ratio",
    "rsi",
    "macd",
    "volume_ratio",
    "volatility",
]


def load_ohlcv(file_path) -> pd.DataFrame:
    """Load klines saved by ``scripts/fetch_data.py`` as a float OHLCV frame indexed by timestamp."""
    data = pd.read_csv(file_path, usecols=["timestamp", *OHLCV_COLUMNS])
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp").sort_index()
    data = data[~data.index.duplicated(keep="last")]
    return data.astype(float)


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"]
    features = pd.DataFrame(index=data.index)
    features["log_return"] = np.log(close).diff()
    features["hl_range"] = np.log(data["high"] / data["low"])
    features["close_sma_ratio"] = close / ta.trend.sma_indicator(close, window=20) - 1
    features["close_ema_ratio"] = close / ta.trend.ema_indicator(close, window=20) - 1
    features["rsi"] = ta.momentum.rsi(close, window=14) / 100
    features["macd"] = ta.trend.macd(close) / close
    features["volume_ratio"] = np.log1p(data["volume"]) - np.log1p(data["volume"].rolling(24).mean())
    features["volatility"] = features["log_return"].rolling(24).std()
    return features[FEATURE_COLUMNS]


def build_frame(data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return the features plus ``close`` and ``target``, the log return over the next ``horizon`` candles.

    Warm-up rows with incomplete indicators are dropped rather than back-filled with future values.
    The last ``horizon`` rows keep a NaN target so they can still feed a live forecast.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    frame = compute_features(data)
    frame["close"] = data["close"]
    frame["target"] = np.log(data["close"].shift(-horizon) / data["close"])
    return frame.dropna(subset=FEATURE_COLUMNS)


def prepare_frame(file_path, horizon: int = 1) -> pd.DataFrame:
    return build_frame(load_ohlcv(file_path), horizon=horizon)
