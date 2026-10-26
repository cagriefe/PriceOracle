import numpy as np
import pandas as pd


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
