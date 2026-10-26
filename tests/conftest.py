import pandas as pd
import pytest

from tests.synthetic import make_ohlcv, write_ohlcv_csv
from utils.helpers import load_config


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def ohlcv_csv(tmp_path, ohlcv):
    return write_ohlcv_csv(ohlcv, tmp_path / "ohlcv.csv")


@pytest.fixture
def tiny_config(tmp_path, ohlcv_csv) -> dict:
    return load_config(
        data_path=str(ohlcv_csv),
        model_path=str(tmp_path / "model.pt"),
        seq_length=16,
        hidden_size=8,
        num_layers=1,
        batch_size=32,
        num_epochs=2,
        patience=2,
    )
