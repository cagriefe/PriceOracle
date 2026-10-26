from pathlib import Path

import numpy as np
import pytest
import torch

from utils.helpers import PROJECT_ROOT, cli_path, get_device, load_config, resolve_path, set_seed
from utils.metrics import price_metrics, up_move_ratio

REQUIRED_CONFIG_KEYS = {
    "data_path",
    "model_path",
    "model_type",
    "horizon",
    "seq_length",
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "num_epochs",
    "patience",
    "val_ratio",
    "test_ratio",
    "seed",
}


def test_default_config_is_complete():
    assert REQUIRED_CONFIG_KEYS <= load_config().keys()


def test_load_config_skips_none_overrides():
    config = load_config(model_type=None, num_epochs=3)
    assert config["model_type"] == load_config()["model_type"]
    assert config["num_epochs"] == 3


def test_resolve_path_uses_project_root_for_relative_paths(tmp_path):
    assert resolve_path("data/btc_usdt.csv") == PROJECT_ROOT / "data" / "btc_usdt.csv"
    assert resolve_path(tmp_path) == tmp_path


def test_cli_path_is_relative_to_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(cli_path("model.pt")) == tmp_path.resolve() / "model.pt"
    assert cli_path(None) is None


def test_set_seed_is_reproducible():
    set_seed(123)
    first = torch.rand(3), np.random.rand(3)
    set_seed(123)
    second = torch.rand(3), np.random.rand(3)
    assert torch.equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_get_device():
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("auto").type in {"cpu", "cuda"}


def test_price_metrics_and_directional_accuracy():
    reference = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 99.0, 102.0, 100.0])
    predicted = np.array([100.5, 99.5, 99.0, 100.0])
    metrics = price_metrics(actual, predicted, reference)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(9.5 / 4))
    assert metrics["mape"] == pytest.approx(np.mean([0.5 / 101, 0.5 / 99, 3 / 102, 0]) * 100)
    # The flat last candle is ignored; the third forecast points the wrong way.
    assert metrics["directional_accuracy"] == pytest.approx(2 / 3)
    assert up_move_ratio(actual, reference) == pytest.approx(2 / 3)


def test_naive_forecast_has_zero_error_on_a_flat_market():
    flat = np.full(5, 100.0)
    metrics = price_metrics(flat, flat, flat)
    assert metrics["mae"] == metrics["rmse"] == 0.0
    assert np.isnan(metrics["directional_accuracy"])
