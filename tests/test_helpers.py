from pathlib import Path

import numpy as np
import torch

from utils.helpers import (
    PROJECT_ROOT,
    cli_path,
    get_device,
    github_annotation,
    github_step_summary,
    load_config,
    resolve_path,
    set_seed,
)

REQUIRED_CONFIG_KEYS = {
    "data_path",
    "model_path",
    "model_type",
    "horizon",
    "quantiles",
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
    "conformal_window",
    "fee_rate",
    "slippage",
}


def test_default_config_is_complete():
    config = load_config()
    assert REQUIRED_CONFIG_KEYS <= config.keys()
    assert 0.5 in config["quantiles"]


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


def test_github_helpers_only_act_inside_actions(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    github_annotation("notice", "hidden")
    github_step_summary("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    github_annotation("warning", "visible")
    github_step_summary("## Summary")
    assert capsys.readouterr().out == "::warning::visible\n"
    assert summary.read_text() == "## Summary\n"
