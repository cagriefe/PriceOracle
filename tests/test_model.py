import warnings

import numpy as np
import pytest
import torch

from data.preprocess import FEATURE_COLUMNS
from models import MODEL_TYPES, MonotoneQuantiles, build_model
from models.layers import PinballLoss
from models.predictor import PricePredictor
from models.tcn_model import CausalConvBlock
from scripts.train import validate_quantiles
from tests.synthetic import make_predictor


@pytest.mark.parametrize("model_type", sorted(MODEL_TYPES))
def test_every_architecture_outputs_sorted_quantiles(model_type):
    torch.manual_seed(0)
    model = build_model(model_type, input_size=8, hidden_size=16, num_layers=2, dropout=0.1, output_size=5)
    model.eval()
    outputs = model(torch.randn(6, 12, 8))
    assert isinstance(model, MonotoneQuantiles)
    assert outputs.shape == (6, 5)
    assert torch.all(outputs[:, 1:] > outputs[:, :-1])


def test_single_output_models_are_plain_regressors():
    model = build_model("gru", input_size=8, hidden_size=16, num_layers=1)
    assert not isinstance(model, MonotoneQuantiles)
    assert model(torch.randn(4, 12, 8)).shape == (4, 1)


@pytest.mark.parametrize("model_type", ["gru", "lstm"])
def test_single_layer_recurrent_models_do_not_warn_about_dropout(model_type):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(model_type, input_size=8, hidden_size=16, num_layers=1, dropout=0.2)


def test_build_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model("xgboost", input_size=8, hidden_size=16, num_layers=1)


def test_transformer_requires_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        build_model("transformer", input_size=8, hidden_size=10, num_layers=1)


def test_tcn_blocks_are_causal():
    torch.manual_seed(0)
    block = CausalConvBlock(3, 4, kernel_size=3, dilation=2, dropout=0.0).eval()
    x = torch.randn(1, 3, 20)
    changed = x.clone()
    changed[:, :, 15:] += 10.0
    torch.testing.assert_close(block(x)[:, :, :15], block(changed)[:, :, :15])


def test_pinball_loss_matches_the_definition():
    quantiles = [0.1, 0.5, 0.9]
    predictions = torch.tensor([[0.0, 1.0, 2.0]])
    target = torch.tensor([1.5])
    diff = 1.5 - np.array([0.0, 1.0, 2.0])
    expected = np.mean(np.maximum(np.array(quantiles) * diff, (np.array(quantiles) - 1) * diff))
    assert PinballLoss(quantiles)(predictions, target).item() == pytest.approx(expected)


@pytest.mark.parametrize("quantiles", [[0.25, 0.75], [0.5, 0.25, 0.75], [0.0, 0.5, 1.0], [0.1, 0.5, 0.8]])
def test_invalid_quantile_sets_are_rejected(quantiles):
    with pytest.raises(ValueError):
        validate_quantiles(quantiles)


def test_valid_quantile_sets_are_accepted():
    assert validate_quantiles([0.1, 0.5, 0.9]) == [0.1, 0.5, 0.9]


def test_checkpoint_round_trip_reproduces_predictions(tmp_path, ohlcv):
    # A Path in the config must be stored as a plain string so the checkpoint loads with weights_only=True.
    predictor, frame, data = make_predictor(ohlcv, data_path=tmp_path / "prices.csv")
    predictor.save(tmp_path / "model.pt")

    restored = PricePredictor.load(tmp_path / "model.pt")
    indices = data.splits["test"]
    np.testing.assert_allclose(restored.predict_quantiles(frame, indices), predictor.predict_quantiles(frame, indices))
    assert restored.boundaries == predictor.boundaries
    assert restored.trained_at == predictor.trained_at
    assert restored.config["data_path"] == str(tmp_path / "prices.csv")
    assert restored.feature_columns == FEATURE_COLUMNS
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(restored.split_indices(frame)[name], data.splits[name])


def test_older_checkpoints_are_rejected(tmp_path, ohlcv):
    predictor, _, _ = make_predictor(ohlcv)
    predictor.save(tmp_path / "model.pt")
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    checkpoint["version"] = 1
    torch.save(checkpoint, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="older version"):
        PricePredictor.load(tmp_path / "old.pt")


def test_median_forecast_is_the_middle_quantile(ohlcv):
    predictor, frame, data = make_predictor(ohlcv)
    indices = data.splits["test"]
    quantiles = predictor.predict_quantiles(frame, indices)
    np.testing.assert_allclose(predictor.predict_log_returns(frame, indices), quantiles[:, 2])
