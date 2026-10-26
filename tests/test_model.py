import warnings

import numpy as np
import pytest
import torch

from data.dataset import prepare_training_data
from data.preprocess import FEATURE_COLUMNS, build_frame
from models import BitcoinPricePredictorGRU, BitcoinPricePredictorLSTM, build_model
from models.predictor import PricePredictor
from utils.helpers import load_config


@pytest.mark.parametrize(
    "model_type, model_class", [("gru", BitcoinPricePredictorGRU), ("lstm", BitcoinPricePredictorLSTM)]
)
def test_build_model_returns_requested_architecture(model_type, model_class):
    model = build_model(model_type, input_size=8, hidden_size=16, num_layers=2, dropout=0.1)
    assert isinstance(model, model_class)
    assert model(torch.randn(4, 12, 8)).shape == (4, 1)


@pytest.mark.parametrize("model_type", ["gru", "lstm"])
def test_single_layer_models_do_not_warn_about_dropout(model_type):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(model_type, input_size=8, hidden_size=16, num_layers=1, dropout=0.2)


def test_build_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model("transformer", input_size=8, hidden_size=16, num_layers=1)


def test_checkpoint_round_trip_reproduces_predictions(tmp_path, ohlcv):
    frame = build_frame(ohlcv)
    data = prepare_training_data(frame, seq_length=16, val_ratio=0.15, test_ratio=0.15)
    # A Path in the config must be stored as a plain string so the checkpoint loads with weights_only=True.
    config = load_config(seq_length=16, hidden_size=8, num_layers=1, data_path=tmp_path / "prices.csv")
    model = build_model("gru", len(FEATURE_COLUMNS), hidden_size=8, num_layers=1)
    predictor = PricePredictor(model, config, FEATURE_COLUMNS, data.feature_scaler, data.target_scaler, data.boundaries)
    predictor.save(tmp_path / "model.pt")

    restored = PricePredictor.load(tmp_path / "model.pt")
    indices = data.splits["test"]
    np.testing.assert_allclose(
        restored.predict_log_returns(frame, indices), predictor.predict_log_returns(frame, indices)
    )
    assert restored.boundaries == predictor.boundaries
    assert restored.config["data_path"] == str(tmp_path / "prices.csv")
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(restored.split_indices(frame)[name], data.splits[name])
