from models.gru_model import BitcoinPricePredictorGRU
from models.layers import MonotoneQuantiles
from models.linear_model import LinearForecaster
from models.lstm_model import BitcoinPricePredictorLSTM
from models.tcn_model import TCNForecaster
from models.transformer_model import TransformerForecaster

MODEL_TYPES = {
    "gru": BitcoinPricePredictorGRU,
    "lstm": BitcoinPricePredictorLSTM,
    "transformer": TransformerForecaster,
    "tcn": TCNForecaster,
    "linear": LinearForecaster,
}


def build_model(
    model_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0, output_size: int = 1
):
    """Build a regressor; with several outputs (quantiles) they are made non-decreasing."""
    try:
        model_class = MODEL_TYPES[model_type.lower()]
    except KeyError:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {sorted(MODEL_TYPES)}") from None
    model = model_class(input_size, hidden_size, num_layers, output_size=output_size, dropout=dropout)
    return MonotoneQuantiles(model) if output_size > 1 else model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
