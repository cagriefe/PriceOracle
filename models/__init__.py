from models.gru_model import BitcoinPricePredictorGRU
from models.lstm_model import BitcoinPricePredictorLSTM

MODEL_TYPES = {"gru": BitcoinPricePredictorGRU, "lstm": BitcoinPricePredictorLSTM}


def build_model(
    model_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0, output_size: int = 1
):
    try:
        model_class = MODEL_TYPES[model_type.lower()]
    except KeyError:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {sorted(MODEL_TYPES)}") from None
    return model_class(input_size, hidden_size, num_layers, output_size=output_size, dropout=dropout)
