import numpy as np


def price_metrics(actual, predicted, reference) -> dict:
    """Error metrics in price units plus directional accuracy.

    ``reference`` is the last known close when the forecast was made; a forecast "gets the direction
    right" when it moves away from the reference in the same direction as the actual price. Candles
    with no actual move are ignored for directional accuracy.
    """
    actual, predicted, reference = (np.asarray(values, dtype=np.float64) for values in (actual, predicted, reference))
    error = predicted - actual
    actual_move = np.sign(actual - reference)
    predicted_move = np.sign(predicted - reference)
    moved = actual_move != 0
    directional_accuracy = float(np.mean(actual_move[moved] == predicted_move[moved])) if moved.any() else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / actual) * 100),
        "directional_accuracy": directional_accuracy,
    }


def up_move_ratio(actual, reference) -> float:
    """Share of candles that closed above the reference: the accuracy of an "always up" guess."""
    move = np.sign(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    moved = move != 0
    return float(np.mean(move[moved] > 0)) if moved.any() else float("nan")
