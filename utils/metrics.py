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


def interval_metrics(actual, lower, upper, level: float) -> dict:
    """Coverage, mean width and the interval (Winkler) score, a proper scoring rule: lower is better.

    The score adds ``2 / alpha`` times the miss distance to the width, so narrow intervals only win
    when they still contain the outcome often enough.
    """
    actual, lower, upper = (np.asarray(values, dtype=np.float64) for values in (actual, lower, upper))
    alpha = 1 - level
    below, above = actual < lower, actual > upper
    width = upper - lower
    score = width + (2 / alpha) * ((lower - actual) * below + (actual - upper) * above)
    return {
        "coverage": float(np.mean(~below & ~above)),
        "width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
    }


def pinball_score(actual, quantile_values, quantiles) -> float:
    """Mean quantile loss of ``quantile_values`` (n, k) at levels ``quantiles``; lower is better."""
    diff = np.asarray(actual, dtype=np.float64)[:, None] - np.asarray(quantile_values, dtype=np.float64)
    taus = np.asarray(quantiles, dtype=np.float64)[None, :]
    return float(np.mean(np.maximum(taus * diff, (taus - 1) * diff)))


def brier_score(probability_up, actual_return) -> float:
    """Mean squared error of the predicted probability of a rise; a constant 50% guess scores 0.25."""
    outcome = (np.asarray(actual_return, dtype=np.float64) > 0).astype(np.float64)
    return float(np.mean((np.asarray(probability_up, dtype=np.float64) - outcome) ** 2))
