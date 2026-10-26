"""Permutation feature importance for the quantile forecaster."""

import numpy as np

from utils.metrics import pinball_score


def permutation_importance(predictor, frame, end_indices, repeats: int = 3, seed: int = 0, max_samples: int = 5000):
    """Percentage increase in pinball loss when one feature is shuffled across time, most important first.

    Shuffling breaks the link between a feature and the outcome while keeping its distribution, so a
    large increase means the model relies on that feature. Scored on the latest ``max_samples`` windows.
    """
    end_indices = np.asarray(end_indices)[-max_samples:]
    features = predictor.scale_features(frame)
    targets = predictor.target_scaler.transform(frame["target"].to_numpy())[end_indices]

    def loss(matrix):
        return pinball_score(targets, predictor.predict_scaled(matrix, end_indices), predictor.quantiles)

    baseline = loss(features)
    rng = np.random.default_rng(seed)
    importance = {}
    for column, name in enumerate(predictor.feature_columns):
        increases = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            increases.append(loss(shuffled) / baseline - 1)
        importance[name] = float(np.mean(increases) * 100)
    return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
