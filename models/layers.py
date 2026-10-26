import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneQuantiles(nn.Module):
    """Make a regressor's outputs non-decreasing, so predicted quantiles can never cross.

    The first output is kept as-is and each following output adds a strictly positive softplus step.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        raw = self.model(x)
        return torch.cat([raw[..., :1], F.softplus(raw[..., 1:])], dim=-1).cumsum(dim=-1)


class PinballLoss(nn.Module):
    """Mean quantile (pinball) loss over a fixed set of quantile levels."""

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("quantiles", torch.as_tensor(quantiles, dtype=torch.float32))

    def forward(self, predictions, targets):
        diff = targets.reshape(-1, 1) - predictions
        return torch.maximum(self.quantiles * diff, (self.quantiles - 1) * diff).mean()
