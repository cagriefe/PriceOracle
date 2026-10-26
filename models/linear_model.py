import torch.nn as nn


class LinearForecaster(nn.Module):
    """Linear baseline on the latest candle's features; ``hidden_size``, ``num_layers`` and ``dropout`` are ignored."""

    def __init__(self, input_size, hidden_size=None, num_layers=None, output_size=1, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x[:, -1, :])
