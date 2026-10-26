import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection (left padding only, so no look-ahead)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.dropout(F.relu(self.conv1(F.pad(x, (self.padding, 0)))))
        out = self.dropout(F.relu(self.conv2(F.pad(out, (self.padding, 0)))))
        return F.relu(out + self.residual(x))


class TCNForecaster(nn.Module):
    """Temporal convolutional network with dilations 1, 2, 4, ...

    With kernel size 3 the receptive field is ``1 + 4 * (2**num_layers - 1)`` candles.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, kernel_size=3):
        super().__init__()
        self.network = nn.Sequential(
            *[
                CausalConvBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 2**i, dropout)
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc(self.network(x.transpose(1, 2))[:, :, -1])
