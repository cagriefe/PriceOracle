import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    """Pre-norm Transformer encoder over the input window; the last position summarises the sequence.

    Attention is bidirectional inside the window, which is safe: the whole window lies in the past.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, num_heads=4, max_len=1024):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.positional = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, dim_feedforward=4 * hidden_size, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.encoder(self.input_proj(x) + self.positional[:, : x.size(1)])
        return self.fc(self.norm(hidden[:, -1]))
