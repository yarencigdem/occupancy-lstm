import torch
import torch.nn as nn

class LSTMOccupancyModel(nn.Module):
    """
    Binary occupancy classification with LSTM.
    Input: (batch, seq_len, n_features)
    Output: logits (batch, 1)
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        out = out[:, -1, :]            # last timestep (B, H)
        logits = self.fc(out)          # (B, 1)
        return logits
