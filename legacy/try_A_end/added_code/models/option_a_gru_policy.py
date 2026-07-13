"""PyTorch models for option A."""

from __future__ import annotations

import torch
from torch import nn


class OptionAGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int = 6,
        hidden_size: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.post_norm = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.num_actions)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        z = self.encoder(x)
        y, hidden = self.gru(z, hidden)
        y = self.post_norm(y)
        return self.head(y), hidden


class OptionAMLP(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        return self.net(x), hidden
