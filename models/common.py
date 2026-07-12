"""Shared model utilities for Overcooked training options."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Configurable multi-layer perceptron with LayerNorm and activation."""

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        output_activation: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = i == len(layer_sizes) - 2
            if not is_last or output_activation:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}[
                    activation.lower()
                ]
                layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to linear layers."""
    if isinstance(module, (nn.Linear, nn.GRU)):
        if hasattr(module, "weight"):
            nn.init.orthogonal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.orthogonal_(param, gain=gain)
        elif "bias" in name:
            nn.init.zeros_(param)
