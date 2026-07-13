"""Macro-action actor-critic network for warm-started Macro-PPO."""

from __future__ import annotations

import torch
from torch import nn

from planning.macro_actions import MACRO_ACTIONS


class MacroActorCritic(nn.Module):
    """Small recurrent actor-critic over macro actions, not primitive moves."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        macro_actions: int = len(MACRO_ACTIONS),
        role_dim: int = 2,
        partner_dim: int = 8,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.macro_actions = int(macro_actions)
        input_dim = self.feature_dim + int(role_dim) + int(partner_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, self.macro_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, features, role_onehot, partner_features, hidden=None, action_mask=None):
        x = torch.cat([features, role_onehot, partner_features], dim=-1)
        encoded = self.encoder(x)
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)
        output, hidden = self.gru(encoded, hidden)
        last = output[:, -1]
        logits = self.actor(last)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        value = self.critic(last).squeeze(-1)
        return logits, value, hidden


def build_default_model(feature_dim: int = 64, hidden_dim: int = 128) -> MacroActorCritic:
    return MacroActorCritic(feature_dim=feature_dim, hidden_dim=hidden_dim)

