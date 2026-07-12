"""Option B: Actor-Critic model for BC warm-start + FCP-style PPO.

Architecture:
    Input: stack of K obs (normalized) + agent_index one-hot + prev_action one-hot
    Actor: MLP(512 -> 256 -> 128 -> 6 logits)
    Critic: MLP(same architecture -> 1 value)

The critic is NOT used during inference in StudentAgent.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import MLP, init_weights_orthogonal


class BCWarmstartActor(nn.Module):
    """MLP actor used for BC warm-start phase.
    
    Input = stack_obs (K * obs_dim) + agent_index_one_hot (2) + prev_action_one_hot (6)
    """

    def __init__(
        self,
        obs_dim: int = 96,
        k_stack: int = 4,
        num_actions: int = 6,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.k_stack = k_stack
        self.num_actions = num_actions

        # Feature sizes
        stack_dim = k_stack * obs_dim
        agent_one_hot_dim = 2
        prev_action_dim = num_actions
        input_dim = stack_dim + agent_one_hot_dim + prev_action_dim

        layer_sizes = [input_dim] + list(hidden_sizes)
        self.encoder = MLP(layer_sizes, activation="relu", use_layer_norm=True, dropout=dropout)
        self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)

        self.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, stack_obs: torch.Tensor, agent_index: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stack_obs: (batch, k * obs_dim)
            agent_index: (batch,) int in {0,1}
            prev_action: (batch,) int in {0,...,5}
        Returns:
            logits: (batch, num_actions)
        """
        agent_oh = F.one_hot(agent_index.long(), num_classes=2).float()
        prev_oh = F.one_hot(prev_action.long(), num_classes=self.num_actions).float()
        x = torch.cat([stack_obs, agent_oh, prev_oh], dim=-1)
        features = self.encoder(x)
        logits = self.actor_head(features)
        return logits


class ActorCritic(nn.Module):
    """Combined Actor-Critic for PPO training.
    
    Both actor and critic share the same encoder structure but have separate heads.
    The critic is only used during training; inference uses only the actor.
    """

    def __init__(
        self,
        obs_dim: int = 96,
        k_stack: int = 4,
        num_actions: int = 6,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.k_stack = k_stack
        self.num_actions = num_actions

        stack_dim = k_stack * obs_dim
        agent_one_hot_dim = 2
        prev_action_dim = num_actions
        input_dim = stack_dim + agent_one_hot_dim + prev_action_dim

        layer_sizes = [input_dim] + list(hidden_sizes)

        # Shared encoder for actor
        self.actor_encoder = MLP(layer_sizes, activation="relu", use_layer_norm=True, dropout=dropout)
        self.actor_head = nn.Linear(hidden_sizes[-1], num_actions)

        # Separate encoder for critic (decentralized - uses same obs)
        self.critic_encoder = MLP(layer_sizes, activation="relu", use_layer_norm=True, dropout=dropout)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

        self.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _build_input(self, stack_obs: torch.Tensor, agent_index: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        agent_oh = F.one_hot(agent_index.long(), num_classes=2).float()
        prev_oh = F.one_hot(prev_action.long(), num_classes=self.num_actions).float()
        return torch.cat([stack_obs, agent_oh, prev_oh], dim=-1)

    def actor_logits(self, stack_obs: torch.Tensor, agent_index: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        x = self._build_input(stack_obs, agent_index, prev_action)
        features = self.actor_encoder(x)
        return self.actor_head(features)

    def critic_value(self, stack_obs: torch.Tensor, agent_index: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        x = self._build_input(stack_obs, agent_index, prev_action)
        features = self.critic_encoder(x)
        return self.critic_head(features).squeeze(-1)

    def forward(self, stack_obs: torch.Tensor, agent_index: torch.Tensor, prev_action: torch.Tensor):
        """Returns (logits, value) for PPO."""
        x = self._build_input(stack_obs, agent_index, prev_action)
        actor_features = self.actor_encoder(x)
        critic_features = self.critic_encoder(x)
        logits = self.actor_head(actor_features)
        value = self.critic_head(critic_features).squeeze(-1)
        return logits, value

    def load_bc_warmstart(self, bc_actor: BCWarmstartActor) -> None:
        """Initialize actor from BC warm-start weights."""
        self.actor_encoder.load_state_dict(bc_actor.encoder.state_dict())
        self.actor_head.load_state_dict(bc_actor.actor_head.state_dict())

    def export_to_numpy(self) -> dict:
        """Export actor weights to numpy dict for inference without PyTorch."""
        params = {}
        for name, param in self.actor_encoder.named_parameters():
            params[f"actor_encoder.{name}"] = param.detach().cpu().numpy()
        for name, param in self.actor_head.named_parameters():
            params[f"actor_head.{name}"] = param.detach().cpu().numpy()
        return params


class NumpyActorInference:
    """Pure NumPy inference for the actor. Used in StudentAgent for deployment.
    
    Applies LayerNorm, ReLU, Linear layers using numpy only.
    """

    def __init__(self, params: dict, config: dict):
        self.params = params
        self.config = config
        self.obs_dim = config["obs_dim"]
        self.k_stack = config["k_stack"]
        self.num_actions = config["num_actions"]
        self.hidden_sizes = config["hidden_sizes"]

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * weight + bias

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _linear(self, x: np.ndarray, key_prefix: str) -> np.ndarray:
        W = self.params[f"{key_prefix}.weight"]
        b = self.params[f"{key_prefix}.bias"]
        return x @ W.T + b

    def _mlp_block(self, x: np.ndarray, layer_idx: int, key_prefix: str) -> np.ndarray:
        """Apply one Linear + LayerNorm + ReLU block."""
        # Linear
        x = self._linear(x, f"{key_prefix}.net.{layer_idx}")
        layer_idx += 1
        # LayerNorm
        x = self._layer_norm(
            x,
            self.params[f"{key_prefix}.net.{layer_idx}.weight"],
            self.params[f"{key_prefix}.net.{layer_idx}.bias"],
        )
        layer_idx += 1
        # ReLU
        x = self._relu(x)
        return x, layer_idx + 1  # skip Dropout (no-op at inference)

    def forward(self, stack_obs: np.ndarray, agent_index: int, prev_action: int) -> np.ndarray:
        """Run actor forward pass. Returns logits (num_actions,)."""
        agent_oh = np.zeros(2, dtype=np.float32)
        agent_oh[agent_index] = 1.0
        prev_oh = np.zeros(self.num_actions, dtype=np.float32)
        prev_oh[prev_action] = 1.0

        x = np.concatenate([stack_obs.flatten(), agent_oh, prev_oh])

        # MLP encoder - each block: Linear + LayerNorm + ReLU + Dropout
        # Count through layers: for each hidden layer = 4 sub-modules (Linear, LN, ReLU, Dropout)
        layer_idx = 0
        for _ in self.hidden_sizes:
            # Linear
            W = self.params[f"actor_encoder.net.{layer_idx}.weight"]
            b = self.params[f"actor_encoder.net.{layer_idx}.bias"]
            x = x @ W.T + b
            layer_idx += 1
            # LayerNorm
            ln_w = self.params[f"actor_encoder.net.{layer_idx}.weight"]
            ln_b = self.params[f"actor_encoder.net.{layer_idx}.bias"]
            x = self._layer_norm(x, ln_w, ln_b)
            layer_idx += 1
            # ReLU
            x = self._relu(x)
            layer_idx += 1
            # Dropout (skip at inference)
            layer_idx += 1

        # Actor head (just Linear)
        W = self.params["actor_head.weight"]
        b = self.params["actor_head.bias"]
        logits = x @ W.T + b
        return logits

    def act(self, stack_obs: np.ndarray, agent_index: int, prev_action: int) -> int:
        logits = self.forward(stack_obs, agent_index, prev_action)
        return int(np.argmax(logits))
