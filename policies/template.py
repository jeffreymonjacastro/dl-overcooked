"""Student agent template for Option B: BC-Warmstarted FCP-style PPO.

This template supports multiple modes:
- mode: option_b  -> loads BC/PPO MLP actor with obs stack
- mode: option_a  -> (future: GRU recurrent policy)
- mode: stay      -> fixed action stay (default fallback)

The agent is purely inference-only. No training, no file writes during act().
"""

from __future__ import annotations

import json
import pathlib
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# NumPy-only forward pass for MLP Actor (Option B)
# ---------------------------------------------------------------------------

class _MLPActorNumpy:
    """Pure NumPy forward pass for the MLP actor. CPU-safe, no PyTorch needed."""

    def __init__(self, params: dict, hidden_sizes: tuple, num_actions: int = 6, dropout: float = 0.0):
        self.params = params
        self.hidden_sizes = hidden_sizes
        self.num_actions = num_actions

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MLP encoder forward: Linear + LayerNorm + ReLU per hidden layer (except last)."""
        layer_idx = 0
        num_layers = len(self.hidden_sizes)
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            W = self.params[f"actor_encoder.net.{layer_idx}.weight"]
            b = self.params[f"actor_encoder.net.{layer_idx}.bias"]
            x = x @ W.T + b
            layer_idx += 1
            if not is_last:
                ln_w = self.params[f"actor_encoder.net.{layer_idx}.weight"]
                ln_b = self.params[f"actor_encoder.net.{layer_idx}.bias"]
                x = self._layer_norm(x, ln_w, ln_b)
                layer_idx += 1
                x = np.maximum(0.0, x)  # ReLU
                layer_idx += 1
                layer_idx += 1  # Dropout (skip at inference)

        W = self.params["actor_head.weight"]
        b = self.params["actor_head.bias"]
        return x @ W.T + b


# ---------------------------------------------------------------------------
# StudentAgent
# ---------------------------------------------------------------------------

class StudentAgent:
    """Student agent supporting Option B (MLP BC/PPO) and fallback stay policy.

    Expected config keys for Option B:
        mode: option_b
        model_config_path: artifacts/option_b/final_policy_config.json
        checkpoint_path: artifacts/option_b/final_policy.npz

    Config for fallback:
        mode: stay  (or mode: random)
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.mode = str(self.config.get("mode", "stay")).lower()
        self._rng = np.random.default_rng(self.config.get("seed", None))

        # ---- Option B: MLP Actor ----
        self._mlp_actor: _MLPActorNumpy | None = None
        self._norm_mean: np.ndarray | None = None
        self._norm_std: np.ndarray | None = None
        self._obs_dim: int = 96
        self._k_stack: int = 4
        self._num_actions: int = 6
        self._hidden_sizes: tuple = (512, 256, 128)

        # Inference state
        self._obs_deque: deque | None = None
        self._prev_action: int = 0

        if self.mode == "option_b":
            self._load_option_b()
        elif self.mode not in {"stay", "random", "north", "south", "east", "west", "interact"}:
            # Graceful fallback
            self.mode = "stay"

        self._reset_state()

    def _load_option_b(self) -> None:
        """Load model config and NumPy weights."""
        model_config_path = self.config.get("model_config_path", "artifacts/option_b/final_policy_config.json")
        checkpoint_path = self.config.get("checkpoint_path", "artifacts/option_b/final_policy.npz")

        # Load model config
        model_config_path = pathlib.Path(model_config_path)
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        with open(model_config_path) as f:
            model_cfg = json.load(f)

        self._obs_dim = int(model_cfg.get("obs_dim", 96))
        self._k_stack = int(model_cfg.get("k_stack", model_cfg.get("obs_stack_k", 4)))
        self._num_actions = int(model_cfg.get("num_actions", 6))
        self._hidden_sizes = tuple(model_cfg.get("hidden_sizes", [512, 256, 128]))

        # Load normalization
        norm_path = model_cfg.get("normalization_path", "artifacts/shared/normalization.json")
        norm_path = pathlib.Path(norm_path)
        if norm_path.exists():
            with open(norm_path) as f:
                norm = json.load(f)
            self._norm_mean = np.array(norm["mean"], dtype=np.float32)
            self._norm_std = np.array(norm["std"], dtype=np.float32)
        else:
            self._norm_mean = np.zeros(self._obs_dim, dtype=np.float32)
            self._norm_std = np.ones(self._obs_dim, dtype=np.float32)

        # Load weights
        checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        params = dict(np.load(checkpoint_path))

        self._mlp_actor = _MLPActorNumpy(params, self._hidden_sizes, self._num_actions)

    def _reset_state(self) -> None:
        """Reset all temporal state (hidden state, obs deque, prev action)."""
        self._prev_action = 0  # BOS token
        if self.mode == "option_b":
            self._obs_deque = deque(
                [np.zeros(self._obs_dim, dtype=np.float32)] * self._k_stack,
                maxlen=self._k_stack
            )

    def reset(self) -> None:
        """Called at the start of each episode."""
        self._reset_state()

    def act(self, obs) -> int:
        """Return action index in {0, 1, 2, 3, 4, 5}."""
        if self.mode == "option_b" and self._mlp_actor is not None:
            return self._act_option_b(obs)
        return self._act_fallback()

    def _act_option_b(self, obs) -> int:
        """Option B MLP actor forward pass."""
        # Extract observation vector and agent index
        if isinstance(obs, dict):
            obs_vec = np.asarray(obs.get("obs", obs.get("state", np.zeros(self._obs_dim))), dtype=np.float32)
            agent_index = int(obs.get("agent_index", 0))
        else:
            obs_vec = np.asarray(obs, dtype=np.float32).flatten()
            agent_index = 0

        # Truncate or pad to expected obs_dim
        if obs_vec.shape[0] != self._obs_dim:
            padded = np.zeros(self._obs_dim, dtype=np.float32)
            min_len = min(obs_vec.shape[0], self._obs_dim)
            padded[:min_len] = obs_vec[:min_len]
            obs_vec = padded

        # Normalize
        obs_norm = (obs_vec - self._norm_mean) / self._norm_std

        # Update deque
        self._obs_deque.append(obs_norm)

        # Build obs stack
        stack = np.concatenate(list(self._obs_deque), axis=0).astype(np.float32)

        # Build full input: stack + agent_one_hot + prev_action_one_hot
        agent_oh = np.zeros(2, dtype=np.float32)
        agent_oh[agent_index] = 1.0
        prev_oh = np.zeros(self._num_actions, dtype=np.float32)
        prev_oh[self._prev_action] = 1.0

        x = np.concatenate([stack, agent_oh, prev_oh])

        # Forward pass
        logits = self._mlp_actor.forward(x)
        action = int(np.argmax(logits))

        # Update previous action
        self._prev_action = action
        return action

    def _act_fallback(self) -> int:
        action_map = {
            "stay": 4, "north": 0, "south": 1, "east": 2, "west": 3, "interact": 5,
        }
        if self.mode == "random":
            return int(self._rng.integers(0, self._num_actions))
        return action_map.get(self.mode, 4)
