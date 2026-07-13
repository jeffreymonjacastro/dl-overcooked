"""Student policy entrypoint.

Students may replace this file with their own policy. The runner expects a class
called StudentAgent with:

    __init__(self, config: dict)
    reset(self)
    act(self, obs) -> int

Action convention:
    0 = north/up
    1 = south/down
    2 = east/right
    3 = west/left
    4 = stay
    5 = interact
"""

from __future__ import annotations

import json
from math import erf
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else _repo_root() / path


def _gelu(x):
    # PyTorch nn.GELU defaults to the exact erf formulation, not the tanh approximation.
    erf_x = np.vectorize(erf, otypes=[np.float32])(x / np.sqrt(2.0))
    return 0.5 * x * (1.0 + erf_x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


class _OptionAGRUNumpy:
    def __init__(self, checkpoint_path, model_config_path):
        self.weights = np.load(checkpoint_path)
        config = json.loads(Path(model_config_path).read_text(encoding="utf-8-sig"))
        norm_path = _resolve(config["normalization_path"])
        norm = json.loads(norm_path.read_text(encoding="utf-8-sig"))
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.hidden_size = int(config["hidden_size"])
        self.hidden = np.zeros(self.hidden_size, dtype=np.float32)
        self.previous_action = 4
        self.timestep = 0

    def reset(self):
        self.hidden = np.zeros(self.hidden_size, dtype=np.float32)
        self.previous_action = 4
        self.timestep = 0

    def _w(self, name):
        return np.asarray(self.weights[name], dtype=np.float32)

    def _linear(self, x, prefix):
        return x @ self._w(f"{prefix}.weight").T + self._w(f"{prefix}.bias")

    def _features(self, obs):
        if isinstance(obs, dict):
            raw_obs = obs.get("obs")
            agent_index = int(obs.get("agent_index", 0))
        else:
            raw_obs = obs
            agent_index = 0
        x = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        x = (x - self.mean) / np.maximum(self.std, 1e-6)
        agent_onehot = np.zeros(2, dtype=np.float32)
        agent_onehot[np.clip(agent_index, 0, 1)] = 1.0
        prev_onehot = np.zeros(6, dtype=np.float32)
        prev_onehot[np.clip(int(self.previous_action), 0, 5)] = 1.0
        start_flag = np.asarray([1.0 if self.timestep == 0 else 0.0], dtype=np.float32)
        return np.concatenate([x, agent_onehot, prev_onehot, start_flag]).astype(np.float32)

    def logits(self, obs):
        x = self._features(obs)
        x = self._linear(x, "encoder.0")
        x = _layer_norm(x, self._w("encoder.1.weight"), self._w("encoder.1.bias"))
        x = _gelu(x)
        x = self._linear(x, "encoder.4")
        x = _gelu(x)

        w_ih = self._w("gru.weight_ih_l0")
        w_hh = self._w("gru.weight_hh_l0")
        b_ih = self._w("gru.bias_ih_l0")
        b_hh = self._w("gru.bias_hh_l0")
        hs = self.hidden_size
        gi = w_ih @ x + b_ih
        gh = w_hh @ self.hidden + b_hh
        i_r, i_z, i_n = gi[:hs], gi[hs : 2 * hs], gi[2 * hs :]
        h_r, h_z, h_n = gh[:hs], gh[hs : 2 * hs], gh[2 * hs :]
        reset_gate = _sigmoid(i_r + h_r)
        update_gate = _sigmoid(i_z + h_z)
        new_gate = np.tanh(i_n + reset_gate * h_n)
        self.hidden = ((1.0 - update_gate) * new_gate + update_gate * self.hidden).astype(np.float32)

        y = _layer_norm(self.hidden, self._w("post_norm.weight"), self._w("post_norm.bias"))
        return self._linear(y, "head")

    def act(self, obs):
        logits = self.logits(obs)
        action = int(np.argmax(logits))
        self.previous_action = action
        self.timestep += 1
        return action


class _OptionAMLPNumpy:
    def __init__(self, checkpoint_path, model_config_path):
        self.weights = np.load(checkpoint_path)
        config = json.loads(Path(model_config_path).read_text(encoding="utf-8-sig"))
        norm = json.loads(_resolve(config["normalization_path"]).read_text(encoding="utf-8-sig"))
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.previous_action = 4
        self.timestep = 0

    def reset(self):
        self.previous_action = 4
        self.timestep = 0

    def _w(self, name):
        return np.asarray(self.weights[name], dtype=np.float32)

    def _linear(self, x, prefix):
        return x @ self._w(f"{prefix}.weight").T + self._w(f"{prefix}.bias")

    def _features(self, obs):
        if isinstance(obs, dict):
            raw_obs = obs.get("obs")
            agent_index = int(obs.get("agent_index", 0))
        else:
            raw_obs = obs
            agent_index = 0
        x = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        x = (x - self.mean) / np.maximum(self.std, 1e-6)
        agent_onehot = np.zeros(2, dtype=np.float32)
        agent_onehot[np.clip(agent_index, 0, 1)] = 1.0
        prev_onehot = np.zeros(6, dtype=np.float32)
        prev_onehot[np.clip(int(self.previous_action), 0, 5)] = 1.0
        start_flag = np.asarray([1.0 if self.timestep == 0 else 0.0], dtype=np.float32)
        return np.concatenate([x, agent_onehot, prev_onehot, start_flag]).astype(np.float32)

    def logits(self, obs):
        x = self._features(obs)
        x = self._linear(x, "net.0")
        x = _layer_norm(x, self._w("net.1.weight"), self._w("net.1.bias"))
        x = _gelu(x)
        x = _gelu(self._linear(x, "net.3"))
        return self._linear(x, "net.5")

    def act(self, obs):
        action = int(np.argmax(self.logits(obs)))
        self.previous_action = action
        self.timestep += 1
        return action


class _RoutedPolicyNumpy:
    def __init__(self, config):
        self.config = config
        experts = config.get("experts", {})
        if not experts:
            raise ValueError("Router config requires experts")
        self.experts = {
            name: _OptionAGRUNumpy(_resolve(spec["checkpoint_path"]), _resolve(spec["model_config_path"]))
            for name, spec in experts.items()
        }
        self.default_expert = str(config.get("default_expert", next(iter(self.experts))))
        self.rules = list(config.get("rules", []))
        self.layout_name = str(config.get("layout_name", "")).lower()
        self.partner_name = str(config.get("partner_name", "")).lower()
        self.active_expert = self._select_expert()

    def _select_expert(self):
        for rule in self.rules:
            layout = str(rule.get("layout_name", "*")).lower()
            partner = str(rule.get("partner_name", "*")).lower()
            layout_ok = layout in {"*", self.layout_name}
            partner_ok = partner in {"*", self.partner_name}
            if layout_ok and partner_ok:
                expert = str(rule["expert"])
                if expert in self.experts:
                    return expert
        return self.default_expert

    def reset(self):
        self.active_expert = self._select_expert()
        for expert in self.experts.values():
            expert.reset()

    def act(self, obs):
        return self.experts[self.active_expert].act(obs)


class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.mode = str(self.config.get("mode", "fixed")).lower()
        if self.mode in {"option_a", "option_a2", "gru_bc", "gru_bc_a2"}:
            checkpoint_path = _resolve(self.config["checkpoint_path"])
            model_config_path = _resolve(self.config["model_config_path"])
            self.policy = _OptionAGRUNumpy(checkpoint_path, model_config_path)
            return
        if self.mode in {"option_a_mlp", "mlp_bc"}:
            checkpoint_path = _resolve(self.config["checkpoint_path"])
            model_config_path = _resolve(self.config["model_config_path"])
            self.policy = _OptionAMLPNumpy(checkpoint_path, model_config_path)
            return
        if self.mode in {"option_a2_router", "router"}:
            self.policy = _RoutedPolicyNumpy(self.config)
            return

        self.fixed_action = str(self.config.get("action", "stay")).lower()
        self.action_map = {
            "north": 0,
            "up": 0,
            "south": 1,
            "down": 1,
            "east": 2,
            "right": 2,
            "west": 3,
            "left": 3,
            "stay": 4,
            "interact": 5,
            "random": -1,
        }
        if self.fixed_action not in self.action_map:
            raise ValueError(f"Unknown fixed action: {self.fixed_action}")
        self.rng = np.random.default_rng(self.config.get("seed", None))

    def reset(self):
        if hasattr(self, "policy"):
            self.policy.reset()

    def act(self, obs):
        """Return an action index in {0, 1, 2, 3, 4, 5}."""
        if hasattr(self, "policy"):
            return self.policy.act(obs)
        action = self.action_map[self.fixed_action]
        if action == -1:
            return int(self.rng.integers(0, 6))
        return action
