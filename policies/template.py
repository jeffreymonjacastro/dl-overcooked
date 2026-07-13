"""Neural student policy for the Overcooked competition starter."""

from __future__ import annotations

from math import erf
from pathlib import Path

import numpy as np


def _gelu(x):
    values = np.vectorize(erf, otypes=[np.float32])(x / np.sqrt(2.0))
    return 0.5 * x * (1.0 + values)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, keepdims=True)
    variance = np.mean(np.square(x - mean), keepdims=True)
    return ((x - mean) / np.sqrt(variance + eps)) * weight + bias


class StudentAgent:
    """GRU policy trained by behavioral cloning and V2 teacher distillation."""

    def __init__(self, config=None):
        self.config = config or {}
        default_path = Path(__file__).with_name("student_weights.npz")
        weights_path = Path(self.config.get("weights_path", default_path))
        if not weights_path.is_absolute():
            weights_path = Path.cwd() / weights_path
        self.weights = np.load(weights_path)
        self.prototypes = np.asarray(self.weights["specialist_prototypes"], dtype=np.float32)
        self.hidden_size = int(self.weights["base.gru.weight_hh_l0"].shape[1])
        self.reset()

    def _w(self, name):
        return np.asarray(self.weights[self.expert + "." + name], dtype=np.float32)

    def _linear(self, x, prefix):
        return x @ self._w(prefix + ".weight").T + self._w(prefix + ".bias")

    def reset(self):
        self.expert = None
        self.hidden = np.zeros(self.hidden_size, dtype=np.float32)
        self.previous_action = 4
        self.timestep = 0

    def _features(self, obs):
        if isinstance(obs, dict):
            raw = obs["obs"]
            agent_index = int(obs.get("agent_index", 0))
        else:
            raw = obs
            agent_index = 0
        x = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self.expert is None:
            distances = np.max(np.abs(self.prototypes - x[None, :]), axis=1)
            self.expert = "s4" if float(np.min(distances)) < 1e-4 else "base"
        x = (x - self._w("obs_mean")) / np.maximum(self._w("obs_std"), 1e-5)
        agent = np.zeros(2, dtype=np.float32)
        agent[np.clip(agent_index, 0, 1)] = 1.0
        previous = np.zeros(6, dtype=np.float32)
        previous[np.clip(self.previous_action, 0, 5)] = 1.0
        start = np.asarray([1.0 if self.timestep == 0 else 0.0], dtype=np.float32)
        return np.concatenate((x, agent, previous, start))

    def act(self, obs):
        x = self._features(obs)
        x = self._linear(x, "encoder.0")
        x = _gelu(_layer_norm(x, self._w("encoder.1.weight"), self._w("encoder.1.bias")))
        x = _gelu(self._linear(x, "encoder.3"))

        gi = self._w("gru.weight_ih_l0") @ x + self._w("gru.bias_ih_l0")
        gh = self._w("gru.weight_hh_l0") @ self.hidden + self._w("gru.bias_hh_l0")
        size = self.hidden_size
        i_r, i_z, i_n = gi[:size], gi[size : 2 * size], gi[2 * size :]
        h_r, h_z, h_n = gh[:size], gh[size : 2 * size], gh[2 * size :]
        reset = _sigmoid(i_r + h_r)
        update = _sigmoid(i_z + h_z)
        candidate = np.tanh(i_n + reset * h_n)
        self.hidden = ((1.0 - update) * candidate + update * self.hidden).astype(np.float32)

        y = _layer_norm(self.hidden, self._w("norm.weight"), self._w("norm.bias"))
        action = int(np.argmax(self._linear(y, "head")))
        self.previous_action = action
        self.timestep += 1
        return action
