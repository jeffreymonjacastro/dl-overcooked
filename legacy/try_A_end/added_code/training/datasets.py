"""Dataset utilities for option A recurrent behavioral cloning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_ACTIONS = 6


@dataclass(frozen=True)
class Episode:
    path: Path
    split: str
    source_group: str
    layout: str
    partner: str
    obs: np.ndarray
    actions: np.ndarray
    agent_indices: np.ndarray
    quality_weight: float


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def quality_weight_from_rewards(data: Any) -> float:
    if "rewards" not in data.files:
        return 0.75
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    positive = int(np.sum(rewards > 0))
    if positive >= 3:
        return 1.50
    if positive == 2:
        return 1.25
    if positive == 1:
        return 1.00
    return 0.40


def load_episodes(
    split_manifest_path: str | Path,
    split_names: set[str],
    max_episodes: int | None = None,
) -> list[Episode]:
    manifest = load_json(split_manifest_path)
    episodes: list[Episode] = []
    for entry in manifest["entries"]:
        if entry["split"] not in split_names:
            continue
        path = resolve_repo_path(entry["source_path"])
        try:
            with np.load(path, allow_pickle=True) as data:
                if "obs" not in data.files or "actions" not in data.files:
                    continue
                obs = np.asarray(data["obs"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.int64)
                if obs.ndim != 2 or len(obs) != len(actions) or len(obs) == 0:
                    continue
                if actions.min(initial=0) < 0 or actions.max(initial=0) >= NUM_ACTIONS:
                    continue
                if "agent_indices" in data.files:
                    agent_indices = np.asarray(data["agent_indices"], dtype=np.int64)
                    if len(agent_indices) != len(actions):
                        agent_indices = np.ones(len(actions), dtype=np.int64)
                else:
                    agent_indices = np.ones(len(actions), dtype=np.int64)
                episodes.append(
                    Episode(
                        path=path,
                        split=entry["split"],
                        source_group=str(entry.get("source_group", "unknown")),
                        layout=str(entry.get("layout", "unknown")),
                        partner=str(entry.get("partner", "unknown")),
                        obs=obs,
                        actions=actions,
                        agent_indices=np.clip(agent_indices, 0, 1),
                        quality_weight=quality_weight_from_rewards(data),
                    )
                )
        except Exception:
            continue
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return episodes


def load_extra_npz_episodes(
    paths: list[str | Path],
    source_group: str = "greedy_teacher",
    min_positive_rewards: int = 0,
) -> list[Episode]:
    episodes: list[Episode] = []
    for raw_path in paths:
        path = resolve_repo_path(raw_path)
        try:
            with np.load(path, allow_pickle=True) as data:
                if "obs" not in data.files or "actions" not in data.files:
                    continue
                obs = np.asarray(data["obs"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.int64)
                if obs.ndim != 2 or len(obs) != len(actions) or len(obs) == 0:
                    continue
                if actions.min(initial=0) < 0 or actions.max(initial=0) >= NUM_ACTIONS:
                    continue
                if min_positive_rewards > 0:
                    rewards = np.asarray(data["rewards"], dtype=np.float32) if "rewards" in data.files else np.zeros(len(actions), dtype=np.float32)
                    if int(np.sum(rewards > 0)) < min_positive_rewards:
                        continue
                agent_indices = (
                    np.asarray(data["agent_indices"], dtype=np.int64)
                    if "agent_indices" in data.files and len(data["agent_indices"]) == len(actions)
                    else np.zeros(len(actions), dtype=np.int64)
                )
                episodes.append(
                    Episode(
                        path=path,
                        split="train",
                        source_group=source_group,
                        layout="teacher_rollout",
                        partner="teacher_rollout",
                        obs=obs,
                        actions=actions,
                        agent_indices=np.clip(agent_indices, 0, 1),
                        quality_weight=min(1.0, quality_weight_from_rewards(data)),
                    )
                )
        except Exception:
            continue
    return episodes


def build_timestep_features(
    obs: np.ndarray,
    actions: np.ndarray,
    agent_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    norm_obs = (obs - mean[None, :]) / np.maximum(std[None, :], 1e-6)
    n = len(actions)
    agent_onehot = np.zeros((n, 2), dtype=np.float32)
    agent_onehot[np.arange(n), np.clip(agent_indices, 0, 1)] = 1.0
    prev_actions = np.full(n, 4, dtype=np.int64)
    if n > 1:
        prev_actions[1:] = actions[:-1]
    prev_onehot = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    prev_onehot[np.arange(n), np.clip(prev_actions, 0, NUM_ACTIONS - 1)] = 1.0
    start_flag = np.zeros((n, 1), dtype=np.float32)
    start_flag[0, 0] = 1.0
    return np.concatenate([norm_obs.astype(np.float32), agent_onehot, prev_onehot, start_flag], axis=1)


class SequenceDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        normalization_path: str | Path,
        seq_len: int = 32,
        stride: int = 16,
    ):
        norm = load_json(normalization_path)
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.seq_len = int(seq_len)
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for ep in episodes:
            x = build_timestep_features(ep.obs, ep.actions, ep.agent_indices, self.mean, self.std)
            y = ep.actions.astype(np.int64, copy=False)
            starts = list(range(0, max(1, len(y)), stride))
            for start in starts:
                end = min(start + self.seq_len, len(y))
                if end <= start:
                    continue
                pad = self.seq_len - (end - start)
                x_win = np.zeros((self.seq_len, x.shape[1]), dtype=np.float32)
                y_win = np.zeros((self.seq_len,), dtype=np.int64)
                mask = np.zeros((self.seq_len,), dtype=np.float32)
                x_win[: end - start] = x[start:end]
                y_win[: end - start] = y[start:end]
                mask[: end - start] = 1.0
                if pad > 0:
                    y_win[end - start :] = 4
                self.items.append((x_win, y_win, mask, float(ep.quality_weight)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y, mask, quality = self.items[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.tensor(quality, dtype=torch.float32),
        )


class EpisodeSequenceDataset(Dataset):
    """Full-episode dataset for recurrent training.

    This avoids training a GRU on mid-episode windows that silently start with a
    zero hidden state. Episodes are padded to `max_seq_len`, and the mask marks
    real timesteps.
    """

    def __init__(
        self,
        episodes: list[Episode],
        normalization_path: str | Path,
        max_seq_len: int = 250,
    ):
        norm = load_json(normalization_path)
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        self.max_seq_len = int(max_seq_len)
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for ep in episodes:
            x = build_timestep_features(ep.obs, ep.actions, ep.agent_indices, self.mean, self.std)
            y = ep.actions.astype(np.int64, copy=False)
            n = min(len(y), self.max_seq_len)
            if n <= 0:
                continue
            x_ep = np.zeros((self.max_seq_len, x.shape[1]), dtype=np.float32)
            y_ep = np.full((self.max_seq_len,), 4, dtype=np.int64)
            mask = np.zeros((self.max_seq_len,), dtype=np.float32)
            x_ep[:n] = x[:n]
            y_ep[:n] = y[:n]
            mask[:n] = 1.0
            self.items.append((x_ep, y_ep, mask, float(ep.quality_weight)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y, mask, quality = self.items[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.tensor(quality, dtype=torch.float32),
        )


def action_class_weights(episodes: list[Episode]) -> torch.Tensor:
    counts = np.ones(NUM_ACTIONS, dtype=np.float64)
    for ep in episodes:
        for action in ep.actions:
            if 0 <= int(action) < NUM_ACTIONS:
                counts[int(action)] += 1.0
    inv = counts.sum() / (NUM_ACTIONS * counts)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)
