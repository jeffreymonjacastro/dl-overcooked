"""Dataset loader for Option B BC warm-start training.

Loads .npz demonstration files recursively, applies quality tiers,
builds observation stacks (K consecutive frames), and returns
DataLoader-compatible datasets.

Design decisions (from GUIA_CORRECCIONES_OPTION_A2.md):
- previous_action[t] = action[t-1], never action[t]
- t=0: previous_action is zero vector (BOS)
- Quality tiers: A (high), B (normal), C (low weight), D (excluded from BC)
- No mixing of npz+pkl duplicates
- Normalization stats computed only from train split
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Quality tier assignment
# ---------------------------------------------------------------------------

def compute_quality_tier(rewards: np.ndarray, actions: np.ndarray) -> tuple[str, float]:
    """Assign quality tier and weight to an episode.
    
    Tier A: multiple soups, high score -> weight 1.5
    Tier B: at least 1 soup, consistent progress -> weight 1.0
    Tier C: 0 soups but useful navigation/interactions -> weight 0.1
    Tier D: inactive/loops/corrupt -> weight 0.0 (excluded)
    """
    deliveries = int((rewards > 0).sum())
    num_steps = len(actions)
    interact_count = int((actions == 5).sum())
    stay_count = int((actions == 4).sum())
    stay_ratio = stay_count / max(num_steps, 1)

    # Check for loops: longest repeated action run
    max_run = 1
    cur_run = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    if deliveries >= 3:
        return "A", 1.5
    elif deliveries >= 2:
        return "A", 1.5
    elif deliveries >= 1:
        return "B", 1.0
    elif interact_count > 5 and stay_ratio < 0.8 and max_run < 50:
        return "C", 0.1
    elif stay_ratio > 0.9 or max_run > 100:
        return "D", 0.0
    else:
        return "C", 0.4


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_episodes_from_npz(npz_path: pathlib.Path) -> list[dict]:
    """Load all episodes from a single .npz file.
    
    Returns list of per-episode dicts with keys:
        obs, actions, rewards, agent_index, source_path, episode_id
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return []

    obs = data["obs"]           # (T, obs_dim)
    actions = data["actions"]   # (T,)
    rewards = data.get("rewards", np.zeros(len(actions), dtype=np.float32))
    dones = data.get("dones", np.zeros(len(actions), dtype=bool))
    if len(dones) > 0:
        dones[-1] = True
    episode_ids = data.get("episode_ids", np.zeros(len(obs), dtype=int))
    agent_indices = data.get("agent_indices", np.zeros(len(obs), dtype=int))

    # Split into individual episodes by done=True boundaries
    episodes = []
    ep_start = 0
    unique_ep_ids = np.unique(episode_ids)

    for ep_id in unique_ep_ids:
        mask = episode_ids == ep_id
        ep_obs = obs[mask]
        ep_actions = actions[mask]
        ep_rewards = rewards[mask]
        ep_agent_idx = int(agent_indices[mask][0]) if agent_indices[mask].size > 0 else 0

        if len(ep_obs) < 5:
            continue

        tier, weight = compute_quality_tier(ep_rewards, ep_actions)
        if tier == "D":
            continue

        episodes.append({
            "obs": ep_obs.astype(np.float32),
            "actions": ep_actions.astype(np.int64),
            "rewards": ep_rewards.astype(np.float32),
            "agent_index": ep_agent_idx,
            "source_path": str(npz_path),
            "episode_id": int(ep_id),
            "quality_tier": tier,
            "quality_weight": weight,
            "reward_sum": float(ep_rewards.sum()),
            "deliveries": int((ep_rewards > 0).sum()),
        })

    return episodes


def load_all_episodes(data_root: str = "data", exclude_tiers: Optional[list[str]] = None) -> list[dict]:
    """Recursively load all episodes from .npz files."""
    if exclude_tiers is None:
        exclude_tiers = ["D"]

    data_path = pathlib.Path(data_root)
    all_episodes = []
    for npz_file in sorted(data_path.rglob("*.npz")):
        episodes = load_episodes_from_npz(npz_file)
        for ep in episodes:
            if ep["quality_tier"] not in exclude_tiers:
                all_episodes.append(ep)

    return all_episodes


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_normalization(episodes: list[dict], eps: float = 1e-8) -> dict:
    """Compute per-feature mean and std from a list of episodes."""
    all_obs = np.concatenate([ep["obs"] for ep in episodes], axis=0)
    mean = all_obs.mean(axis=0)
    std = all_obs.std(axis=0)
    std = np.maximum(std, eps)
    return {"mean": mean.tolist(), "std": std.tolist()}


def load_normalization(norm_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load normalization stats from JSON file."""
    with open(norm_path) as f:
        norm = json.load(f)
    return np.array(norm["mean"], dtype=np.float32), np.array(norm["std"], dtype=np.float32)


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / std


# ---------------------------------------------------------------------------
# Train/Val splits
# ---------------------------------------------------------------------------

def make_splits(
    episodes: list[dict],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split episodes into train/val/test. Unit is full episode (no timestep leakage)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(episodes)).tolist()

    n = len(episodes)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [episodes[i] for i in train_idx],
        "val": [episodes[i] for i in val_idx],
        "test": [episodes[i] for i in test_idx],
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ObsStackBCDataset(Dataset):
    """Dataset for BC warm-start training with observation stacking.
    
    For each timestep t in each episode, provides:
        stack_obs: K consecutive normalized observations (K * obs_dim,)
        agent_index: int in {0, 1}
        prev_action: action at t-1 (0 for t=0, BOS token)
        target_action: action at t
        weight: episode quality weight * class weight
    """

    def __init__(
        self,
        episodes: list[dict],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        k_stack: int = 4,
        action_weights: Optional[np.ndarray] = None,
    ):
        self.k_stack = k_stack
        self.obs_dim = norm_mean.shape[0]
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Pre-compute class weights for balanced training
        if action_weights is None:
            # Count action frequencies
            all_actions = np.concatenate([ep["actions"] for ep in episodes])
            counts = np.bincount(all_actions, minlength=6).astype(float)
            counts = np.maximum(counts, 1.0)
            # Inverse frequency weighting
            action_weights = counts.sum() / (6 * counts)
            action_weights = action_weights / action_weights.mean()
            # Clip to avoid extreme weights
            action_weights = np.clip(action_weights, 0.5, 3.0)
        self.action_weights = action_weights.astype(np.float32)

        # Flatten all timesteps into a list for fast indexing
        self.samples: list[tuple] = []
        for ep in episodes:
            obs_raw = ep["obs"]  # (T, obs_dim)
            actions = ep["actions"]  # (T,)
            agent_idx = ep["agent_index"]
            ep_weight = ep["quality_weight"]
            T = len(obs_raw)

            # Normalize
            obs_norm = normalize_obs(obs_raw, norm_mean, norm_std)

            for t in range(T):
                # Build K-stack: pad with first obs if t < K
                stack_frames = []
                for k in range(k_stack - 1, -1, -1):
                    frame_t = max(0, t - k)
                    stack_frames.append(obs_norm[frame_t])
                stack_obs = np.concatenate(stack_frames, axis=0).astype(np.float32)

                # previous_action: action[t-1], BOS=0 for t=0
                prev_action = int(actions[t - 1]) if t > 0 else 0
                target_action = int(actions[t])

                # Sample weight
                class_w = float(self.action_weights[target_action])
                sample_w = ep_weight * class_w
                sample_w = float(np.clip(sample_w, 0.05, 5.0))

                self.samples.append((stack_obs, agent_idx, prev_action, target_action, sample_w))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stack_obs, agent_idx, prev_action, target_action, weight = self.samples[idx]
        return (
            torch.from_numpy(stack_obs),
            torch.tensor(agent_idx, dtype=torch.long),
            torch.tensor(prev_action, dtype=torch.long),
            torch.tensor(target_action, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )


def make_dataloaders(
    splits: dict[str, list[dict]],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    k_stack: int = 4,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val DataLoaders."""
    loaders = {}
    for split_name, episodes in splits.items():
        if not episodes:
            continue
        dataset = ObsStackBCDataset(episodes, norm_mean, norm_std, k_stack=k_stack)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=split_name == "train",
        )
    return loaders
