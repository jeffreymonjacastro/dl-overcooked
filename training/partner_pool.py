"""Partner pool construction for FCP-style PPO training.

Provides a registry of partner policies and sampling strategies.

Partners included:
- bc_clone_t<N>: frozen Behavior Cloning partner sampled at temperature N
- historical_ppo_<N>: frozen PPO checkpoints (added during PPO training)

Sampling strategies:
- uniform: equal probability over all partners
- adaptive_cole: higher probability for partners where ego performs poorly
"""

from __future__ import annotations

import json
import pathlib
from collections import deque
from typing import Any, Optional

import numpy as np


class PartnerConfig:
    """Configuration for a single partner."""

    def __init__(
        self,
        partner_id: str,
        partner_type: str,  # "builtin" or "bc_checkpoint"
        layout: Optional[str] = None,
        seed: Optional[int] = None,
        random_action_prob: float = 0.0,
        sticky_action_prob: float = 0.0,
        checkpoint_path: Optional[str] = None,
        temperature: float = 1.0,
        snapshot_step: Optional[int] = None,
        skill_estimate: float = 0.5,
        behavior_stats: Optional[dict] = None,
    ):
        self.partner_id = partner_id
        self.partner_type = partner_type
        self.layout = layout
        self.seed = seed
        self.random_action_prob = random_action_prob
        self.sticky_action_prob = sticky_action_prob
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.snapshot_step = snapshot_step
        self.skill_estimate = skill_estimate
        self.behavior_stats = behavior_stats or {}

    def to_dict(self) -> dict:
        return {
            "partner_id": self.partner_id,
            "type": self.partner_type,
            "layout": self.layout,
            "seed": self.seed,
            "random_action_prob": self.random_action_prob,
            "sticky_action_prob": self.sticky_action_prob,
            "checkpoint_path": self.checkpoint_path,
            "temperature": self.temperature,
            "snapshot_step": self.snapshot_step,
            "skill_estimate": self.skill_estimate,
            "behavior_stats": self.behavior_stats,
        }


class NeuralPartner:
    """Overcooked-compatible agent wrapper around a frozen PyTorch policy."""

    def __init__(
        self,
        model: Any,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        env: Any = None,
        agent_index: int = 1,
        temperature: float = 1.0,
        seed: int = 42,
        device: Any = None,
        k_stack: int = 4,
        obs_dim: int = 96,
    ):
        import torch

        self.model = model
        self.model.eval()
        self.norm_mean = np.asarray(norm_mean, dtype=np.float32)
        self.norm_std = np.asarray(norm_std, dtype=np.float32)
        self.env = env
        self.agent_index = int(agent_index)
        self.temperature = max(float(temperature), 1e-6)
        self.rng = np.random.default_rng(seed)
        self.device = device or next(model.parameters()).device
        self.k_stack = int(k_stack)
        self.obs_dim = int(obs_dim)
        self.prev_action = 0
        self.obs_stack: deque[np.ndarray] = deque(maxlen=self.k_stack)
        self.reset()

    def reset(self) -> None:
        self.prev_action = 0
        self.obs_stack.clear()
        for _ in range(self.k_stack):
            self.obs_stack.append(np.zeros(self.obs_dim, dtype=np.float32))

    def set_agent_index(self, agent_index: int) -> None:
        if int(agent_index) != self.agent_index:
            self.agent_index = int(agent_index)
            self.reset()

    def set_env(self, env: Any) -> None:
        if env is not self.env:
            self.env = env
            self.reset()

    def set_mdp(self, mdp: Any) -> None:
        self.mdp = mdp

    def action(self, state: Any):
        import torch
        from overcooked_ai_py.mdp.actions import Action

        if self.env is None:
            raise ValueError("NeuralPartner requires env before action()")
        obs = self.env.featurize_state_mdp(state)[self.agent_index].astype(np.float32)
        self.obs_stack.append((obs - self.norm_mean) / self.norm_std)
        stack = np.concatenate(list(self.obs_stack)).astype(np.float32)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(stack).unsqueeze(0).to(self.device),
                torch.tensor([self.agent_index], dtype=torch.long, device=self.device),
                torch.tensor([self.prev_action], dtype=torch.long, device=self.device),
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits[0] / self.temperature, dim=-1).cpu().numpy()
        action_idx = int(self.rng.choice(len(probs), p=probs))
        self.prev_action = action_idx
        return Action.INDEX_TO_ACTION[action_idx], {}


class PartnerPool:
    """Registry of training partners and sampling logic."""

    def __init__(self, rng_seed: int = 42):
        self.partners: list[PartnerConfig] = []
        self.partner_scores: dict[str, list[float]] = {}  # pid -> recent scores
        self.rng = np.random.default_rng(rng_seed)

    def add(self, partner: PartnerConfig) -> None:
        self.partners.append(partner)
        self.partner_scores[partner.partner_id] = []

    def build_default_pool(self, temperatures: tuple[float, ...] = (0.75, 1.0, 1.5)) -> None:
        """Add BC clone partners; no GreedyHumanModel dependency."""
        for temp in temperatures:
            self.add(PartnerConfig(
                f"bc_clone_t{str(temp).replace('.', '_')}",
                "bc_clone",
                temperature=temp,
                skill_estimate=0.5,
            ))

    def add_historical_checkpoint(self, checkpoint_path: str, step: int, seed: int) -> str:
        """Add a frozen PPO checkpoint as a partner."""
        pid = f"historical_ppo_step{step}_seed{seed}"
        self.add(PartnerConfig(
            pid,
            partner_type="historical_ppo",
            checkpoint_path=checkpoint_path,
            snapshot_step=step,
            skill_estimate=0.5,
            behavior_stats={"training_step": step, "seed": seed},
        ))
        return pid

    def update_score(self, partner_id: str, score: float, max_history: int = 20) -> None:
        """Update recent performance score for a partner."""
        if partner_id not in self.partner_scores:
            self.partner_scores[partner_id] = []
        self.partner_scores[partner_id].append(float(score))
        if len(self.partner_scores[partner_id]) > max_history:
            self.partner_scores[partner_id] = self.partner_scores[partner_id][-max_history:]

    def sample_partner(
        self,
        strategy: str = "uniform",
        epsilon: float = 0.20,
        beta: float = 1.0,
    ) -> PartnerConfig:
        """Sample a partner for the next episode.
        
        Strategies:
        - "uniform": equal probability
        - "adaptive_cole": higher probability for harder partners
        """
        if not self.partners:
            raise ValueError("No partners in pool")

        if strategy == "uniform" or len(self.partners) == 1:
            idx = int(self.rng.integers(0, len(self.partners)))
            return self.partners[idx]

        if strategy == "adaptive_cole":
            # Compute average recent score per partner
            avg_scores = []
            for p in self.partners:
                hist = self.partner_scores.get(p.partner_id, [])
                avg_scores.append(float(np.mean(hist)) if hist else 0.5)

            avg_scores = np.array(avg_scores)
            # Normalize to [0, 1]
            score_range = avg_scores.max() - avg_scores.min()
            if score_range > 0:
                norm_scores = (avg_scores - avg_scores.min()) / score_range
            else:
                norm_scores = np.ones(len(avg_scores)) * 0.5

            # Higher probability for harder partners (lower score)
            probs = (1 - epsilon) * self._softmax(-beta * norm_scores)
            probs += epsilon / len(self.partners)
            probs /= probs.sum()

            idx = int(self.rng.choice(len(self.partners), p=probs))
            return self.partners[idx]

        raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def save_manifest(self, path: str) -> None:
        """Save pool manifest to JSON."""
        manifest = {
            "num_partners": len(self.partners),
            "partners": [p.to_dict() for p in self.partners],
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[PartnerPool] Saved manifest to {path}")

    def __len__(self) -> int:
        return len(self.partners)

    def __repr__(self) -> str:
        ids = [p.partner_id for p in self.partners]
        return f"PartnerPool({len(self.partners)} partners: {ids})"
