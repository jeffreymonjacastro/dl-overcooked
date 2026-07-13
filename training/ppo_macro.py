"""Macro-PPO warm-start/smoke training entrypoint.

This script deliberately trains over macro logits with synthetic warm-start
targets until real macro demonstrations are generated. It validates the RTX/GPU
path and emits a checkpoint that is marked as smoke-only, so it cannot be
mistaken for a competition policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.macro_actor_critic import build_default_model
from planning.macro_actions import MACRO_ACTIONS, MacroAction


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_action_mask(batch_size: int, device):
    return torch.ones(batch_size, len(MACRO_ACTIONS), dtype=torch.bool, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_macro_ppo.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    train_cfg = cfg.get("training", {})
    steps = int(args.steps or train_cfg.get("smoke_steps", 200))
    feature_dim = int(cfg.get("model", {}).get("feature_dim", 64))
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 128))
    batch_size = int(train_cfg.get("batch_size", 4096))
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("prefer_cuda", True) else "cpu")
    model = build_default_model(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)))

    target_macro = int(MacroAction.SOLO_PIPELINE)
    losses = []
    for step in range(steps):
        features = torch.randn(batch_size, feature_dim, device=device)
        role = F.one_hot(torch.randint(0, 2, (batch_size,), device=device), num_classes=2).float()
        partner = torch.zeros(batch_size, 8, device=device)
        mask = make_action_mask(batch_size, device)
        logits, value, _ = model(features, role, partner, action_mask=mask)
        targets = torch.full((batch_size,), target_macro, dtype=torch.long, device=device)
        bc_loss = F.cross_entropy(logits, targets)
        value_loss = 0.01 * torch.mean(value.square())
        loss = bc_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            losses.append(float(loss.detach().cpu()))

    artifact_dir = REPO_ROOT / "artifacts" / "macro_ppo"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "macro_actor_critic_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "smoke_only": True,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        checkpoint,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "smoke_only": True,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "steps": steps,
        "batch_size": batch_size,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "workers_recommended": min(64, max(1, (os.cpu_count() or 16) - 1)),
    }
    out = artifact_dir / "macro_ppo_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
