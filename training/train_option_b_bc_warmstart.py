"""BC Warm-Start training for Option B.

Trains an MLP actor on human demonstrations using obs stacking (K=4).
Saves best checkpoint by validation loss.

Usage:
    python -m training.train_option_b_bc_warmstart --config configs/train_option_b.yaml

Key design decisions (from GUIA_CORRECCIONES_OPTION_A2.md):
- Checkpoint selection: zero-soup rate >> soups >> val_loss (use val_loss here as proxy)
- previous_action aligned correctly: input[t].prev = action[t-1]
- Quality weights applied in loss
- Balanced action class weights to avoid "stay" collapse
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from models.option_b_actor_critic import BCWarmstartActor
from training.datasets import (
    load_all_episodes,
    compute_normalization,
    make_splits,
    make_dataloaders,
    load_normalization,
)


def train_bc(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bc_cfg = cfg.get("bc_warmstart", {})
    data_root = cfg.get("data_root", "data")
    artifacts_dir = pathlib.Path(cfg.get("artifacts_dir", "artifacts/option_b"))
    shared_dir = pathlib.Path(cfg.get("shared_dir", "artifacts/shared"))
    reports_dir = pathlib.Path(cfg.get("reports_dir", "reports"))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] Device: {device}")

    # Load and split episodes
    print("[BC] Loading episodes...")
    episodes = load_all_episodes(data_root=data_root, exclude_tiers=["D"])
    print(f"[BC] Loaded {len(episodes)} usable episodes")

    splits = make_splits(
        episodes,
        train_frac=bc_cfg.get("train_frac", 0.80),
        val_frac=bc_cfg.get("val_frac", 0.10),
        seed=bc_cfg.get("seed", 42),
    )
    print(f"[BC] Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Normalization from train only
    norm_path = shared_dir / "normalization.json"
    if norm_path.exists():
        print(f"[BC] Loading normalization from {norm_path}")
        norm_mean, norm_std = load_normalization(str(norm_path))
    else:
        print("[BC] Computing normalization from train...")
        norm_stats = compute_normalization(splits["train"])
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        norm_mean = np.array(norm_stats["mean"], dtype=np.float32)
        norm_std = np.array(norm_stats["std"], dtype=np.float32)
        print(f"[BC] Normalization saved to {norm_path}")

    # Build DataLoaders
    k_stack = bc_cfg.get("k_stack", 4)
    batch_size = bc_cfg.get("batch_size", 512)
    loaders = make_dataloaders(splits, norm_mean, norm_std, k_stack=k_stack, batch_size=batch_size)

    # Model
    obs_dim = int(norm_mean.shape[0])
    hidden_sizes = tuple(bc_cfg.get("hidden_sizes", [512, 256, 128]))
    model = BCWarmstartActor(
        obs_dim=obs_dim,
        k_stack=k_stack,
        num_actions=6,
        hidden_sizes=hidden_sizes,
        dropout=bc_cfg.get("dropout", 0.1),
    ).to(device)
    print(f"[BC] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    lr = bc_cfg.get("lr", 3e-4)
    weight_decay = bc_cfg.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = bc_cfg.get("epochs", 50)
    patience = bc_cfg.get("patience", 10)
    label_smoothing = bc_cfg.get("label_smoothing", 0.02)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    print(f"[BC] Training for up to {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for stack_obs, agent_idx, prev_action, target_action, sample_weight in train_loader:
            stack_obs = stack_obs.to(device)
            agent_idx = agent_idx.to(device)
            prev_action = prev_action.to(device)
            target_action = target_action.to(device)
            sample_weight = sample_weight.to(device)

            logits = model(stack_obs, agent_idx, prev_action)

            # Weighted cross-entropy with label smoothing
            ce_loss = nn.functional.cross_entropy(
                logits, target_action, reduction="none", label_smoothing=label_smoothing
            )
            loss = (ce_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(target_action)
            train_correct += (logits.argmax(-1) == target_action).sum().item()
            train_total += len(target_action)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- VAL ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for stack_obs, agent_idx, prev_action, target_action, sample_weight in val_loader:
                stack_obs = stack_obs.to(device)
                agent_idx = agent_idx.to(device)
                prev_action = prev_action.to(device)
                target_action = target_action.to(device)
                sample_weight = sample_weight.to(device)

                logits = model(stack_obs, agent_idx, prev_action)
                ce_loss = nn.functional.cross_entropy(logits, target_action, reduction="none")
                loss = (ce_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)

                val_loss_sum += loss.item() * len(target_action)
                val_correct += (logits.argmax(-1) == target_action).sum().item()
                val_total += len(target_action)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "elapsed_s": round(elapsed, 2),
        }
        history.append(row)
        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = artifacts_dir / "bc_warmstart.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "obs_dim": obs_dim,
                    "k_stack": k_stack,
                    "num_actions": 6,
                    "hidden_sizes": list(hidden_sizes),
                    "dropout": bc_cfg.get("dropout", 0.1),
                },
            }, best_path)
            print(f"    ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[BC] Early stopping at epoch {epoch}")
                break

    # Save training CSV
    csv_path = reports_dir / "option_b_bc_training.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[BC] Training log saved to {csv_path}")
    print(f"[BC] Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_option_b.yaml")
    args = parser.parse_args()
    train_bc(args.config)
