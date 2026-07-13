"""Train Option A2: full-episode recurrent behavioral cloning."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.option_a2_gru_policy import OptionAGRU
from training.datasets import (
    EpisodeSequenceDataset,
    action_class_weights,
    load_episodes,
    load_extra_npz_episodes,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_option_a2.yaml")
    return parser.parse_args()


def masked_weighted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    weights = class_weights[targets] * quality[:, None] * mask
    return (nll * weights).sum() / torch.clamp(weights.sum(), min=1.0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, class_weights: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    correct = 0.0
    total = 0.0
    per_action_correct = np.zeros(6, dtype=np.float64)
    per_action_total = np.zeros(6, dtype=np.float64)
    for x, y, mask, quality in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        quality = quality.to(device)
        logits, _ = model(x)
        loss = masked_weighted_ce(logits, y, mask, quality, class_weights)
        batch_weight = float(mask.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
        pred = logits.argmax(dim=-1)
        valid = mask.bool()
        correct += float(((pred == y) & valid).sum().item())
        total += batch_weight
        for action in range(6):
            action_mask = (y == action) & valid
            per_action_total[action] += float(action_mask.sum().item())
            per_action_correct[action] += float(((pred == y) & action_mask).sum().item())
    per_action_accuracy = {
        str(i): float(per_action_correct[i] / per_action_total[i]) if per_action_total[i] else 0.0
        for i in range(6)
    }
    return {
        "loss": total_loss / max(total_weight, 1.0),
        "accuracy": correct / max(total, 1.0),
        "per_action_accuracy": per_action_accuracy,
    }


def export_npz(model: OptionAGRU, output_path: Path) -> None:
    arrays = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in model.state_dict().items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 67))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    split_manifest_path = config["split_manifest_path"]
    normalization_path = config["normalization_path"]
    train_episodes = load_episodes(split_manifest_path, {"train"}, config.get("max_train_episodes"))
    extra_paths: list[str] = []
    for pattern in config.get("extra_train_npz_globs", []) or []:
        extra_paths.extend(sorted(glob.glob(str(pattern))))
    if extra_paths:
        train_episodes.extend(
            load_extra_npz_episodes(
                extra_paths,
                min_positive_rewards=int(config.get("extra_min_positive_rewards", 0)),
            )
        )
    val_episodes = load_episodes(
        split_manifest_path,
        {"validation_seen_layout", "validation_unseen_layout", "validation_combined"},
        config.get("max_val_episodes"),
    )
    if not train_episodes:
        raise RuntimeError("No train episodes available")
    if not val_episodes:
        val_episodes = train_episodes[: min(16, len(train_episodes))]

    max_seq_len = int(config.get("max_seq_len", 250))
    train_ds = EpisodeSequenceDataset(train_episodes, normalization_path, max_seq_len=max_seq_len)
    val_ds = EpisodeSequenceDataset(val_episodes, normalization_path, max_seq_len=max_seq_len)
    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = train_ds.items[0][0].shape[1]
    model = OptionAGRU(
        input_dim=input_dim,
        hidden_size=int(config.get("hidden_size", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)
    class_weights = action_class_weights(train_episodes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    output_dir = Path(config.get("output_dir", "artifacts/option_a2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(config.get("training_log_path", "reports/option_a2/training.csv"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_path = output_dir / "best_by_val_loss.pt"
    latest_path = output_dir / "latest.pt"

    epochs = int(config.get("epochs", 10))
    best_val = float("inf")
    best_epoch = -1
    started = time.time()
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        for x, y, mask, quality in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            quality = quality.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = masked_weighted_ce(logits, y, mask, quality, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            weight = float(mask.sum().item())
            train_loss += float(loss.item()) * weight
            train_weight += weight
        val = evaluate(model, val_loader, device, class_weights)
        train_loss = train_loss / max(train_weight, 1.0)
        row = {
            "epoch": epoch,
            "model_type": "gru_bc_a2_full_episode",
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "per_action_accuracy_json": json.dumps(val["per_action_accuracy"], sort_keys=True),
            "device": str(device),
            "seconds_elapsed": time.time() - started,
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "max_seq_len": max_seq_len,
        }
        rows.append(row)
        print(json.dumps(row))
        checkpoint_payload = {
            "model_type": "gru_bc_a2_full_episode",
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "config": config,
            "class_weights": class_weights.detach().cpu(),
            "epoch": epoch,
            "val_loss": val["loss"],
        }
        torch.save(checkpoint_payload, latest_path)
        if val["loss"] < best_val:
            best_val = float(val["loss"])
            best_epoch = epoch
            torch.save(checkpoint_payload | {"best_val_loss": best_val}, best_val_path)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = torch.load(best_val_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.cpu()
    export_npz(model, output_dir / "final_policy.npz")

    norm = load_json(normalization_path)
    final_config = {
        "option": "A2.1",
        "model_type": "gru_bc_a2_full_episode",
        "obs_dim": int(norm["obs_dim"]),
        "input_dim": int(input_dim),
        "num_actions": 6,
        "agent_index_encoding": "one_hot",
        "previous_action": True,
        "start_flag": True,
        "hidden_size": int(config.get("hidden_size", 128)),
        "num_layers": 1,
        "normalization_path": str(normalization_path),
        "checkpoint_path": "artifacts/option_a2/final_policy.npz",
        "max_seq_len": max_seq_len,
        "training_unit": "full_episode_padded",
        "action_mapping": {
            "0": "north",
            "1": "south",
            "2": "east",
            "3": "west",
            "4": "stay",
            "5": "interact",
        },
        "training_device": str(device),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
    }
    config_path = output_dir / "final_policy_config.json"
    config_path.write_text(json.dumps(final_config, indent=2), encoding="utf-8")

    if bool(config.get("publish_final", True)):
        shutil.copy2(output_dir / "final_policy.npz", "artifacts/final_policy.npz")
        shutil.copy2(config_path, "artifacts/final_policy_config.json")

    summary = {
        "best_checkpoint": str(best_val_path),
        "exported_npz": str(output_dir / "final_policy.npz"),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "device": str(device),
    }
    (report_path.parent / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
