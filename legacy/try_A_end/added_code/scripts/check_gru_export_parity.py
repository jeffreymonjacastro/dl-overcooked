"""Check PyTorch GRU checkpoint vs NumPy NPZ export parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.option_a_gru_policy import OptionAGRU
from policies.template import _OptionAGRUNumpy
from training.datasets import build_timestep_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", default="artifacts/shared/split_manifest.json")
    parser.add_argument("--normalization", default="artifacts/shared/normalization.json")
    parser.add_argument("--output", default="reports/diagnostics/parity_metrics_a2.json")
    return parser.parse_args()


def load_episode(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if entry["split"] != "train":
            continue
        path = Path(entry["source_path"])
        with np.load(path, allow_pickle=True) as data:
            if "obs" not in data.files or "actions" not in data.files:
                continue
            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.int64)
            if obs.ndim == 2 and len(obs) >= 64 and len(obs) == len(actions):
                return obs[:250], actions[:250]
    raise RuntimeError("No usable episode found")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = OptionAGRU(
        input_dim=int(checkpoint["input_dim"]),
        hidden_size=int(config.get("hidden_size", 128)),
        dropout=float(config.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    norm = json.loads(Path(args.normalization).read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    obs, actions = load_episode(Path(args.manifest))

    metrics = {
        "max_abs_error_logits": 0.0,
        "max_abs_error_hidden": 0.0,
        "num_steps": 0,
        "num_action_matches": 0,
    }
    for agent_index in (0, 1):
        agent_indices = np.full(len(actions), agent_index, dtype=np.int64)
        features = build_timestep_features(obs, actions, agent_indices, mean, std)
        np_policy = _OptionAGRUNumpy(args.npz, args.model_config)
        hidden = None
        for t in range(len(actions)):
            with torch.no_grad():
                torch_logits, hidden = model(torch.from_numpy(features[t]).view(1, 1, -1), hidden)
            np_policy.previous_action = 4 if t == 0 else int(actions[t - 1])
            np_policy.timestep = t
            np_logits = np_policy.logits({"obs": obs[t], "agent_index": agent_index})
            torch_logits_np = torch_logits[0, 0].detach().numpy()
            torch_hidden_np = hidden[0, 0].detach().numpy()
            metrics["max_abs_error_logits"] = max(metrics["max_abs_error_logits"], float(np.max(np.abs(torch_logits_np - np_logits))))
            metrics["max_abs_error_hidden"] = max(metrics["max_abs_error_hidden"], float(np.max(np.abs(torch_hidden_np - np_policy.hidden))))
            metrics["num_steps"] += 1
            metrics["num_action_matches"] += int(np.argmax(torch_logits_np) == np.argmax(np_logits))
    metrics["action_match_rate"] = metrics["num_action_matches"] / max(metrics["num_steps"], 1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if metrics["max_abs_error_logits"] > 1e-5 or metrics["max_abs_error_hidden"] > 1e-5 or metrics["action_match_rate"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

