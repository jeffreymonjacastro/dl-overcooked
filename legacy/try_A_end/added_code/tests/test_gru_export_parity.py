from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.option_a_gru_policy import OptionAGRU
from policies.template import _OptionAGRUNumpy
from training.datasets import build_timestep_features


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "option_a_baseline_before_a2" / "option_a" / "best_checkpoint.pt"
NPZ = ROOT / "artifacts" / "option_a_baseline_before_a2" / "option_a" / "final_policy.npz"
MODEL_CONFIG = ROOT / "artifacts" / "option_a_baseline_before_a2" / "option_a" / "final_policy_config.json"
NORMALIZATION = ROOT / "artifacts" / "shared" / "normalization.json"
MANIFEST = ROOT / "artifacts" / "shared" / "split_manifest.json"
METRICS_PATH = ROOT / "reports" / "diagnostics" / "parity_metrics.json"


def _load_episode() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if entry["split"] != "train":
            continue
        path = ROOT / entry["source_path"]
        with np.load(path, allow_pickle=True) as data:
            if "obs" not in data.files or "actions" not in data.files:
                continue
            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.int64)
            if obs.ndim == 2 and len(obs) >= 64 and len(obs) == len(actions):
                return obs[:250], actions[:250]
    raise RuntimeError("No usable episode found for parity test")


class GRUExportParityTest(unittest.TestCase):
    def test_pytorch_checkpoint_matches_numpy_export(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = OptionAGRU(
            input_dim=int(checkpoint["input_dim"]),
            hidden_size=int(checkpoint["config"].get("hidden_size", 128)),
            dropout=float(checkpoint["config"].get("dropout", 0.10)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        norm = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        obs, actions = _load_episode()

        aggregate = {
            "max_abs_error_logits": 0.0,
            "max_abs_error_hidden": 0.0,
            "num_steps": 0,
            "num_action_matches": 0,
        }

        for agent_index in (0, 1):
            agent_indices = np.full(len(actions), agent_index, dtype=np.int64)
            features = build_timestep_features(obs, actions, agent_indices, mean, std)
            np_policy = _OptionAGRUNumpy(NPZ, MODEL_CONFIG)
            hidden = None

            for t in range(len(actions)):
                x_t = torch.from_numpy(features[t]).view(1, 1, -1)
                with torch.no_grad():
                    torch_logits, hidden = model(x_t, hidden)
                if t == 0:
                    np_policy.previous_action = 4
                else:
                    np_policy.previous_action = int(actions[t - 1])
                np_policy.timestep = t
                np_logits = np_policy.logits({"obs": obs[t], "agent_index": agent_index})

                torch_logits_np = torch_logits[0, 0].detach().numpy()
                torch_hidden_np = hidden[0, 0].detach().numpy()
                aggregate["max_abs_error_logits"] = max(
                    aggregate["max_abs_error_logits"],
                    float(np.max(np.abs(torch_logits_np - np_logits))),
                )
                aggregate["max_abs_error_hidden"] = max(
                    aggregate["max_abs_error_hidden"],
                    float(np.max(np.abs(torch_hidden_np - np_policy.hidden))),
                )
                aggregate["num_steps"] += 1
                aggregate["num_action_matches"] += int(np.argmax(torch_logits_np) == np.argmax(np_logits))

        aggregate["action_match_rate"] = aggregate["num_action_matches"] / max(aggregate["num_steps"], 1)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

        self.assertLessEqual(aggregate["max_abs_error_logits"], 1e-5)
        self.assertLessEqual(aggregate["max_abs_error_hidden"], 1e-5)
        self.assertEqual(aggregate["action_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

