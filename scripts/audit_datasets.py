"""Audit all .npz demonstration files in the data/ directory.

Creates reports/dataset_inventory.csv and reports/dataset_duplicates.csv
following the schema from CODEX_GUIA_ENTRENAMIENTO_AB.md Section 6.2.

Usage:
    python -m scripts.audit_datasets
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np


def compute_quality_flags(obs, actions, rewards):
    """Compute quality flags for an episode."""
    flags = []
    stay_ratio = float((actions == 4).mean())
    interact_count = int((actions == 5).sum())
    reward_sum = float(rewards.sum())

    if stay_ratio > 0.8:
        flags.append("high_stay")
    if interact_count == 0:
        flags.append("no_interact")
    if reward_sum == 0:
        flags.append("zero_reward")

    max_run = 1
    cur_run = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    if max_run > 50:
        flags.append(f"long_repeat_{max_run}")

    return "|".join(flags) if flags else "ok"


def load_metadata_from_npz(npz_path: pathlib.Path, metadata_path: pathlib.Path | None = None) -> dict:
    """Extract metadata from a .npz file and its .metadata.json if available."""
    result = {
        "source_path": str(npz_path),
        "stem": npz_path.stem,
        "source_group": npz_path.parent.parent.name,
        "layout": "unknown",
        "partner": "unknown",
        "seed": -1,
        "agent_index": -1,
        "role_swap": False,
    }

    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            env_cfg = meta.get("environment", {})
            result["layout"] = env_cfg.get("layout_name", result["layout"])
            result["seed"] = meta.get("seed", result["seed"])
        except Exception:
            pass

    return result


def audit_datasets(data_root: str = "data", output_dir: str = "reports") -> None:
    """Run full dataset audit."""
    data_path = pathlib.Path(data_root)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    duplicates = []
    seen_stems = {}

    npz_files = sorted(data_path.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files")

    for npz_path in npz_files:
        stem = npz_path.stem

        # Check for duplicate
        if stem in seen_stems:
            duplicates.append({
                "stem": stem,
                "path_1": seen_stems[stem],
                "path_2": str(npz_path),
            })
            continue
        seen_stems[stem] = str(npz_path)

        # Look for metadata
        metadata_path = npz_path.with_suffix(".npz").parent / (stem + ".metadata.json")
        if not metadata_path.exists():
            metadata_path = npz_path.with_suffix("").with_suffix(".metadata.json")

        meta = load_metadata_from_npz(npz_path, metadata_path if metadata_path.exists() else None)

        try:
            data = np.load(npz_path, allow_pickle=True)
            obs = data["obs"]
            actions = data["actions"]
            rewards = data.get("rewards", np.zeros(len(actions)))
            episode_ids = data.get("episode_ids", np.zeros(len(obs), dtype=int))
            agent_indices = data.get("agent_indices", np.zeros(len(obs), dtype=int))

            for ep_id in np.unique(episode_ids):
                mask = episode_ids == ep_id
                ep_obs = obs[mask]
                ep_acts = actions[mask]
                ep_rews = rewards[mask]
                ep_agent = int(agent_indices[mask][0]) if agent_indices[mask].size > 0 else 0

                reward_sum = float(ep_rews.sum())
                soups = reward_sum / 20.0

                # First and last delivery
                delivery_ts = [i for i, r in enumerate(ep_rews) if r > 0]
                first_soup = delivery_ts[0] if delivery_ts else -1
                last_soup = delivery_ts[-1] if delivery_ts else -1

                flags = compute_quality_flags(ep_obs, ep_acts, ep_rews)

                # Quality tier
                if soups >= 2:
                    tier = "A"
                elif soups >= 1:
                    tier = "B"
                elif (ep_acts == 5).sum() > 5 and (ep_acts == 4).mean() < 0.8:
                    tier = "C"
                else:
                    tier = "D"

                rows.append({
                    "episode_id": f"{stem}_{int(ep_id)}",
                    "source_group": meta["source_group"],
                    "source_path": meta["source_path"],
                    "stem": stem,
                    "format": "npz",
                    "obs_shape": f"{ep_obs.shape}",
                    "obs_dim": ep_obs.shape[1] if ep_obs.ndim > 1 else ep_obs.shape[0],
                    "num_steps": len(ep_obs),
                    "action_min": int(ep_acts.min()),
                    "action_max": int(ep_acts.max()),
                    "layout": meta["layout"],
                    "partner": meta["partner"],
                    "seed": meta["seed"],
                    "agent_index": ep_agent,
                    "role_swap": meta["role_swap"],
                    "reward_sum": round(reward_sum, 3),
                    "num_deliveries": round(soups, 2),
                    "first_delivery_t": first_soup,
                    "last_delivery_t": last_soup,
                    "stay_ratio": round(float((ep_acts == 4).mean()), 4),
                    "interact_ratio": round(float((ep_acts == 5).mean()), 4),
                    "has_npz": True,
                    "has_pkl": npz_path.with_suffix(".pkl").exists(),
                    "has_metadata_json": (npz_path.parent / f"{stem}.metadata.json").exists(),
                    "has_layout_file": False,
                    "quality_flags": flags,
                    "quality_tier": tier,
                    "usable_for_bc": tier in ("A", "B", "C"),
                    "usable_for_environment_rollout": True,
                })
        except Exception as e:
            print(f"  Error loading {npz_path}: {e}")
            rows.append({
                "episode_id": f"{stem}_error",
                "source_group": meta["source_group"],
                "source_path": meta["source_path"],
                "stem": stem,
                "format": "npz",
                "obs_shape": "error",
                "obs_dim": -1,
                "num_steps": -1,
                "quality_flags": f"error:{e}",
                "quality_tier": "D",
                "usable_for_bc": False,
                "usable_for_environment_rollout": False,
            })

    # Write inventory
    inventory_path = output_path / "dataset_inventory.csv"
    if rows:
        with open(inventory_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Inventory: {len(rows)} episodes -> {inventory_path}")

    # Write duplicates
    dup_path = output_path / "dataset_duplicates.csv"
    with open(dup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "path_1", "path_2"])
        writer.writeheader()
        writer.writerows(duplicates)
    print(f"Duplicates: {len(duplicates)} -> {dup_path}")

    # Summary stats
    tiers = {}
    for r in rows:
        t = r.get("quality_tier", "D")
        tiers[t] = tiers.get(t, 0) + 1
    print(f"Quality tiers: {tiers}")
    usable = sum(1 for r in rows if r.get("usable_for_bc"))
    print(f"Usable for BC: {usable}/{len(rows)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    audit_datasets(args.data_root, args.output_dir)
