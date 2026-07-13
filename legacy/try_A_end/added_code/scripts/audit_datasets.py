"""Audit Overcooked demonstration datasets and create shared artifacts.

This script is intentionally read-only with respect to data/. It writes reports
and manifests used by the training scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BUILTIN_LAYOUTS = {
    "asymmetric_advantages",
    "coordination_ring",
    "counter_circuit",
    "cramped_room",
    "forced_coordination",
    "large_room",
    "simple_o",
    "simple_tomato",
    "small_corridor",
    "soup_coordination",
    "tutorial_0",
    "tutorial_1",
    "tutorial_2",
    "tutorial_3",
}

PARTNER_NAMES = ("greedy_full_task", "random_motion", "stay", "random", "human_keyboard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--artifacts-dir", default="artifacts/shared")
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


def json_loads_maybe(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        try:
            value = value.item()
        except Exception:
            return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def read_sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def extract_layout(metadata: dict[str, Any], path: Path) -> str:
    layout = metadata.get("layout", {}) if isinstance(metadata.get("layout"), dict) else {}
    env = metadata.get("environment", {}) if isinstance(metadata.get("environment"), dict) else {}
    value = first_present(layout.get("layout_name"), env.get("layout_name"))
    if value:
        return str(value)
    stem = path.stem
    marker = "_2026"
    return stem.split(marker)[0] if marker in stem else stem


def extract_partner(metadata: dict[str, Any], path: Path) -> str:
    policies = metadata.get("policies", {}) if isinstance(metadata.get("policies"), dict) else {}
    for cfg in policies.values():
        if isinstance(cfg, dict):
            name = cfg.get("name")
            if name and str(name) != "human_keyboard":
                return str(name)
    lower_parts = [part.lower() for part in path.parts]
    for name in PARTNER_NAMES:
        if name in lower_parts:
            return name
    stem_lower = path.stem.lower()
    for name in PARTNER_NAMES:
        if name in stem_lower:
            return name
    return "unknown"


def source_group(path: Path, data_dir: Path) -> str:
    try:
        rel = path.relative_to(data_dir)
        return rel.parts[0]
    except Exception:
        return "unknown"


def file_hash(obs: np.ndarray, actions: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(obs).view(np.uint8))
    h.update(np.ascontiguousarray(actions).view(np.uint8))
    return h.hexdigest()


def layout_type(layout: str, has_layout_file: bool) -> str:
    if layout in BUILTIN_LAYOUTS:
        return "tier1_builtin"
    if has_layout_file:
        return "tier2_custom_with_layout"
    return "tier3_custom_without_layout"


def audit_file(path: Path, data_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_id": "",
        "source_group": source_group(path, data_dir),
        "source_path": str(path),
        "stem": path.stem,
        "format": ".npz",
        "obs_shape": "",
        "num_steps": 0,
        "action_min": "",
        "action_max": "",
        "layout": "",
        "layout_type": "",
        "partner": "unknown",
        "seed": "",
        "agent_index": "",
        "role_swap": "",
        "reward_sum": "",
        "num_deliveries": "",
        "first_delivery_t": "",
        "last_delivery_t": "",
        "stay_ratio": "",
        "interact_ratio": "",
        "has_npz": True,
        "has_pkl": path.with_suffix(".pkl").exists(),
        "has_metadata_json": path.with_suffix(".metadata.json").exists(),
        "has_layout_file": False,
        "quality_flags": "",
        "usable_for_bc": False,
        "usable_for_environment_rollout": False,
        "content_hash": "",
    }
    flags: list[str] = []
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            if "obs" not in keys or "actions" not in keys:
                flags.append("missing_obs_or_actions")
                row["quality_flags"] = ";".join(flags)
                return row
            obs = np.asarray(data["obs"])
            actions = np.asarray(data["actions"])
            row["obs_shape"] = "x".join(str(x) for x in obs.shape)
            row["num_steps"] = int(actions.shape[0])
            row["episode_id"] = str(path.stem)
            if actions.size:
                row["action_min"] = int(np.min(actions))
                row["action_max"] = int(np.max(actions))
                row["stay_ratio"] = float(np.mean(actions == 4))
                row["interact_ratio"] = float(np.mean(actions == 5))
            if obs.shape[0] != actions.shape[0]:
                flags.append("obs_action_length_mismatch")
            if actions.size and (np.min(actions) < 0 or np.max(actions) > 5):
                flags.append("action_out_of_range")

            metadata = json_loads_maybe(data["metadata_json"]) if "metadata_json" in keys else {}
            sidecar = read_sidecar_metadata(path)
            metadata = {**sidecar, **metadata}
            row["layout"] = extract_layout(metadata, path)
            row["partner"] = extract_partner(metadata, path)

            if "episode_ids" in keys and len(data["episode_ids"]):
                row["episode_id"] = f"{path.stem}:{int(data['episode_ids'][0])}"
            if "episode_seeds" in keys and len(data["episode_seeds"]):
                seed = int(data["episode_seeds"][0])
                row["seed"] = "" if seed < 0 else seed
            if "agent_indices" in keys and len(data["agent_indices"]):
                vals = sorted(set(int(x) for x in data["agent_indices"].tolist()))
                row["agent_index"] = "|".join(str(x) for x in vals)
            if "role_swaps" in keys and len(data["role_swaps"]):
                vals = sorted(set(bool(x) for x in data["role_swaps"].tolist()))
                row["role_swap"] = "|".join(str(x).lower() for x in vals)
            if "rewards" in keys:
                rewards = np.asarray(data["rewards"], dtype=np.float64)
                row["reward_sum"] = float(np.sum(rewards))
                positive = np.where(rewards > 0)[0]
                row["num_deliveries"] = int(len(positive))
                if len(positive):
                    timesteps = np.asarray(data["timesteps"]) if "timesteps" in keys else np.arange(len(rewards))
                    row["first_delivery_t"] = int(timesteps[positive[0]])
                    row["last_delivery_t"] = int(timesteps[positive[-1]])
                else:
                    flags.append("zero_positive_rewards")
            else:
                flags.append("missing_rewards")

            layout_files = list(data_dir.rglob(f"{row['layout']}.layout")) if row["layout"] else []
            row["has_layout_file"] = bool(layout_files)
            row["layout_type"] = layout_type(str(row["layout"]), bool(layout_files))
            if row["layout_type"] != "tier3_custom_without_layout":
                row["usable_for_environment_rollout"] = True
            row["usable_for_bc"] = not flags or set(flags).issubset({"zero_positive_rewards", "missing_rewards"})
            row["content_hash"] = file_hash(obs, actions)
    except Exception as exc:
        flags.append(f"load_error:{type(exc).__name__}")
    row["quality_flags"] = ";".join(flags)
    return row


def assign_splits(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    usable = [r for r in rows if str(r["usable_for_bc"]).lower() == "true" or r["usable_for_bc"] is True]
    layouts = sorted({str(r["layout"]) for r in usable if r["layout"]})
    rng = np.random.default_rng(seed)
    shuffled_layouts = layouts[:]
    rng.shuffle(shuffled_layouts)
    n_unseen = max(1, int(round(0.15 * len(shuffled_layouts)))) if shuffled_layouts else 0
    unseen_layouts = set(shuffled_layouts[:n_unseen])

    groups = sorted({str(r["source_group"]) for r in usable})
    shuffled_groups = groups[:]
    rng.shuffle(shuffled_groups)
    val_groups = set(shuffled_groups[: max(1, int(round(0.15 * len(shuffled_groups))))]) if shuffled_groups else set()
    test_groups = set(shuffled_groups[-max(1, int(round(0.10 * len(shuffled_groups)))) :]) if shuffled_groups else set()

    entries = []
    split_counts = Counter()
    for row in usable:
        group = str(row["source_group"])
        layout = str(row["layout"])
        if group in test_groups:
            split = "internal_test"
        elif group in val_groups and layout in unseen_layouts:
            split = "validation_combined"
        elif layout in unseen_layouts:
            split = "validation_unseen_layout"
        elif group in val_groups:
            split = "validation_seen_layout"
        else:
            split = "train"
        split_counts[split] += 1
        entries.append(
            {
                "episode_id": row["episode_id"],
                "source_path": row["source_path"],
                "source_group": row["source_group"],
                "layout": row["layout"],
                "partner": row["partner"],
                "split": split,
            }
        )
    return {
        "seed": seed,
        "entries": entries,
        "split_counts": dict(split_counts),
        "unseen_layouts": sorted(unseen_layouts),
        "validation_groups": sorted(val_groups),
        "internal_test_groups": sorted(test_groups),
    }


def write_normalization(manifest: dict[str, Any], output_path: Path) -> dict[str, Any]:
    train_paths = [Path(e["source_path"]) for e in manifest["entries"] if e["split"] == "train"]
    n = 0
    sum_x = None
    sum_sq = None
    obs_dim = None
    action_counts = Counter()
    for path in train_paths:
        with np.load(path, allow_pickle=True) as data:
            obs = np.asarray(data["obs"], dtype=np.float64)
            if obs.ndim != 2:
                continue
            actions = np.asarray(data["actions"], dtype=np.int64)
            obs_dim = int(obs.shape[1])
            if sum_x is None:
                sum_x = np.zeros(obs.shape[1], dtype=np.float64)
                sum_sq = np.zeros(obs.shape[1], dtype=np.float64)
            sum_x += obs.sum(axis=0)
            sum_sq += np.square(obs).sum(axis=0)
            n += obs.shape[0]
            action_counts.update(int(x) for x in actions.tolist())
    if n == 0 or sum_x is None or sum_sq is None:
        raise RuntimeError("No train observations found for normalization")
    mean = sum_x / n
    var = np.maximum(sum_sq / n - np.square(mean), 1e-8)
    std = np.sqrt(var)
    payload = {
        "obs_dim": obs_dim,
        "num_observations": int(n),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "epsilon": 1e-6,
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir = Path(args.artifacts_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.rglob("*.npz"))
    rows = [audit_file(path, data_dir) for path in paths]

    inventory_path = reports_dir / "dataset_inventory.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_hash = defaultdict(list)
    for row in rows:
        if row.get("content_hash"):
            by_hash[row["content_hash"]].append(row)
    duplicates = [items for items in by_hash.values() if len(items) > 1]
    dup_path = reports_dir / "dataset_duplicates.csv"
    with dup_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["content_hash", "source_path", "episode_id", "source_group", "layout"])
        writer.writeheader()
        for group in duplicates:
            for row in group:
                writer.writerow({k: row.get(k, "") for k in ["content_hash", "source_path", "episode_id", "source_group", "layout"]})

    manifest = assign_splits(rows, args.seed)
    manifest_path = artifacts_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    normalization = write_normalization(manifest, artifacts_dir / "normalization.json")
    schema = {
        "obs_source": "npz:obs",
        "action_source": "npz:actions",
        "observation_type": "featurized",
        "include_agent_index": True,
        "obs_dim": normalization["obs_dim"],
        "num_actions": 6,
    }
    (artifacts_dir / "dataset_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    usable = sum(1 for r in rows if r["usable_for_bc"])
    layout_counts = Counter(str(r["layout"]) for r in rows if r["layout"])
    partner_counts = Counter(str(r["partner"]) for r in rows if r["partner"])
    summary = [
        "# Dataset summary",
        "",
        f"- NPZ files discovered: {len(paths)}",
        f"- Usable for BC: {usable}",
        f"- Duplicate content groups: {len(duplicates)}",
        f"- Observation dim: {normalization['obs_dim']}",
        f"- Train observations for normalization: {normalization['num_observations']}",
        f"- Split counts: {manifest['split_counts']}",
        f"- Top layouts: {layout_counts.most_common(10)}",
        f"- Partners: {partner_counts.most_common()}",
        "",
        "Notes:",
        "- `num_deliveries` is positive reward event count, not a guaranteed official soup count.",
        "- Files with only `obs/actions` are kept for BC only when action mapping is valid.",
    ]
    (reports_dir / "dataset_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"inventory": str(inventory_path), "usable": usable, "splits": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
