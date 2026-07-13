"""Create A2 dataset quality reports from the existing split manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/shared/split_manifest.json")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--artifacts-dir", default="artifacts/shared")
    return parser.parse_args()


def longest_run(actions: np.ndarray) -> int:
    if len(actions) == 0:
        return 0
    best = 1
    current = 1
    for prev, cur in zip(actions[:-1], actions[1:]):
        if int(prev) == int(cur):
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def quality_tier(soups: int, percent_stay: float, percent_interact: float, repeated_run: int, has_required_keys: bool) -> str:
    if not has_required_keys:
        return "D"
    if soups >= 2:
        return "A"
    if soups == 1:
        return "B"
    if percent_stay < 0.70 and repeated_run < 30 and percent_interact > 0.01:
        return "C"
    return "D"


def read_episode(entry: dict[str, object]) -> dict[str, object]:
    path = Path(str(entry["source_path"]))
    row = {
        "episode_id": entry.get("episode_id", path.stem),
        "group_id": entry.get("source_group", "unknown"),
        "layout": entry.get("layout", "unknown"),
        "partner_type": entry.get("partner", "unknown"),
        "source_type": "human",
        "source_path": str(path),
        "split": entry.get("split", "unknown"),
        "agent_index": "",
        "role_swap": "",
        "seed": "",
        "num_steps": 0,
        "soups": 0,
        "official_score_proxy": 0.0,
        "first_soup_timestep": "",
        "last_soup_timestep": "",
        "percent_stay": 0.0,
        "percent_interact": 0.0,
        "longest_repeated_action_run": 0,
        "changed_state_rate": "",
        "has_custom_layout": False,
        "quality_tier": "D",
        "exclude_from_bc": False,
        "exclusion_reason": "",
    }
    try:
        with np.load(path, allow_pickle=True) as data:
            required = {"obs", "actions"}.issubset(set(data.files))
            if not required:
                row["exclude_from_bc"] = True
                row["exclusion_reason"] = "missing_obs_or_actions"
                return row
            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.int64)
            row["num_steps"] = int(len(actions))
            if len(actions) == 0 or obs.ndim != 2 or len(obs) != len(actions):
                row["exclude_from_bc"] = True
                row["exclusion_reason"] = "invalid_shape"
                return row
            if actions.min(initial=0) < 0 or actions.max(initial=0) > 5:
                row["exclude_from_bc"] = True
                row["exclusion_reason"] = "invalid_action_range"
                return row
            rewards = np.asarray(data["rewards"], dtype=np.float32) if "rewards" in data.files else np.zeros(len(actions), dtype=np.float32)
            positives = np.where(rewards > 0)[0]
            timesteps = np.asarray(data["timesteps"], dtype=np.int64) if "timesteps" in data.files else np.arange(len(actions))
            row["soups"] = int(len(positives))
            if len(positives):
                first_t = int(timesteps[positives[0]])
                last_t = int(timesteps[positives[-1]])
                row["first_soup_timestep"] = first_t
                row["last_soup_timestep"] = last_t
                row["official_score_proxy"] = float(10000 * len(positives) + 10 * (250 - last_t) + (250 - first_t))
            row["percent_stay"] = float(np.mean(actions == 4))
            row["percent_interact"] = float(np.mean(actions == 5))
            row["longest_repeated_action_run"] = longest_run(actions)
            if "agent_indices" in data.files:
                row["agent_index"] = "|".join(str(x) for x in sorted(set(np.asarray(data["agent_indices"], dtype=np.int64).tolist())))
            if "role_swaps" in data.files:
                row["role_swap"] = "|".join(str(bool(x)).lower() for x in sorted(set(np.asarray(data["role_swaps"]).tolist())))
            if "episode_seeds" in data.files and len(data["episode_seeds"]):
                seed = int(data["episode_seeds"][0])
                row["seed"] = "" if seed < 0 else seed
            row["quality_tier"] = quality_tier(
                int(row["soups"]),
                float(row["percent_stay"]),
                float(row["percent_interact"]),
                int(row["longest_repeated_action_run"]),
                True,
            )
            row["exclude_from_bc"] = row["quality_tier"] == "D"
            row["exclusion_reason"] = "tier_d_low_quality" if row["exclude_from_bc"] else ""
    except Exception as exc:
        row["exclude_from_bc"] = True
        row["exclusion_reason"] = f"load_error:{type(exc).__name__}"
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = [read_episode(entry) for entry in manifest["entries"]]
    reports_dir = Path(args.reports_dir)
    artifacts_dir = Path(args.artifacts_dir)
    write_csv(reports_dir / "dataset_inventory_a2.csv", rows)
    write_csv(reports_dir / "dataset_quality_a2.csv", rows)

    entries = []
    for row in rows:
        if row["exclude_from_bc"]:
            continue
        entries.append(
            {
                "episode_id": row["episode_id"],
                "source_path": row["source_path"],
                "source_group": row["group_id"],
                "layout": row["layout"],
                "partner": row["partner_type"],
                "source_type": row["source_type"],
                "quality_tier": row["quality_tier"],
                "split": row["split"],
            }
        )
    manifest_a2 = {
        "base_manifest": args.manifest,
        "entries": entries,
        "excluded_count": len(rows) - len(entries),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "split_manifest_a2.json").write_text(json.dumps(manifest_a2, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "usable_a2": len(entries), "excluded": len(rows) - len(entries)}, indent=2))


if __name__ == "__main__":
    main()

