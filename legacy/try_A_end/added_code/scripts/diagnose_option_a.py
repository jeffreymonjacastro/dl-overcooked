"""Generate A2 diagnostic reports from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", default="artifacts/option_a_baseline_before_a2/option_a/teacher_rollouts")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def quality_tier(num_soups: int, percent_stay: float, longest_run: int) -> str:
    if num_soups >= 2:
        return "A"
    if num_soups == 1:
        return "B"
    if longest_run <= 25 and percent_stay < 0.70:
        return "C"
    return "D"


def longest_repeated_action_run(actions: np.ndarray) -> int:
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


def audit_teacher_rollouts(teacher_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(teacher_dir.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            actions = np.asarray(data["actions"], dtype=np.int64) if "actions" in data.files else np.asarray([], dtype=np.int64)
            rewards = np.asarray(data["rewards"], dtype=np.float32) if "rewards" in data.files else np.asarray([], dtype=np.float32)
            positives = np.where(rewards > 0)[0]
            parts = path.stem.split("_")
            layout = "_".join(parts[:-2]) if len(parts) > 2 else path.stem
            rows.append(
                {
                    "source_path": str(path),
                    "layout_guess": layout,
                    "num_steps": int(len(actions)),
                    "soups": int(len(positives)),
                    "zero_soup": int(len(positives) == 0),
                    "reward_sum": float(np.sum(rewards)) if len(rewards) else 0.0,
                    "percent_stay": float(np.mean(actions == 4)) if len(actions) else 0.0,
                    "percent_interact": float(np.mean(actions == 5)) if len(actions) else 0.0,
                    "longest_repeated_action_run": longest_repeated_action_run(actions),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    teacher_rows = audit_teacher_rollouts(Path(args.teacher_dir))
    write_csv(reports_dir / "teacher_rollouts_audit.csv", teacher_rows)

    tier_counts = Counter(
        quality_tier(
            int(row["soups"]),
            float(row["percent_stay"]),
            int(row["longest_repeated_action_run"]),
        )
        for row in teacher_rows
    )
    summary = {
        "teacher_rollouts": len(teacher_rows),
        "teacher_zero_soup": sum(int(row["zero_soup"]) for row in teacher_rows),
        "teacher_quality_tiers": dict(sorted(tier_counts.items())),
    }
    (reports_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "diagnostics" / "behavior_analysis.md").write_text(
        "\n".join(
            [
                "# Behavior analysis A2",
                "",
                "This diagnostic summarizes generated teacher rollouts preserved from A1.",
                "Detailed visual failure tracing still requires running `scripts/trace_episode.py` with rendering or trajectory pickles.",
                "",
                f"- Teacher rollouts audited: {summary['teacher_rollouts']}",
                f"- Zero-soup teacher rollouts: {summary['teacher_zero_soup']}",
                f"- Quality tiers: {summary['teacher_quality_tiers']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

