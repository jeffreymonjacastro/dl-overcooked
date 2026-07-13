"""Evaluate short-training candidates and write score-first reports."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "shortppo"


def run_eval(policy: str, output: str, layouts: str, partners: str, seeds: str) -> Path:
    out_path = REPORT_DIR / output
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_competition_protocol.py"),
        "--policy",
        policy,
        "--layouts",
        layouts,
        "--partners",
        partners,
        "--seeds",
        seeds,
        "--output",
        str(out_path.relative_to(REPO_ROOT)),
        "--keep-outputs",
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    return out_path


def summarize(csv_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    return {
        "rows": float(len(rows)),
        "mean_soups": sum(float(row["mean_soups"]) for row in rows) / len(rows),
        "official_score_mean": sum(float(row["official_score_mean"]) for row in rows) / len(rows),
        "zero_rate": sum(float(row["zero_rate"]) for row in rows) / len(rows),
        "worst_role_mean": min(float(row["worst_role_mean_soups"]) for row in rows),
    }


def write_summary(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    lines = ["# ShortPPO Evaluation Summary", ""]
    for name, summary in summaries.items():
        lines.append(f"## {name}")
        lines.append("")
        for key, value in summary.items():
            lines.append(f"- `{key}`: `{value:.4f}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default="cramped_room,coordination_ring,forced_coordination")
    parser.add_argument("--partners", default="greedy_full_task,greedy_full_task_noise_015,random_motion,stay")
    parser.add_argument("--seeds", default="67,68,69")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "hybrid_official_score": run_eval(
            "hybrid_official_score", "pretrain_hybrid_official_score.csv", args.layouts, args.partners, args.seeds
        ),
        "adaptive_competition": run_eval(
            "adaptive_competition", "pretrain_adaptive_competition.csv", args.layouts, args.partners, args.seeds
        ),
        "adaptive_competition_shortppo": run_eval(
            "adaptive_competition_shortppo", "final_evaluation.csv", args.layouts, args.partners, args.seeds
        ),
    }
    summaries = {name: summarize(path) for name, path in outputs.items()}
    write_summary(REPORT_DIR / "pretrain_summary.md", summaries)
    with (REPORT_DIR / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "policy", "mean_soups", "official_score_mean", "zero_rate", "worst_role_mean"])
        writer.writeheader()
        for name, summary in summaries.items():
            writer.writerow(
                {
                    "checkpoint": "score_first_planner_v1" if name == "adaptive_competition_shortppo" else "baseline",
                    "policy": name,
                    "mean_soups": summary["mean_soups"],
                    "official_score_mean": summary["official_score_mean"],
                    "zero_rate": summary["zero_rate"],
                    "worst_role_mean": summary["worst_role_mean"],
                }
            )
    print(summaries)


if __name__ == "__main__":
    main()

