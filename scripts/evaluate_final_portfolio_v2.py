"""Compare score_first_portfolio against score_first_portfolio_v2 on target layouts."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_competition_protocol import summarize_attempts
from scripts.score_official import score_output_dir
from src.runner import run_from_config


DEFAULT_LAYOUTS = "centre_objects,cramped_room_o_3orders,scenario2_s,unident,large_room"


def parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def builtin_policy(name: str, layout: str, partner: str, seed: int) -> dict[str, object]:
    cfg: dict[str, object] = {
        "type": "builtin",
        "name": name,
        "seed": seed,
        "random_action_prob": 0.0,
        "sticky_action_prob": 0.0,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    if name in {
        "hybrid_official_score",
        "adaptive_competition",
        "adaptive_competition_shortppo",
        "score_first_portfolio",
        "score_first_portfolio_v2",
    }:
        cfg["config"] = {"layout_name": layout, "partner_name": partner}
    return cfg


def run_cell(policy: str, layout: str, seeds: list[int], output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cfg = {
        "seed": seeds[0],
        "mode": "evaluation",
        "environment": {
            "layout_name": layout,
            "layout_file": None,
            "horizon": 250,
            "old_dynamics": True,
        },
        "policies": {
            "agent_0": builtin_policy(policy, layout, "greedy_full_task", seeds[0]),
            "agent_1": builtin_policy("greedy_full_task", layout, "greedy_full_task", seeds[0]),
        },
        "execution": {
            "num_episodes": len(seeds),
            "episode_seeds": seeds,
            "swap_agent_positions": False,
        },
        "observation": {"type": "featurized", "include_agent_index": True},
        "rendering": {"mode": "none", "fps": 0, "save_gif": False},
        "logging": {
            "output_dir": str(output_dir),
            "save_step_log": True,
            "save_episode_summary": True,
            "save_trajectory_pickle": False,
        },
    }
    aggregate = run_from_config(cfg)
    summary = summarize_attempts(score_output_dir(output_dir, horizon=250))
    return {"num_rollouts": aggregate["num_rollouts"], **summary, "output_dir": str(output_dir)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], seeds: list[int]) -> None:
    lines = [
        "# Portfolio V2 Target Layout Comparison",
        "",
        "Partner: `greedy_full_task`",
        f"Seeds: `{seeds[0]}..{seeds[-1]}`",
        "",
        "| Layout | Baseline soups | V2 soups | Delta soups | Baseline score | V2 score | Delta score | Promoted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['layout']}` | {float(row['baseline_mean_soups']):.4f} | "
            f"{float(row['v2_mean_soups']):.4f} | {float(row['delta_soups']):.4f} | "
            f"{float(row['baseline_official_score_mean']):.2f} | {float(row['v2_official_score_mean']):.2f} | "
            f"{float(row['delta_score']):.2f} | {row['promoted']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default=DEFAULT_LAYOUTS)
    parser.add_argument("--seeds", default="67-76")
    parser.add_argument("--output", default="reports/final_pass/portfolio_v2_results.csv")
    parser.add_argument("--report", default="reports/final_pass/portfolio_v2_summary.md")
    args = parser.parse_args()

    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, object]] = []
    base_out = REPO_ROOT / "outputs" / "final_portfolio_v2"
    for layout in layouts:
        baseline = run_cell("score_first_portfolio", layout, seeds, base_out / f"baseline_{layout}")
        v2 = run_cell("score_first_portfolio_v2", layout, seeds, base_out / f"v2_{layout}")
        delta_soups = float(v2["mean_soups"]) - float(baseline["mean_soups"])
        delta_score = float(v2["official_score_mean"]) - float(baseline["official_score_mean"])
        rows.append(
            {
                "layout": layout,
                "baseline_mean_soups": baseline["mean_soups"],
                "v2_mean_soups": v2["mean_soups"],
                "delta_soups": delta_soups,
                "baseline_official_score_mean": baseline["official_score_mean"],
                "v2_official_score_mean": v2["official_score_mean"],
                "delta_score": delta_score,
                "baseline_zero_rate": baseline["zero_rate"],
                "v2_zero_rate": v2["zero_rate"],
                "promoted": delta_score > 0.0 and float(v2["mean_soups"]) >= 3.0,
            }
        )
        print(rows[-1])
    write_csv(REPO_ROOT / args.output, rows)
    write_report(REPO_ROOT / args.report, rows, seeds)


if __name__ == "__main__":
    main()
