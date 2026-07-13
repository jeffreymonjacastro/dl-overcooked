"""Evaluate policies with three-seed groups, role swap and official score."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_official import score_output_dir
from src.runner import run_from_config


def builtin_policy(name: str, layout: str, partner: str, seed: int):
    cfg = {
        "type": "builtin",
        "name": name,
        "seed": seed,
        "random_action_prob": 0.0,
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
        if name == "adaptive_competition_shortppo":
            cfg["config"]["params_path"] = "artifacts/shortppo/best_params.json"
    return cfg


def partner_policy(name: str, seed: int):
    if name == "greedy_full_task_noise_015":
        cfg = builtin_policy("greedy_full_task", "", "", seed)
        cfg["random_action_prob"] = 0.15
        return cfg
    return builtin_policy(name, "", "", seed)


def make_config(policy_name: str, layout: str, partner: str, seeds: list[int], output_dir: Path, swap: bool):
    return {
        "seed": seeds[0],
        "mode": "evaluation",
        "environment": {
            "layout_name": layout,
            "layout_file": None,
            "horizon": 250,
            "old_dynamics": True,
        },
        "policies": {
            "agent_0": builtin_policy(policy_name, layout, partner, seeds[0]),
            "agent_1": partner_policy(partner, seeds[0]),
        },
        "execution": {
            "num_episodes": len(seeds),
            "episode_seeds": seeds,
            "swap_agent_positions": swap,
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


def summarize_attempts(attempts: list[dict[str, object]]) -> dict[str, float]:
    soups = np.asarray([float(row["soups"]) for row in attempts], dtype=np.float64)
    scores = np.asarray([float(row["attempt_score"]) for row in attempts], dtype=np.float64)
    if len(soups) == 0:
        return {}
    groups = [soups[i : i + 3] for i in range(0, len(soups) - 2, 3)]
    group_means = np.asarray([float(np.mean(group)) for group in groups], dtype=np.float64) if groups else soups
    return {
        "attempts": float(len(soups)),
        "p_at_least_1_soup": float(np.mean(soups >= 1.0)),
        "p_group_mean_ge_1": float(np.mean(group_means >= 1.0)),
        "p_group_mean_ge_2": float(np.mean(group_means >= 2.0)),
        "p_group_mean_ge_3": float(np.mean(group_means >= 3.0)),
        "mean_soups": float(np.mean(soups)),
        "median_soups": float(np.median(soups)),
        "p10_soups": float(np.percentile(soups, 10)),
        "official_score_mean": float(np.mean(scores)),
        "official_score_p10": float(np.percentile(scores, 10)),
        "zero_rate": float(np.mean(soups == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="adaptive_competition")
    parser.add_argument("--layouts", default="cramped_room,coordination_ring,forced_coordination")
    parser.add_argument("--partners", default="greedy_full_task,greedy_full_task_noise_015,random_motion,stay")
    parser.add_argument("--seeds", default="67,68,69")
    parser.add_argument("--swap", action="store_true", default=True)
    parser.add_argument("--output", default="reports/competition_protocol_results.csv")
    parser.add_argument("--keep-outputs", action="store_true")
    args = parser.parse_args()

    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    partners = [item.strip() for item in args.partners.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    results = []
    base_out = REPO_ROOT / "outputs" / "competition_protocol"
    if base_out.exists() and not args.keep_outputs:
        shutil.rmtree(base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    for layout in layouts:
        for partner in partners:
            output_dir = base_out / f"{args.policy}_{layout}_{partner}"
            cfg = make_config(args.policy, layout, partner, seeds, output_dir, args.swap)
            aggregate = run_from_config(cfg)
            attempts = score_output_dir(output_dir, horizon=250)
            overall = summarize_attempts(attempts)
            by_role = {}
            for role_swap in ["False", "True"]:
                role_attempts = [row for row in attempts if str(row.get("role_swap")) == role_swap]
                by_role[role_swap] = summarize_attempts(role_attempts)
            row = {
                "policy": args.policy,
                "layout": layout,
                "partner": partner,
                "num_rollouts": aggregate["num_rollouts"],
                **overall,
                "role0_mean_soups": by_role.get("False", {}).get("mean_soups", 0.0),
                "role1_mean_soups": by_role.get("True", {}).get("mean_soups", 0.0),
                "worst_role_mean_soups": min(
                    by_role.get("False", {}).get("mean_soups", 0.0),
                    by_role.get("True", {}).get("mean_soups", 0.0),
                ),
                "output_dir": str(output_dir),
            }
            results.append(row)
            print(row)

    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print({"results": str(out), "rows": len(results)})


if __name__ == "__main__":
    main()
