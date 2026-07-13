"""Evaluate the revealed competition scenarios without training."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_competition_protocol import summarize_attempts
from scripts.score_official import score_output_dir
from src.runner import run_from_config


@dataclass(frozen=True)
class RevealedScenario:
    scenario_id: int
    layout: str
    partner: str
    sticky_action_prob: float
    random_action_prob: float
    pass_kind: str
    base_points: int


SCENARIOS = (
    RevealedScenario(1, "asymmetric_advantages", "greedy_full_task", 0.0, 0.0, "any_soup", 6),
    RevealedScenario(2, "coordination_ring", "greedy_full_task", 0.10, 0.0, "mean_ge_2", 9),
    RevealedScenario(3, "counter_circuit", "greedy_full_task", 0.10, 0.15, "mean_ge_2", 11),
)


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
        if name == "adaptive_competition_shortppo":
            cfg["config"]["params_path"] = "artifacts/shortppo/best_params.json"
    return cfg


def make_config(policy: str, scenario: RevealedScenario, seeds: list[int], output_dir: Path) -> dict[str, object]:
    agent = builtin_policy(policy, scenario.layout, scenario.partner, seeds[0])
    partner = builtin_policy(scenario.partner, scenario.layout, scenario.partner, seeds[0])
    partner["sticky_action_prob"] = scenario.sticky_action_prob
    partner["random_action_prob"] = scenario.random_action_prob
    return {
        "seed": seeds[0],
        "mode": "evaluation",
        "environment": {
            "layout_name": scenario.layout,
            "layout_file": None,
            "horizon": 250,
            "old_dynamics": True,
        },
        "policies": {"agent_0": agent, "agent_1": partner},
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


def group_attempts(attempts: list[dict[str, object]], group_size: int) -> list[list[dict[str, object]]]:
    return [attempts[i : i + group_size] for i in range(0, len(attempts), group_size) if len(attempts[i : i + group_size]) == group_size]


def group_passes(scenario: RevealedScenario, soups: list[float]) -> bool:
    if scenario.pass_kind == "any_soup":
        return sum(soups) >= 1.0
    if scenario.pass_kind == "mean_ge_2":
        return float(np.mean(soups)) >= 2.0
    raise ValueError(f"Unknown pass kind: {scenario.pass_kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="adaptive_competition")
    parser.add_argument("--seeds", default="67-96")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--output", default="reports/revealed_scenarios_results.csv")
    parser.add_argument("--groups-output", default="reports/revealed_scenarios_groups.csv")
    parser.add_argument("--report", default="reports/REVEALED_SCENARIOS_EVAL_REPORT.md")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        output_dir = REPO_ROOT / "outputs" / "revealed_scenarios" / args.policy / f"s{scenario.scenario_id}_{scenario.layout}"
        cfg = make_config(args.policy, scenario, seeds, output_dir)
        aggregate = run_from_config(cfg)
        attempts = score_output_dir(output_dir, horizon=250)
        summary = summarize_attempts(attempts)
        groups = group_attempts(attempts, args.group_size)
        pass_count = 0
        for idx, group in enumerate(groups, start=1):
            soups = [float(row["soups"]) for row in group]
            scores = [float(row["attempt_score"]) for row in group]
            passed = group_passes(scenario, soups)
            pass_count += int(passed)
            group_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "group_id": idx,
                    "layout": scenario.layout,
                    "partner": scenario.partner,
                    "sticky_action_prob": scenario.sticky_action_prob,
                    "random_action_prob": scenario.random_action_prob,
                    "seeds": "|".join(str(row["seed"]) for row in group),
                    "soups": "|".join(str(row["soups"]) for row in group),
                    "mean_soups": float(np.mean(soups)),
                    "total_soups": float(np.sum(soups)),
                    "mean_score": float(np.mean(scores)),
                    "pass_rule": scenario.pass_kind,
                    "passed": passed,
                    "base_points_if_passed": scenario.base_points if passed else 0,
                }
            )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "policy": args.policy,
                "layout": scenario.layout,
                "partner": scenario.partner,
                "sticky_action_prob": scenario.sticky_action_prob,
                "random_action_prob": scenario.random_action_prob,
                "num_rollouts": aggregate["num_rollouts"],
                **summary,
                "groups": len(groups),
                "groups_passed": pass_count,
                "group_pass_rate": pass_count / max(1, len(groups)),
                "pass_rule": scenario.pass_kind,
                "base_points_if_all_groups_pass": scenario.base_points if pass_count == len(groups) else 0,
                "output_dir": str(output_dir),
            }
        )
        print(rows[-1])

    write_csv(REPO_ROOT / args.output, rows)
    write_csv(REPO_ROOT / args.groups_output, group_rows)
    write_report(REPO_ROOT / args.report, rows, group_rows, args.policy, seeds, args.group_size)


def parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], group_rows: list[dict[str, object]], policy: str, seeds: list[int], group_size: int) -> None:
    lines = [
        "# Revealed Scenarios Evaluation Report",
        "",
        "No training was run. This evaluation uses the current policy as-is.",
        "",
        "## Protocol",
        "",
        "```text",
        f"policy = {policy}",
        f"seeds = {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes)",
        f"group_size = {group_size}",
        "role_swap = false",
        "horizon = 250",
        "sticky_action_prob = 0.10 where sticky is requested",
        "random_action_prob = 0.15 where random actions are requested",
        "```",
        "",
        "## Summary",
        "",
        "| Scenario | Layout | Noise | Mean soups | Score mean | Groups passed | Base points |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        noise = f"sticky={row['sticky_action_prob']}, random={row['random_action_prob']}"
        lines.append(
            f"| {row['scenario_id']} | `{row['layout']}` | {noise} | "
            f"{float(row['mean_soups']):.4f} | {float(row['official_score_mean']):.2f} | "
            f"{int(row['groups_passed'])}/{int(row['groups'])} | {int(row['base_points_if_all_groups_pass'])} |"
        )
    lines.extend(["", "## Groups", ""])
    for row in rows:
        lines.append(f"### Scenario {row['scenario_id']} - `{row['layout']}`")
        lines.append("")
        lines.append("| Group | Seeds | Soups | Mean soups | Mean score | Passed |")
        lines.append("| ---: | --- | --- | ---: | ---: | --- |")
        for group in [g for g in group_rows if g["scenario_id"] == row["scenario_id"]]:
            lines.append(
                f"| {group['group_id']} | {group['seeds']} | {group['soups']} | "
                f"{float(group['mean_soups']):.4f} | {float(group['mean_score']):.2f} | {group['passed']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
