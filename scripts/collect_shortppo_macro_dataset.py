"""Collect a compact macro-decision dataset from the adaptive teacher."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from overcooked_ai_py.agents.agent import AgentPair

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import build_env
from src.observations import ObservationBuilder
from src.policy_loader import build_two_policies


def teacher_policy(layout: str, partner: str):
    return {
        "type": "builtin",
        "name": "adaptive_competition",
        "config": {"layout_name": layout, "partner_name": partner},
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }


def partner_policy(name: str):
    if name == "greedy_full_task_noise_015":
        return {
            "type": "builtin",
            "name": "greedy_full_task",
            "ingredient": "onion",
            "avoid_teammate": True,
            "random_action_prob": 0.15,
            "max_action_time_ms": 100,
            "invalid_action": "stay",
            "timeout_action": "stay",
        }
    return {
        "type": "builtin",
        "name": name,
        "ingredient": "onion",
        "avoid_teammate": True,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }


def collect(layouts: list[str], partners: list[str], seeds: list[int], output: Path) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    by_macro: dict[str, int] = {}
    with output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layout",
            "partner",
            "seed",
            "role_swap",
            "timestep",
            "teacher_role",
            "macro_label",
            "topology",
            "partner_activity",
            "primitive_action",
            "reward",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layout in layouts:
            for partner in partners:
                env_cfg = {"layout_name": layout, "layout_file": None, "horizon": 250, "old_dynamics": True}
                for seed in seeds:
                    for role_swap in (False, True):
                        cfg = {
                            "environment": env_cfg,
                            "policies": {"agent_0": teacher_policy(layout, partner), "agent_1": partner_policy(partner)},
                            "observation": {"type": "featurized", "include_agent_index": True},
                        }
                        env = build_env(env_cfg)
                        obs_builder = ObservationBuilder(env, cfg["observation"])
                        agent0, agent1 = build_two_policies(cfg, env, obs_builder, seed=seed)
                        if role_swap:
                            agent0, agent1 = agent1, agent0
                        pair = AgentPair(agent0, agent1)
                        env.reset(regen_mdp=False)
                        pair.reset()
                        pair.set_mdp(env.mdp)
                        done = False
                        while not done:
                            state = env.state
                            joint_action_infos = pair.joint_action(state)
                            joint_action, joint_infos = zip(*joint_action_infos)
                            teacher_idx = 1 if role_swap else 0
                            info = dict(joint_infos[teacher_idx] or {})
                            next_state, reward, done, _ = env.step(joint_action, joint_infos)
                            macro = str(info.get("adaptive_mode", info.get("selected_mode", "unknown")))
                            by_macro[macro] = by_macro.get(macro, 0) + 1
                            writer.writerow(
                                {
                                    "layout": layout,
                                    "partner": partner,
                                    "seed": seed,
                                    "role_swap": role_swap,
                                    "timestep": state.timestep,
                                    "teacher_role": teacher_idx,
                                    "macro_label": macro,
                                    "topology": info.get("topology", ""),
                                    "partner_activity": info.get("partner_activity", ""),
                                    "primitive_action": joint_action[teacher_idx],
                                    "reward": float(reward),
                                }
                            )
                            rows += 1
    return {"rows": rows, **{f"macro_{key}": value for key, value in sorted(by_macro.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default="cramped_room,coordination_ring,forced_coordination")
    parser.add_argument("--partners", default="greedy_full_task,greedy_full_task_noise_015,random_motion")
    parser.add_argument("--seeds", default="67,68,69")
    parser.add_argument("--output", default="artifacts/shortppo/macro_dataset.csv")
    args = parser.parse_args()
    summary = collect(
        [item.strip() for item in args.layouts.split(",") if item.strip()],
        [item.strip() for item in args.partners.split(",") if item.strip()],
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()],
        REPO_ROOT / args.output,
    )
    summary_path = REPO_ROOT / "reports" / "shortppo" / "macro_dataset_summary.md"
    lines = ["# Macro Dataset Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

