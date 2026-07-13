"""Collect greedy teacher rollouts using the official runner."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_from_config


def base_config(layout: str, seed: int, role: int, partner: str) -> dict:
    greedy = {
        "type": "builtin",
        "name": "greedy_full_task",
        "ingredient": "onion",
        "avoid_teammate": True,
        "random_action_prob": 0.0,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    partner_cfg = {
        "type": "builtin",
        "name": partner,
        "ingredient": "onion",
        "avoid_teammate": True,
        "random_action_prob": 0.0,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    if role == 0:
        agent_0, agent_1 = greedy, partner_cfg
        record = [0]
    else:
        agent_0, agent_1 = partner_cfg, greedy
        record = [1]
    return {
        "seed": seed,
        "mode": "collect_teacher_rollouts",
        "environment": {"layout_name": layout, "layout_file": None, "horizon": 250, "old_dynamics": True},
        "policies": {"agent_0": agent_0, "agent_1": agent_1},
        "execution": {"num_episodes": 1, "episode_seeds": [seed], "swap_agent_positions": False},
        "observation": {"type": "featurized", "include_agent_index": True},
        "rendering": {"mode": "none", "fps": 0, "save_gif": False},
        "logging": {
            "output_dir": f"outputs/teacher_rollouts/{layout}_{partner}_role{role}_{seed}",
            "save_step_log": False,
            "save_episode_summary": True,
            "save_trajectory_pickle": False,
        },
        "data_collection": {
            "enabled": True,
            "record_agent_indices": record,
            "include_next_obs": True,
            "include_info": False,
            "output_dir": "artifacts/option_a/teacher_rollouts",
            "auto_name": True,
            "overwrite": False,
            "save_npz": True,
        },
    }


def main() -> None:
    layouts = ["cramped_room", "coordination_ring", "forced_coordination", "large_room", "small_corridor"]
    partners = ["stay", "random_motion", "greedy_full_task"]
    seeds = [67, 68]
    results = []
    Path("artifacts/option_a/teacher_rollouts").mkdir(parents=True, exist_ok=True)
    for layout in layouts:
        for partner in partners:
            for role in [0, 1]:
                for seed in seeds:
                    cfg = base_config(layout, seed, role, partner)
                    result = run_from_config(cfg)
                    result.update({"layout": layout, "partner": partner, "role": role, "seed": seed})
                    results.append(result)
                    print(json.dumps(result))
    Path("reports/option_a_teacher_rollouts.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
