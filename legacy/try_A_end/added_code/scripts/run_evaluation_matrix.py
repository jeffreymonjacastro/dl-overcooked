"""Run a compact evaluation matrix for option A."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.config import load_yaml
from src.runner import run_from_config
from training.metrics import summarize_step_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate_option_a.yaml")
    parser.add_argument("--output", default="reports/option_a_evaluation.csv")
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def read_steps(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    base = load_yaml(args.config)
    layouts = ["cramped_room", "coordination_ring", "forced_coordination"]
    partners = [
        ("stay", {}),
        ("random_motion", {}),
        ("greedy_full_task", {}),
        ("greedy_full_task_noise_015", {"name": "greedy_full_task", "random_action_prob": 0.15}),
    ]
    seeds = [67 + i for i in range(args.episodes)]
    rows = []
    for layout in layouts:
        for partner_label, partner_overrides in partners:
            cfg = copy.deepcopy(base)
            cfg["environment"]["layout_name"] = layout
            cfg["execution"]["num_episodes"] = args.episodes
            cfg["execution"]["episode_seeds"] = seeds
            agent0_cfg = cfg["policies"]["agent_0"].setdefault("config", {})
            agent0_cfg["layout_name"] = layout
            agent0_cfg["partner_name"] = partner_label
            partner_cfg = cfg["policies"]["agent_1"]
            partner_cfg.update(partner_overrides)
            if not partner_overrides:
                partner_cfg["name"] = partner_label
            out_dir = Path("outputs") / Path(args.output).stem / f"{layout}_{partner_label}"
            cfg["logging"]["output_dir"] = str(out_dir)
            result = run_from_config(cfg)
            steps = read_steps(out_dir / "steps.csv")
            summary = summarize_step_rows(steps, int(cfg["environment"].get("horizon", 250)))
            row = {
                "layout": layout,
                "partner": partner_label,
                "seeds": "|".join(str(s) for s in seeds),
                "mean_return_sparse": result.get("mean_return_sparse", 0.0),
                "std_return_sparse": result.get("std_return_sparse", 0.0),
                **summary,
            }
            rows.append(row)
            print(json.dumps(row))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
