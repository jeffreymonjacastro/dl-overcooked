"""Run one configured episode and keep detailed logs for manual inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_yaml
from src.runner import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate_option_a2.yaml")
    parser.add_argument("--output-dir", default="outputs/trace_episode")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--partner", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["execution"]["num_episodes"] = 1
    cfg["execution"]["episode_seeds"] = [args.seed]
    cfg["logging"]["output_dir"] = args.output_dir
    cfg["logging"]["save_step_log"] = True
    cfg["logging"]["save_episode_summary"] = True
    cfg["logging"]["save_trajectory_pickle"] = True
    if args.layout:
        cfg["environment"]["layout_name"] = args.layout
    if args.partner:
        cfg["policies"]["agent_1"]["name"] = args.partner
    result = run_from_config(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

