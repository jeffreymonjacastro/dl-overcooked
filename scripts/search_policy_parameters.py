"""Small parameter-search harness for revealed scenarios.

The current deterministic planner exposes only a few safe knobs. This script is
kept intentionally simple so it can be expanded to grid/random/CEM search once a
scenario layout is revealed.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="forced_coordination")
    parser.add_argument("--partner", default="greedy_full_task")
    parser.add_argument("--policy", default="adaptive_competition")
    parser.add_argument("--seeds", default="67,68,69")
    args = parser.parse_args()
    candidates = list(itertools.product(["nearest"], ["default"]))
    print({"candidate_count": len(candidates), "note": "planner knobs are placeholders until revealed-scenario tuning"})
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_competition_protocol.py"),
        "--policy",
        args.policy,
        "--layouts",
        args.layout,
        "--partners",
        args.partner,
        "--seeds",
        args.seeds,
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()

