"""Prepare a revealed layout/partner scenario without replacing the final config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-file")
    parser.add_argument("--layout-name", default="forced_coordination")
    parser.add_argument("--partner", default="greedy_full_task")
    parser.add_argument("--scenario", default="1")
    args = parser.parse_args()
    print({"scenario": args.scenario, "layout_name": args.layout_name, "layout_file": args.layout_file, "partner": args.partner})
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "catalog_layouts.py")], cwd=str(REPO_ROOT))
    subprocess.check_call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_competition_protocol.py"),
            "--policy",
            "adaptive_competition",
            "--layouts",
            args.layout_name,
            "--partners",
            args.partner,
            "--seeds",
            "67,68,69,70,71,72",
        ],
        cwd=str(REPO_ROOT),
    )


if __name__ == "__main__":
    main()

