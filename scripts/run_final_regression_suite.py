"""Run protected revealed-scenario regression for a candidate policy."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(path: Path, rows: list[dict[str, str]], policy: str) -> bool:
    passed = True
    lines = [
        "# Protected Revealed Scenario Regression",
        "",
        f"Policy: `{policy}`",
        "",
        "| Scenario | Layout | Mean soups | Score mean | Groups passed | Gate |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        groups = int(float(row["groups"]))
        groups_passed = int(float(row["groups_passed"]))
        gate = groups_passed == groups
        passed = passed and gate
        lines.append(
            f"| {row['scenario_id']} | `{row['layout']}` | {float(row['mean_soups']):.4f} | "
            f"{float(row['official_score_mean']):.2f} | {groups_passed}/{groups} | {gate} |"
        )
    lines.extend(["", f"Overall gate: `{passed}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="score_first_portfolio_v2")
    parser.add_argument("--seeds", default="67-96")
    parser.add_argument("--group-size", type=int, default=3)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "reports" / "final_pass"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / f"protected_{args.policy}.csv"
    groups = out_dir / f"protected_{args.policy}_groups.csv"
    report = out_dir / f"protected_{args.policy}.md"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_revealed_scenarios.py"),
        "--policy",
        args.policy,
        "--seeds",
        args.seeds,
        "--group-size",
        str(args.group_size),
        "--output",
        str(results.relative_to(REPO_ROOT)),
        "--groups-output",
        str(groups.relative_to(REPO_ROOT)),
        "--report",
        str(report.relative_to(REPO_ROOT)),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    rows = read_csv(results)
    passed = write_report(report, rows, args.policy)
    print({"policy": args.policy, "passed": passed, "report": str(report)})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
