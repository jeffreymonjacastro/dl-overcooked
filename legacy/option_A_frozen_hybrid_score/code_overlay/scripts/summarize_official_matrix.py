"""Create attempt-level and scenario-level official score summaries."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from score_official import score_output_dir


LAYOUTS = ["forced_coordination", "coordination_ring", "cramped_room"]


def parse_cell_name(name: str) -> tuple[str, str]:
    for layout in LAYOUTS:
        prefix = layout + "_"
        if name.startswith(prefix):
            return layout, name[len(prefix) :]
    return "", name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--attempts-output", required=True)
    parser.add_argument("--scenarios-output", required=True)
    args = parser.parse_args()

    root = Path(args.matrix_output_dir)
    attempts: list[dict[str, object]] = []
    scenarios: list[dict[str, object]] = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        layout, partner = parse_cell_name(cell_dir.name)
        cell_attempts = score_output_dir(cell_dir, args.horizon)
        for row in cell_attempts:
            row["layout"] = layout or row.get("layout", "")
            row["partner"] = partner
            row["agent_index"] = 0 if str(row.get("role_swap", "")).lower() != "true" else 1
            attempts.append(row)
        scores = [float(row["attempt_score"]) for row in cell_attempts]
        soups = [float(row["soups"]) for row in cell_attempts]
        scenarios.append(
            {
                "layout": layout,
                "partner": partner,
                "num_attempts": len(cell_attempts),
                "score_seed_1": scores[0] if len(scores) > 0 else 0.0,
                "score_seed_2": scores[1] if len(scores) > 1 else 0.0,
                "score_seed_3": scores[2] if len(scores) > 2 else 0.0,
                "scenario_score_mean": statistics.fmean(scores) if scores else 0.0,
                "scenario_score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "scenario_score_min": min(scores) if scores else 0.0,
                "mean_soups": statistics.fmean(soups) if soups else 0.0,
                "zero_soup_attempts": sum(1 for value in soups if value == 0),
            }
        )

    write_csv(Path(args.attempts_output), attempts)
    write_csv(Path(args.scenarios_output), scenarios)
    print({"attempts": len(attempts), "scenarios": len(scenarios)})


if __name__ == "__main__":
    main()
