"""Classify low-performing layouts from the layout sweep report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

STRUCTURALLY_FORCED = {
    "forced_coordination",
    "forced_coordination_tomato",
    "soup_coordination",
    "pipeline",
    "schelling",
    "schelling_s",
    "small_corridor",
}
BOTTLENECKS = {"bottleneck", "cramped_corridor", "m_shaped_s", "scenario1_s", "scenario4"}
RECIPE_AWARE = {"simple_tomato", "cramped_room_tomato", "cramped_room_o_3orders", "bonus_order_test"}
SLOW_BUT_WORKING = {"centre_objects", "scenario2_s", "unident"}
COMPETITION_LIKE = {"large_room", "chavez_room", "jamcy_room", "m_room", "diagonal_run"}
INVALID_OR_PATHOLOGICAL = {
    "corridor",
    "you_shall_not_pass",
    "multiplayer_schelling",
    "maze_kitchen",
    "tutorial_1",
    "cramped_room_single",
}


def classify(layout: str, mean_soups: float, omitted: bool = False) -> tuple[str, int, str]:
    if omitted or layout in INVALID_OR_PATHOLOGICAL:
        return "D_or_E_planner_pathological_or_invalid", 5, "Timeout, memory blow-up, invalid recipe config, or incompatible test layout."
    if layout in SLOW_BUT_WORKING or mean_soups > 0.0:
        return "B_slow_but_working", 2, "Already delivers soups but below target; good candidate for specialist routing."
    if layout in RECIPE_AWARE:
        return "A_competition_like_and_solvable_recipe", 1, "Likely recipe/order mismatch or partner contamination."
    if layout in STRUCTURALLY_FORCED:
        return "C_structurally_forced_handoff", 3, "Needs handoff, role symmetry, or partner cooperation."
    if layout in BOTTLENECKS:
        return "A_competition_like_bottleneck", 1, "Likely deadlock, narrow corridor, or yield/replan issue."
    if layout in COMPETITION_LIKE:
        return "A_competition_like_custom_geometry", 1, "Custom geometry not covered by current route table."
    return "C_unknown_low_layout", 4, "Needs manual inspection before promotion."


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Low Layout Triage",
        "",
        "| Layout | Class | Priority | Mean soups | Reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (int(item["priority"]), str(item["layout"]))):
        lines.append(
            f"| `{row['layout']}` | {row['class']} | {row['priority']} | "
            f"{float(row['mean_soups']):.3f} | {row['reason']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="reports/layout_sweep_score_first_portfolio.csv")
    parser.add_argument("--omitted", default="reports/layout_sweep_omitted.csv")
    parser.add_argument("--output", default="reports/final_pass/low_layout_triage.csv")
    parser.add_argument("--report", default="reports/final_pass/low_layout_triage.md")
    args = parser.parse_args()

    sweep_rows = read_rows(REPO_ROOT / args.sweep)
    omitted_rows = read_rows(REPO_ROOT / args.omitted)
    rows: list[dict[str, object]] = []

    for row in sweep_rows:
        mean_soups = float(row.get("mean_soups", 0.0) or 0.0)
        if mean_soups >= 3.0:
            continue
        family, priority, reason = classify(row["layout"], mean_soups)
        rows.append(
            {
                "layout": row["layout"],
                "class": family,
                "priority": priority,
                "mean_soups": mean_soups,
                "zero_rate": float(row.get("zero_rate", 0.0) or 0.0),
                "reason": reason,
            }
        )

    for row in omitted_rows:
        family, priority, reason = classify(row["layout"], 0.0, omitted=True)
        rows.append(
            {
                "layout": row["layout"],
                "class": family,
                "priority": priority,
                "mean_soups": 0.0,
                "zero_rate": 1.0,
                "reason": f"{reason} Detail: {row.get('reason', '')}",
            }
        )

    if not rows:
        raise SystemExit("No low layouts found to classify.")
    write_csv(REPO_ROOT / args.output, rows)
    write_markdown(REPO_ROOT / args.report, rows)
    print({"rows": len(rows), "output": args.output, "report": args.report})


if __name__ == "__main__":
    main()
