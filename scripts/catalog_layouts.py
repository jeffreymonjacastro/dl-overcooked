"""Build a topology catalog for built-in and repository layouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.layout_graph import LayoutGraphAnalyzer
from src.environment import build_mdp


DEFAULT_BUILTINS = {
    "asymmetric_advantages",
    "coordination_ring",
    "counter_circuit",
    "cramped_room",
    "forced_coordination",
    "large_room",
    "simple_o",
    "simple_tomato",
    "small_corridor",
    "soup_coordination",
    "tutorial_3",
}


def collect_layout_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    names = set(DEFAULT_BUILTINS)
    for meta_path in REPO_ROOT.glob("data/**/*.metadata.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for key in ("layout_name", "layout"):
            value = data.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    for name in sorted(names):
        candidates.append({"source": "builtin_or_metadata", "layout_name": name, "layout_file": ""})
    for layout_path in sorted((REPO_ROOT / "configs" / "layouts").glob("*.layout")):
        candidates.append({"source": "configs_layouts", "layout_name": layout_path.stem, "layout_file": str(layout_path)})
    for layout_path in sorted((REPO_ROOT / "data").glob("**/*.layout")):
        candidates.append({"source": "data_layouts", "layout_name": layout_path.stem, "layout_file": str(layout_path)})
    return candidates


def build_one(candidate: dict[str, str]) -> dict[str, object]:
    env_cfg = {"old_dynamics": True}
    if candidate["layout_file"]:
        env_cfg["layout_file"] = candidate["layout_file"]
        env_cfg["layout_name"] = None
    else:
        env_cfg["layout_name"] = candidate["layout_name"]
        env_cfg["layout_file"] = None
    mdp = build_mdp(env_cfg)
    analysis = LayoutGraphAnalyzer().analyze(mdp)
    recipe = "mixed"
    if analysis.onion_disp and not analysis.tomato_disp:
        recipe = "onion"
    elif analysis.tomato_disp and not analysis.onion_disp:
        recipe = "tomato"
    return {
        "source": candidate["source"],
        "layout_name": analysis.layout_name or candidate["layout_name"],
        "layout_file": candidate["layout_file"],
        "width": analysis.width,
        "height": analysis.height,
        "components": len(analysis.components),
        "bottlenecks": len(analysis.bottlenecks),
        "onion_dispensers": len(analysis.onion_disp),
        "tomato_dispensers": len(analysis.tomato_disp),
        "dish_dispensers": len(analysis.dish_disp),
        "pots": len(analysis.pots),
        "serving_locations": len(analysis.serving),
        "handoff_counters": len(analysis.handoff_counters),
        "solo_capable": analysis.solo_capable,
        "forced_handoff": analysis.forced_handoff,
        "recipe_type": recipe,
        "topology_family": analysis.topology.value,
        "split": assign_split(analysis.layout_name or candidate["layout_name"], analysis.topology.value),
    }


def assign_split(layout_name: str, topology: str) -> str:
    key = layout_name.lower()
    if "forced" in key or topology == "FORCED_HANDOFF":
        return "adversarial_layouts"
    if "maze" in key or "corridor" in key or topology == "BOTTLENECK":
        return "heldout_layouts"
    if key in {"cramped_room", "coordination_ring", "counter_circuit"}:
        return "validation_layouts"
    return "train_layouts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/layout_catalog.csv")
    args = parser.parse_args()
    rows = []
    errors = []
    seen = set()
    for candidate in collect_layout_candidates():
        dedupe = (candidate["layout_name"], candidate["layout_file"])
        if dedupe in seen:
            continue
        seen.add(dedupe)
        try:
            rows.append(build_one(candidate))
        except Exception as exc:
            errors.append({"layout_name": candidate["layout_name"], "layout_file": candidate["layout_file"], "error": repr(exc)})
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if errors:
        err_path = out.with_name(out.stem + "_errors.csv")
        with err_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["layout_name", "layout_file", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print({"catalog": str(out), "rows": len(rows), "errors": len(errors)})


if __name__ == "__main__":
    main()
