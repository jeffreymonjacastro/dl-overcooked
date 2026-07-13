"""Evaluate the current policy across a list of Overcooked layouts.

This is a broad smoke/evidence sweep, not a training script. It runs each
layout in a separate subprocess so slow layouts can be marked as timeouts
without blocking the whole report.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from scripts.evaluate_competition_protocol import summarize_attempts
from scripts.score_official import score_output_dir
from src.environment import load_custom_layout_dict
from src.runner import run_from_config


DEFAULT_LAYOUTS = [
    "asymmetric_advantages",
    "asymmetric_advantages_tomato",
    "bonus_order_test",
    "bottleneck",
    "centre_objects",
    "centre_pots",
    "chavez_room",
    "coordination_ring",
    "corridor",
    "counter_circuit",
    "counter_circuit_o_1order",
    "cramped_corridor",
    "cramped_room",
    "cramped_room_o_3orders",
    "cramped_room_single",
    "cramped_room_tomato",
    "diagonal_run",
    "five_by_five",
    "forced_coordination",
    "forced_coordination_tomato",
    "inverse_marshmallow_experiment",
    "jamcy_room",
    "large_room",
    "long_cook_time",
    "marshmallow_experiment",
    "marshmallow_experiment_coordination",
    "maze_kitchen",
    "mdp_test",
    "multiplayer_schelling",
    "m_room",
    "m_shaped_s",
    "pipeline",
    "scenario1_s",
    "scenario2",
    "scenario2_s",
    "scenario3",
    "scenario4",
    "schelling",
    "schelling_s",
    "simple_o",
    "simple_o_t",
    "simple_tomato",
    "small_corridor",
    "soup_coordination",
    "tutorial_0",
    "tutorial_1",
    "tutorial_2",
    "tutorial_3",
    "unident",
    "you_shall_not_pass",
]


def parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def builtin_policy(name: str, layout: str, partner: str, seed: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "builtin",
        "name": name,
        "seed": seed,
        "random_action_prob": 0.0,
        "sticky_action_prob": 0.0,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    if name in {
        "hybrid_official_score",
        "adaptive_competition",
        "adaptive_competition_shortppo",
        "score_first_portfolio",
    }:
        cfg["config"] = {"layout_name": layout, "partner_name": partner}
    return cfg


def find_repo_layouts() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for root in [REPO_ROOT / "configs", REPO_ROOT / "data"]:
        if not root.exists():
            continue
        for path in root.rglob("*.layout"):
            paths.setdefault(path.stem, path)
    return paths


def can_build_builtin(layout: str) -> tuple[bool, str]:
    try:
        OvercookedGridworld.from_layout_name(layout, old_dynamics=True)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report all layout loader failures
        return False, f"{type(exc).__name__}: {exc}"


def resolve_layout(layout: str, repo_layouts: dict[str, Path]) -> tuple[str | None, Path | None, str | None]:
    ok, err = can_build_builtin(layout)
    if ok:
        return "builtin", None, None
    if layout not in repo_layouts:
        return None, None, f"not found as built-in and not found under configs/data; built-in error was {err}"
    layout_file = repo_layouts[layout]
    try:
        load_custom_layout_dict(layout_file)
    except Exception as exc:  # noqa: BLE001
        return None, layout_file, f"layout file found but invalid: {type(exc).__name__}: {exc}"
    return "file", layout_file, None


def make_config(
    *,
    policy: str,
    partner: str,
    layout: str,
    source: str,
    layout_file: Path | None,
    seeds: list[int],
    horizon: int,
    output_dir: Path,
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "layout_name": layout if source == "builtin" else None,
        "layout_file": None if source == "builtin" else str(layout_file),
        "horizon": horizon,
        "old_dynamics": True,
    }
    return {
        "seed": seeds[0],
        "mode": "evaluation",
        "environment": environment,
        "policies": {
            "agent_0": builtin_policy(policy, layout, partner, seeds[0]),
            "agent_1": builtin_policy(partner, layout, partner, seeds[0]),
        },
        "execution": {
            "num_episodes": len(seeds),
            "episode_seeds": seeds,
            "swap_agent_positions": False,
        },
        "observation": {"type": "featurized", "include_agent_index": True},
        "rendering": {"mode": "none", "fps": 0, "save_gif": False},
        "logging": {
            "output_dir": str(output_dir),
            "save_step_log": True,
            "save_episode_summary": True,
            "save_trajectory_pickle": False,
        },
    }


def evaluate_one(args: argparse.Namespace) -> int:
    seeds = parse_seeds(args.seeds)
    layout_file = Path(args.layout_file) if args.layout_file else None
    output_dir = REPO_ROOT / "outputs" / "layout_sweep_score_first_portfolio" / args.layout
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg = make_config(
            policy=args.policy,
            partner=args.partner,
            layout=args.layout,
            source=args.source,
            layout_file=layout_file,
            seeds=seeds,
            horizon=args.horizon,
            output_dir=output_dir,
        )
        aggregate = run_from_config(cfg)
        attempts = score_output_dir(output_dir, horizon=args.horizon)
        summary = summarize_attempts(attempts)
        soups = [float(row["soups"]) for row in attempts]
        row = {
            "layout": args.layout,
            "source": args.source,
            "layout_file": str(layout_file or ""),
            "episodes": int(aggregate["num_rollouts"]),
            "mean_soups": summary.get("mean_soups", 0.0),
            "median_soups": summary.get("median_soups", 0.0),
            "p_at_least_1_soup": summary.get("p_at_least_1_soup", 0.0),
            "zero_rate": summary.get("zero_rate", 0.0),
            "official_score_mean": summary.get("official_score_mean", 0.0),
            "official_score_p10": summary.get("official_score_p10", 0.0),
            "min_soups": min(soups) if soups else 0.0,
            "max_soups": max(soups) if soups else 0.0,
            "output_dir": str(output_dir),
            "status": "LOW" if summary.get("mean_soups", 0.0) < 2.0 else "OK",
            "notes": "",
        }
        if row["status"] == "LOW":
            row["notes"] = low_score_note(row)
        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001
        result = {
            "layout": args.layout,
            "status": "FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 1


def low_score_note(row: dict[str, Any]) -> str:
    if float(row["mean_soups"]) == 0.0:
        return "0 soups in every measured seed; current routing/partner pairing cannot complete this layout."
    if float(row["zero_rate"]) > 0.0:
        return "Below 2 soups with some zero-soup seeds; coordination/routing is unstable on this layout."
    return "It delivers soups but too slowly; this layout needs layout-specific routing optimization."


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    omitted: list[dict[str, Any]],
    policy: str,
    partner: str,
    seeds: list[int],
    horizon: int,
    timeout_seconds: int,
) -> None:
    lines = [
        "# Layout Sweep - Score First Portfolio",
        "",
        "No training was run. This evaluates the current policy as-is.",
        "",
        "## Protocol",
        "",
        "```text",
        f"policy = {policy}",
        f"partner = {partner}",
        f"seeds = {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes per completed layout)",
        f"horizon = {horizon}",
        "role_swap = false",
        "noise = none",
        f"timeout_seconds_per_layout = {timeout_seconds}",
        "```",
        "",
        "## Summary",
        "",
        "| Layout | Source | Mean soups | Mean score | P(>=1 soup) | Zero rate | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['layout']}` | {row['source']} | {float(row['mean_soups']):.3f} | "
            f"{float(row['official_score_mean']):.2f} | {float(row['p_at_least_1_soup']):.2f} | "
            f"{float(row['zero_rate']):.2f} | {row['status']} |"
        )

    low_rows = [row for row in rows if row["status"] == "LOW"]
    lines.extend(["", "## Layouts Below 2 Soups", ""])
    if low_rows:
        lines.extend(["| Layout | Mean soups | Mean score | Zero rate | Likely reason |", "| --- | ---: | ---: | ---: | --- |"])
        for row in sorted(low_rows, key=lambda item: (float(item["mean_soups"]), item["layout"])):
            lines.append(
                f"| `{row['layout']}` | {float(row['mean_soups']):.3f} | "
                f"{float(row['official_score_mean']):.2f} | {float(row['zero_rate']):.2f} | {row['notes']} |"
            )
    else:
        lines.append("All completed layouts averaged at least 2 soups.")

    lines.extend(["", "## Omitted Or Timed Out", ""])
    if omitted:
        lines.extend(["| Layout | Reason | Path |", "| --- | --- | --- |"])
        for row in omitted:
            lines.append(f"| `{row['layout']}` | {row['reason']} | `{row.get('path', '')}` |")
    else:
        lines.append("No requested layouts were omitted or timed out.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace) -> int:
    layouts = [item.strip().removesuffix(".layout") for item in args.layouts.split(",") if item.strip()]
    repo_layouts = find_repo_layouts()
    seeds = parse_seeds(args.seeds)
    worker_dir = REPO_ROOT / "reports" / "_layout_sweep_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    script = Path(__file__).resolve()

    for index, layout in enumerate(layouts, start=1):
        source, layout_file, reason = resolve_layout(layout, repo_layouts)
        if reason:
            print(f"[{index}/{len(layouts)}] OMIT {layout}: {reason}", flush=True)
            omitted.append({"layout": layout, "reason": reason, "path": str(layout_file or "")})
            continue

        result_path = worker_dir / f"{layout}.json"
        if result_path.exists():
            result_path.unlink()
        cmd = [
            sys.executable,
            str(script),
            "--worker",
            "--layout",
            layout,
            "--source",
            str(source),
            "--layout-file",
            str(layout_file or ""),
            "--policy",
            args.policy,
            "--partner",
            args.partner,
            "--seeds",
            args.seeds,
            "--horizon",
            str(args.horizon),
            "--result-path",
            str(result_path),
        ]
        print(f"[{index}/{len(layouts)}] RUN {layout} ({source})", flush=True)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, timeout=args.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {args.timeout_seconds}s", flush=True)
            omitted.append(
                {
                    "layout": layout,
                    "reason": f"timeout after {args.timeout_seconds}s; current planner/policy is too slow here",
                    "path": str(layout_file or ""),
                }
            )
            continue
        if not result_path.exists():
            omitted.append({"layout": layout, "reason": "worker produced no result file", "path": str(layout_file or "")})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "FAILED":
            omitted.append({"layout": layout, "reason": result.get("reason", "worker failed"), "path": str(layout_file or "")})
            print(f"    FAILED {result.get('reason')}", flush=True)
            continue
        rows.append(result)
        print(
            f"    soups={float(result['mean_soups']):.3f} "
            f"score={float(result['official_score_mean']):.2f} status={result['status']}",
            flush=True,
        )

    report_dir = REPO_ROOT / "reports"
    result_csv = report_dir / "layout_sweep_score_first_portfolio.csv"
    omitted_csv = report_dir / "layout_sweep_omitted.csv"
    report_md = report_dir / "LAYOUT_SWEEP_SCORE_FIRST_PORTFOLIO.md"

    if rows:
        write_csv(
            result_csv,
            rows,
            [
                "layout",
                "source",
                "layout_file",
                "episodes",
                "mean_soups",
                "median_soups",
                "p_at_least_1_soup",
                "zero_rate",
                "official_score_mean",
                "official_score_p10",
                "min_soups",
                "max_soups",
                "status",
                "notes",
                "output_dir",
            ],
        )
    if omitted:
        write_csv(omitted_csv, omitted, ["layout", "reason", "path"])

    write_report(
        report_md,
        rows=rows,
        omitted=omitted,
        policy=args.policy,
        partner=args.partner,
        seeds=seeds,
        horizon=args.horizon,
        timeout_seconds=args.timeout_seconds,
    )
    print({"evaluated": len(rows), "omitted_or_timed_out": len(omitted), "report": str(report_md)}, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default=",".join(DEFAULT_LAYOUTS))
    parser.add_argument("--policy", default="score_first_portfolio")
    parser.add_argument("--partner", default="greedy_full_task")
    parser.add_argument("--seeds", default="67-76")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--layout")
    parser.add_argument("--source")
    parser.add_argument("--layout-file", default="")
    parser.add_argument("--result-path")
    args = parser.parse_args()

    if args.worker:
        return evaluate_one(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
