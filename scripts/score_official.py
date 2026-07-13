"""Official score calculation for Overcooked evaluation attempts.

The competition objective is lexicographic in practice: soups first, then
robustness/timing. This module keeps the numeric score calculation in one
place so training and reports do not drift.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOUP_REWARD = 20.0


@dataclass(frozen=True)
class AttemptScore:
    soups: int
    horizon: int
    first_soup_timestep: int | None
    last_soup_timestep: int | None
    timeouts: int
    penalty: int
    attempt_score: float


@dataclass(frozen=True)
class ScenarioScore:
    score_seed_1: float
    score_seed_2: float
    score_seed_3: float
    scenario_score_mean: float
    scenario_score_std: float
    scenario_score_min: float
    zero_soup_attempts: int


def calculate_attempt_score(
    soups: int,
    horizon: int,
    first_soup_timestep: int | None,
    last_soup_timestep: int | None,
    timeouts: int = 0,
) -> AttemptScore:
    """Calculate the official score for one attempt."""
    soups = int(soups)
    horizon = int(horizon)
    timeouts = int(timeouts)
    if soups < 0:
        raise ValueError("soups must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if timeouts < 0:
        raise ValueError("timeouts must be non-negative")
    penalty = min(100 * timeouts, 5000)
    if soups == 0:
        if first_soup_timestep is not None or last_soup_timestep is not None:
            raise ValueError("soup timesteps must be None when soups == 0")
        return AttemptScore(soups, horizon, None, None, timeouts, penalty, 0.0)
    if first_soup_timestep is None or last_soup_timestep is None:
        raise ValueError("first and last soup timesteps are required when soups > 0")
    first = int(first_soup_timestep)
    last = int(last_soup_timestep)
    if not (0 <= first <= last <= horizon):
        raise ValueError("soup timesteps must satisfy 0 <= first <= last <= horizon")
    score = 10000 * soups + 10 * (horizon - last) + (horizon - first) - penalty
    return AttemptScore(soups, horizon, first, last, timeouts, penalty, float(score))


def calculate_scenario_score(attempts: Iterable[AttemptScore]) -> ScenarioScore:
    """Aggregate exactly three attempts into one official scenario score."""
    attempts = list(attempts)
    if len(attempts) != 3:
        raise ValueError("official scenario score requires exactly three attempts")
    scores = [a.attempt_score for a in attempts]
    mean = sum(scores) / 3.0
    var = sum((score - mean) ** 2 for score in scores) / 3.0
    return ScenarioScore(
        score_seed_1=float(scores[0]),
        score_seed_2=float(scores[1]),
        score_seed_3=float(scores[2]),
        scenario_score_mean=float(mean),
        scenario_score_std=float(math.sqrt(var)),
        scenario_score_min=float(min(scores)),
        zero_soup_attempts=sum(1 for a in attempts if a.soups == 0),
    )


def score_steps_rows(rows: list[dict[str, str]], horizon: int, timeouts: int = 0) -> AttemptScore:
    """Score one rollout from logged steps.csv rows.

    The local runner logs sparse step reward but not an explicit delivery event.
    In Overcooked-AI, a soup delivery gives sparse reward 20, so positive sparse
    reward / 20 is the validated soup count proxy used here.
    """
    soup_timesteps: list[int] = []
    soup_count = 0
    for row in rows:
        reward = float(row.get("reward", 0.0) or 0.0)
        if reward <= 0:
            continue
        delivered = int(round(reward / SOUP_REWARD))
        if delivered <= 0:
            continue
        soup_count += delivered
        soup_timesteps.extend([int(row["timestep"])] * delivered)
    if soup_count == 0:
        return calculate_attempt_score(0, horizon, None, None, timeouts)
    return calculate_attempt_score(soup_count, horizon, soup_timesteps[0], soup_timesteps[-1], timeouts)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_output_dir(output_dir: Path, horizon: int) -> list[dict[str, object]]:
    """Score every episode in an evaluation output directory."""
    steps_path = output_dir / "steps.csv"
    episodes_path = output_dir / "episodes.csv"
    steps = read_csv(steps_path)
    episodes = {int(row["episode_id"]): row for row in read_csv(episodes_path)}
    by_episode: dict[int, list[dict[str, str]]] = {}
    for row in steps:
        by_episode.setdefault(int(row["episode_id"]), []).append(row)
    scored = []
    for episode_id, episode_steps in sorted(by_episode.items()):
        episode = episodes.get(episode_id, {})
        attempt = score_steps_rows(episode_steps, horizon)
        scored.append(
            {
                "episode_id": episode_id,
                "layout": episode.get("layout_name", episode_steps[0].get("layout_name", "")),
                "seed": episode.get("seed", ""),
                "role_swap": episode.get("role_swap", ""),
                "soups": attempt.soups,
                "first_soup_timestep": attempt.first_soup_timestep,
                "last_soup_timestep": attempt.last_soup_timestep,
                "timeouts": attempt.timeouts,
                "penalty": attempt.penalty,
                "attempt_score": attempt.attempt_score,
            }
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--csv-output")
    args = parser.parse_args()
    rows = score_output_dir(Path(args.output_dir), args.horizon)
    if args.csv_output:
        out = Path(args.csv_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
