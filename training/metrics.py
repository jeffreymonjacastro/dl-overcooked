"""Shared metrics for Overcooked evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from scripts.score_official import SOUP_REWARD, calculate_attempt_score


def official_score_from_soups(
    soups: int,
    horizon: int,
    first_soup_t: int | None,
    last_soup_t: int | None,
    penalty: float = 0.0,
) -> float:
    timeouts = int(round(float(penalty) / 100.0)) if penalty else 0
    return calculate_attempt_score(soups, horizon, first_soup_t, last_soup_t, timeouts).attempt_score


def summarize_step_rows(rows: Iterable[dict[str, object]], horizon: int = 250) -> dict[str, float]:
    by_ep: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_ep[str(row["episode_id"])].append(row)
    soups = []
    scores = []
    first_ts = []
    last_ts = []
    zero = 0
    for ep_rows in by_ep.values():
        reward_events = []
        num_soups = 0
        for row in ep_rows:
            reward = float(row.get("reward", 0.0) or 0.0)
            if reward <= 0.0:
                continue
            delivered = int(round(reward / SOUP_REWARD))
            if delivered <= 0:
                continue
            reward_events.append(row)
            num_soups += delivered
        soups.append(float(num_soups))
        if num_soups == 0:
            zero += 1
            scores.append(0.0)
            continue
        first_t = int(reward_events[0]["timestep"])
        last_t = int(reward_events[-1]["timestep"])
        first_ts.append(float(first_t))
        last_ts.append(float(last_t))
        scores.append(calculate_attempt_score(num_soups, horizon, first_t, last_t).attempt_score)
    n = max(len(by_ep), 1)
    return {
        "num_rollouts": float(len(by_ep)),
        "mean_soups": sum(soups) / n if soups else 0.0,
        "median_soups": sorted(soups)[len(soups) // 2] if soups else 0.0,
        "zero_soup_rate": zero / n,
        "official_score_mean": sum(scores) / n if scores else 0.0,
        "first_soup_t_mean": sum(first_ts) / len(first_ts) if first_ts else -1.0,
        "last_soup_t_mean": sum(last_ts) / len(last_ts) if last_ts else -1.0,
    }
