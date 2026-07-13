"""Vector environment placeholders for CPU-bound Overcooked rollouts.

The real environment simulation remains CPU-bound. This module centralizes the
worker-count configuration used by Macro-PPO scripts so GPU updates can be
batched without pretending rollouts themselves are GPU accelerated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorEnvConfig:
    num_workers: int = 16
    rollout_horizon: int = 250
    batch_size: int = 4096


def choose_worker_count(requested: int | None, cpu_count: int | None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    if cpu_count is None or cpu_count <= 0:
        return 16
    return max(1, min(64, int(cpu_count) - 1))

