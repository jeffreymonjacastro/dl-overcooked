"""Benchmark StudentAgent inference latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from policies.template import StudentAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate_option_a.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="reports/option_a_latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    agent_cfg = cfg["policies"]["agent_0"]["config"]
    agent = StudentAgent(agent_cfg)
    rng = np.random.default_rng(67)
    times = []
    agent.reset()
    for i in range(args.steps):
        obs = {"obs": rng.normal(size=96).astype(np.float32), "agent_index": i % 2}
        start = time.perf_counter()
        action = agent.act(obs)
        times.append(1000.0 * (time.perf_counter() - start))
        if action < 0 or action > 5:
            raise RuntimeError(f"Invalid action: {action}")
    arr = np.asarray(times, dtype=np.float64)
    payload = {
        "steps": args.steps,
        "latency_mean_ms": float(np.mean(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_max_ms": float(np.max(arr)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
