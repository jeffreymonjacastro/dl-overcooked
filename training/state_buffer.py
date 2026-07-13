"""Failure-state buffer for later macro-policy fine tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureStateRecord:
    reason: str
    layout_name: str
    timestep: int
    agent_index: int
    held_self: str | None
    held_partner: str | None
    note: str = ""


class FailureStateBuffer:
    def __init__(self):
        self.records: list[FailureStateRecord] = []

    def add(self, record: FailureStateRecord):
        self.records.append(record)

    def extend(self, records):
        self.records.extend(records)

    def write_jsonl(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def summary(self) -> dict[str, Any]:
        by_reason: dict[str, int] = {}
        for record in self.records:
            by_reason[record.reason] = by_reason.get(record.reason, 0) + 1
        return {"records": len(self.records), "by_reason": by_reason}

