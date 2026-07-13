"""Heuristic partner-activity tracker for adaptive macro routing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class PartnerActivity(str, Enum):
    UNKNOWN = "UNKNOWN"
    SEEKING_INGREDIENT = "SEEKING_INGREDIENT"
    FILLING_POT = "FILLING_POT"
    SEEKING_DISH = "SEEKING_DISH"
    DELIVERING = "DELIVERING"
    PASSIVE = "PASSIVE"
    RANDOM = "RANDOM"
    BLOCKING = "BLOCKING"
    HANDOFF = "HANDOFF"


@dataclass(frozen=True)
class PartnerObservation:
    timestep: int
    position: tuple[int, int]
    held: str | None
    moved: bool
    useful_change: bool


class PartnerTracker:
    """Small deterministic tracker, used before any learned partner model."""

    def __init__(self, window: int = 20):
        self.window = int(window)
        self.history: deque[PartnerObservation] = deque(maxlen=self.window)
        self.prev_position = None
        self.prev_held = None

    def reset(self):
        self.history.clear()
        self.prev_position = None
        self.prev_held = None

    def update(self, state, partner_index: int, task_state=None) -> PartnerActivity:
        partner = state.players[partner_index]
        held = None if partner.held_object is None else partner.held_object.name
        position = tuple(partner.position)
        moved = self.prev_position is not None and position != self.prev_position
        useful_change = self.prev_held is not None and held != self.prev_held
        obs = PartnerObservation(int(state.timestep), position, held, moved, useful_change)
        self.history.append(obs)
        self.prev_position = position
        self.prev_held = held
        return self.activity(task_state=task_state)

    def activity(self, task_state=None) -> PartnerActivity:
        if len(self.history) < 3:
            return PartnerActivity.UNKNOWN
        recent = list(self.history)
        moved_count = sum(obs.moved for obs in recent)
        useful_count = sum(obs.useful_change for obs in recent)
        held_values = [obs.held for obs in recent[-5:]]
        unique_positions = len({obs.position for obs in recent[-10:]})
        if moved_count == 0 and useful_count == 0:
            return PartnerActivity.PASSIVE
        if unique_positions >= 7 and useful_count == 0:
            return PartnerActivity.RANDOM
        current_held = held_values[-1]
        if current_held in {"onion", "tomato"}:
            return PartnerActivity.FILLING_POT
        if current_held == "dish":
            return PartnerActivity.SEEKING_DISH
        if current_held == "soup":
            return PartnerActivity.DELIVERING
        if useful_count > 0 and task_state is not None and task_state.handoff_required:
            return PartnerActivity.HANDOFF
        return PartnerActivity.UNKNOWN

