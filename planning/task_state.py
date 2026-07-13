"""Coordinate-invariant task-state extraction for macro policies."""

from __future__ import annotations

from dataclasses import dataclass

from overcooked_ai_py.mdp.overcooked_mdp import Recipe

from planning.layout_graph import LayoutAnalysis, Position, manhattan


@dataclass(frozen=True)
class PotSnapshot:
    position: Position
    status: str


@dataclass(frozen=True)
class TaskState:
    held_self: str | None
    held_partner: str | None
    pot_states: tuple[PotSnapshot, ...]
    counter_objects: dict[Position, str]
    ready_pots: tuple[Position, ...]
    cooking_pots: tuple[Position, ...]
    ingredient_pots: tuple[Position, ...]
    empty_pots: tuple[Position, ...]
    self_component: int | None
    partner_component: int | None
    best_handoff_counter: Position | None
    handoff_required: bool


class TaskStateExtractor:
    def __init__(self, mdp, analysis: LayoutAnalysis):
        self.mdp = mdp
        self.analysis = analysis

    def extract(self, state, agent_index: int) -> TaskState:
        partner_index = 1 - int(agent_index)
        player = state.players[agent_index]
        partner = state.players[partner_index]
        pot_states_raw = self.mdp.get_pot_states(state)
        pot_snapshots: list[PotSnapshot] = []
        for status, positions in pot_states_raw.items():
            for pos in positions:
                pot_snapshots.append(PotSnapshot(tuple(pos), str(status)))
        ingredient_pots = []
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            ingredient_pots.extend(tuple(p) for p in pot_states_raw.get(f"{k}_items", []))
        counter_objects = {tuple(obj.position): obj.name for obj in state.objects.values()}
        best_handoff = self.analysis.handoff_counters[0].position if self.analysis.handoff_counters else None
        return TaskState(
            held_self=None if player.held_object is None else player.held_object.name,
            held_partner=None if partner.held_object is None else partner.held_object.name,
            pot_states=tuple(sorted(pot_snapshots, key=lambda p: p.position)),
            counter_objects=counter_objects,
            ready_pots=tuple(tuple(p) for p in pot_states_raw.get("ready", [])),
            cooking_pots=tuple(tuple(p) for p in pot_states_raw.get("cooking", [])),
            ingredient_pots=tuple(ingredient_pots),
            empty_pots=tuple(tuple(p) for p in pot_states_raw.get("empty", [])),
            self_component=self.analysis.component_by_pos.get(player.position),
            partner_component=self.analysis.component_by_pos.get(partner.position),
            best_handoff_counter=best_handoff,
            handoff_required=self.analysis.forced_handoff,
        )


def nearest(origin: Position, positions):
    positions = list(positions)
    if not positions:
        return None
    return min(positions, key=lambda p: manhattan(origin, p))

