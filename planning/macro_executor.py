"""Low-level executor for target-oriented macro actions."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from overcooked_ai_py.mdp.actions import Action, Direction

from planning.layout_graph import Position, manhattan


class MacroExecutor:
    """Turn a selected interaction target into one Overcooked primitive action."""

    def __init__(self, mdp, avoid_teammate: bool = True):
        self.mdp = mdp
        self.avoid_teammate = bool(avoid_teammate)

    def move_or_interact(self, state, agent_index: int, target: Position | None):
        if target is None:
            return Action.STAY
        player = state.players[agent_index]
        pos = player.position
        if manhattan(pos, target) == 1:
            desired = direction_from_to(pos, target)
            if player.orientation == desired:
                return Action.INTERACT
            return desired
        next_pos = self.next_step_to_interaction(state, agent_index, target)
        if next_pos is None:
            return Action.STAY
        return Action.determine_action_for_change_in_pos(pos, next_pos)

    def wait_near(self, state, agent_index: int, targets: Iterable[Position]):
        targets = list(targets)
        if not targets:
            return Action.STAY
        player = state.players[agent_index]
        if any(manhattan(player.position, target) == 1 for target in targets):
            return Action.STAY
        nearest = min(targets, key=lambda p: manhattan(player.position, p))
        next_pos = self.next_step_to_interaction(state, agent_index, nearest)
        if next_pos is None:
            return Action.STAY
        return Action.determine_action_for_change_in_pos(player.position, next_pos)

    def next_step_to_interaction(self, state, agent_index: int, target: Position):
        player = state.players[agent_index]
        valid = set(self.mdp.get_valid_player_positions())
        blocked = set()
        if self.avoid_teammate:
            for idx, other in enumerate(state.players):
                if idx != agent_index:
                    blocked.add(other.position)
        goals = [p for p in adjacent_positions(target) if p in valid and p not in blocked]
        if not goals:
            goals = [p for p in adjacent_positions(target) if p in valid]
        if not goals:
            return None
        path = shortest_path(player.position, set(goals), valid, blocked)
        if path is None or len(path) < 2:
            return None
        return path[1]


def shortest_path(start: Position, goals: set[Position], valid: set[Position], blocked: set[Position]):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos in goals:
            return path
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(pos, direction)
            if nxt not in valid:
                continue
            if nxt in blocked and nxt not in goals:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def adjacent_positions(pos: Position) -> list[Position]:
    return [Action.move_in_direction(pos, direction) for direction in Direction.ALL_DIRECTIONS]


def direction_from_to(a: Position, b: Position):
    direction = (b[0] - a[0], b[1] - a[1])
    if direction not in Direction.ALL_DIRECTIONS:
        raise ValueError(f"Positions are not adjacent: {a} -> {b}")
    return direction

