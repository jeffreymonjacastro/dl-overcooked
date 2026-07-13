"""Topology analysis for Overcooked layouts.

The analyzer intentionally depends on graph structure instead of layout names.
It extracts connected walkable regions, resource accessibility and counters that
can be used as handoff points between separated players.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from overcooked_ai_py.mdp.actions import Action, Direction


Position = tuple[int, int]


class TopologyFamily(str, Enum):
    SOLO_CAPABLE = "SOLO_CAPABLE"
    SHARED_OPEN = "SHARED_OPEN"
    BOTTLENECK = "BOTTLENECK"
    FORCED_HANDOFF = "FORCED_HANDOFF"
    ASYMMETRIC_RESOURCES = "ASYMMETRIC_RESOURCES"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HandoffCounter:
    position: Position
    adjacent_components: tuple[int, ...]
    adjacent_walkable: tuple[Position, ...]
    score: float


@dataclass(frozen=True)
class LayoutAnalysis:
    layout_name: str
    width: int
    height: int
    walkable: frozenset[Position]
    counters: frozenset[Position]
    onion_disp: tuple[Position, ...]
    tomato_disp: tuple[Position, ...]
    dish_disp: tuple[Position, ...]
    pots: tuple[Position, ...]
    serving: tuple[Position, ...]
    starts: tuple[Position, ...]
    components: tuple[frozenset[Position], ...]
    component_by_pos: dict[Position, int]
    player_components: tuple[int | None, ...]
    bottlenecks: tuple[Position, ...]
    handoff_counters: tuple[HandoffCounter, ...]
    accessible_by_component: dict[int, dict[str, bool]]
    topology: TopologyFamily

    @property
    def solo_capable(self) -> bool:
        return self.topology in {TopologyFamily.SOLO_CAPABLE, TopologyFamily.SHARED_OPEN}

    @property
    def forced_handoff(self) -> bool:
        return self.topology == TopologyFamily.FORCED_HANDOFF


class LayoutGraphAnalyzer:
    """Build a compact structural summary from an OvercookedGridworld."""

    RESOURCE_KEYS = ("ingredient", "dish", "pot", "serve")

    def analyze(self, mdp) -> LayoutAnalysis:
        walkable = frozenset(tuple(p) for p in mdp.get_valid_player_positions())
        counters = frozenset(tuple(p) for p in mdp.get_counter_locations())
        onion_disp = tuple(tuple(p) for p in mdp.get_onion_dispenser_locations())
        tomato_disp = tuple(tuple(p) for p in mdp.get_tomato_dispenser_locations())
        dish_disp = tuple(tuple(p) for p in mdp.get_dish_dispenser_locations())
        pots = tuple(tuple(p) for p in mdp.get_pot_locations())
        serving = tuple(tuple(p) for p in mdp.get_serving_locations())
        starts = tuple(tuple(p) for p in getattr(mdp, "start_player_positions", ()))
        components, component_by_pos = self._connected_components(walkable)
        player_components = tuple(component_by_pos.get(start) for start in starts[:2])
        bottlenecks = tuple(sorted(self._find_bottlenecks(walkable)))
        accessible = self._resource_accessibility(
            component_by_pos,
            onion_disp + tomato_disp,
            dish_disp,
            pots,
            serving,
        )
        handoffs = tuple(
            sorted(
                self._handoff_counters(counters, walkable, component_by_pos, onion_disp + tomato_disp, dish_disp, pots, serving),
                key=lambda h: h.score,
            )
        )
        topology = self._classify(components, player_components, bottlenecks, handoffs, accessible)
        height = len(getattr(mdp, "terrain_mtx", []))
        width = len(getattr(mdp, "terrain_mtx", [[]])[0]) if height else 0
        return LayoutAnalysis(
            layout_name=str(getattr(mdp, "layout_name", "")),
            width=width,
            height=height,
            walkable=walkable,
            counters=counters,
            onion_disp=onion_disp,
            tomato_disp=tomato_disp,
            dish_disp=dish_disp,
            pots=pots,
            serving=serving,
            starts=starts,
            components=components,
            component_by_pos=component_by_pos,
            player_components=player_components,
            bottlenecks=bottlenecks,
            handoff_counters=handoffs,
            accessible_by_component=accessible,
            topology=topology,
        )

    def _connected_components(self, walkable: frozenset[Position]):
        components: list[frozenset[Position]] = []
        component_by_pos: dict[Position, int] = {}
        unvisited = set(walkable)
        while unvisited:
            start = unvisited.pop()
            queue = deque([start])
            group = {start}
            while queue:
                pos = queue.popleft()
                for nxt in self._neighbors(pos):
                    if nxt not in unvisited:
                        continue
                    unvisited.remove(nxt)
                    group.add(nxt)
                    queue.append(nxt)
            idx = len(components)
            frozen = frozenset(group)
            components.append(frozen)
            for pos in frozen:
                component_by_pos[pos] = idx
        return tuple(components), component_by_pos

    def _resource_accessibility(
        self,
        component_by_pos: dict[Position, int],
        ingredients: Iterable[Position],
        dishes: Iterable[Position],
        pots: Iterable[Position],
        serving: Iterable[Position],
    ) -> dict[int, dict[str, bool]]:
        accessible: dict[int, dict[str, bool]] = {}
        resource_sets = {
            "ingredient": tuple(ingredients),
            "dish": tuple(dishes),
            "pot": tuple(pots),
            "serve": tuple(serving),
        }
        for comp in set(component_by_pos.values()):
            accessible[comp] = {key: False for key in self.RESOURCE_KEYS}
        for key, positions in resource_sets.items():
            for pos in positions:
                for adj in self._neighbors(pos):
                    comp = component_by_pos.get(adj)
                    if comp is not None:
                        accessible.setdefault(comp, {k: False for k in self.RESOURCE_KEYS})[key] = True
        return accessible

    def _handoff_counters(
        self,
        counters: Iterable[Position],
        walkable: frozenset[Position],
        component_by_pos: dict[Position, int],
        ingredients: Iterable[Position],
        dishes: Iterable[Position],
        pots: Iterable[Position],
        serving: Iterable[Position],
    ) -> list[HandoffCounter]:
        resource_points = tuple(ingredients) + tuple(dishes) + tuple(pots) + tuple(serving)
        handoffs: list[HandoffCounter] = []
        for counter in counters:
            adj_walkable = tuple(p for p in self._neighbors(counter) if p in walkable)
            comps = tuple(sorted({component_by_pos[p] for p in adj_walkable}))
            if len(comps) < 2:
                continue
            dist = min((manhattan(counter, r) for r in resource_points), default=0)
            congestion = max(0, 3 - len(adj_walkable))
            score = float(dist + 2 * congestion)
            handoffs.append(HandoffCounter(counter, comps, adj_walkable, score))
        return handoffs

    def _find_bottlenecks(self, walkable: frozenset[Position]) -> set[Position]:
        bottlenecks = set()
        for pos in walkable:
            degree = sum(1 for nxt in self._neighbors(pos) if nxt in walkable)
            if degree <= 2:
                bottlenecks.add(pos)
        return bottlenecks

    def _classify(
        self,
        components,
        player_components,
        bottlenecks,
        handoffs,
        accessible,
    ) -> TopologyFamily:
        if not components:
            return TopologyFamily.UNKNOWN

        any_complete = any(
            access.get("ingredient") and access.get("dish") and access.get("pot") and access.get("serve")
            for access in accessible.values()
        )
        player_comps = {comp for comp in player_components if comp is not None}
        if len(player_comps) > 1 and not any_complete and handoffs:
            return TopologyFamily.FORCED_HANDOFF
        if any_complete and len(components) == 1:
            ratio = len(bottlenecks) / max(1, sum(len(c) for c in components))
            return TopologyFamily.BOTTLENECK if ratio > 0.45 else TopologyFamily.SHARED_OPEN
        if any_complete:
            return TopologyFamily.SOLO_CAPABLE
        if len(player_comps) > 1:
            return TopologyFamily.ASYMMETRIC_RESOURCES
        return TopologyFamily.UNKNOWN

    @staticmethod
    def _neighbors(pos: Position) -> list[Position]:
        return [Action.move_in_direction(pos, direction) for direction in Direction.ALL_DIRECTIONS]


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

