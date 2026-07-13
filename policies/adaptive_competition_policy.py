"""Adaptive competition policy built on topology and macro actions.

This policy preserves the score-first baseline as a fallback, but adds a
topology-driven handoff mode for layouts where no single component has every
resource needed to cook and deliver soup.
"""

from __future__ import annotations

from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

from models.partner_tracker import PartnerTracker
from planning.layout_graph import LayoutGraphAnalyzer, LayoutAnalysis, Position, TopologyFamily, manhattan
from planning.macro_executor import MacroExecutor
from planning.task_state import TaskStateExtractor, nearest
from policies.basic_policies import GreedyFullTaskPolicy, HybridOfficialScorePolicy


class AdaptiveCompetitionPolicy(Agent):
    """Portfolio router using topology first and baseline specialists second."""

    def __init__(
        self,
        mlam,
        layout_name: str | None = None,
        partner_name: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.layout_name = str(layout_name or "").lower()
        self.partner_name = str(partner_name or "").lower()
        self.baseline = HybridOfficialScorePolicy(mlam, layout_name=layout_name, partner_name=partner_name, seed=seed)
        self.greedy_local = GreedyFullTaskPolicy(seed=seed)
        self.greedy_human = GreedyHumanModel(mlam)
        self.tracker = PartnerTracker(window=20)
        self.analysis: LayoutAnalysis | None = None
        self.extractor: TaskStateExtractor | None = None
        self.executor: MacroExecutor | None = None

    def reset(self):
        super().reset()
        self.analysis = None
        self.extractor = None
        self.executor = None
        if hasattr(self, "tracker"):
            self.tracker.reset()
        if hasattr(self, "baseline"):
            self.baseline.reset()
        if hasattr(self, "greedy_local"):
            self.greedy_local.reset()
        if hasattr(self, "greedy_human"):
            self.greedy_human.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.baseline.set_agent_index(agent_index)
        self.greedy_local.set_agent_index(agent_index)
        self.greedy_human.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.baseline.set_mdp(mdp)
        self.greedy_local.set_mdp(mdp)
        self.greedy_human.set_mdp(mdp)

    def _ensure_runtime(self):
        if self.analysis is None:
            self.analysis = LayoutGraphAnalyzer().analyze(self.mdp)
            self.extractor = TaskStateExtractor(self.mdp, self.analysis)
            self.executor = MacroExecutor(self.mdp, avoid_teammate=True)

    def action(self, state):
        self._ensure_runtime()
        assert self.analysis is not None
        assert self.extractor is not None
        assert self.executor is not None
        task_state = self.extractor.extract(state, self.agent_index)
        partner_activity = self.tracker.update(state, 1 - self.agent_index, task_state=task_state)

        if self.analysis.topology == TopologyFamily.FORCED_HANDOFF:
            action, mode = self._forced_handoff_action(state, task_state)
        elif self.partner_name in {"stay", "random_motion"} and self.analysis.solo_capable:
            action, mode = self.greedy_local.action(state)[0], "solo_pipeline"
        else:
            action, info = self.baseline.action(state)
            mode = "baseline:" + str(info.get("selected_mode", "unknown"))

        return action, {
            "policy_name": "adaptive_competition",
            "adaptive_mode": mode,
            "topology": self.analysis.topology.value,
            "partner_activity": partner_activity.value,
        }

    def _forced_handoff_action(self, state, task_state):
        comp = task_state.self_component
        access = self.analysis.accessible_by_component.get(comp, {}) if comp is not None else {}
        can_supply = bool(access.get("ingredient") or access.get("dish"))
        can_cook = bool(access.get("pot") or access.get("serve"))
        if can_cook and not access.get("ingredient"):
            return self._forced_cooker_action(state, task_state), "forced_cooker"
        if can_supply and not access.get("pot"):
            return self._forced_supplier_action(state, task_state), "forced_supplier"
        if can_cook:
            return self._forced_cooker_action(state, task_state), "forced_cooker_mixed"
        if can_supply:
            return self._forced_supplier_action(state, task_state), "forced_supplier_mixed"
        return self.baseline.action(state)[0], "forced_fallback"

    def _forced_cooker_action(self, state, task_state):
        player = state.players[self.agent_index]
        held = player.held_object
        held_name = None if held is None else held.name

        if held_name == "soup":
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.serving))
        if held_name == "dish":
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, task_state.ready_pots))
        if held_name in {"onion", "tomato"}:
            target = nearest(player.position, self._pots_accepting_ingredients(state))
            return self.executor.move_or_interact(state, self.agent_index, target)

        if task_state.ready_pots:
            dish = self._nearest_accessible_counter_object(player.position, "dish", task_state.counter_objects)
            if dish is not None:
                return self.executor.move_or_interact(state, self.agent_index, dish)
            if self._component_can(task_state.self_component, "dish"):
                return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.dish_disp))
            return self.executor.wait_near(state, self.agent_index, self._handoffs_for_component(task_state.self_component))

        ingredient = self._nearest_accessible_counter_object(player.position, "onion", task_state.counter_objects)
        if ingredient is None:
            ingredient = self._nearest_accessible_counter_object(player.position, "tomato", task_state.counter_objects)
        if ingredient is not None and self._pots_accepting_ingredients(state):
            return self.executor.move_or_interact(state, self.agent_index, ingredient)

        if self._pots_accepting_ingredients(state):
            return self.executor.wait_near(state, self.agent_index, self._handoffs_for_component(task_state.self_component))

        if self._component_can(task_state.self_component, "dish"):
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.dish_disp))
        return Action.STAY

    def _forced_supplier_action(self, state, task_state):
        player = state.players[self.agent_index]
        held = player.held_object
        held_name = None if held is None else held.name
        handoffs = self._handoffs_for_component(task_state.self_component)
        counter_objects = task_state.counter_objects
        dish_handoffs = self._dish_handoffs(task_state.self_component)
        ingredient_handoffs = self._ingredient_handoffs(task_state.self_component)

        if held_name == "dish":
            empty_dish_handoffs = [pos for pos in dish_handoffs if pos not in counter_objects]
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, empty_dish_handoffs or dish_handoffs))
        if held_name in {"onion", "tomato"}:
            empty_ingredient_handoffs = [pos for pos in ingredient_handoffs if pos not in counter_objects]
            if not empty_ingredient_handoffs:
                return Action.STAY
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, empty_ingredient_handoffs))

        desired = self._supplier_desired_object(state, task_state)
        if desired in {"onion", "tomato"} and not any(pos not in counter_objects for pos in ingredient_handoffs):
            if self._ready_soup_exists(state) and self._component_can(task_state.self_component, "dish"):
                desired = "dish"
            else:
                return Action.STAY
        if desired == "dish" and self._component_can(task_state.self_component, "dish"):
            if not any(pos not in counter_objects for pos in dish_handoffs):
                return Action.STAY
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.dish_disp))
        if desired == "tomato" and self.analysis.tomato_disp:
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.tomato_disp))
        if self.analysis.onion_disp:
            return self.executor.move_or_interact(state, self.agent_index, nearest(player.position, self.analysis.onion_disp))
        return Action.STAY

    def _supplier_desired_object(self, state, task_state) -> str:
        handoff_objects = [name for pos, name in task_state.counter_objects.items() if pos in self._all_handoffs()]
        dish_handoffs = set(self._dish_handoffs(task_state.self_component))
        ingredient_handoffs = set(self._ingredient_handoffs(task_state.self_component))
        dish_waiting = any(task_state.counter_objects.get(pos) == "dish" for pos in dish_handoffs)
        ingredient_waiting = sum(1 for pos in ingredient_handoffs if task_state.counter_objects.get(pos) in {"onion", "tomato"})
        pot_states = self.mdp.get_pot_states(state)
        ready_or_cooking = bool(pot_states.get("ready") or pot_states.get("cooking") or self._ready_soup_exists(state))
        full_not_cooking = bool(pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items"))
        if (ready_or_cooking or full_not_cooking) and not dish_waiting:
            return "dish"
        if ingredient_waiting >= max(1, len(ingredient_handoffs)) and not dish_waiting and self._component_can(task_state.self_component, "dish"):
            return "dish"
        return "onion"

    def _pots_accepting_ingredients(self, state) -> list[Position]:
        pot_states = self.mdp.get_pot_states(state)
        positions: list[Position] = []
        positions.extend(tuple(p) for p in pot_states.get("empty", []))
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            positions.extend(tuple(p) for p in pot_states.get(f"{k}_items", []))
        return positions

    def _nearest_accessible_counter_object(self, origin: Position, object_name: str, counter_objects: dict[Position, str]):
        candidates = []
        own_comp = self.analysis.component_by_pos.get(origin)
        for pos, name in counter_objects.items():
            if name != object_name:
                continue
            if any(self.analysis.component_by_pos.get(adj) == own_comp for adj in _adjacent(pos)):
                candidates.append(pos)
        return nearest(origin, candidates)

    def _component_can(self, component: int | None, resource: str) -> bool:
        if component is None:
            return False
        return bool(self.analysis.accessible_by_component.get(component, {}).get(resource))

    def _handoffs_for_component(self, component: int | None) -> list[Position]:
        if component is None:
            return []
        return [handoff.position for handoff in self.analysis.handoff_counters if component in handoff.adjacent_components]

    def _dish_handoffs(self, component: int | None) -> list[Position]:
        handoffs = self._handoffs_for_component(component)
        if not handoffs:
            return []
        if self.analysis.dish_disp:
            return [min(handoffs, key=lambda pos: min(manhattan(pos, dish) for dish in self.analysis.dish_disp))]
        return [handoffs[-1]]

    def _ingredient_handoffs(self, component: int | None) -> list[Position]:
        handoffs = self._handoffs_for_component(component)
        dish_slots = set(self._dish_handoffs(component))
        remaining = [pos for pos in handoffs if pos not in dish_slots]
        return remaining or handoffs

    def _all_handoffs(self) -> set[Position]:
        return {handoff.position for handoff in self.analysis.handoff_counters}

    def _ready_soup_exists(self, state) -> bool:
        pot_positions = set(self.analysis.pots)
        for obj in state.objects.values():
            if obj.name == "soup" and tuple(obj.position) in pot_positions:
                return True
        return False


def _adjacent(pos: Position):
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
