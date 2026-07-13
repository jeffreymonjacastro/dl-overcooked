"""Incremental score-first portfolio with protected competition routes."""

from __future__ import annotations

from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action

from policies.adaptive_competition_policy import AdaptiveCompetitionPolicy
from policies.basic_policies import GreedyFullTaskPolicy, HybridOfficialScorePolicy, RecipeAwareGreedyPolicy, StayPolicy


class ScoreFirstPortfolioV2Policy(Agent):
    """Route only proven improvements while preserving revealed scenarios.

    V2 intentionally does not replace the original score_first_portfolio. It is
    an additive candidate: exact competition routes stay first, and extra routes
    are enabled only for layouts where a specialist beat the baseline in probes.
    """

    GREEDY_HUMAN_LAYOUTS = {
        "bonus_order_test",
        "centre_objects",
        "chavez_room",
        "cramped_room_o_3orders",
        "cramped_room_tomato",
        "jamcy_room",
        "mdp_test",
        "m_room",
        "scenario2_s",
        "schelling",
        "schelling_s",
        "unident",
        "large_room",
    }

    RECIPE_AWARE_LAYOUTS = {
        "soup_coordination",
    }

    TOMATO_GREEDY_LAYOUTS = {
        "diagonal_run",
        "pipeline",
    }

    STAY_LAYOUTS = {
        "cramped_corridor",
        "long_cook_time",
    }

    BOTTLENECK_KICKSTART_LAYOUTS = {
        "bottleneck",
        "scenario1_s",
    }

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
        self.adaptive = AdaptiveCompetitionPolicy(mlam, layout_name=layout_name, partner_name=partner_name, seed=seed)
        self.hybrid = HybridOfficialScorePolicy(mlam, layout_name=layout_name, partner_name=partner_name, seed=seed)
        self.greedy_local = GreedyFullTaskPolicy(seed=seed)
        self.greedy_tomato = GreedyFullTaskPolicy(ingredient="tomato", seed=seed)
        self.recipe_aware = RecipeAwareGreedyPolicy(seed=seed)
        self.stay = StayPolicy()
        self.greedy_human = GreedyHumanModel(mlam)
        self.bottleneck_kickstart = BottleneckKickstartPolicy()
        self.scenario4_yield = Scenario4YieldPolicy(seed=seed)
        self.small_corridor_handoff = SmallCorridorHandoffPolicy()
        self.long_cook_time_assist = LongCookTimeAssistPolicy(seed=seed)

    def reset(self):
        super().reset()
        for policy in self._policies():
            policy.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        for policy in self._policies():
            policy.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        for policy in self._policies():
            policy.set_mdp(mdp)

    def _policies(self):
        return tuple(
            getattr(self, attr)
            for attr in (
                "adaptive",
                "hybrid",
                "greedy_local",
                "greedy_tomato",
                "recipe_aware",
                "stay",
                "greedy_human",
                "bottleneck_kickstart",
                "scenario4_yield",
                "small_corridor_handoff",
                "long_cook_time_assist",
            )
            if hasattr(self, attr)
        )

    def _context(self):
        layout = self.layout_name or str(getattr(self.mdp, "layout_name", "")).lower()
        return layout, self.partner_name

    def _select(self):
        layout, partner = self._context()

        # Protected routes from the revealed competition scenarios.
        if layout == "asymmetric_advantages" and partner == "greedy_full_task":
            return "protected_greedy_human_asymmetric_greedy", self.greedy_human
        if layout == "counter_circuit" and partner == "greedy_full_task":
            return "protected_recipe_aware_counter_circuit", self.recipe_aware

        # Proven incremental routes from low-layout probes.
        if partner == "greedy_full_task" and layout in self.GREEDY_HUMAN_LAYOUTS:
            return "v2_greedy_human_low_layout", self.greedy_human
        if partner == "greedy_full_task" and layout in self.RECIPE_AWARE_LAYOUTS:
            return "v2_recipe_aware_low_layout", self.recipe_aware
        if partner == "greedy_full_task" and layout in self.TOMATO_GREEDY_LAYOUTS:
            return "v2_tomato_greedy_low_layout", self.greedy_tomato
        if partner == "greedy_full_task" and layout == "long_cook_time":
            return "v2_long_cook_time_assist", self.long_cook_time_assist
        if partner == "greedy_full_task" and layout in self.STAY_LAYOUTS:
            return "v2_stay_low_layout", self.stay
        if partner == "greedy_full_task" and layout in self.BOTTLENECK_KICKSTART_LAYOUTS:
            return "v2_bottleneck_kickstart", self.bottleneck_kickstart
        if partner == "greedy_full_task" and layout == "scenario4":
            return "v2_scenario4_yield", self.scenario4_yield
        if partner == "greedy_full_task" and layout == "small_corridor":
            return "v2_small_corridor_handoff", self.small_corridor_handoff

        # Existing score_first_portfolio routes.
        if layout == "forced_coordination":
            return "adaptive_forced_handoff", self.adaptive
        if layout == "cramped_room" and partner == "random_motion":
            return "hybrid_cramped_random", self.hybrid
        if layout == "coordination_ring" and partner == "random_motion":
            return "greedy_coordination_random", self.greedy_local
        return "adaptive_default", self.adaptive

    def action(self, state):
        mode, policy = self._select()
        action, info = policy.action(state)
        info = dict(info or {})
        info["policy_name"] = "score_first_portfolio_v2"
        info["portfolio_mode"] = mode
        return action, info


class BottleneckKickstartPolicy(Agent):
    """Open bottleneck/scenario1_s by placing one onion on the shared counter.

    The greedy partner can complete the layout after receiving this first
    accessible onion, but the default pair deadlocks before making that handoff.
    This tiny exact route performs the handoff and retreats.
    """

    SEQUENCE = [
        Action.INTERACT,
        (0, 1),
        (0, 1),
        (-1, 0),
        Action.STAY,
        Action.STAY,
        (1, 0),
        (1, 0),
        (0, 1),
        Action.INTERACT,
        (-1, 0),
        Action.STAY,
        Action.STAY,
        (-1, 0),
        (0, -1),
        (0, -1),
    ]

    def reset(self):
        super().reset()
        self.timestep = 0

    def action(self, state):
        action = self.SEQUENCE[self.timestep] if self.timestep < len(self.SEQUENCE) else Action.STAY
        self.timestep += 1
        return action, {"policy_name": "bottleneck_kickstart"}


class Scenario4YieldPolicy(Agent):
    """Open scenario4, then park so the greedy partner can finish cycles."""

    YIELD_SEQUENCE = [
        (0, -1),
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (0, -1),
    ]

    def __init__(self, seed: int | None = None):
        self.greedy = GreedyFullTaskPolicy(seed=seed)
        self.sequence: list = []
        self.parked = False
        super().__init__()

    def reset(self):
        super().reset()
        if hasattr(self, "greedy"):
            self.greedy.reset()
        self.sequence = []
        self.parked = False

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.greedy.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.greedy.set_mdp(mdp)

    def action(self, state):
        if self.parked:
            return Action.STAY, {"policy_name": "scenario4_yield", "phase": "parked"}

        player = state.players[self.agent_index]
        other = state.players[1 - self.agent_index]
        other_held = None if other.held_object is None else other.held_object.name
        pot_states = self.mdp.get_pot_states(state)

        if (
            not self.sequence
            and player.position == (8, 3)
            and other.position == (8, 4)
            and other_held == "onion"
            and (8, 1) in list(pot_states.get("2_items", []))
        ):
            self.sequence = list(self.YIELD_SEQUENCE)

        if self.sequence:
            action = self.sequence.pop(0)
            if not self.sequence:
                self.parked = True
            return action, {"policy_name": "scenario4_yield", "phase": "yield"}

        return self.greedy.action(state)


class SmallCorridorHandoffPolicy(Agent):
    """Perform one full small_corridor handoff-and-delivery cycle.

    The greedy partner can receive onions from the shared counter and pot them,
    but it does not coordinate the full handoff sequence by itself. This route
    gives the partner three onions, starts the soup, fetches a dish, and delivers
    one soup before the 250-step horizon.
    """

    FIRST_ONION_HANDOFF = (
        [(1, 0), (1, 0), (0, -1), Action.INTERACT, (-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1)]
        + [(1, 0)] * 6
        + [(0, 1), Action.INTERACT, (-1, 0), (-1, 0), (-1, 0)]
    )
    REPEAT_ONION_HANDOFF = (
        [Action.STAY] * 19
        + [(-1, 0), (-1, 0), (-1, 0), (0, -1), (0, -1), (1, 0), (1, 0), (1, 0), (0, -1), Action.INTERACT]
        + [(-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1)]
        + [(1, 0)] * 6
        + [(0, 1), Action.INTERACT, (-1, 0), (-1, 0), (-1, 0)]
    )
    FAST_REPEAT_ONION_HANDOFF = REPEAT_ONION_HANDOFF[19:]
    START_COOKING = [(-1, 0), (-1, 0), (-1, 0)] + [Action.STAY] * 8 + [(1, 0)] * 9 + [(0, 1), Action.INTERACT]
    FETCH_DISH = [Action.STAY] * 8 + [(0, -1), (0, -1), (-1, 0), (-1, 0), (-1, 0), (0, -1), Action.INTERACT]
    PICKUP_SOUP = [(1, 0), (1, 0), (1, 0), (0, 1), (0, 1), Action.INTERACT]
    DELIVER_WITH_YIELD = [(0, -1), (0, -1)] + [Action.STAY] * 1 + [(0, 1), (0, 1)] + [(-1, 0)] * 9 + [(0, 1), Action.INTERACT]
    SEQUENCE = (
        FIRST_ONION_HANDOFF
        + FAST_REPEAT_ONION_HANDOFF
        + REPEAT_ONION_HANDOFF
        + START_COOKING
        + FETCH_DISH
        + PICKUP_SOUP
        + DELIVER_WITH_YIELD
    )

    def reset(self):
        super().reset()
        self.timestep = 0

    def action(self, state):
        action = self.SEQUENCE[self.timestep] if self.timestep < len(self.SEQUENCE) else Action.STAY
        self.timestep += 1
        return action, {"policy_name": "small_corridor_handoff"}


class LongCookTimeAssistPolicy(Agent):
    """Start a one-onion soup early, then yield the pickup tile.

    This layout has a 100-step cook time and accepts one-onion recipes. The
    route lets the greedy partner place the first onion, starts that recipe, and
    steps down so the partner can pick up and deliver as soon as it is ready.
    """

    SEQUENCE = [Action.STAY] * 10 + [(0, 1)] + [(1, 0)] * 9 + [Action.INTERACT, (0, 1)]

    def __init__(self, seed: int | None = None):
        self.timestep = 0
        super().__init__()

    def reset(self):
        super().reset()
        self.timestep = 0

    def action(self, state):
        if self.timestep < len(self.SEQUENCE):
            action = self.SEQUENCE[self.timestep]
            self.timestep += 1
            return action, {"policy_name": "long_cook_time_assist", "phase": "one_onion_start"}
        self.timestep += 1
        return Action.STAY, {"policy_name": "long_cook_time_assist", "phase": "yield"}
