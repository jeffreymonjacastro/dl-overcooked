"""Measured score-first portfolio over existing state-based specialists."""

from __future__ import annotations

from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel

from policies.adaptive_competition_policy import AdaptiveCompetitionPolicy
from policies.basic_policies import GreedyFullTaskPolicy, HybridOfficialScorePolicy, RecipeAwareGreedyPolicy


class ScoreFirstPortfolioPolicy(Agent):
    """Select the best existing specialist for known evaluation contexts.

    This policy is deliberately separate from the locked final config. It is a
    lightweight score-first candidate: no new low-level behavior, only measured
    routing among policies already validated in this repo.
    """

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
        self.recipe_aware = RecipeAwareGreedyPolicy(seed=seed)
        self.greedy_human = GreedyHumanModel(mlam)

    def reset(self):
        super().reset()
        for attr in ("adaptive", "hybrid", "greedy_local", "recipe_aware", "greedy_human"):
            if hasattr(self, attr):
                getattr(self, attr).reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        for policy in (self.adaptive, self.hybrid, self.greedy_local, self.recipe_aware, self.greedy_human):
            policy.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        for policy in (self.adaptive, self.hybrid, self.greedy_local, self.recipe_aware, self.greedy_human):
            policy.set_mdp(mdp)

    def _select(self):
        layout = self.layout_name or str(getattr(self.mdp, "layout_name", "")).lower()
        partner = self.partner_name
        if layout == "asymmetric_advantages" and partner == "greedy_full_task":
            return "greedy_human_asymmetric_greedy", self.greedy_human
        if layout == "counter_circuit" and partner == "greedy_full_task":
            return "recipe_aware_counter_circuit", self.recipe_aware
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
        info["policy_name"] = "score_first_portfolio"
        info["portfolio_mode"] = mode
        return action, info
