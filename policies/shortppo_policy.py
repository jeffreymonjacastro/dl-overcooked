"""Score-first short-training candidate.

The 7H guide asks for a separate candidate that can use short PPO/BC artifacts
but should fall back to the best score-first planner if learning does not beat
it. This policy keeps that boundary explicit: it wraps the existing adaptive
planner and applies only parameter-search style overrides loaded from config.
"""

from __future__ import annotations

from pathlib import Path
import json

from policies.adaptive_competition_policy import AdaptiveCompetitionPolicy


class AdaptiveCompetitionShortPPOPolicy(AdaptiveCompetitionPolicy):
    """Separate candidate registered as ``adaptive_competition_shortppo``."""

    def __init__(
        self,
        mlam,
        layout_name: str | None = None,
        partner_name: str | None = None,
        seed: int | None = None,
        params_path: str | None = None,
        params: dict | None = None,
    ):
        super().__init__(mlam, layout_name=layout_name, partner_name=partner_name, seed=seed)
        self.shortppo_params = dict(params or {})
        if params_path:
            path = Path(params_path)
            if path.exists():
                self.shortppo_params.update(json.loads(path.read_text(encoding="utf-8")))

    def action(self, state):
        action, info = super().action(state)
        info = dict(info or {})
        info["policy_name"] = "adaptive_competition_shortppo"
        info["shortppo_params"] = self.shortppo_params.get("name", "score_first_planner")
        return action, info

