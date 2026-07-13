from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policies.template import StudentAgent


class AgentResetTest(unittest.TestCase):
    def test_gru_agent_reset_restores_initial_state(self):
        cfg = {
            "mode": "option_a",
            "checkpoint_path": "artifacts/option_a_baseline_before_a2/option_a/final_policy.npz",
            "model_config_path": "artifacts/option_a_baseline_before_a2/option_a/final_policy_config.json",
        }
        agent = StudentAgent(cfg)
        obs = {"obs": np.zeros(96, dtype=np.float32), "agent_index": 0}

        first_action = agent.act(obs)
        agent.act(obs)
        agent.reset()
        reset_action = agent.act(obs)

        fresh_action = StudentAgent(cfg).act(obs)

        self.assertEqual(first_action, reset_action)
        self.assertEqual(first_action, fresh_action)
        self.assertEqual(agent.policy.timestep, 1)
        self.assertEqual(agent.policy.previous_action, reset_action)


if __name__ == "__main__":
    unittest.main()

