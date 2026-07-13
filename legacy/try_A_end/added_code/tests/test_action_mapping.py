from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policies.template import StudentAgent


class ActionMappingTest(unittest.TestCase):
    def test_fixed_policy_action_mapping(self):
        expected = {
            "north": 0,
            "up": 0,
            "south": 1,
            "down": 1,
            "east": 2,
            "right": 2,
            "west": 3,
            "left": 3,
            "stay": 4,
            "interact": 5,
        }
        for name, index in expected.items():
            with self.subTest(name=name):
                self.assertEqual(StudentAgent({"action": name}).act({}), index)

    def test_random_action_is_in_range(self):
        agent = StudentAgent({"action": "random", "seed": 67})
        for _ in range(100):
            action = agent.act({})
            self.assertGreaterEqual(action, 0)
            self.assertLessEqual(action, 5)


if __name__ == "__main__":
    unittest.main()

