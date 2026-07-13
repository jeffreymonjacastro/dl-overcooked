from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.datasets import NUM_ACTIONS, build_timestep_features


class PreviousActionAlignmentTest(unittest.TestCase):
    def test_previous_action_is_shifted_by_one(self):
        obs = np.zeros((5, 3), dtype=np.float32)
        actions = np.asarray([0, 1, 2, 3, 5], dtype=np.int64)
        agent_indices = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)

        features = build_timestep_features(obs, actions, agent_indices, mean, std)
        prev_slice = features[:, 3 + 2 : 3 + 2 + NUM_ACTIONS]
        decoded_previous = prev_slice.argmax(axis=1).tolist()

        self.assertEqual(decoded_previous, [4, 0, 1, 2, 3])

    def test_start_flag_only_marks_first_timestep(self):
        obs = np.zeros((4, 2), dtype=np.float32)
        actions = np.asarray([4, 4, 4, 4], dtype=np.int64)
        agent_indices = np.zeros(4, dtype=np.int64)
        features = build_timestep_features(obs, actions, agent_indices, np.zeros(2), np.ones(2))
        self.assertEqual(features[:, -1].tolist(), [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

