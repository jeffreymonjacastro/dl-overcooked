from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.metrics import official_score_from_soups, summarize_step_rows


class RewardAndSoupMetricsTest(unittest.TestCase):
    def test_official_score_proxy(self):
        self.assertEqual(official_score_from_soups(0, 250, None, None), 0.0)
        self.assertEqual(official_score_from_soups(2, 250, 20, 180), 2 * 10000 + 10 * 70 + 230)

    def test_summarize_step_rows_counts_positive_rewards_as_deliveries(self):
        rows = [
            {"episode_id": 0, "timestep": 0, "reward": 0.0},
            {"episode_id": 0, "timestep": 10, "reward": 20.0},
            {"episode_id": 0, "timestep": 50, "reward": 20.0},
            {"episode_id": 1, "timestep": 0, "reward": 0.0},
        ]
        summary = summarize_step_rows(rows, horizon=250)
        self.assertEqual(summary["num_rollouts"], 2.0)
        self.assertEqual(summary["mean_soups"], 1.0)
        self.assertEqual(summary["zero_soup_rate"], 0.5)
        self.assertEqual(summary["first_soup_t_mean"], 10.0)
        self.assertEqual(summary["last_soup_t_mean"], 50.0)


if __name__ == "__main__":
    unittest.main()

