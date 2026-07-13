import unittest

from scripts.score_official import calculate_attempt_score, calculate_scenario_score


class ScoreOfficialTests(unittest.TestCase):
    def test_zero_soups_scores_zero(self):
        score = calculate_attempt_score(0, 250, None, None, 0)
        self.assertEqual(score.attempt_score, 0)

    def test_one_soup_without_timeouts(self):
        score = calculate_attempt_score(1, 250, 100, 100, 0)
        self.assertEqual(score.attempt_score, 10000 + 10 * 150 + 150)

    def test_multiple_soups(self):
        score = calculate_attempt_score(3, 250, 50, 210, 0)
        self.assertEqual(score.attempt_score, 30000 + 10 * 40 + 200)

    def test_timeout_penalty_below_cap(self):
        score = calculate_attempt_score(1, 250, 20, 120, 7)
        self.assertEqual(score.penalty, 700)
        self.assertEqual(score.attempt_score, 10000 + 10 * 130 + 230 - 700)

    def test_timeout_penalty_caps_at_5000(self):
        score = calculate_attempt_score(1, 250, 20, 120, 999)
        self.assertEqual(score.penalty, 5000)
        self.assertEqual(score.attempt_score, 10000 + 10 * 130 + 230 - 5000)

    def test_first_and_last_same_timestep(self):
        score = calculate_attempt_score(1, 250, 249, 249, 0)
        self.assertEqual(score.attempt_score, 10000 + 10 * 1 + 1)

    def test_three_seed_average(self):
        attempts = [
            calculate_attempt_score(1, 250, 100, 100, 0),
            calculate_attempt_score(0, 250, None, None, 0),
            calculate_attempt_score(2, 250, 80, 200, 0),
        ]
        scenario = calculate_scenario_score(attempts)
        expected = sum(a.attempt_score for a in attempts) / 3.0
        self.assertEqual(scenario.scenario_score_mean, expected)
        self.assertEqual(scenario.zero_soup_attempts, 1)

    def test_invalid_missing_timesteps_for_positive_soups(self):
        with self.assertRaises(ValueError):
            calculate_attempt_score(1, 250, None, 100, 0)


if __name__ == "__main__":
    unittest.main()
