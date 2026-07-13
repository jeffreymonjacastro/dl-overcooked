from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.option_a_gru_policy import OptionAGRU
from training.datasets import Episode, EpisodeSequenceDataset, SequenceDataset

import numpy as np


class SequenceStateHandlingTest(unittest.TestCase):
    def test_full_episode_dataset_does_not_create_mid_episode_windows(self):
        ep = Episode(
            path=Path("synthetic.npz"),
            split="train",
            source_group="test",
            layout="test",
            partner="test",
            obs=np.zeros((10, 3), dtype=np.float32),
            actions=np.arange(10, dtype=np.int64) % 6,
            agent_indices=np.zeros(10, dtype=np.int64),
            quality_weight=1.0,
        )
        norm_path = Path("reports/diagnostics/test_norm.json")
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        norm_path.write_text('{"mean":[0,0,0],"std":[1,1,1],"obs_dim":3}', encoding="utf-8")

        windowed = SequenceDataset([ep], norm_path, seq_len=4, stride=2)
        full = EpisodeSequenceDataset([ep], norm_path, max_seq_len=10)

        self.assertGreater(len(windowed), 1)
        self.assertEqual(len(full), 1)
        self.assertEqual(float(full[0][2].sum().item()), 10.0)

    def test_segmented_hidden_matches_continuous_when_hidden_is_carried(self):
        torch.manual_seed(67)
        model = OptionAGRU(input_dim=12, hidden_size=8, dropout=0.0)
        model.eval()
        x = torch.randn(1, 10, 12)

        with torch.no_grad():
            logits_full, hidden_full = model(x)
            logits_a, hidden_a = model(x[:, :4, :])
            logits_b, hidden_b = model(x[:, 4:, :], hidden_a)

        logits_joined = torch.cat([logits_a, logits_b], dim=1)
        self.assertTrue(torch.allclose(logits_full, logits_joined, atol=1e-6))
        self.assertTrue(torch.allclose(hidden_full, hidden_b, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

