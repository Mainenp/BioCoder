from __future__ import annotations

import unittest

import numpy as np

from multimodal_science.baselines.dataset import SequenceSplit
from multimodal_science.baselines.sequence_model import SequenceModelSpec
from multimodal_science.baselines.sequence_training import (
    SequenceTrainConfig,
    _prediction_records,
    _subset_split,
)
from multimodal_science.baselines.train_sequence_cli import parser


def split_fixture() -> SequenceSplit:
    return SequenceSplit(
        split="train",
        signals=np.arange(80, dtype=np.float32).reshape(10, 8),
        scalar_features=np.zeros((10, 7), dtype=np.float32),
        targets=np.zeros((10, 3), dtype=np.float32),
        asset_ids=tuple(f"asset-{index}" for index in range(10)),
        group_ids=tuple(f"source-{index // 2}" for index in range(10)),
        roi_width_minutes=np.ones(10),
        dataset_report_sha256="a" * 64,
        asset_index_sha256="b" * 64,
    )


class SequenceBaselineContractTests(unittest.TestCase):
    def test_default_training_never_claims_or_selects_cuda(self) -> None:
        config = SequenceTrainConfig()

        config.validate()
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.smoke_test)
        self.assertIsNone(config.max_train_samples)

    def test_sample_caps_require_explicit_smoke_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "smoke"):
            SequenceTrainConfig(max_train_samples=128).validate()

        SequenceTrainConfig(
            smoke_test=True,
            max_train_samples=128,
            max_validation_samples=64,
        ).validate()

    def test_even_subsample_preserves_identity_alignment_and_coverage(self) -> None:
        original = split_fixture()
        subset = _subset_split(original, 4)

        self.assertEqual(subset.signals.shape, (4, 8))
        self.assertEqual(subset.asset_ids, ("asset-0", "asset-3", "asset-6", "asset-9"))
        self.assertEqual(subset.group_ids, ("source-0", "source-1", "source-3", "source-4"))
        self.assertEqual(subset.signals[1, 0], original.signals[3, 0])

    def test_model_modalities_share_one_validated_contract(self) -> None:
        sequence = SequenceModelSpec(modality="sequence")
        metadata = SequenceModelSpec(modality="sequence_metadata")

        self.assertEqual(sequence.as_dict()["input_points"], 160)
        self.assertEqual(sequence.as_dict()["position_bins"], 10)
        self.assertEqual(metadata.as_dict()["scalar_features"], 7)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            SequenceModelSpec(modality="image").validate()

    def test_cli_defaults_to_cpu_and_exposes_no_internal_test_surface(self) -> None:
        command = parser()
        arguments = command.parse_args(
            ["--dataset-root", "dataset", "--output-dir", "run"]
        )
        destinations = {action.dest for action in command._actions}

        self.assertEqual(arguments.device, "cpu")
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)

    def test_prediction_evidence_keeps_identity_and_human_targets(self) -> None:
        split = split_fixture()
        split.targets[0] = [1.0, 0.2, 0.4]
        records = _prediction_records(
            split,
            np.linspace(0.1, 0.9, 10),
            np.tile([0.25, 0.5], (10, 1)),
        )

        self.assertEqual(records[0]["asset_id"], "asset-0")
        self.assertEqual(records[0]["group_id"], "source-0")
        self.assertTrue(records[0]["target_peak_present"])
        self.assertAlmostEqual(records[0]["target_start_normalized"], 0.2)
        self.assertIsNone(records[1]["target_start_normalized"])


if __name__ == "__main__":
    unittest.main()
