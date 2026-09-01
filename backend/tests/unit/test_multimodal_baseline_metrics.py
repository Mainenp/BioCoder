from __future__ import annotations

import math
import unittest

import numpy as np

from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
    select_threshold,
)


class MultimodalBaselineMetricTests(unittest.TestCase):
    def test_binary_metrics_match_hand_calculated_example(self) -> None:
        result = binary_metrics(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.35, 0.8],
            threshold=0.5,
        )

        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 0, 2, 1))
        self.assertAlmostEqual(result["macro_f1"], (2.0 / 3.0 + 4.0 / 5.0) / 2.0)
        self.assertAlmostEqual(result["mcc"], 2.0 / math.sqrt(12.0))
        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["auprc"], 5.0 / 6.0)

    def test_threshold_selection_is_deterministic_on_ties(self) -> None:
        result = select_threshold(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.6, 0.7],
            objective="macro_f1",
        )

        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["value"], 1.0)

    def test_boundary_metrics_report_normalized_and_physical_errors(self) -> None:
        result = boundary_metrics(
            [[0.2, 0.4], [0.5, 0.8], [-1.0, -1.0]],
            [[0.1, 0.5], [0.6, 0.9], [0.0, 0.0]],
            [True, True, False],
            roi_width_minutes=[1.0, 2.0, 1.0],
        )

        self.assertEqual(result["positive_samples"], 2)
        self.assertEqual(result["valid_prediction_rate"], 1.0)
        self.assertAlmostEqual(result["start_mae_normalized"], 0.1)
        self.assertAlmostEqual(result["end_mae_normalized"], 0.1)
        self.assertAlmostEqual(result["width_mae_normalized"], 0.1)
        self.assertAlmostEqual(result["mean_interval_iou"], 0.5)
        self.assertAlmostEqual(result["boundary_mae_seconds"], 9.0)

    def test_grouped_bootstrap_is_seeded_and_resamples_sources(self) -> None:
        labels = np.asarray([0, 1] * 4)
        scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

        first = grouped_bootstrap_binary(
            labels,
            scores,
            groups,
            threshold=0.5,
            iterations=50,
            seed=7,
        )
        second = grouped_bootstrap_binary(
            labels,
            scores,
            groups,
            threshold=0.5,
            iterations=50,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["group_count"], 4)
        self.assertEqual(first["intervals"]["macro_f1"]["median"], 1.0)

    def test_scores_outside_probability_range_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "probabilities"):
            binary_metrics([0, 1], [-0.1, 1.1])

    def test_reversed_boundary_is_flagged_and_not_silently_reordered(self) -> None:
        result = boundary_metrics([[0.2, 0.4]], [[0.5, 0.1]], [True])

        self.assertEqual(result["valid_prediction_rate"], 0.0)
        self.assertEqual(result["mean_interval_iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
