from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from multimodal_science.baselines.comparison import build_development_comparison
from multimodal_science.data.manifest import sha256_file


def classification_metrics(value: float, threshold: float) -> dict[str, float]:
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "precision": value,
        "recall": value,
        "specificity": value,
        "macro_f1": value,
        "mcc": value,
        "auroc": value,
        "auprc": value,
        "false_positive_rate": 1.0 - value,
        "threshold": threshold,
    }


class DevelopmentAblationComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.asset_index_sha256 = "a" * 64
        self.dataset_report_sha256 = "b" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def sequence_report(self, modality: str, value: float) -> dict[str, Any]:
        return {
            "schema_version": "chrompeak-sequence-baseline-report-v1",
            "evaluation_scope": "train_validation_baseline",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "dataset": {
                "dataset_report_sha256": self.dataset_report_sha256,
                "asset_index_sha256": self.asset_index_sha256,
                "used_train_samples": 10,
                "used_validation_samples": 4,
                "validation_source_groups": 1,
            },
            "config": {"modality": modality, "smoke_test": False},
            "threshold_selection": {
                "threshold": 0.4,
                "objective": "macro_f1",
                "value": value,
            },
            "validation": {
                "fixed_threshold_0_5": classification_metrics(value, 0.5),
                "selected_threshold": classification_metrics(value + 0.01, 0.4),
                "positive_boundary_metrics": {
                    "mean_interval_iou": value,
                    "boundary_mae_seconds": 2.0,
                    "valid_prediction_rate": 1.0,
                },
            },
        }

    def verification(self, report_path: Path) -> dict[str, Any]:
        return {
            "schema_version": "chrompeak-sequence-run-verification-v1",
            "quality_gate_passed": True,
            "sealed_internal_test_confirmed": True,
            "source_report": {"sha256": sha256_file(report_path)},
            "source_dataset": {
                "report_sha256": self.dataset_report_sha256,
                "asset_index_sha256": self.asset_index_sha256,
            },
        }

    def fixture(self) -> dict[str, Path]:
        detector_dataset = self.write_json(
            "detector_dataset.json",
            {
                "schema_version": "chrompeak-detector-dataset-v1",
                "asset_index_sha256": self.asset_index_sha256,
                "quality_gate_passed": True,
                "leakage_audit": {"passed": True, "overlap": []},
                "splits": {
                    "train": {"assets": 10, "source_groups": 2},
                    "validation": {"assets": 4, "source_groups": 1},
                },
            },
        )
        detector = self.write_json(
            "detector_evaluation.json",
            {
                "schema_version": "chrompeak-detector-evaluation-v1",
                "evaluation_split": "validation",
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
                "classification": {
                    "fixed_threshold": classification_metrics(0.8, 0.5),
                    "validation_selected_threshold": {
                        "threshold": 0.4,
                        "objective": "macro_f1",
                        "value": 0.81,
                        "metrics": classification_metrics(0.81, 0.4),
                    },
                },
                "localization": {"fixed_threshold": {"mean_best_iou": 0.7}},
                "coco": {"ap_50_95": 0.5, "ap_50": 0.8, "ap_75": 0.6},
                "provenance": {
                    "detector_dataset_report_sha256": sha256_file(detector_dataset)
                },
            },
        )
        sequence = self.write_json("sequence.json", self.sequence_report("sequence", 0.9))
        sequence_metadata = self.write_json(
            "sequence_metadata.json",
            self.sequence_report("sequence_metadata", 0.92),
        )
        return {
            "detector_evaluation_path": detector,
            "detector_dataset_report_path": detector_dataset,
            "sequence_report_path": sequence,
            "sequence_verification_path": self.write_json(
                "sequence_verification.json", self.verification(sequence)
            ),
            "sequence_metadata_report_path": sequence_metadata,
            "sequence_metadata_verification_path": self.write_json(
                "sequence_metadata_verification.json",
                self.verification(sequence_metadata),
            ),
        }

    def test_builds_hash_bound_fixed_and_selected_threshold_comparison(self) -> None:
        inputs = self.fixture()
        output = self.root / "comparison"

        result = build_development_comparison(**inputs, output_dir=output)

        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["development_comparison_eligible"])
        self.assertFalse(report["final_benchmark_eligible"])
        self.assertEqual(report["dataset"]["asset_index_sha256"], self.asset_index_sha256)
        self.assertAlmostEqual(
            report["sequence_metadata_minus_sequence"]["fixed_threshold_0_5"]["macro_f1"],
            0.02,
        )
        self.assertEqual(report["detector_only_coco"]["ap_50"], 0.8)
        self.assertIn("ChromPeakFormer image detector", result.markdown_path.read_text())
        self.assertTrue(result.manifest_path.is_file())

    def test_rejects_cross_modality_dataset_mismatch(self) -> None:
        inputs = self.fixture()
        metadata_path = inputs["sequence_metadata_report_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["dataset"]["asset_index_sha256"] = "c" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "do not share one Dataset"):
            build_development_comparison(**inputs, output_dir=self.root / "comparison")

    def test_rejects_unverified_sequence_report(self) -> None:
        inputs = self.fixture()
        verification_path = inputs["sequence_verification_path"]
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification["source_report"]["sha256"] = "f" * 64
        verification_path.write_text(json.dumps(verification), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "does not bind"):
            build_development_comparison(**inputs, output_dir=self.root / "comparison")


if __name__ == "__main__":
    unittest.main()
