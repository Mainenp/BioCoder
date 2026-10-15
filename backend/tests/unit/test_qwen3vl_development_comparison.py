from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.development_comparison import (
    build_cross_family_development_comparison,
)
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA


DATASET_HASH = "d" * 64
MODEL_HASH = "m" * 64
INSTRUCTION_HASH = "i" * 64


def classification(value: float) -> dict[str, float]:
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
    }


def language_metrics(value: float) -> dict[str, Any]:
    return {
        "peak_presence": {"classification": classification(value)},
        "peak_presence_metadata": {"classification": classification(value - 0.01)},
        "peak_grounding": {
            "grounding": {
                "mean_bbox_iou_all": value - 0.2,
                "iou_at_0_5_rate_all": value - 0.3,
                "x_boundary_mae_pixels_schema_valid": 20.0,
            }
        },
        "scientific_qc": {"exact_match_rate": value - 0.05},
    }


class QwenDevelopmentComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def generation(self, *, adapter: bool) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_REPORT_SCHEMA,
            "source": {
                "instruction_report_sha256": INSTRUCTION_HASH,
                "validation_prompts_sha256": "p" * 64,
                "source_dataset_report_sha256": DATASET_HASH,
            },
            "model": {
                "artifact_sha256": MODEL_HASH,
                "identity_immutable": True,
                "adapter": (
                    {"development_training_complete": True, "training_records": 100}
                    if adapter
                    else None
                ),
            },
            "counts": {"predictions": 16, "unique_images_opened": 4},
            "development_comparison_candidate": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }

    def evaluation(self, generation_hash: str, value: float) -> dict[str, Any]:
        return {
            "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
            "inputs": {
                "generation_report_sha256": generation_hash,
                "instruction_report_sha256": INSTRUCTION_HASH,
                "source_dataset_report_sha256": DATASET_HASH,
                "validation_prompts_sha256": "p" * 64,
                "validation_answers_sha256": "a" * 64,
                "instruction_manifest_sha256": "f" * 64,
            },
            "counts": {
                "predictions": 16,
                "independent_validation_assets": 4,
                "validation_source_groups": 2,
                "by_language": {"en": 8, "zh-CN": 8},
            },
            "metrics_by_language": {
                "en": language_metrics(value),
                "zh-CN": language_metrics(value - 0.02),
            },
            "cross_language_consistency": {"peak_presence": {"pairs": 4}},
            "language_variants_are_not_independent_source_assets": True,
            "prediction_generation_provenance_verified": True,
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }

    def fixture(self) -> dict[str, Path]:
        specialist = self.write(
            "specialist.json",
            {
                "schema_version": "chrompeak-development-ablation-v1",
                "dataset": {
                    "dataset_report_sha256": DATASET_HASH,
                    "asset_index_sha256": "x" * 64,
                    "validation_assets": 4,
                    "validation_source_groups": 2,
                },
                "fixed_threshold_0_5": {"sequence": classification(0.9)},
                "localization": {"sequence": {"mean_iou": 0.8}},
                "detector_only_coco": {"ap_50": 0.8},
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
        )
        zero_generation = self.write("zero_generation.json", self.generation(adapter=False))
        lora_generation = self.write("lora_generation.json", self.generation(adapter=True))
        zero_evaluation = self.write(
            "zero_evaluation.json",
            self.evaluation(sha256_file(zero_generation), 0.6),
        )
        lora_evaluation = self.write(
            "lora_evaluation.json",
            self.evaluation(sha256_file(lora_generation), 0.8),
        )
        return {
            "specialist_comparison_path": specialist,
            "zero_shot_generation_path": zero_generation,
            "zero_shot_evaluation_path": zero_evaluation,
            "lora_generation_path": lora_generation,
            "lora_evaluation_path": lora_evaluation,
        }

    def test_builds_hash_bound_per_language_comparison(self) -> None:
        result = build_cross_family_development_comparison(
            **self.fixture(), output_dir=self.root / "comparison"
        )

        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["dataset"]["bilingual_prompts_are_paired_views"])
        self.assertAlmostEqual(
            report["lora_minus_zero_shot"]["en"]["classification"]
            ["peak_presence"]["macro_f1"],
            0.2,
        )
        self.assertIn("never counted as independent", result.markdown_path.read_text())
        self.assertTrue(result.manifest_path.is_file())

    def test_rejects_lora_without_adapter(self) -> None:
        inputs = self.fixture()
        path = inputs["lora_generation_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model"]["adapter"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")
        evaluation = inputs["lora_evaluation_path"]
        evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
        evaluation_payload["inputs"]["generation_report_sha256"] = sha256_file(path)
        evaluation.write_text(json.dumps(evaluation_payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "adapter identity"):
            build_cross_family_development_comparison(
                **inputs, output_dir=self.root / "comparison"
            )

    def test_rejects_unpaired_bilingual_counts(self) -> None:
        inputs = self.fixture()
        path = inputs["zero_shot_evaluation_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["counts"]["by_language"]["zh-CN"] = 7
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "coverage is not paired"):
            build_cross_family_development_comparison(
                **inputs, output_dir=self.root / "comparison"
            )

    def test_slurm_runner_keeps_comparison_scoped_and_scheduled(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_compare_development.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("compare_development_cli", script)
        self.assertIn("bilingual_prompts_are_paired_views", script)
        self.assertNotIn("internal-test", script)


if __name__ == "__main__":
    unittest.main()
