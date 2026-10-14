from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.qwen3vl.v2_rl_evidence_summary import (
    summarize_v2_rl_evidence,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, *names: str) -> None:
    (root / "artifact_manifest.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _analysis() -> dict:
    constraints = {
        model: {
            scope: {
                "presence_macro_f1": 0.90 if model == "c0" else 0.91,
                "presence_mcc": 0.8,
                "presence_false_positive_rate": 0.1,
                "presence_recall": 0.9,
                "metadata_presence_macro_f1": 0.9,
                "scientific_qc_exact_match": 0.9,
                "json_valid_rate": 1.0,
            }
            for scope in ("overall", "en", "zh-CN")
        }
        for model in ("c0", "c1", "rl")
    }
    consistency = {
        model: {
            "peak_grounding": {
                "pairs": 2,
                "both_schema_valid_rate": 1.0,
                "exact_prediction_consistency_rate_all": 0.5,
                "mean_prediction_bbox_iou_all": 0.8 if model == "c0" else 0.9,
                "prediction_bbox_iou_at_0_9_rate_all": 0.5,
            }
        }
        for model in ("c0", "c1", "rl")
    }
    localization = {
        model: {
            scope: {
                "positive_grounding_rows": 2,
                "mean_iou": 0.7,
                "iou_at_0_5": 0.8,
                "iou_at_0_75": 0.6,
                "start_error_pixels": 1.0,
                "end_error_pixels": 2.0,
                "start_error_seconds": None,
                "end_error_seconds": None,
                "boundary_error_rows": 2,
                "seconds_error_rows": 0,
                "schema_invalid_or_unusable_rows": 0,
            }
            for scope in ("overall", "en", "zh-CN")
        }
        for model in ("c0", "c1", "rl")
    }
    strata = {
        model: {
            scope: {
                variable: {
                    level: localization[model][scope]
                    for level in ("low", "middle", "high")
                }
                for variable in (
                    "true_peak_width_pixels",
                    "log1p_raw_max",
                    "min_edge_distance_pixels",
                )
            }
            for scope in ("overall", "en", "zh-CN")
        }
        for model in ("c0", "c1", "rl")
    }

    def metric(delta: float, low: float) -> dict:
        return {
            "rl_minus_reference": delta,
            "rl_minus_reference_ci_95": [low, delta + 0.01],
            "higher_is_better": True,
            "probability_rl_better": 1.0,
        }

    comparisons = {}
    for name in ("rl_minus_c0", "rl_minus_c1"):
        comparisons[name] = {}
        for scope in ("overall", "en", "zh-CN"):
            comparisons[name][scope] = {
                "source_groups": 2,
                "resampling": "ratio_of_sums",
                "mean_iou": metric(0.03, 0.02),
                "iou_at_0_5": metric(0.02, 0.01),
                "iou_at_0_75": metric(0.04, 0.03),
            }
    return {
        "schema_version": "chrompeak-qwen3vl-v2-localization-analysis-v1",
        "evaluation_scope": "development_validation",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "localization": localization,
        "constraints": constraints,
        "cross_language_consistency": consistency,
        "stratification": {
            "boundaries": {
                variable: {
                    "lower_boundary": 1.0,
                    "upper_boundary": 2.0,
                    "population_rows": 2,
                    "tie_policy": "low_le_lower; middle_gt_lower_and_le_upper; high_gt_upper",
                }
                for variable in (
                    "true_peak_width_pixels",
                    "log1p_raw_max",
                    "min_edge_distance_pixels",
                )
            },
            "metrics": strata,
        },
        "paired_group_bootstrap": {
            "iterations": 10_000,
            "seed": 17,
            "confidence_level": 0.95,
            "resampling_unit": "source_group",
            "estimator": "ratio_of_sums",
            "comparisons": comparisons,
        },
        "failure_cases": {
            "comparison": "rl_minus_c1",
            "sampling": "deterministic_sha256_order",
            "seed": 17,
            "maximum_records": 20,
            "records": 1,
            "artifact": "localization_failure_cases.jsonl",
        },
        "contracts": {
            "positive_grounding_only": True,
            "shared_record_identity": True,
            "shared_fixed_empirical_tertiles": True,
            "paired_source_group_bootstrap": True,
            "internal_test_accessed": False,
            "validation_metadata_hash_bound": True,
        },
    }


class V2RLEvidenceSummaryTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        analysis_root = root / "analysis"
        run_root = root / "runs"
        slurm_root = root / "slurm"
        analysis_root.mkdir()
        slurm_root.mkdir()
        _write_json(analysis_root / "localization_analysis.json", _analysis())
        (analysis_root / "localization_analysis.md").write_text("# result\n", encoding="utf-8")
        (analysis_root / "localization_failure_cases.jsonl").write_text(
            json.dumps({"instruction_id": "failure-1"}) + "\n", encoding="utf-8"
        )
        _manifest(
            analysis_root,
            "localization_analysis.json",
            "localization_analysis.md",
            "localization_failure_cases.jsonl",
        )

        training = run_root / "qwen3vl" / "rl" / "runs"
        for name in ("R1", "R2", "R3", "c1-original", "c1-1e-6"):
            report_name = (
                "sft_continue_training_report.json"
                if name.startswith("c1-")
                else "fusion_training_report.json"
            )
            _write_json(
                training / f"v2-{name}-formal-seed17-revision" / report_name,
                {
                    "schema_version": "training-v1",
                    "runtime": {
                        "global_steps": 300,
                        "optimizer_updates": 300,
                        "wall_time_seconds": 100.0,
                        "seconds_per_update": 0.333,
                        "peak_cuda_gib": 9.0,
                    },
                    "final_diagnostics": {
                        "final_loss": 0.1,
                        "final_kl_mean": 0.01,
                        "final_grounding_mean_iou": 0.7,
                        "final_grounding_mean_giou": 0.6,
                        "final_zero_variance_group_fraction": 0.1,
                    },
                    "stopped_early": False,
                    "stop_reason": None,
                },
            )

        candidates = [
            "C0:xic-only-sft-seed17:step-00000000",
            *[f"R1:config:step-{step:08d}" for step in (100, 200, 300)],
            *[f"R2:config:step-{step:08d}" for step in (100, 200, 300)],
            *[f"R3:config:step-{step:08d}" for step in (100, 200, 300)],
            *[f"C1:original:step-{step:08d}" for step in (100, 200, 300)],
            *[f"C1:lora-lr-1e-6:step-{step:08d}" for step in (100, 200, 300)],
        ]
        for index, candidate in enumerate(candidates):
            evaluation_root = root / "evaluations" / str(index)
            _write_json(
                evaluation_root / "qwen_evaluation_report.json",
                {
                    "schema_version": "evaluation-v2",
                    "counts": {"predictions": 100, "valid_json": 100},
                    "development_comparison_eligible": True,
                    "final_benchmark_eligible": False,
                    "internal_test_accessed": False,
                    "metrics": {
                        "peak_presence": {
                            "classification": {"macro_f1": 0.9, "mcc": 0.8}
                        },
                        "peak_grounding": {
                            "grounding": {
                                "mean_bbox_iou_all": 0.7,
                                "iou_at_0_5_rate_all": 0.8,
                            }
                        },
                    },
                },
            )
            _manifest(evaluation_root, "qwen_evaluation_report.json")
            (slurm_root / f"job-{index}.out").write_text(
                f"CANDIDATE_ID={candidate}\nPERSISTENT_EVALUATION={evaluation_root}\n",
                encoding="utf-8",
            )
        return analysis_root, run_root, slurm_root

    def test_summarizes_sealed_evidence_and_preregistered_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis_root, run_root, slurm_root = self._fixture(Path(temporary))
            result = summarize_v2_rl_evidence(
                analysis_root=analysis_root,
                run_root=run_root,
                slurm_root=slurm_root,
            )
            self.assertTrue(result["hypotheses"]["h1"]["passed"])
            self.assertTrue(result["hypotheses"]["h2"]["passed"])
            self.assertTrue(result["hypotheses"]["h3"]["passed"])
            self.assertTrue(result["hypotheses"]["h4"]["passed"])
            self.assertEqual(len(result["formal_training"]), 5)
            self.assertEqual(len(result["validation_candidates"]), 16)
            self.assertFalse(result["internal_test_accessed"])

    def test_rejects_tampered_localization_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis_root, run_root, slurm_root = self._fixture(Path(temporary))
            (analysis_root / "localization_analysis.md").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Manifest artifact drift"):
                summarize_v2_rl_evidence(
                    analysis_root=analysis_root,
                    run_root=run_root,
                    slurm_root=slurm_root,
                )

    def test_rejects_missing_validation_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            analysis_root, run_root, slurm_root = self._fixture(Path(temporary))
            (slurm_root / "job-15.out").unlink()
            with self.assertRaisesRegex(ValueError, "Expected 16 validation candidates"):
                summarize_v2_rl_evidence(
                    analysis_root=analysis_root,
                    run_root=run_root,
                    slurm_root=slurm_root,
                )


if __name__ == "__main__":
    unittest.main()
