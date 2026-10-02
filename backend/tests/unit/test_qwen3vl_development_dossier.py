from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.qwen3vl.development_dossier import (
    REPORT_SCHEMA,
    build_development_dossier,
)


DATASET_SHA256 = "1" * 64
INSTRUCTION_SHA256 = "2" * 64
PROMPTS_SHA256 = "3" * 64
ANSWERS_SHA256 = "4" * 64
INSTRUCTION_MANIFEST_SHA256 = "5" * 64
MODEL_SHA256 = "6" * 64
FUSION_BUNDLE_SHA256 = "7" * 64
TRAINING_REPORT_SHA256 = "a" * 64
TRAINING_MANIFEST_SHA256 = "b" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, names: list[str]) -> str:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _identity_sha256(rows: list[dict]) -> str:
    identities = sorted(
        (
            row.get("instruction_id"),
            row.get("pair_id"),
            row.get("asset_id"),
            row.get("group_id"),
            row.get("task"),
            row.get("language"),
            row.get("target_peak_present"),
            tuple(row.get("expected_bbox_2d") or ()),
        )
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stat(value: float) -> dict:
    return {
        "values_by_seed": {"17": value, "29": value + 0.01, "43": value - 0.01},
        "n": 3,
        "mean": value,
        "sample_standard_deviation": 0.01,
        "standard_deviation_ddof": 1,
        "minimum": value - 0.01,
        "maximum": value + 0.01,
    }


def _qwen_language(value: float) -> dict:
    classification = {
        "accuracy": value,
        "balanced_accuracy": value,
        "precision": value,
        "recall": value,
        "specificity": value,
        "macro_f1": value,
        "mcc": value - 0.1,
        "auroc": value,
        "auprc": value,
        "false_positive_rate": 1.0 - value,
    }
    return {
        "classification": {
            "peak_presence": classification,
            "peak_presence_metadata": classification,
        },
        "grounding": {
            "mean_bbox_iou_all": value - 0.2,
            "iou_at_0_5_rate_all": value - 0.1,
            "x_boundary_mae_pixels_schema_valid": 5.0,
        },
        "scientific_qc_exact_match": value,
    }


def _evaluation_source(en_value: float, zh_value: float) -> dict:
    return {
        "schema_version": "chrompeak-qwen3vl-evaluation-v2",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "prediction_generation_provenance_verified": True,
        "inputs": {
            "source_dataset_report_sha256": DATASET_SHA256,
            "instruction_report_sha256": INSTRUCTION_SHA256,
            "validation_prompts_sha256": PROMPTS_SHA256,
            "validation_answers_sha256": ANSWERS_SHA256,
            "instruction_manifest_sha256": INSTRUCTION_MANIFEST_SHA256,
        },
        "metrics_by_language": {
            language: {
                "peak_presence": {
                    "classification": language_row["classification"][
                        "peak_presence"
                    ]
                },
                "peak_presence_metadata": {
                    "classification": language_row["classification"][
                        "peak_presence_metadata"
                    ]
                },
                "peak_grounding": {"grounding": language_row["grounding"]},
                "scientific_qc": {
                    "exact_match_rate": language_row["scientific_qc_exact_match"]
                },
            }
            for language, language_row in {
                "en": _qwen_language(en_value),
                "zh-CN": _qwen_language(zh_value),
            }.items()
        },
    }
def _specialist_metric(value: float) -> dict:
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "precision": value,
        "recall": value,
        "specificity": value,
        "negative_predictive_value": value,
        "positive_f1": value,
        "negative_f1": value,
        "macro_f1": value,
        "mcc": value - 0.1,
        "auroc": value,
        "auprc": value,
        "false_positive_rate": 1.0 - value,
    }


def _record(
    *,
    pair_id: str,
    task: str,
    language: str,
    exact: bool,
    predicted: bool | None = None,
    bbox: list[float] | None = None,
    bbox_iou: float | None = None,
    qc_state: str | None = None,
) -> dict:
    row = {
        "schema_version": "chrompeak-qwen3vl-evaluation-record-v2",
        "instruction_id": f"{pair_id}-{language}",
        "pair_id": pair_id,
        "task": task,
        "language": language,
        "asset_id": "asset-1",
        "group_id": "group-1",
        "valid_json": True,
        "schema_valid": True,
        "exact_match": exact,
    }
    if task in {"peak_presence", "peak_presence_metadata"}:
        row.update(
            {
                "target_peak_present": True,
                "predicted_peak_present": predicted,
                "classification_score": float(bool(predicted)),
            }
        )
    elif task == "peak_grounding":
        row.update(
            {
                "expected_bbox_2d": [0.0, 0.0, 10.0, 10.0],
                "predicted_bbox_2d": bbox,
                "bbox_iou": bbox_iou,
                "iou_at_0_5": bool(bbox_iou is not None and bbox_iou >= 0.5),
            }
        )
    else:
        row.update(
            {
                "predicted_qc_state": qc_state,
                "predicted_reason": "reason",
                "qc_state_correct": exact,
                "reason_correct": True,
            }
        )
    return row


class DevelopmentDossierTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        evaluation_root = root / "evaluation"
        evaluation_root.mkdir()
        records = [
            _record(
                pair_id="presence",
                task="peak_presence",
                language="en",
                exact=True,
                predicted=True,
            ),
            _record(
                pair_id="presence",
                task="peak_presence",
                language="zh-CN",
                exact=False,
                predicted=False,
            ),
            _record(
                pair_id="metadata",
                task="peak_presence_metadata",
                language="en",
                exact=True,
                predicted=True,
            ),
            _record(
                pair_id="metadata",
                task="peak_presence_metadata",
                language="zh-CN",
                exact=True,
                predicted=True,
            ),
            _record(
                pair_id="grounding",
                task="peak_grounding",
                language="en",
                exact=True,
                bbox=[0.0, 0.0, 10.0, 10.0],
                bbox_iou=1.0,
            ),
            _record(
                pair_id="grounding",
                task="peak_grounding",
                language="zh-CN",
                exact=False,
                bbox=[5.0, 0.0, 10.0, 10.0],
                bbox_iou=0.5,
            ),
            _record(
                pair_id="qc",
                task="scientific_qc",
                language="en",
                exact=True,
                qc_state="pass",
            ),
            _record(
                pair_id="qc",
                task="scientific_qc",
                language="zh-CN",
                exact=False,
                qc_state="review",
            ),
        ]
        records_path = evaluation_root / "evaluation_records.jsonl"
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        evaluation = {
            "schema_version": "chrompeak-qwen3vl-evaluation-v2",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "prediction_generation_provenance_verified": True,
            "inputs": {
                "source_dataset_report_sha256": DATASET_SHA256,
                "instruction_report_sha256": INSTRUCTION_SHA256,
                "validation_prompts_sha256": PROMPTS_SHA256,
                "validation_answers_sha256": ANSWERS_SHA256,
                "instruction_manifest_sha256": INSTRUCTION_MANIFEST_SHA256,
            },
            "counts": {
                "predictions": len(records),
                "independent_validation_assets": 1,
                "validation_source_groups": 1,
            },
        }
        evaluation_path = evaluation_root / "qwen_evaluation_report.json"
        _write_json(evaluation_path, evaluation)
        evaluation_manifest = _write_manifest(
            evaluation_root,
            ["qwen_evaluation_report.json", "evaluation_records.jsonl"],
        )

        cross_root = root / "cross"
        cross_root.mkdir()
        specialist_context = {
            "fixed_threshold_0_5": {
                "chrompeakformer": _specialist_metric(0.8),
                "sequence": _specialist_metric(0.9),
                "sequence_metadata": _specialist_metric(0.92),
            },
            "localization": {
                "chrompeakformer": {"mean_iou": 0.78, "metric": "mean_best_iou"},
                "sequence": {"mean_iou": 0.79, "metric": "mean_interval_iou"},
                "sequence_metadata": {
                    "mean_iou": 0.80,
                    "metric": "mean_interval_iou",
                },
            },
            "detector_only_coco": {"ap_50_95": 0.52},
        }
        specialist = {
            "schema_version": "chrompeak-development-ablation-v1",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "dataset": {"dataset_report_sha256": DATASET_SHA256},
            **specialist_context,
        }
        source_payloads = {
            "specialist_comparison": specialist,
            "zero_shot_generation": {
                "model": {"artifact_sha256": MODEL_SHA256},
                "source": {
                    "source_dataset_report_sha256": DATASET_SHA256,
                    "instruction_report_sha256": INSTRUCTION_SHA256,
                    "validation_prompts_sha256": PROMPTS_SHA256,
                },
                "internal_test_accessed": False,
            },
            "zero_shot_evaluation": _evaluation_source(0.6, 0.5),
            "lora_generation": {
                "model": {"artifact_sha256": MODEL_SHA256},
                "source": {
                    "source_dataset_report_sha256": DATASET_SHA256,
                    "instruction_report_sha256": INSTRUCTION_SHA256,
                    "validation_prompts_sha256": PROMPTS_SHA256,
                },
                "internal_test_accessed": False,
            },
            "lora_evaluation": _evaluation_source(0.8, 0.7),
        }
        for prefix in ("zero_shot", "lora"):
            generation_path = root / f"{prefix}_generation.json"
            _write_json(generation_path, source_payloads[f"{prefix}_generation"])
            source_payloads[f"{prefix}_evaluation"]["inputs"][
                "generation_report_sha256"
            ] = hashlib.sha256(generation_path.read_bytes()).hexdigest()
        source_entries = {}
        for name, payload in source_payloads.items():
            source_path = root / f"{name}.json"
            _write_json(source_path, payload)
            source_entries[name] = {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        cross = {
            "schema_version": "chrompeak-cross-family-development-comparison-v1",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "dataset": {
                "dataset_report_sha256": DATASET_SHA256,
                "validation_assets": 1,
                "validation_source_groups": 1,
            },
            "sources": source_entries,
            "qwen": {
                "zero_shot": {
                    "en": _qwen_language(0.6),
                    "zh-CN": _qwen_language(0.5),
                },
                "lora": {
                    "en": _qwen_language(0.8),
                    "zh-CN": _qwen_language(0.7),
                },
            },
            "specialist_context": specialist_context,
        }
        cross_path = cross_root / "cross_family_development_report.json"
        _write_json(cross_path, cross)
        (cross_root / "cross_family_development_table.md").write_text("table\n")
        _write_manifest(
            cross_root,
            ["cross_family_development_report.json", "cross_family_development_table.md"],
        )

        intervention_root = root / "intervention"
        intervention_root.mkdir()
        intervention = {
            "schema_version": "chrompeak-qwen3vl-xic-intervention-analysis-v1",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "shared_provenance": {
                "source_dataset_report_sha256": DATASET_SHA256,
                "fusion_training_report_sha256": "c" * 64,
                "fusion_manifest_sha256": "d" * 64,
                "base_model_artifact_sha256": MODEL_SHA256,
                "instruction_report_sha256": INSTRUCTION_SHA256,
                "validation_prompts_sha256": PROMPTS_SHA256,
                "validation_answers_sha256": ANSWERS_SHA256,
            },
            "sources": {
                "aligned": {
                    "evaluation_root": str(evaluation_root),
                    "evaluation_report_sha256": hashlib.sha256(
                        evaluation_path.read_bytes()
                    ).hexdigest(),
                    "evaluation_records_sha256": hashlib.sha256(
                        records_path.read_bytes()
                    ).hexdigest(),
                }
            },
            "observed_metrics": {"overall": {}},
            "paired_group_bootstrap": {"iterations": 1000, "comparisons": {}},
        }
        intervention_path = intervention_root / "xic_intervention_analysis.json"
        _write_json(intervention_path, intervention)
        (intervention_root / "xic_intervention_analysis.md").write_text("analysis\n")
        _write_manifest(
            intervention_root,
            ["xic_intervention_analysis.json", "xic_intervention_analysis.md"],
        )

        matrix_root = root / "matrix"
        matrix_root.mkdir()
        aggregate = {}
        for scope in ("overall", "en", "zh-CN"):
            aggregate[scope] = {
                "peak_presence": {
                    metric: _stat(value)
                    for metric, value in (
                        ("balanced_accuracy", 0.9),
                        ("macro_f1", 0.91),
                        ("mcc", 0.82),
                        ("false_positive_rate", 0.1),
                    )
                },
                "peak_presence_metadata": {
                    metric: _stat(value)
                    for metric, value in (
                        ("balanced_accuracy", 0.91),
                        ("macro_f1", 0.92),
                        ("mcc", 0.83),
                        ("false_positive_rate", 0.09),
                    )
                },
                "peak_grounding": {
                    "mean_bbox_iou_all": _stat(0.6),
                    "iou_at_0_5_rate_all": _stat(0.72),
                },
                "scientific_qc": {"exact_match_rate": _stat(0.94)},
            }
        selected_hashes = {
            "training_report": TRAINING_REPORT_SHA256,
            "training_manifest": TRAINING_MANIFEST_SHA256,
            "generation_report": "c" * 64,
            "generation_manifest": "d" * 64,
            "evaluation_report": hashlib.sha256(
                evaluation_path.read_bytes()
            ).hexdigest(),
            "evaluation_manifest": evaluation_manifest,
            "evaluation_records": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        }

        def matrix_row(label: str, tokens: int, seed: int) -> dict:
            hashes = {
                name: hashlib.sha256(f"{label}-{name}".encode()).hexdigest()
                for name in (
                    "training_report",
                    "training_manifest",
                    "generation_report",
                    "generation_manifest",
                    "evaluation_report",
                    "evaluation_manifest",
                    "evaluation_records",
                )
            }
            if label == "primary-seed17":
                hashes = selected_hashes
            return {
                "sensor_tokens": tokens,
                "training_seed": seed,
                "sha256": hashes,
            }

        matrix = {
            "schema_version": "chrompeak-qwen3vl-xic-fusion-matrix-analysis-v1",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "shared_provenance": {
                "dataset_report_sha256": DATASET_SHA256,
                "fusion_bundle_report_sha256": FUSION_BUNDLE_SHA256,
                "base_model_artifact_sha256": MODEL_SHA256,
                "instruction_report_sha256": INSTRUCTION_SHA256,
                "validation_prompts_sha256": PROMPTS_SHA256,
                "validation_answers_sha256": ANSWERS_SHA256,
                "evaluation_record_identity_sha256": _identity_sha256(records),
            },
            "matrix": {
                "primary-seed17": matrix_row("primary-seed17", 4, 17),
                "primary-seed29": matrix_row("primary-seed29", 4, 29),
                "primary-seed43": matrix_row("primary-seed43", 4, 43),
                "tokens1-seed17": matrix_row("tokens1-seed17", 1, 17),
                "tokens8-seed17": matrix_row("tokens8-seed17", 8, 17),
            },
            "primary_seed_aggregate": aggregate,
            "primary_output_quality": {},
            "primary_en_minus_zh_cn": {},
            "primary_cross_language_consistency": {},
            "token_ablation": {
                "training_seed": 17,
                "baseline_sensor_tokens": 4,
                "runs": {
                    "1": {"sensor_tokens": 1, "training_seed": 17},
                    "4": {"sensor_tokens": 4, "training_seed": 17},
                    "8": {"sensor_tokens": 8, "training_seed": 17},
                },
            },
            "gate_audit": {},
            "contracts": {
                "same_leakage_safe_validation": True,
                "same_base_model": True,
                "same_training_data_and_initial_adapter": True,
                "three_independent_primary_training_seeds": True,
                "token_ablation_uses_seed17": True,
                "evaluation_metrics_recomputed_from_records": True,
                "evaluation_record_identities_match_across_runs": True,
            },
        }
        matrix_path = matrix_root / "fusion_matrix_analysis.json"
        _write_json(matrix_path, matrix)
        (matrix_root / "fusion_matrix_analysis.md").write_text("matrix\n")
        _write_manifest(
            matrix_root,
            ["fusion_matrix_analysis.json", "fusion_matrix_analysis.md"],
        )
        return cross_path, matrix_path, intervention_path, evaluation_root

    def test_builds_main_table_and_failure_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross, matrix, intervention, evaluation = self._fixture(root)
            output = root / "dossier"
            result = build_development_dossier(
                cross_family_report_path=cross,
                fusion_matrix_report_path=matrix,
                xic_intervention_report_path=intervention,
                selected_fusion_evaluation_root=evaluation,
                output_dir=output,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(len(report["main_table"]), 9)
            self.assertEqual(
                report["failure_summary"]["peak_presence"][
                    "exact_prediction_consistency_rate_all"
                ],
                0.0,
            )
            self.assertEqual(
                report["failure_summary"]["peak_grounding"]["localization"][
                    "either_language_iou_below_0_5_rate"
                ],
                0.0,
            )
            self.assertEqual(
                report["failure_summary"]["peak_grounding"][
                    "both_languages_correct_rate"
                ],
                1.0,
            )
            self.assertGreater(report["failure_cases"]["records"], 0)
            self.assertFalse(report["internal_test_accessed"])
            self.assertFalse(report["pre_internal_test_readiness"]["ready"])
            self.assertIn(
                "mean ± sample SD",
                result.main_table_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Highest-severity",
                result.failure_analysis_path.read_text(encoding="utf-8"),
            )

    def test_rejects_tampered_selected_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross, matrix, intervention, evaluation = self._fixture(root)
            with (evaluation / "evaluation_records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                build_development_dossier(
                    cross_family_report_path=cross,
                    fusion_matrix_report_path=matrix,
                    xic_intervention_report_path=intervention,
                    selected_fusion_evaluation_root=evaluation,
                    output_dir=root / "dossier",
                )

    def test_exact_checkpoint_intervention_unlocks_pre_test_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross, matrix_path, intervention_path, evaluation = self._fixture(root)
            checkpoint_sha256 = "a" * 64
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix["matrix"]["primary-seed17"]["sha256"][
                "training_report"
            ] = checkpoint_sha256
            _write_json(matrix_path, matrix)
            _write_manifest(
                matrix_path.parent,
                ["fusion_matrix_analysis.json", "fusion_matrix_analysis.md"],
            )
            intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
            intervention["shared_provenance"][
                "fusion_training_report_sha256"
            ] = checkpoint_sha256
            intervention["shared_provenance"][
                "fusion_manifest_sha256"
            ] = TRAINING_MANIFEST_SHA256
            _write_json(intervention_path, intervention)
            _write_manifest(
                intervention_path.parent,
                ["xic_intervention_analysis.json", "xic_intervention_analysis.md"],
            )
            result = build_development_dossier(
                cross_family_report_path=cross,
                fusion_matrix_report_path=matrix_path,
                xic_intervention_report_path=intervention_path,
                selected_fusion_evaluation_root=evaluation,
                output_dir=root / "dossier",
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["pre_internal_test_readiness"]["ready"])
            self.assertTrue(
                report["causal_xic_intervention_evidence"][
                    "selected_checkpoint_match"
                ]
            )

    def test_rejects_incomplete_five_cell_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross, matrix_path, intervention, evaluation = self._fixture(root)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            del matrix["matrix"]["primary-seed43"]
            _write_json(matrix_path, matrix)
            _write_manifest(
                matrix_path.parent,
                ["fusion_matrix_analysis.json", "fusion_matrix_analysis.md"],
            )
            with self.assertRaisesRegex(ValueError, "exactly the five declared cells"):
                build_development_dossier(
                    cross_family_report_path=cross,
                    fusion_matrix_report_path=matrix_path,
                    xic_intervention_report_path=intervention,
                    selected_fusion_evaluation_root=evaluation,
                    output_dir=root / "dossier",
                )

    def test_rejects_bound_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cross_path, matrix, intervention, evaluation = self._fixture(root)
            cross = json.loads(cross_path.read_text(encoding="utf-8"))
            source = Path(cross["sources"]["zero_shot_evaluation"]["path"])
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["inputs"]["validation_answers_sha256"] = "f" * 64
            _write_json(source, payload)
            cross["sources"]["zero_shot_evaluation"]["sha256"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            _write_json(cross_path, cross)
            _write_manifest(
                cross_path.parent,
                [
                    "cross_family_development_report.json",
                    "cross_family_development_table.md",
                ],
            )
            with self.assertRaisesRegex(ValueError, "validation_answers_sha256 drift"):
                build_development_dossier(
                    cross_family_report_path=cross_path,
                    fusion_matrix_report_path=matrix,
                    xic_intervention_report_path=intervention,
                    selected_fusion_evaluation_root=evaluation,
                    output_dir=root / "dossier",
                )

    def test_cli_and_slurm_have_no_internal_test_surface(self) -> None:
        package_root = Path(__file__).parents[2] / "multimodal_science" / "qwen3vl"
        cli = (package_root / "build_development_dossier_cli.py").read_text(
            encoding="utf-8"
        )
        slurm = (
            package_root / "slurm" / "coder_build_development_dossier.sbatch"
        ).read_text(encoding="utf-8")
        self.assertNotIn("internal-test", cli)
        self.assertNotIn("INTERNAL_TEST_ROOT", slurm)
        self.assertIn("#SBATCH --job-name=coder", slurm)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", slurm)
        self.assertIn("DEVELOPMENT_DOSSIER_CONTRACT=OK", slurm)
        self.assertIn('git -C "$repo_root" archive "$code_revision"', slurm)
        self.assertIn('PYTHONPATH="$code_root/backend"', slurm)


if __name__ == "__main__":
    unittest.main()
