from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_matrix_analysis import (
    MATRIX_CONFIGURATIONS,
    REPORT_SCHEMA,
    FusionMatrixRun,
    _recomputed_metrics,
    analyze_fusion_matrix,
)
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, names: list[str]) -> None:
    (root / "artifact_manifest.sha256").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _records(error_stride: int, grounding_iou: float) -> list[dict]:
    rows = []
    for asset_index in range(1815):
        asset_id = f"asset-{asset_index:04d}"
        group_id = f"group-{asset_index % 11:02d}"
        truth = asset_index < 1409
        predicted = truth if not (truth and asset_index % error_stride == 0) else False
        for language in ("en", "zh-CN"):
            for task in ("peak_presence", "peak_presence_metadata"):
                rows.append(
                    {
                        "instruction_id": f"{asset_id}-{language}-{task}",
                        "pair_id": f"{asset_id}-{task}",
                        "asset_id": asset_id,
                        "group_id": group_id,
                        "task": task,
                        "language": language,
                        "target_peak_present": truth,
                        "classification_score": float(predicted),
                        "schema_valid": True,
                        "exact_match": predicted == truth,
                    }
                )
            if truth:
                rows.append(
                    {
                        "instruction_id": f"{asset_id}-{language}-peak_grounding",
                        "pair_id": f"{asset_id}-peak_grounding",
                        "asset_id": asset_id,
                        "group_id": group_id,
                        "task": "peak_grounding",
                        "language": language,
                        "target_peak_present": True,
                        "expected_bbox_2d": [10.0, 0.0, 20.0, 30.0],
                        "bbox_iou": grounding_iou,
                        "iou_at_0_5": grounding_iou >= 0.5,
                        "schema_valid": True,
                        "exact_match": False,
                    }
                )
            rows.append(
                {
                    "instruction_id": f"{asset_id}-{language}-scientific_qc",
                    "pair_id": f"{asset_id}-scientific_qc",
                    "asset_id": asset_id,
                    "group_id": group_id,
                    "task": "scientific_qc",
                    "language": language,
                    "target_peak_present": truth,
                    "schema_valid": True,
                    "exact_match": predicted == truth,
                }
            )
    assert len(rows) == 13708
    return rows


def _task_metrics(rows: list[dict], scope: str) -> dict:
    selected = _recomputed_metrics(rows, scope)
    return {
        "peak_presence": {"classification": selected["peak_presence"]},
        "peak_presence_metadata": {
            "classification": selected["peak_presence_metadata"]
        },
        "peak_grounding": {"grounding": selected["peak_grounding"]},
        "scientific_qc": {
            "exact_match_rate": selected["scientific_qc"]["exact_match_rate"]
        },
    }


def _make_run(
    root: Path,
    label: str,
    sensor_tokens: int,
    training_seed: int,
    *,
    error_stride: int,
    grounding_iou: float,
) -> FusionMatrixRun:
    training_root = root / label / "training"
    generation_root = root / label / "generation"
    evaluation_root = root / label / "evaluation"
    (training_root / "adapter").mkdir(parents=True)
    generation_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    (training_root / "adapter/adapter_model.safetensors").write_bytes(b"adapter")
    (training_root / "sensor_projector.safetensors").write_bytes(b"projector")
    projector = {
        "input_points": 160,
        "hidden_size": 64,
        "sensor_tokens": sensor_tokens,
        "base_channels": 32,
        "dropout": 0.0,
    }
    gate_probability = 0.018 + training_seed / 1_000_000 + sensor_tokens / 10_000_000
    sources = {
        "fusion_bundle_report_sha256": "a" * 64,
        "lora_bundle_report_sha256": "b" * 64,
        "fusion_lora_content_binding": {"sha256": "c" * 64},
        "dataset_report_sha256": "d" * 64,
        "adapter_train_rows_sha256": "e" * 64,
        "initial_adapter": {"training_report_sha256": "f" * 64},
        "auxiliary_pretraining": None,
    }
    training = {
        "schema_version": FUSION_TRAINING_REPORT_SCHEMA,
        "code_revision": "1" * 40,
        "sources": sources,
        "model": {
            "name_or_path": "model",
            "revision": "modelscope-master",
            "base_artifact_sha256": "2" * 64,
            "sensor_projector": projector,
            "sensor_gate": {
                "initial_probability": 0.017986,
                "final_probability": gate_probability,
                "absolute_probability_change": gate_probability - 0.017986,
            },
        },
        "training": {
            "seed": training_seed,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "training_records": 54335,
            "optimizer_updates": 3396,
        },
        "contracts": {
            "train_split_only": True,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "image_and_xic_forward": True,
            "lora_and_projector_backward": True,
            "parameter_updates_verified": True,
        },
        "development_training_complete": True,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    training_path = training_root / "fusion_training_report.json"
    _write_json(training_path, training)
    _write_manifest(
        training_root,
        [
            "fusion_training_report.json",
            "adapter/adapter_model.safetensors",
            "sensor_projector.safetensors",
        ],
    )
    training_sha = sha256_file(training_path)
    training_manifest_sha = sha256_file(training_root / "artifact_manifest.sha256")

    intervention = {
        "mode": "aligned",
        "seed": 17,
        "answer_key_used": False,
        "language_variants_share_one_asset_intervention": True,
    }
    predictions_path = generation_root / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    generation = {
        "schema_version": GENERATION_REPORT_SCHEMA,
        "source": {
            "bundle_report_sha256": "3" * 64,
            "prompt_artifact_sha256": "4" * 64,
            "instruction_report_sha256": "5" * 64,
            "validation_prompts_sha256": "6" * 64,
            "source_dataset_report_sha256": "d" * 64,
        },
        "model": {
            "artifact_sha256": "2" * 64,
            "identity_immutable": True,
            "adapter": {
                "training_report_sha256": training_sha,
                "manifest_sha256": training_manifest_sha,
                "code_revision": "1" * 40,
                "development_training_complete": True,
                "sensor_projector": projector,
                "xic_intervention": intervention,
            },
        },
        "runtime": {
            "backend": "transformers",
            "manual_cached_greedy_decode": True,
            "explicit_fused_mrope_positions": True,
            "sensor_projector": projector,
            "sensor_gate": {"probability": gate_probability},
            "xic_intervention": intervention,
        },
        "generation": {"batch_size": 1, "do_sample": False, "seed": 17},
        "counts": {"predictions": 13708},
        "artifacts": {
            "predictions": {
                "path": predictions_path.name,
                "sha256": sha256_file(predictions_path),
                "records": 13708,
            }
        },
        "development_comparison_candidate": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    generation_path = generation_root / "generation_report.json"
    _write_json(generation_path, generation)
    _write_manifest(generation_root, ["generation_report.json", "predictions.jsonl"])

    rows = _records(error_stride, grounding_iou)
    records_path = evaluation_root / "evaluation_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    cross_language = {
        "peak_presence": {
            "pairs": 1815,
            "both_schema_valid_rate": 1.0,
            "exact_prediction_consistency_rate_all": 1.0,
        },
        "peak_presence_metadata": {
            "pairs": 1815,
            "both_schema_valid_rate": 1.0,
            "exact_prediction_consistency_rate_all": 1.0,
        },
        "peak_grounding": {
            "pairs": 1409,
            "both_schema_valid_rate": 1.0,
            "exact_prediction_consistency_rate_all": 1.0,
            "mean_prediction_bbox_iou_all": 1.0,
            "prediction_bbox_iou_at_0_9_rate_all": 1.0,
        },
        "scientific_qc": {
            "pairs": 1815,
            "both_schema_valid_rate": 1.0,
            "exact_prediction_consistency_rate_all": 1.0,
        },
    }
    evaluation = {
        "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
        "inputs": {
            "instruction_report_sha256": "5" * 64,
            "source_dataset_report_sha256": "d" * 64,
            "validation_prompts_sha256": "6" * 64,
            "validation_answers_sha256": "7" * 64,
            "instruction_manifest_sha256": "8" * 64,
            "generation_report_sha256": sha256_file(generation_path),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "counts": {
            "predictions": 13708,
            "valid_json": 13708,
            "schema_valid": 13708,
            "independent_validation_assets": 1815,
            "validation_source_groups": 11,
            "by_language": {"en": 6854, "zh-CN": 6854},
        },
        "metrics": _task_metrics(rows, "overall"),
        "metrics_by_language": {
            language: _task_metrics(rows, language)
            for language in ("en", "zh-CN")
        },
        "cross_language_consistency": cross_language,
        "evaluation": {"bootstrap_iterations": 1000, "seed": 17},
        "artifacts": {
            "evaluation_records": {
                "path": records_path.name,
                "sha256": sha256_file(records_path),
                "records": len(rows),
            }
        },
        "generation_provenance": {
            "report_sha256": sha256_file(generation_path),
            "prediction_records": 13708,
        },
        "prediction_generation_provenance_verified": True,
        "language_variants_are_not_independent_source_assets": True,
        "answer_key_file_separate_from_prompts": True,
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(evaluation_root / "qwen_evaluation_report.json", evaluation)
    _write_manifest(
        evaluation_root,
        ["qwen_evaluation_report.json", "evaluation_records.jsonl"],
    )
    return FusionMatrixRun(training_root, generation_root, evaluation_root)


def _matrix(root: Path) -> dict[str, FusionMatrixRun]:
    runs = {}
    for index, (label, (tokens, seed)) in enumerate(MATRIX_CONFIGURATIONS.items()):
        runs[label] = _make_run(
            root,
            label,
            tokens,
            seed,
            error_stride=25 + index * 5,
            grounding_iou=0.60 + index * 0.05,
        )
    return runs


class FusionMatrixAnalysisTests(unittest.TestCase):
    def test_builds_three_seed_and_token_ablation_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = analyze_fusion_matrix(
                runs=_matrix(root), output_dir=root / "analysis"
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            primary = report["primary_seed_aggregate"]["overall"][
                "peak_grounding"
            ]["mean_bbox_iou_all"]
            self.assertEqual(primary["n"], 3)
            self.assertEqual(primary["standard_deviation_ddof"], 1)
            self.assertAlmostEqual(primary["mean"], 0.65)
            self.assertAlmostEqual(primary["sample_standard_deviation"], 0.05)
            self.assertEqual(
                set(report["token_ablation"]["runs"]), {"1", "4", "8"}
            )
            self.assertIn(
                "effect_toward_better",
                report["token_ablation"]["comparisons_to_four_tokens"]["1"]
                ["overall"]["peak_presence"]["false_positive_rate"],
            )
            self.assertTrue(
                report["contracts"]["evaluation_metrics_recomputed_from_records"]
            )
            self.assertFalse(report["internal_test_accessed"])
            self.assertTrue(result.markdown_path.is_file())

    def test_rejects_evaluation_record_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = _matrix(root)
            evaluation_root = runs["tokens8-seed17"].evaluation_root
            records_path = evaluation_root / "evaluation_records.jsonl"
            rows = records_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["pair_id"] = "drifted-pair"
            rows[0] = json.dumps(first, sort_keys=True)
            records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report_path = evaluation_root / "qwen_evaluation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifacts"]["evaluation_records"]["sha256"] = sha256_file(
                records_path
            )
            _write_json(report_path, report)
            _write_manifest(
                evaluation_root,
                ["qwen_evaluation_report.json", "evaluation_records.jsonl"],
            )
            with self.assertRaisesRegex(ValueError, "identity drift"):
                analyze_fusion_matrix(runs=runs, output_dir=root / "analysis")

    def test_rejects_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = _matrix(Path(directory))
            runs.pop("primary-seed43")
            with self.assertRaisesRegex(ValueError, "Expected matrix labels"):
                analyze_fusion_matrix(
                    runs=runs, output_dir=Path(directory) / "analysis"
                )

    def test_slurm_contracts_are_explicit(self) -> None:
        slurm_root = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm"
        )
        evaluation = (slurm_root / "coder_fusion_evaluation_matrix.sbatch").read_text(
            encoding="utf-8"
        )
        analysis = (slurm_root / "coder_analyze_fusion_matrix.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --array=0-4%2", evaluation)
        self.assertIn("export BIOCODER_SEED=17", evaluation)
        self.assertIn("BIOCODER_MATRIX_EVALUATION_REVISION", analysis)
        self.assertIn("evaluation_metrics_recomputed_from_records", analysis)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", analysis)


if __name__ == "__main__":
    unittest.main()
