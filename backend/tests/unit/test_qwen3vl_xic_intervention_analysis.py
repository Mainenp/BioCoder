from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_inference import XIC_INTERVENTIONS
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.xic_intervention_analysis import (
    REPORT_SCHEMA,
    XicInterventionRun,
    _selected_metrics,
    analyze_xic_interventions,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluation_metrics(rows: list[dict], scope: str) -> dict:
    selected = _selected_metrics(rows, scope)
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


def _records(label: str) -> list[dict]:
    rows = []
    for group_index in range(2):
        group_id = f"group-{group_index}"
        asset_id = f"asset-{group_index}"
        truth = bool(group_index)
        for language in ("en", "zh-CN"):
            if label == "aligned":
                predicted = truth
                iou = 0.9
                qc = True
            elif label == "shuffled":
                predicted = not truth
                iou = 0.2
                qc = False
            elif label == "zero":
                predicted = truth if language == "en" else not truth
                iou = 0.5
                qc = language == "en"
            else:
                predicted = truth if group_index else not truth
                iou = 0.4
                qc = group_index == 1
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
                        "expected_bbox_2d": [10.0, 0.0, 20.0, 30.0],
                        "bbox_iou": iou,
                        "iou_at_0_5": iou >= 0.5,
                        "schema_valid": True,
                        "exact_match": iou == 1.0,
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
                    "schema_valid": True,
                    "exact_match": qc,
                }
            )
    return rows


def _make_run(root: Path, label: str) -> XicInterventionRun:
    generation_root = root / label / "generation"
    evaluation_root = root / label / "evaluation"
    generation_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    intervention = {
        "mode": label,
        "seed": 17,
        "language_variants_share_one_asset_intervention": True,
    }
    generation = {
        "schema_version": GENERATION_REPORT_SCHEMA,
        "source": {
            "bundle_report_sha256": "a" * 64,
            "prompt_artifact_sha256": "b" * 64,
            "instruction_report_sha256": "c" * 64,
            "validation_prompts_sha256": "d" * 64,
            "source_dataset_report_sha256": "e" * 64,
        },
        "model": {
            "artifact_sha256": "f" * 64,
            "identity_immutable": True,
            "adapter": {
                "training_report_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "code_revision": "3" * 40,
                "development_training_complete": True,
                "sensor_projector": {
                    "input_points": 160,
                    "hidden_size": 64,
                    "sensor_tokens": 4,
                    "base_channels": 32,
                    "dropout": 0.0,
                },
                "xic_intervention": intervention,
            },
        },
        "runtime": {"backend": "transformers", "xic_intervention": intervention},
        "generation": {"batch_size": 2, "do_sample": False, "seed": 17},
        "counts": {"predictions": 16},
        "development_comparison_candidate": True,
        "internal_test_accessed": False,
        "final_benchmark_eligible": False,
    }
    generation_path = generation_root / "generation_report.json"
    _write_json(generation_path, generation)

    rows = _records(label)
    records_path = evaluation_root / "evaluation_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    evaluation = {
        "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
        "inputs": {
            "instruction_report_sha256": "c" * 64,
            "source_dataset_report_sha256": "e" * 64,
            "validation_prompts_sha256": "d" * 64,
            "validation_answers_sha256": "4" * 64,
            "instruction_manifest_sha256": "5" * 64,
            "generation_report_sha256": sha256_file(generation_path),
            "predictions_sha256": label.encode().hex().ljust(64, "0")[:64],
        },
        "counts": {
            "predictions": len(rows),
            "schema_valid": len(rows),
            "validation_source_groups": 2,
            "by_language": {"en": len(rows) // 2, "zh-CN": len(rows) // 2},
        },
        "metrics": _evaluation_metrics(rows, "overall"),
        "metrics_by_language": {
            language: _evaluation_metrics(rows, language)
            for language in ("en", "zh-CN")
        },
        "cross_language_consistency": {},
        "artifacts": {
            "evaluation_records": {
                "path": records_path.name,
                "sha256": sha256_file(records_path),
                "records": len(rows),
            }
        },
        "generation_provenance": {
            "report_sha256": sha256_file(generation_path),
            "prediction_records": len(rows),
        },
        "prediction_generation_provenance_verified": True,
        "development_comparison_eligible": True,
        "language_variants_are_not_independent_source_assets": True,
        "internal_test_accessed": False,
        "final_benchmark_eligible": False,
    }
    _write_json(evaluation_root / "qwen_evaluation_report.json", evaluation)
    return XicInterventionRun(generation_root, evaluation_root)


class XicInterventionAnalysisTests(unittest.TestCase):
    def test_builds_paired_group_bootstrap_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {label: _make_run(root, label) for label in XIC_INTERVENTIONS}
            result = analyze_xic_interventions(
                runs=runs,
                output_dir=root / "analysis",
                bootstrap_iterations=200,
                seed=29,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(
                report["paired_group_bootstrap"]["resampling_unit"], "source_group"
            )
            self.assertEqual(report["paired_group_bootstrap"]["independent_units"], 2)
            shuffled = report["paired_group_bootstrap"]["comparisons"]["shuffled"]
            self.assertEqual(
                shuffled["overall"]["peak_presence"]["source_groups"], 2
            )
            self.assertEqual(
                shuffled["overall"]["peak_grounding"]["source_groups"], 1
            )
            self.assertGreater(
                shuffled["overall"]["peak_presence"]["macro_f1"][
                    "direction_adjusted_aligned_benefit"
                ],
                0.0,
            )
            self.assertTrue(report["contracts"]["same_fusion_checkpoint"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertTrue(result.markdown_path.is_file())
            manifest = result.manifest_path.read_text(encoding="utf-8")
            self.assertIn("xic_intervention_analysis.json", manifest)
            self.assertIn("xic_intervention_analysis.md", manifest)

    def test_rejects_evaluation_record_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {label: _make_run(root, label) for label in XIC_INTERVENTIONS}
            evaluation_root = runs["zero"].evaluation_root
            records_path = evaluation_root / "evaluation_records.jsonl"
            rows = [json.loads(line) for line in records_path.read_text().splitlines()]
            rows[0]["asset_id"] = "drifted-asset"
            records_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            report_path = evaluation_root / "qwen_evaluation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifacts"]["evaluation_records"]["sha256"] = sha256_file(
                records_path
            )
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "identity drift"):
                analyze_xic_interventions(
                    runs=runs,
                    output_dir=root / "analysis",
                    bootstrap_iterations=100,
                )

    def test_slurm_contract_is_explicit(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_analyze_xic_interventions.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("BIOCODER_ALIGNED_GENERATION_ROOT", script)
        self.assertIn("BIOCODER_AVAILABILITY_OFF_EVALUATION_ROOT", script)
        self.assertIn("paired_source_group_bootstrap", script)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)


if __name__ == "__main__":
    unittest.main()
