from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.audit_zero_shot_cli import parser
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_RECORD_SCHEMA,
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.inference import (
    GENERATION_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
)
from multimodal_science.qwen3vl.zero_shot_audit import (
    ZERO_SHOT_AUDIT_SCHEMA,
    audit_zero_shot_failures,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def make_formal_results(root: Path) -> tuple[Path, str, Path, str]:
    generation = root / "generation"
    evaluation = root / "evaluation"
    generation.mkdir()
    evaluation.mkdir()
    predictions = []
    records = []
    instruction_number = 0

    for group_number in (1, 2):
        group_id = f"group-{group_number}"
        for language in ("en", "zh-CN"):
            task_responses: dict[str, dict[str, object]] = {
                "peak_presence": {"peak_present": True},
                "peak_presence_metadata": {
                    "peak_present": language == "en"
                },
                "scientific_qc": {
                    "qc_state": "no_peak",
                    "reason": "no_visible_peak",
                },
                "peak_grounding": {
                    "bbox_2d": (
                        [400, 0, 700, 300]
                        if language == "en"
                        else [160, 0, 280, 300]
                    )
                },
            }
            for task, response_object in task_responses.items():
                instruction_number += 1
                instruction_id = f"instruction-{instruction_number:02d}"
                predictions.append(
                    {
                        "schema_version": PREDICTION_SCHEMA,
                        "instruction_id": instruction_id,
                        "response": json.dumps(response_object, separators=(",", ":")),
                    }
                )
                row: dict[str, object] = {
                    "schema_version": BILINGUAL_EVALUATION_RECORD_SCHEMA,
                    "instruction_id": instruction_id,
                    "task": task,
                    "asset_id": f"asset-{group_number}",
                    "group_id": group_id,
                    "pair_id": f"pair-{group_number}-{task}",
                    "language": language,
                    "valid_json": True,
                    "schema_valid": True,
                    "exact_match": False,
                }
                if task in {"peak_presence", "peak_presence_metadata"}:
                    row["predicted_peak_present"] = response_object["peak_present"]
                elif task == "scientific_qc":
                    row["predicted_qc_state"] = response_object["qc_state"]
                    row["predicted_reason"] = response_object["reason"]
                else:
                    english = language == "en"
                    row.update(
                        {
                            "image_width": 400,
                            "image_height": 300,
                            "expected_bbox_2d": [160.0, 0.0, 280.0, 300.0],
                            "predicted_bbox_2d": (
                                None if english else [160.0, 0.0, 280.0, 300.0]
                            ),
                            "schema_valid": not english,
                            "bbox_iou": 0.0 if english else 1.0,
                        }
                    )
                records.append(row)

    predictions_path = generation / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    generation_report_path = generation / "generation_report.json"
    write_json(
        generation_report_path,
        {
            "schema_version": GENERATION_REPORT_SCHEMA,
            "artifacts": {
                "predictions": {
                    "path": predictions_path.name,
                    "sha256": sha256_file(predictions_path),
                    "records": len(predictions),
                }
            },
            "internal_test_accessed": False,
        },
    )
    generation_hash = sha256_file(generation_report_path)

    records_path = evaluation / "evaluation_records.jsonl"
    write_jsonl(records_path, records)
    evaluation_report_path = evaluation / "qwen_evaluation_report.json"
    write_json(
        evaluation_report_path,
        {
            "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
            "inputs": {
                "generation_report_sha256": generation_hash,
                "predictions_sha256": sha256_file(predictions_path),
            },
            "artifacts": {
                "evaluation_records": {
                    "path": records_path.name,
                    "sha256": sha256_file(records_path),
                    "records": len(records),
                }
            },
            "prediction_generation_provenance_verified": True,
            "internal_test_accessed": False,
        },
    )
    return (
        generation_report_path,
        generation_hash,
        evaluation_report_path,
        sha256_file(evaluation_report_path),
    )


class Qwen3VLZeroShotAuditTests(unittest.TestCase):
    def test_reports_shortcuts_and_coordinate_counterfactuals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation, generation_hash, evaluation, evaluation_hash = (
                make_formal_results(root)
            )
            result = audit_zero_shot_failures(
                generation,
                evaluation,
                root / "audit",
                expected_generation_report_sha256=generation_hash,
                expected_evaluation_report_sha256=evaluation_hash,
                bootstrap_iterations=20,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            distributions = report["prediction_distributions"]
            grounding = report["grounding_coordinate_counterfactual"][
                "by_language"
            ]
            self.assertEqual(report["schema_version"], ZERO_SHOT_AUDIT_SCHEMA)
            self.assertEqual(result.coordinate_records, 4)
            self.assertEqual(result.source_pixel_invalid_records, 2)
            self.assertTrue(distributions["peak_presence"]["en"]["single_output"])
            self.assertEqual(
                distributions["peak_presence_metadata"]["en"]["outputs"][0][
                    "value"
                ],
                "true",
            )
            self.assertEqual(
                distributions["peak_presence_metadata"]["zh-CN"]["outputs"][0][
                    "value"
                ],
                "false",
            )
            self.assertEqual(
                distributions["scientific_qc"]["en"]["outputs"][0]["qc_state"],
                "no_peak",
            )
            self.assertEqual(
                grounding["en"]["source_pixel"]["mean_bbox_iou_all"], 0.0
            )
            self.assertEqual(
                grounding["en"]["normalized_x_0_1000_source_y"][
                    "mean_bbox_iou_all"
                ],
                1.0,
            )
            self.assertEqual(
                grounding["en"]["normalized_x_0_1000_source_y"][
                    "source_pixel_invalid_rescued"
                ],
                2,
            )
            self.assertEqual(
                grounding["zh-CN"]["source_pixel"]["mean_bbox_iou_all"], 1.0
            )
            self.assertFalse(report["development_comparison_eligible"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertTrue(
                report["contracts"]["does_not_replace_formal_zero_shot_metrics"]
            )
            manifest = (result.output_dir / "artifact_manifest.sha256").read_text(
                encoding="utf-8"
            )
            self.assertIn("coordinate_counterfactual_records.jsonl", manifest)
            self.assertIn("zero_shot_failure_audit.json", manifest)

    def test_rejects_hash_mismatch_and_tampered_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation, generation_hash, evaluation, evaluation_hash = (
                make_formal_results(root)
            )
            with self.assertRaisesRegex(ValueError, "Generation report SHA-256 mismatch"):
                audit_zero_shot_failures(
                    generation,
                    evaluation,
                    root / "wrong-hash",
                    expected_generation_report_sha256="0" * 64,
                    expected_evaluation_report_sha256=evaluation_hash,
                    bootstrap_iterations=20,
                )

            predictions = generation.parent / "predictions.jsonl"
            predictions.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predictions SHA-256 mismatch"):
                audit_zero_shot_failures(
                    generation,
                    evaluation,
                    root / "tampered",
                    expected_generation_report_sha256=generation_hash,
                    expected_evaluation_report_sha256=evaluation_hash,
                    bootstrap_iterations=20,
                )

    def test_cli_does_not_accept_model_or_internal_test_arguments(self) -> None:
        command = parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            [
                "--generation-report",
                "generation.json",
                "--generation-report-sha256",
                "a" * 64,
                "--evaluation-report",
                "evaluation.json",
                "--evaluation-report-sha256",
                "b" * 64,
                "--output-dir",
                "audit",
            ]
        )

        self.assertEqual(arguments.bootstrap_iterations, 1000)
        self.assertNotIn("model", destinations)
        self.assertNotIn("model_name_or_path", destinations)
        self.assertNotIn("internal_test", destinations)
        self.assertNotIn("test", destinations)


if __name__ == "__main__":
    unittest.main()
