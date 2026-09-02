from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
    select_threshold,
)
from multimodal_science.baselines.run_validation import validate_sequence_run
from multimodal_science.baselines.sequence_training import (
    REPORT_SCHEMA,
    SequenceTrainConfig,
)
from multimodal_science.baselines.validate_sequence_run_cli import parser
from multimodal_science.data.manifest import sha256_file


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def make_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    dataset = root / "dataset"
    dataset.mkdir()
    config = SequenceTrainConfig(
        epochs=1,
        batch_size=4,
        bootstrap_iterations=20,
    )
    config_payload = config.as_dict()
    write_json(run / "config.json", config_payload)
    (run / "best_model.pt").write_bytes(b"checkpoint evidence only")
    history = [
        {
            "epoch": 1,
            "learning_rate": config.learning_rate,
            "train": {"loss": 0.7},
            "validation": {"loss": 0.6},
            "validation_macro_f1_at_0_5": 0.5,
            "validation_mcc_at_0_5": 0.0,
        }
    ]
    write_jsonl(run / "history.jsonl", history)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.4, 0.7, 0.6, 0.8, 0.2, 0.45])
    predicted = np.asarray(
        [
            [0.10, 0.20],
            [0.20, 0.40],
            [0.30, 0.50],
            [0.25, 0.55],
            [0.15, 0.35],
            [0.40, 0.70],
            [0.50, 0.80],
            [0.35, 0.60],
        ]
    )
    true_boundaries = np.asarray(
        [
            [-1.0, -1.0],
            [0.20, 0.42],
            [-1.0, -1.0],
            [0.27, 0.52],
            [-1.0, -1.0],
            [0.38, 0.68],
            [-1.0, -1.0],
            [0.33, 0.61],
        ]
    )
    widths = np.asarray([1.5] * len(labels))
    groups = tuple(f"source-{index // 2}" for index in range(len(labels)))
    prediction_records = []
    for row in range(len(labels)):
        present = bool(labels[row])
        prediction_records.append(
            {
                "schema_version": "chrompeak-sequence-prediction-v1",
                "row": row,
                "asset_id": f"asset-{row}",
                "group_id": groups[row],
                "presence_probability": float(scores[row]),
                "start_normalized": float(predicted[row, 0]),
                "end_normalized": float(predicted[row, 1]),
                "roi_width_minutes": float(widths[row]),
                "target_peak_present": present,
                "target_start_normalized": (
                    float(true_boundaries[row, 0]) if present else None
                ),
                "target_end_normalized": (
                    float(true_boundaries[row, 1]) if present else None
                ),
            }
        )
    write_jsonl(run / "validation_predictions.jsonl", prediction_records)
    threshold_selection = select_threshold(labels, scores, objective="macro_f1")
    threshold = float(threshold_selection["threshold"])
    asset_index_sha256 = "b" * 64
    write_json(
        dataset / "dataset_report.json",
        {
            "schema_version": "chrompeak-multimodal-dataset-v1",
            "splits": ["train", "validation"],
            "asset_index_sha256": asset_index_sha256,
            "counts": {
                "by_split": {
                    "train": {"assets": 16},
                    "validation": {"assets": len(labels)},
                }
            },
        },
    )
    dataset_report_sha256 = sha256_file(dataset / "dataset_report.json")
    write_json(
        run / "frozen_threshold.json",
        {
            "schema_version": "chrompeak-frozen-threshold-v1",
            "selected_on_split": "validation",
            "objective": "macro_f1",
            "threshold": threshold,
            "dataset_report_sha256": dataset_report_sha256,
            "asset_index_sha256": asset_index_sha256,
            "internal_test_accessed": False,
        },
    )
    validation = {
        "fixed_threshold_0_5": binary_metrics(labels, scores, threshold=0.5),
        "selected_threshold": binary_metrics(labels, scores, threshold=threshold),
        "positive_boundary_metrics": boundary_metrics(
            true_boundaries,
            predicted,
            labels == 1,
            roi_width_minutes=widths,
        ),
        "source_grouped_bootstrap_95": grouped_bootstrap_binary(
            labels,
            scores,
            groups,
            threshold=threshold,
            iterations=config.bootstrap_iterations,
            seed=config.seed + 10_000,
        ),
    }
    artifacts = {}
    for key, filename in (
        ("config", "config.json"),
        ("checkpoint", "best_model.pt"),
        ("history", "history.jsonl"),
        ("validation_predictions", "validation_predictions.jsonl"),
        ("frozen_threshold", "frozen_threshold.json"),
    ):
        artifacts[key] = {"path": filename, "sha256": sha256_file(run / filename)}
    artifacts["validation_predictions"]["records"] = len(labels)
    report = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_scope": "train_validation_baseline",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "promotion_eligible": False,
        "internal_test_accessed": False,
        "evidence_gate": {
            "scientific_gate_complete": False,
            "missing_evidence": [
                "sealed_internal_test_metrics",
                "blank_stratified_false_positive_rate",
                "quantification_metrics",
                "declared_multimodal_ablations",
            ],
        },
        "dataset": {
            "dataset_report_sha256": dataset_report_sha256,
            "asset_index_sha256": asset_index_sha256,
            "original_train_samples": 16,
            "original_validation_samples": len(labels),
            "used_train_samples": 16,
            "used_validation_samples": len(labels),
            "train_source_groups": 8,
            "validation_source_groups": len(set(groups)),
        },
        "config": config_payload,
        "training": {
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_loss": 0.6,
            "positive_weight": 1.0,
            "early_stopped": False,
        },
        "threshold_selection": threshold_selection,
        "validation": validation,
        "artifacts": artifacts,
    }
    write_json(run / "scientific_report.json", report)
    return run


class SequenceRunValidationTests(unittest.TestCase):
    def test_cli_requires_source_dataset_and_exposes_no_internal_test(self) -> None:
        command = parser()
        arguments = command.parse_args(
            ["--run-dir", "run", "--dataset-root", "dataset"]
        )
        destinations = {action.dest for action in command._actions}

        self.assertEqual(arguments.dataset_root, Path("dataset"))
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)

    def test_recomputes_saved_evidence_without_loading_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = make_run(Path(temporary))
            output = run / "verification.json"

            result = validate_sequence_run(
                run,
                run.parent / "dataset",
                output_path=output,
            )
            verification = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(result.prediction_records, 8)
            self.assertEqual(result.validation_source_groups, 4)
            self.assertTrue(verification["quality_gate_passed"])
            self.assertFalse(verification["checkpoint_deserialized"])
            self.assertEqual(verification["run_dir"], ".")

    def test_rejects_tampered_artifact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = make_run(Path(temporary))
            with (run / "validation_predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_sequence_run(run, run.parent / "dataset")

    def test_rejects_metric_not_supported_by_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = make_run(Path(temporary))
            report_path = run / "scientific_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["validation"]["fixed_threshold_0_5"]["macro_f1"] += 0.1
            write_json(report_path, report)

            with self.assertRaisesRegex(ValueError, "recomputed evidence"):
                validate_sequence_run(run, run.parent / "dataset")

    def test_rejects_internal_test_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = make_run(Path(temporary))
            report_path = run / "scientific_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["internal_test_accessed"] = True
            write_json(report_path, report)

            with self.assertRaisesRegex(ValueError, "Internal test"):
                validate_sequence_run(run, run.parent / "dataset")

    def test_rejects_artifact_path_escape_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            report_path = run / "scientific_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifacts"]["config"]["path"] = "../config.json"
            write_json(report_path, report)

            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_sequence_run(run, run.parent / "dataset")

    def test_rejects_tampered_source_dataset_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            dataset_report = root / "dataset" / "dataset_report.json"
            report = json.loads(dataset_report.read_text(encoding="utf-8"))
            report["counts"]["by_split"]["train"]["assets"] = 15
            write_json(dataset_report, report)

            with self.assertRaisesRegex(ValueError, "source Dataset report"):
                validate_sequence_run(run, root / "dataset")


if __name__ == "__main__":
    unittest.main()
