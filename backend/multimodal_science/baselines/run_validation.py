"""Independently verify a published sequence-baseline run from saved evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
    select_threshold,
)
from multimodal_science.baselines.sequence_training import (
    REPORT_SCHEMA,
    SequenceTrainConfig,
)
from multimodal_science.data.manifest import sha256_file

VERIFICATION_SCHEMA = "chrompeak-sequence-run-verification-v1"
_REQUIRED_ARTIFACTS = (
    "config",
    "checkpoint",
    "history",
    "validation_predictions",
    "frozen_threshold",
)
_REQUIRED_MISSING_EVIDENCE = {
    "sealed_internal_test_metrics",
    "blank_stratified_false_positive_rate",
    "quantification_metrics",
    "declared_multimodal_ablations",
}


@dataclass(frozen=True)
class SequenceRunVerification:
    run_dir: Path
    dataset_root: Path
    source_report_path: Path
    source_report_sha256: str
    dataset_report_sha256: str
    verification_path: Path | None
    verification_sha256: str | None
    prediction_records: int
    validation_source_groups: int
    selected_threshold: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, description: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{description} must be an object")
    return value


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), description)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} JSON: {path}") from exc


def _artifact_path(run_dir: Path, artifact: dict[str, Any], name: str) -> Path:
    relative = Path(str(artifact.get("path") or ""))
    _require(relative != Path("."), f"Artifact path is empty: {name}")
    _require(not relative.is_absolute(), f"Artifact path must be relative: {name}")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes run directory: {name}") from exc
    _require(path.is_file(), f"Run artifact is missing: {name}")
    expected_hash = artifact.get("sha256")
    _require(
        isinstance(expected_hash, str) and len(expected_hash) == 64,
        f"Artifact SHA-256 is invalid: {name}",
    )
    _require(sha256_file(path) == expected_hash, f"Run artifact hash mismatch: {name}")
    return path


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank line in {description} at line {line_number}")
            try:
                records.append(_object(json.loads(line), f"{description} line {line_number}"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {description} JSON at line {line_number}"
                ) from exc
    _require(bool(records), f"{description} must not be empty")
    return records


def _number(value: Any, description: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{description} must be finite")
    return result


def _assert_equivalent(actual: Any, expected: Any, description: str) -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{description} must be an object")
        _require(set(actual) == set(expected), f"{description} keys do not match")
        for key in expected:
            _assert_equivalent(actual[key], expected[key], f"{description}.{key}")
        return
    if isinstance(expected, list):
        _require(isinstance(actual, list), f"{description} must be a list")
        _require(len(actual) == len(expected), f"{description} length does not match")
        for index, item in enumerate(expected):
            _assert_equivalent(actual[index], item, f"{description}[{index}]")
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        expected_number = float(expected)
        actual_number = float(actual)
        equal = (
            math.isnan(expected_number)
            and math.isnan(actual_number)
            or math.isclose(actual_number, expected_number, rel_tol=1e-10, abs_tol=1e-12)
        )
        _require(equal, f"{description} does not match recomputed evidence")
        return
    _require(actual == expected, f"{description} does not match recomputed evidence")


def _validate_history(
    records: list[dict[str, Any]],
    report: dict[str, Any],
    config: SequenceTrainConfig,
) -> None:
    training = _object(report.get("training"), "training")
    _require(training.get("epochs_completed") == len(records), "History length mismatch")
    _require(len(records) <= config.epochs, "History exceeds configured epochs")
    losses = []
    for expected_epoch, record in enumerate(records, start=1):
        _require(record.get("epoch") == expected_epoch, "History epochs are not contiguous")
        validation = _object(record.get("validation"), f"history epoch {expected_epoch} validation")
        losses.append(_number(validation.get("loss"), f"history epoch {expected_epoch} loss"))
    best_epoch = training.get("best_epoch")
    _require(
        isinstance(best_epoch, int) and not isinstance(best_epoch, bool),
        "Best epoch must be an integer",
    )
    _require(1 <= best_epoch <= len(records), "Best epoch is outside history")
    best_loss = _number(training.get("best_validation_loss"), "best validation loss")
    _require(
        math.isclose(best_loss, losses[best_epoch - 1], rel_tol=1e-10, abs_tol=1e-12),
        "Best validation loss does not match history",
    )
    _require(best_loss <= min(losses) + 1e-8, "Best epoch is not a minimum-loss epoch")
    _require(
        training.get("early_stopped") == (len(records) < config.epochs),
        "Early-stop flag does not match history",
    )


def _prediction_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    labels = []
    probabilities = []
    predictions = []
    truth = []
    roi_widths = []
    asset_ids = set()
    group_ids = []
    for expected_row, record in enumerate(records):
        _require(
            record.get("schema_version") == "chrompeak-sequence-prediction-v1",
            f"Unsupported prediction schema at row {expected_row}",
        )
        _require(record.get("row") == expected_row, "Prediction rows are not contiguous")
        asset_id = record.get("asset_id")
        group_id = record.get("group_id")
        _require(isinstance(asset_id, str) and asset_id != "", "Prediction asset ID is missing")
        _require(asset_id not in asset_ids, f"Duplicate prediction asset ID: {asset_id}")
        _require(isinstance(group_id, str) and group_id != "", "Prediction group ID is missing")
        asset_ids.add(asset_id)
        group_ids.append(group_id)
        probability = _number(record.get("presence_probability"), "presence probability")
        _require(0.0 <= probability <= 1.0, "Presence probability is outside [0, 1]")
        start = _number(record.get("start_normalized"), "predicted start")
        end = _number(record.get("end_normalized"), "predicted end")
        _require(0.0 <= start < end <= 1.0, "Predicted boundary is invalid")
        roi_width = _number(record.get("roi_width_minutes"), "ROI width")
        _require(roi_width > 0.0, "ROI width must be positive")
        present = record.get("target_peak_present")
        _require(isinstance(present, bool), "Target presence must be boolean")
        target_start = record.get("target_start_normalized")
        target_end = record.get("target_end_normalized")
        if present:
            target_start = _number(target_start, "target start")
            target_end = _number(target_end, "target end")
            _require(
                0.0 <= target_start < target_end <= 1.0,
                "Positive target boundary is invalid",
            )
        else:
            _require(
                target_start is None and target_end is None,
                "Negative target boundaries must be null",
            )
            target_start = -1.0
            target_end = -1.0
        labels.append(int(present))
        probabilities.append(probability)
        predictions.append([start, end])
        truth.append([target_start, target_end])
        roi_widths.append(roi_width)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
        np.asarray(truth, dtype=np.float64),
        np.asarray(roi_widths, dtype=np.float64),
        tuple(group_ids),
    )


def _validate_scope(
    report: dict[str, Any], config: SequenceTrainConfig, prediction_count: int
) -> None:
    smoke = config.smoke_test
    _require(
        report.get("evaluation_scope")
        == ("smoke_test" if smoke else "train_validation_baseline"),
        "Evaluation scope does not match config",
    )
    _require(
        report.get("development_comparison_eligible") is (not smoke),
        "Development-comparison eligibility is invalid",
    )
    _require(report.get("final_benchmark_eligible") is False, "Final benchmark must stay sealed")
    _require(report.get("promotion_eligible") is False, "Promotion must stay gated")
    _require(report.get("internal_test_accessed") is False, "Internal test must stay sealed")
    evidence = _object(report.get("evidence_gate"), "evidence gate")
    _require(
        evidence.get("scientific_gate_complete") is False,
        "Scientific gate cannot be complete",
    )
    missing = evidence.get("missing_evidence")
    _require(isinstance(missing, list), "Missing evidence must be a list")
    _require(
        _REQUIRED_MISSING_EVIDENCE.issubset(set(missing)),
        "Required scientific evidence is not declared missing",
    )
    dataset = _object(report.get("dataset"), "dataset")
    original_train = dataset.get("original_train_samples")
    original_validation = dataset.get("original_validation_samples")
    used_train = dataset.get("used_train_samples")
    used_validation = dataset.get("used_validation_samples")
    for name, value in (
        ("original train samples", original_train),
        ("original validation samples", original_validation),
        ("used train samples", used_train),
        ("used validation samples", used_validation),
    ):
        _require(isinstance(value, int) and value > 0, f"{name} must be positive")
    _require(used_train <= original_train, "Used train samples exceed Dataset")
    _require(used_validation <= original_validation, "Used validation samples exceed Dataset")
    _require(used_validation == prediction_count, "Prediction count does not match used validation")
    if not smoke:
        _require(used_train == original_train, "Non-smoke run did not use the full train split")
        _require(
            used_validation == original_validation,
            "Non-smoke run did not use the full validation split",
        )


def _validate_source_dataset(
    dataset_root: Path, run_dataset: dict[str, Any]
) -> tuple[Path, str]:
    report_path = dataset_root / "dataset_report.json"
    _require(report_path.is_file(), f"Source Dataset report not found: {report_path}")
    report = _read_json(report_path, "source Dataset report")
    _require(
        report.get("schema_version") == "chrompeak-multimodal-dataset-v1",
        "Unsupported source Dataset schema",
    )
    _require(
        set(report.get("splits") or []) == {"train", "validation"},
        "Source Dataset must expose train and validation only",
    )
    report_sha256 = sha256_file(report_path)
    _require(
        report_sha256 == run_dataset.get("dataset_report_sha256"),
        "Run does not match the supplied source Dataset report",
    )
    _require(
        report.get("asset_index_sha256") == run_dataset.get("asset_index_sha256"),
        "Run does not match the supplied source asset index",
    )
    split_counts = _object(
        _object(report.get("counts"), "source Dataset counts").get("by_split"),
        "source Dataset split counts",
    )
    for split in ("train", "validation"):
        declared = _object(split_counts.get(split), f"source Dataset {split} counts")
        _require(
            declared.get("assets") == run_dataset.get(f"original_{split}_samples"),
            f"Run original {split} count does not match source Dataset",
        )
    return report_path, report_sha256


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"Verification output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_sequence_run(
    run_dir: Path,
    dataset_root: Path,
    *,
    output_path: Path | None = None,
) -> SequenceRunVerification:
    """Validate immutable artifacts and recompute every reported validation metric."""

    run_dir = run_dir.resolve()
    dataset_root = dataset_root.resolve()
    report_path = run_dir / "scientific_report.json"
    _require(report_path.is_file(), f"Scientific report not found: {report_path}")
    report = _read_json(report_path, "scientific report")
    _require(report.get("schema_version") == REPORT_SCHEMA, "Unsupported scientific report schema")
    artifacts = _object(report.get("artifacts"), "artifacts")
    _require(
        set(_REQUIRED_ARTIFACTS) == set(artifacts),
        "Scientific report artifact set is invalid",
    )
    paths = {
        name: _artifact_path(run_dir, _object(artifacts[name], f"artifact {name}"), name)
        for name in _REQUIRED_ARTIFACTS
    }
    config_payload = _read_json(paths["config"], "config")
    _require(report.get("config") == config_payload, "Report and config artifact disagree")
    try:
        config = SequenceTrainConfig(**config_payload)
    except TypeError as exc:
        raise ValueError("Config artifact does not match the training schema") from exc
    config.validate()
    history = _read_jsonl(paths["history"], "history")
    _validate_history(history, report, config)
    predictions = _read_jsonl(paths["validation_predictions"], "validation predictions")
    prediction_artifact = _object(artifacts["validation_predictions"], "prediction artifact")
    _require(
        prediction_artifact.get("records") == len(predictions),
        "Declared prediction count mismatch",
    )
    labels, scores, predicted_boundaries, true_boundaries, roi_widths, group_ids = (
        _prediction_arrays(predictions)
    )
    _require(
        np.any(labels == 0) and np.any(labels == 1),
        "Validation predictions need both classes",
    )
    _require(len(set(group_ids)) >= 2, "Validation predictions need at least two source groups")
    _validate_scope(report, config, len(predictions))

    dataset = _object(report.get("dataset"), "dataset")
    dataset_report_path, dataset_report_sha256 = _validate_source_dataset(
        dataset_root, dataset
    )
    _require(
        dataset.get("validation_source_groups") == len(set(group_ids)),
        "Validation source-group count mismatch",
    )
    threshold_report = _object(report.get("threshold_selection"), "threshold selection")
    _require(threshold_report.get("objective") == "macro_f1", "Unexpected threshold objective")
    recomputed_threshold = select_threshold(labels, scores, objective="macro_f1")
    _assert_equivalent(threshold_report, recomputed_threshold, "threshold selection")
    selected_threshold = float(recomputed_threshold["threshold"])
    validation = _object(report.get("validation"), "validation")
    recomputed_validation = {
        "fixed_threshold_0_5": binary_metrics(labels, scores, threshold=0.5),
        "selected_threshold": binary_metrics(
            labels, scores, threshold=selected_threshold
        ),
        "positive_boundary_metrics": boundary_metrics(
            true_boundaries,
            predicted_boundaries,
            labels == 1,
            roi_width_minutes=roi_widths,
        ),
        "source_grouped_bootstrap_95": grouped_bootstrap_binary(
            labels,
            scores,
            group_ids,
            threshold=selected_threshold,
            iterations=config.bootstrap_iterations,
            seed=config.seed + 10_000,
        ),
    }
    _assert_equivalent(validation, recomputed_validation, "validation")

    threshold = _read_json(paths["frozen_threshold"], "frozen threshold")
    _require(
        threshold.get("schema_version") == "chrompeak-frozen-threshold-v1",
        "Unsupported frozen-threshold schema",
    )
    _require(
        threshold.get("selected_on_split") == "validation",
        "Threshold was not selected on validation",
    )
    _require(threshold.get("objective") == "macro_f1", "Frozen threshold objective mismatch")
    _assert_equivalent(threshold.get("threshold"), selected_threshold, "frozen threshold")
    _require(
        threshold.get("internal_test_accessed") is False,
        "Frozen threshold accessed internal test",
    )
    for hash_name in ("dataset_report_sha256", "asset_index_sha256"):
        expected_hash = dataset.get(hash_name)
        _require(
            isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"Dataset {hash_name} is invalid",
        )
        _require(
            threshold.get(hash_name) == expected_hash,
            f"Frozen threshold {hash_name} mismatch",
        )

    source_report_sha256 = sha256_file(report_path)
    verification_payload = {
        "schema_version": VERIFICATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_gate_passed": True,
        "run_dir": ".",
        "source_report": {
            "path": report_path.name,
            "sha256": source_report_sha256,
            "schema_version": REPORT_SCHEMA,
        },
        "source_dataset": {
            "report_path": dataset_report_path.name,
            "report_sha256": dataset_report_sha256,
            "asset_index_sha256": dataset["asset_index_sha256"],
        },
        "verified_artifacts": {
            name: {
                "path": str(paths[name].relative_to(run_dir)),
                "sha256": sha256_file(paths[name]),
            }
            for name in _REQUIRED_ARTIFACTS
        },
        "recomputed_evidence": {
            "prediction_records": len(predictions),
            "validation_source_groups": len(set(group_ids)),
            "selected_threshold": selected_threshold,
            "threshold_selection": recomputed_threshold,
            "validation": recomputed_validation,
        },
        "sealed_internal_test_confirmed": True,
        "checkpoint_deserialized": False,
    }
    resolved_output = output_path.resolve() if output_path is not None else None
    if resolved_output is not None:
        _atomic_write_json(resolved_output, verification_payload)
    return SequenceRunVerification(
        run_dir=run_dir,
        dataset_root=dataset_root,
        source_report_path=report_path,
        source_report_sha256=source_report_sha256,
        dataset_report_sha256=dataset_report_sha256,
        verification_path=resolved_output,
        verification_sha256=(sha256_file(resolved_output) if resolved_output else None),
        prediction_records=len(predictions),
        validation_source_groups=len(set(group_ids)),
        selected_threshold=selected_threshold,
    )
