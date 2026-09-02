"""Hash-bound evaluation for answer-separated Qwen3-VL validation predictions."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from multimodal_science.baselines.metrics import binary_metrics, grouped_bootstrap_binary
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.instruction_data import (
    DATASET_SCHEMA,
    MANIFEST_SCHEMA,
    TASKS,
    VALIDATION_ANSWER_SCHEMA,
    VALIDATION_PROMPT_SCHEMA,
)

PREDICTION_SCHEMA = "chrompeak-qwen3vl-prediction-v1"
EVALUATION_RECORD_SCHEMA = "chrompeak-qwen3vl-evaluation-record-v1"
EVALUATION_REPORT_SCHEMA = "chrompeak-qwen3vl-evaluation-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GROUNDING_SIZE = re.compile(r"for a ([1-9][0-9]*)x([1-9][0-9]*) image")


@dataclass(frozen=True)
class QwenEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    prediction_records: int
    valid_json_records: int
    schema_valid_records: int
    validation_source_groups: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _safe_child(root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Invalid path for {context}")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected a relative path for {context}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes instruction root for {context}") from error
    return candidate


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), context)


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            records.append(
                _object(json.loads(line), f"{context} at {path.name}:{line_number}")
            )
    _require(bool(records), f"{context} is empty: {path}")
    return records


def _artifact(
    instruction_root: Path,
    report: dict[str, Any],
    name: str,
) -> tuple[Path, list[dict[str, Any]]]:
    artifacts = _object(report.get("artifacts"), "instruction artifacts")
    artifact = _object(artifacts.get(name), f"instruction artifact {name}")
    path = _safe_child(instruction_root, artifact.get("path"), name)
    expected_hash = artifact.get("sha256")
    _require(
        isinstance(expected_hash, str) and len(expected_hash) == 64,
        f"Invalid SHA-256 for {name}",
    )
    _require(path.is_file(), f"Instruction artifact not found: {path}")
    actual_hash = sha256_file(path)
    _require(actual_hash == expected_hash, f"Instruction artifact hash mismatch: {name}")
    records = _read_jsonl(path, name)
    expected_records = artifact.get("records")
    _require(
        isinstance(expected_records, int) and expected_records == len(records),
        f"Instruction artifact record count mismatch: {name}",
    )
    return path, records


def _unique_by_id(
    records: Iterable[dict[str, Any]], context: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        instruction_id = record.get("instruction_id")
        _require(
            isinstance(instruction_id, str) and bool(_HEX_24.fullmatch(instruction_id)),
            f"Invalid instruction_id in {context}: {instruction_id}",
        )
        _require(instruction_id not in result, f"Duplicate instruction_id in {context}")
        result[instruction_id] = record
    return result


def _strict_json_object(response: str) -> tuple[dict[str, Any] | None, bool]:
    try:
        parsed = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (parsed, True) if isinstance(parsed, dict) else (None, True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _grounding_dimensions(prompt: dict[str, Any]) -> tuple[int, int]:
    conversations = prompt.get("conversations")
    _require(
        isinstance(conversations, list) and len(conversations) == 1,
        "Validation prompt must contain exactly one human message",
    )
    message = _object(conversations[0], "validation conversation")
    value = message.get("value")
    _require(message.get("from") == "human" and isinstance(value, str), "Invalid prompt")
    match = _GROUNDING_SIZE.search(value)
    _require(match is not None, "Grounding prompt does not declare image dimensions")
    return int(match.group(1)), int(match.group(2))


def _valid_bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(_is_number(x) for x in value):
        return None
    x1, y1, x2, y2 = (float(x) for x in value)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return [x1, y1, x2, y2]


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _grouped_mean_bootstrap(
    values: list[float],
    groups: list[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    _require(array.ndim == 1 and array.size == group_array.size, "Invalid grouped values")
    unique_groups = np.unique(group_array)
    _require(unique_groups.size >= 2, "Grouped bootstrap requires at least two groups")
    indices = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected])
        samples.append(float(np.mean(array[selected_indices])))
    return {
        "iterations": iterations,
        "seed": seed,
        "group_count": int(unique_groups.size),
        "lower_95": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.5)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _mean(flags: Iterable[bool]) -> float:
    values = list(flags)
    return float(sum(values) / len(values)) if values else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    _json_safe(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def evaluate_qwen_predictions(
    instruction_root: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    expected_instruction_report_sha256: str,
    bootstrap_iterations: int = 1000,
    seed: int = 17,
) -> QwenEvaluationResult:
    """Evaluate a complete prediction file without exposing answers during generation."""

    instruction_root = instruction_root.resolve()
    predictions_path = predictions_path.resolve()
    output_dir = output_dir.resolve()
    _require(instruction_root.is_dir(), f"Instruction root not found: {instruction_root}")
    _require(predictions_path.is_file(), f"Predictions not found: {predictions_path}")
    _require(not output_dir.exists(), f"Evaluation output already exists: {output_dir}")
    _require(bootstrap_iterations >= 2, "Bootstrap iterations must be at least two")
    _require(
        bool(_HEX_64.fullmatch(expected_instruction_report_sha256)),
        "Expected instruction report SHA-256 must be lowercase hexadecimal",
    )

    report_path = instruction_root / "instruction_dataset_report.json"
    instruction_report = _read_json(report_path, "instruction report")
    instruction_report_sha256 = sha256_file(report_path)
    _require(
        instruction_report_sha256 == expected_instruction_report_sha256,
        "Instruction report hash mismatch",
    )
    _require(
        instruction_report.get("schema_version") == DATASET_SCHEMA,
        "Unsupported instruction report schema",
    )
    _require(
        instruction_report.get("internal_test_accessed") is False,
        "Instruction report must not access internal test data",
    )
    _, prompts = _artifact(instruction_root, instruction_report, "validation_prompts")
    _, answers = _artifact(instruction_root, instruction_report, "validation_answers")
    _, full_manifest = _artifact(instruction_root, instruction_report, "instruction_manifest")
    validation_manifest = [
        record for record in full_manifest if record.get("split") == "validation"
    ]
    _require(
        all(record.get("split") in {"train", "validation"} for record in full_manifest),
        "Instruction manifest exposes an unsupported split",
    )

    prompt_by_id = _unique_by_id(prompts, "validation prompts")
    answer_by_id = _unique_by_id(answers, "validation answers")
    manifest_by_id = _unique_by_id(validation_manifest, "validation manifest")
    expected_ids = set(prompt_by_id)
    _require(
        expected_ids == set(answer_by_id) == set(manifest_by_id),
        "Validation artifacts disagree",
    )
    split_counts = _object(
        _object(instruction_report.get("counts"), "instruction counts").get("by_split"),
        "instruction split counts",
    )
    validation_counts = _object(split_counts.get("validation"), "validation counts")
    expected_count = validation_counts.get("instructions")
    _require(expected_count == len(expected_ids), "Validation report count mismatch")

    for instruction_id in expected_ids:
        prompt = prompt_by_id[instruction_id]
        answer = answer_by_id[instruction_id]
        manifest = manifest_by_id[instruction_id]
        task = manifest.get("task")
        _require(task in TASKS, f"Unsupported validation task: {task}")
        _require(prompt.get("schema_version") == VALIDATION_PROMPT_SCHEMA, "Invalid prompt schema")
        _require(answer.get("schema_version") == VALIDATION_ANSWER_SCHEMA, "Invalid answer schema")
        _require(prompt.get("task") == answer.get("task") == task, "Validation task mismatch")
        _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "Invalid manifest schema")
        _require(
            isinstance(manifest.get("asset_id"), str) and bool(manifest["asset_id"]),
            "Validation manifest asset_id is missing",
        )
        _require(
            isinstance(manifest.get("group_id"), str) and bool(manifest["group_id"]),
            "Validation manifest group_id is missing",
        )
        _require(
            manifest.get("source_dataset_report_sha256")
            == _object(instruction_report.get("source_dataset"), "source dataset").get(
                "dataset_report_sha256"
            ),
            "Validation manifest source Dataset hash mismatch",
        )
        expected_response = answer.get("expected_response")
        _require(isinstance(expected_response, str), "Validation answer must be a string")
        _require(
            hashlib.sha256(expected_response.encode("utf-8")).hexdigest()
            == manifest.get("response_sha256"),
            "Validation answer hash mismatch",
        )
        _require("expected_response" not in prompt, "Validation prompt leaks an answer")

    predictions = _read_jsonl(predictions_path, "Qwen predictions")
    prediction_by_id = _unique_by_id(predictions, "Qwen predictions")
    _require(
        set(prediction_by_id) == expected_ids,
        "Prediction IDs do not match validation prompts",
    )
    for prediction in predictions:
        _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "Invalid prediction schema")
        _require(
            isinstance(prediction.get("response"), str),
            "Prediction response must be a string",
        )

    evidence = []
    task_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    ordered_ids = [record["instruction_id"] for record in prompts]
    for instruction_id in ordered_ids:
        prompt = prompt_by_id[instruction_id]
        answer = answer_by_id[instruction_id]
        manifest = manifest_by_id[instruction_id]
        prediction = prediction_by_id[instruction_id]
        task = str(manifest["task"])
        expected, expected_json = _strict_json_object(answer["expected_response"])
        _require(expected_json and expected is not None, "Builder answer is not a JSON object")
        parsed, valid_json = _strict_json_object(prediction["response"])
        row: dict[str, Any] = {
            "schema_version": EVALUATION_RECORD_SCHEMA,
            "instruction_id": instruction_id,
            "task": task,
            "asset_id": manifest.get("asset_id"),
            "group_id": manifest.get("group_id"),
            "valid_json": valid_json,
            "schema_valid": False,
            "exact_match": False,
        }

        if task in {"peak_presence", "peak_presence_metadata"}:
            _require(
                set(expected) == {"peak_present"}
                and isinstance(expected["peak_present"], bool),
                "Invalid presence answer",
            )
            schema_valid = (
                parsed is not None
                and set(parsed) == {"peak_present"}
                and isinstance(parsed["peak_present"], bool)
            )
            truth = bool(expected["peak_present"])
            predicted = bool(parsed["peak_present"]) if schema_valid else None
            row.update(
                {
                    "schema_valid": schema_valid,
                    "exact_match": schema_valid and predicted == truth,
                    "target_peak_present": truth,
                    "predicted_peak_present": predicted,
                    "classification_score": float(predicted) if schema_valid else float(not truth),
                }
            )
        elif task == "peak_grounding":
            width, height = _grounding_dimensions(prompt)
            expected_box = _valid_bbox(expected.get("bbox_2d"), width, height)
            _require(
                set(expected) == {"bbox_2d"} and expected_box is not None,
                "Invalid grounding answer",
            )
            predicted_box = (
                _valid_bbox(parsed.get("bbox_2d"), width, height)
                if parsed is not None and set(parsed) == {"bbox_2d"}
                else None
            )
            schema_valid = predicted_box is not None
            iou = _bbox_iou(expected_box, predicted_box) if predicted_box is not None else 0.0
            row.update(
                {
                    "schema_valid": schema_valid,
                    "exact_match": schema_valid and predicted_box == expected_box,
                    "image_width": width,
                    "image_height": height,
                    "expected_bbox_2d": expected_box,
                    "predicted_bbox_2d": predicted_box,
                    "bbox_iou": iou,
                    "iou_at_0_5": iou >= 0.5,
                    "x_boundary_absolute_error_pixels": (
                        [
                            abs(predicted_box[0] - expected_box[0]),
                            abs(predicted_box[2] - expected_box[2]),
                        ]
                        if predicted_box is not None
                        else None
                    ),
                    "full_height": (
                        predicted_box is not None
                        and predicted_box[1] == 0.0
                        and predicted_box[3] == float(height)
                    ),
                }
            )
        else:
            _require(
                set(expected) == {"qc_state", "reason"}
                and all(isinstance(expected[key], str) for key in expected),
                "Invalid scientific QC answer",
            )
            schema_valid = (
                parsed is not None
                and set(parsed) == {"qc_state", "reason"}
                and all(isinstance(parsed[key], str) and bool(parsed[key]) for key in parsed)
            )
            row.update(
                {
                    "schema_valid": schema_valid,
                    "exact_match": schema_valid and parsed == expected,
                    "qc_state_correct": schema_valid and parsed["qc_state"] == expected["qc_state"],
                    "reason_correct": schema_valid and parsed["reason"] == expected["reason"],
                }
            )
        evidence.append(row)
        task_rows[task].append(row)

    metrics: dict[str, Any] = {}
    for task, rows in task_rows.items():
        if not rows:
            continue
        groups = [str(row["group_id"]) for row in rows]
        common = {
            "instructions": len(rows),
            "source_groups": len(set(groups)),
            "valid_json_rate": _mean(bool(row["valid_json"]) for row in rows),
            "schema_valid_rate": _mean(bool(row["schema_valid"]) for row in rows),
            "exact_match_rate": _mean(bool(row["exact_match"]) for row in rows),
        }
        if task in {"peak_presence", "peak_presence_metadata"}:
            labels = [int(bool(row["target_peak_present"])) for row in rows]
            scores = [float(row["classification_score"]) for row in rows]
            common["classification"] = binary_metrics(labels, scores, threshold=0.5)
            if len(set(groups)) >= 2 and len(set(labels)) == 2:
                common["grouped_bootstrap"] = grouped_bootstrap_binary(
                    labels,
                    scores,
                    groups,
                    threshold=0.5,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
        elif task == "peak_grounding":
            ious = [float(row["bbox_iou"]) for row in rows]
            valid_rows = [row for row in rows if row["schema_valid"]]
            boundary_errors = [
                error
                for row in valid_rows
                for error in row["x_boundary_absolute_error_pixels"]
            ]
            normalized_errors = [
                error / float(row["image_width"])
                for row in valid_rows
                for error in row["x_boundary_absolute_error_pixels"]
            ]
            common["grounding"] = {
                "mean_bbox_iou_all": float(np.mean(ious)),
                "iou_at_0_5_rate_all": _mean(bool(row["iou_at_0_5"]) for row in rows),
                "mean_bbox_iou_schema_valid": (
                    float(np.mean([float(row["bbox_iou"]) for row in valid_rows]))
                    if valid_rows
                    else None
                ),
                "x_boundary_mae_pixels_schema_valid": (
                    float(np.mean(boundary_errors)) if boundary_errors else None
                ),
                "x_boundary_mae_normalized_schema_valid": (
                    float(np.mean(normalized_errors)) if normalized_errors else None
                ),
                "full_height_rate_schema_valid": (
                    _mean(bool(row["full_height"]) for row in valid_rows)
                    if valid_rows
                    else None
                ),
            }
            if len(set(groups)) >= 2:
                common["grouped_bootstrap"] = _grouped_mean_bootstrap(
                    ious,
                    groups,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
        else:
            exact = [float(bool(row["exact_match"])) for row in rows]
            common["scientific_qc"] = {
                "qc_state_accuracy_all": _mean(bool(row["qc_state_correct"]) for row in rows),
                "reason_accuracy_all": _mean(bool(row["reason_correct"]) for row in rows),
            }
            if len(set(groups)) >= 2:
                common["grouped_bootstrap"] = _grouped_mean_bootstrap(
                    exact,
                    groups,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
        metrics[task] = common

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        evidence_path = staging / "evaluation_records.jsonl"
        _write_jsonl(evidence_path, evidence)
        final_report_path = staging / "qwen_evaluation_report.json"
        source_groups = {str(row["group_id"]) for row in evidence}
        evaluation_report = {
            "schema_version": EVALUATION_REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "instruction_report_sha256": instruction_report_sha256,
                "source_dataset_report_sha256": _object(
                    instruction_report.get("source_dataset"), "source dataset"
                ).get("dataset_report_sha256"),
                "validation_prompts_sha256": _object(
                    _object(instruction_report["artifacts"], "artifacts")["validation_prompts"],
                    "validation prompts artifact",
                )["sha256"],
                "validation_answers_sha256": _object(
                    _object(instruction_report["artifacts"], "artifacts")["validation_answers"],
                    "validation answers artifact",
                )["sha256"],
                "instruction_manifest_sha256": _object(
                    _object(instruction_report["artifacts"], "artifacts")["instruction_manifest"],
                    "instruction manifest artifact",
                )["sha256"],
                "predictions_sha256": sha256_file(predictions_path),
            },
            "counts": {
                "predictions": len(evidence),
                "valid_json": sum(bool(row["valid_json"]) for row in evidence),
                "schema_valid": sum(bool(row["schema_valid"]) for row in evidence),
                "validation_source_groups": len(source_groups),
                "by_task": dict(sorted(Counter(row["task"] for row in evidence).items())),
            },
            "metrics": metrics,
            "evaluation": {
                "bootstrap_iterations": bootstrap_iterations,
                "seed": seed,
                "invalid_or_schema_invalid_predictions_are_scored_as_failures": True,
                "combined_cross_task_score_reported": False,
            },
            "artifacts": {
                "evaluation_records": {
                    "path": evidence_path.name,
                    "sha256": sha256_file(evidence_path),
                    "records": len(evidence),
                }
            },
            "answer_key_file_separate_from_prompts": True,
            "prediction_generation_provenance_verified": False,
            "internal_test_accessed": False,
            "development_comparison_eligible": False,
            "final_benchmark_eligible": False,
        }
        _write_json(final_report_path, evaluation_report)
        staging.replace(output_dir)

    final_report_path = output_dir / "qwen_evaluation_report.json"
    return QwenEvaluationResult(
        output_dir=output_dir,
        report_path=final_report_path,
        report_sha256=sha256_file(final_report_path),
        prediction_records=len(evidence),
        valid_json_records=sum(bool(row["valid_json"]) for row in evidence),
        schema_valid_records=sum(bool(row["schema_valid"]) for row in evidence),
        validation_source_groups=len({str(row["group_id"]) for row in evidence}),
    )
