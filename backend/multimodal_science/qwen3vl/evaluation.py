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
from multimodal_science.qwen3vl.generation_provenance import (
    verify_generation_provenance,
)
from multimodal_science.qwen3vl.instruction_data import (
    BILINGUAL_DATASET_SCHEMA,
    BILINGUAL_MANIFEST_SCHEMA,
    BILINGUAL_VALIDATION_ANSWER_SCHEMA,
    BILINGUAL_VALIDATION_PROMPT_SCHEMA,
    DATASET_SCHEMA,
    LANGUAGES,
    MANIFEST_SCHEMA,
    TASKS,
    VALIDATION_ANSWER_SCHEMA,
    VALIDATION_PROMPT_SCHEMA,
)

PREDICTION_SCHEMA = "chrompeak-qwen3vl-prediction-v1"
EVALUATION_RECORD_SCHEMA = "chrompeak-qwen3vl-evaluation-record-v1"
EVALUATION_REPORT_SCHEMA = "chrompeak-qwen3vl-evaluation-v1"
BILINGUAL_EVALUATION_RECORD_SCHEMA = "chrompeak-qwen3vl-evaluation-record-v2"
BILINGUAL_EVALUATION_REPORT_SCHEMA = "chrompeak-qwen3vl-evaluation-v2"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GROUNDING_SIZE = re.compile(r"for a ([1-9][0-9]*)x([1-9][0-9]*) image")
_SUPPORTED_DATASET_SCHEMAS = {DATASET_SCHEMA, BILINGUAL_DATASET_SCHEMA}


@dataclass(frozen=True)
class QwenEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    prediction_records: int
    valid_json_records: int
    schema_valid_records: int
    validation_source_groups: int
    generation_provenance_verified: bool
    development_comparison_eligible: bool


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


def _grounding_dimensions(
    prompt: dict[str, Any], manifest: dict[str, Any]
) -> tuple[int, int]:
    width = manifest.get("image_width")
    height = manifest.get("image_height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    ):
        return width, height
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


def _metrics_for_tasks(
    task_rows: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
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
                "qc_state_accuracy_all": _mean(
                    bool(row["qc_state_correct"]) for row in rows
                ),
                "reason_accuracy_all": _mean(
                    bool(row["reason_correct"]) for row in rows
                ),
            }
            if len(set(groups)) >= 2:
                common["grouped_bootstrap"] = _grouped_mean_bootstrap(
                    exact,
                    groups,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
        metrics[task] = common
    return metrics


def _prediction_signature(row: dict[str, Any]) -> Any:
    if not row["schema_valid"]:
        return None
    if row["task"] in {"peak_presence", "peak_presence_metadata"}:
        return row["predicted_peak_present"]
    if row["task"] == "peak_grounding":
        return tuple(row["predicted_bbox_2d"])
    return row["predicted_qc_state"], row["predicted_reason"]


def _cross_language_consistency(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    pair_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evidence:
        key = str(row["task"]), str(row["pair_id"])
        pair_rows.setdefault(key, []).append(row)
    by_task: dict[str, list[list[dict[str, Any]]]] = {}
    for (task, _), rows in pair_rows.items():
        _require(len(rows) == 2, "Evaluation evidence contains an incomplete language pair")
        _require(
            {row["language"] for row in rows} == set(LANGUAGES),
            "Evaluation evidence language pair is invalid",
        )
        by_task.setdefault(task, []).append(rows)

    result = {}
    for task, pairs in sorted(by_task.items()):
        both_valid = [all(bool(row["schema_valid"]) for row in pair) for pair in pairs]
        exact = [
            valid and _prediction_signature(pair[0]) == _prediction_signature(pair[1])
            for valid, pair in zip(both_valid, pairs)
        ]
        task_result: dict[str, Any] = {
            "pairs": len(pairs),
            "both_schema_valid_rate": _mean(both_valid),
            "exact_prediction_consistency_rate_all": _mean(exact),
        }
        if task == "peak_grounding":
            pair_ious = [
                _bbox_iou(pair[0]["predicted_bbox_2d"], pair[1]["predicted_bbox_2d"])
                if valid
                else 0.0
                for valid, pair in zip(both_valid, pairs)
            ]
            task_result.update(
                {
                    "mean_prediction_bbox_iou_all": float(np.mean(pair_ious)),
                    "prediction_bbox_iou_at_0_9_rate_all": _mean(
                        value >= 0.9 for value in pair_ious
                    ),
                }
            )
        result[task] = task_result
    return result


def evaluate_qwen_predictions(
    instruction_root: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    expected_instruction_report_sha256: str,
    bootstrap_iterations: int = 1000,
    seed: int = 17,
    generation_report_path: Path | None = None,
    expected_generation_report_sha256: str | None = None,
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
    _require(
        (generation_report_path is None) == (expected_generation_report_sha256 is None),
        "Generation report path and SHA-256 must be provided together",
    )

    report_path = instruction_root / "instruction_dataset_report.json"
    instruction_report = _read_json(report_path, "instruction report")
    instruction_report_sha256 = sha256_file(report_path)
    _require(
        instruction_report_sha256 == expected_instruction_report_sha256,
        "Instruction report hash mismatch",
    )
    instruction_schema = instruction_report.get("schema_version")
    _require(
        instruction_schema in _SUPPORTED_DATASET_SCHEMAS,
        "Unsupported instruction report schema",
    )
    bilingual = instruction_schema == BILINGUAL_DATASET_SCHEMA
    prompt_schema = (
        BILINGUAL_VALIDATION_PROMPT_SCHEMA if bilingual else VALIDATION_PROMPT_SCHEMA
    )
    answer_schema = (
        BILINGUAL_VALIDATION_ANSWER_SCHEMA if bilingual else VALIDATION_ANSWER_SCHEMA
    )
    manifest_schema = BILINGUAL_MANIFEST_SCHEMA if bilingual else MANIFEST_SCHEMA
    if bilingual:
        contracts = _object(instruction_report.get("contracts"), "instruction contracts")
        _require(contracts.get("language_profile") == "bilingual", "Invalid language profile")
        _require(
            contracts.get("languages") == list(LANGUAGES),
            "Unsupported bilingual language set",
        )
        _require(
            contracts.get("validation_has_parallel_language_pairs") is True,
            "Bilingual validation pairing contract is missing",
        )
        _require(
            contracts.get("language_variants_are_not_independent_source_assets") is True,
            "Bilingual source-asset counting contract is missing",
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
        _require(prompt.get("schema_version") == prompt_schema, "Invalid prompt schema")
        _require(answer.get("schema_version") == answer_schema, "Invalid answer schema")
        _require(prompt.get("task") == answer.get("task") == task, "Validation task mismatch")
        _require(manifest.get("schema_version") == manifest_schema, "Invalid manifest schema")
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
        if bilingual:
            language = manifest.get("language")
            pair_id = manifest.get("pair_id")
            _require(language in LANGUAGES, "Invalid validation language")
            _require(
                isinstance(pair_id, str) and bool(_HEX_24.fullmatch(pair_id)),
                "Invalid validation pair_id",
            )
            _require(
                prompt.get("language") == answer.get("language") == language,
                "Validation language mismatch",
            )
            _require(
                prompt.get("pair_id") == answer.get("pair_id") == pair_id,
                "Validation pair mismatch",
            )

    if bilingual:
        pair_members: dict[str, list[dict[str, Any]]] = {}
        for manifest in validation_manifest:
            pair_members.setdefault(str(manifest["pair_id"]), []).append(manifest)
        for pair in pair_members.values():
            _require(len(pair) == 2, "Bilingual validation pair must contain two records")
            _require(
                {record["language"] for record in pair} == set(LANGUAGES),
                "Bilingual validation pair is missing a language",
            )
            for name in (
                "task",
                "asset_id",
                "record_id",
                "group_id",
                "image_path",
                "image_sha256",
                "response_sha256",
            ):
                _require(
                    len({record.get(name) for record in pair}) == 1,
                    f"Bilingual validation pair disagrees on {name}",
                )
        declared_languages = _object(
            validation_counts.get("by_language"), "validation language counts"
        )
        actual_languages = Counter(record["language"] for record in validation_manifest)
        _require(
            dict(sorted(actual_languages.items())) == declared_languages,
            "Validation language count mismatch",
        )

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

    generation_verification = None
    if generation_report_path is not None:
        expected_generation_records = {}
        for instruction_id in expected_ids:
            prompt = prompt_by_id[instruction_id]
            manifest = manifest_by_id[instruction_id]
            message = _object(prompt["conversations"][0], "validation prompt message")
            expected_generation_records[instruction_id] = {
                "task": manifest["task"],
                "image": manifest["image_path"],
                "image_sha256": manifest["image_sha256"],
                "prompt_sha256": hashlib.sha256(
                    str(message["value"]).encode("utf-8")
                ).hexdigest(),
                "language": manifest.get("language"),
                "pair_id": manifest.get("pair_id"),
            }
        generation_verification = verify_generation_provenance(
            generation_report_path,
            predictions_path,
            expected_generation_report_sha256=str(
                expected_generation_report_sha256
            ),
            instruction_report_sha256=instruction_report_sha256,
            validation_prompts_sha256=_object(
                _object(instruction_report["artifacts"], "artifacts")[
                    "validation_prompts"
                ],
                "validation prompts artifact",
            )["sha256"],
            source_dataset_report_sha256=_object(
                instruction_report.get("source_dataset"), "source dataset"
            )["dataset_report_sha256"],
            expected_prediction_records=len(expected_ids),
            expected_records_by_id=expected_generation_records,
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
            "schema_version": (
                BILINGUAL_EVALUATION_RECORD_SCHEMA
                if bilingual
                else EVALUATION_RECORD_SCHEMA
            ),
            "instruction_id": instruction_id,
            "task": task,
            "asset_id": manifest.get("asset_id"),
            "group_id": manifest.get("group_id"),
            "valid_json": valid_json,
            "schema_valid": False,
            "exact_match": False,
        }
        if bilingual:
            row["language"] = manifest["language"]
            row["pair_id"] = manifest["pair_id"]

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
            width, height = _grounding_dimensions(prompt, manifest)
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
            if bilingual:
                row["predicted_qc_state"] = parsed["qc_state"] if schema_valid else None
                row["predicted_reason"] = parsed["reason"] if schema_valid else None
        evidence.append(row)
        task_rows[task].append(row)

    metrics = _metrics_for_tasks(
        task_rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    metrics_by_language = {}
    cross_language_consistency = {}
    if bilingual:
        for language in LANGUAGES:
            rows_for_language = {
                task: [row for row in rows if row["language"] == language]
                for task, rows in task_rows.items()
            }
            metrics_by_language[language] = _metrics_for_tasks(
                rows_for_language,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        cross_language_consistency = _cross_language_consistency(evidence)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        evidence_path = staging / "evaluation_records.jsonl"
        _write_jsonl(evidence_path, evidence)
        final_report_path = staging / "qwen_evaluation_report.json"
        source_groups = {str(row["group_id"]) for row in evidence}
        report_counts: dict[str, Any] = {
            "predictions": len(evidence),
            "valid_json": sum(bool(row["valid_json"]) for row in evidence),
            "schema_valid": sum(bool(row["schema_valid"]) for row in evidence),
            "validation_source_groups": len(source_groups),
            "by_task": dict(sorted(Counter(row["task"] for row in evidence).items())),
        }
        if bilingual:
            report_counts["by_language"] = dict(
                sorted(Counter(row["language"] for row in evidence).items())
            )
            report_counts["independent_validation_assets"] = len(
                {str(row["asset_id"]) for row in evidence}
            )
            report_counts["semantic_language_pairs"] = len(
                {str(row["pair_id"]) for row in evidence}
            )
        evaluation_report = {
            "schema_version": (
                BILINGUAL_EVALUATION_REPORT_SCHEMA
                if bilingual
                else EVALUATION_REPORT_SCHEMA
            ),
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
            "counts": report_counts,
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
        if generation_verification is not None:
            evaluation_report["inputs"]["generation_report_sha256"] = (
                generation_verification.report_sha256
            )
            evaluation_report["generation_provenance"] = {
                "report_sha256": generation_verification.report_sha256,
                "backend": generation_verification.backend,
                "prediction_records": generation_verification.prediction_records,
                "generation_records": generation_verification.generation_records,
                "model_identity_immutable": (
                    generation_verification.model_identity_immutable
                ),
            }
            evaluation_report["prediction_generation_provenance_verified"] = True
            evaluation_report["development_comparison_eligible"] = (
                generation_verification.development_comparison_eligible
            )
        if bilingual:
            evaluation_report["evaluation"].update(
                {
                    "primary_bilingual_metrics": "metrics_by_language",
                    "combined_metrics_are_prompt_weighted": True,
                    "invalid_output_makes_language_pair_inconsistent": True,
                }
            )
            evaluation_report.update(
                {
                    "metrics_by_language": metrics_by_language,
                    "cross_language_consistency": cross_language_consistency,
                    "language_variants_are_not_independent_source_assets": True,
                }
            )
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
        generation_provenance_verified=generation_verification is not None,
        development_comparison_eligible=(
            generation_verification.development_comparison_eligible
            if generation_verification is not None
            else False
        ),
    )
