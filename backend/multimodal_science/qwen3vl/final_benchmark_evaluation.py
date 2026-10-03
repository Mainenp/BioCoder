"""Evaluate sealed-test Qwen predictions under the frozen metric protocol."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import (
    _bbox_iou,
    _cross_language_consistency,
    _grounding_dimensions,
    _metrics_for_tasks,
    _strict_json_object,
    _valid_bbox,
    _write_json,
    _write_jsonl,
)
from multimodal_science.qwen3vl.final_benchmark_data import (
    FINAL_ANSWER_RECORD_SCHEMA,
    FINAL_ANSWER_REPORT_SCHEMA,
    FINAL_INFERENCE_BUNDLE_SCHEMA,
    FINAL_INSTRUCTION_MANIFEST_SCHEMA,
)
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FINAL_BENCHMARK_PROTOCOL_SCHEMA,
    verify_final_benchmark_access,
)
from multimodal_science.qwen3vl.inference import (
    BUNDLE_PROMPT_SCHEMA,
    GENERATION_CONFIG_SCHEMA,
    GENERATION_RECORD_SCHEMA,
    GENERATION_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
)
from multimodal_science.qwen3vl.instruction_data import LANGUAGES, TASKS


FINAL_QWEN_EVALUATION_SCHEMA = "chrompeak-qwen3vl-final-evaluation-v1"
FINAL_QWEN_EVALUATION_RECORD_SCHEMA = "chrompeak-qwen3vl-final-evaluation-record-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_40_TO_64 = re.compile(r"^[0-9a-f]{40,64}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FinalQwenEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    prediction_records: int
    valid_json_records: int
    schema_valid_records: int
    source_groups: int
    candidate_name: str
    access_id: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {label} line: {line_number}")
            try:
                rows.append(_object(json.loads(line), f"{label} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {label} line: {line_number}") from error
    _require(bool(rows), f"{label} is empty")
    return rows


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative != "", f"Invalid {label} path")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute() and str(relative_path) != ".", f"Unsafe {label}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    return path


def _artifact(
    root: Path, report: dict[str, Any], name: str, label: str
) -> tuple[Path, dict[str, Any]]:
    descriptor = _object(_object(report.get("artifacts"), f"{label} artifacts").get(name), name)
    path = _safe_child(root, descriptor.get("path"), f"{label} {name}")
    digest = descriptor.get("sha256")
    _require(isinstance(digest, str) and bool(_HEX_64.fullmatch(digest)), "Bad digest")
    _require(path.is_file() and sha256_file(path) == digest, f"{label} artifact drift: {name}")
    return path, descriptor


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        instruction_id = row.get("instruction_id")
        _require(
            isinstance(instruction_id, str) and bool(_HEX_24.fullmatch(instruction_id)),
            f"Invalid instruction ID in {label}",
        )
        _require(instruction_id not in result, f"Duplicate instruction ID in {label}")
        result[instruction_id] = row
    return result


def _protocol_context(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    candidate_name: str,
) -> tuple[str, int, int]:
    state = verify_final_benchmark_access(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    _require(not state.completed, "Final benchmark access is already completed")
    protocol = _read_json(
        protocol_root / "final_benchmark_protocol.json", "final benchmark protocol"
    )
    _require(
        protocol.get("schema_version") == FINAL_BENCHMARK_PROTOCOL_SCHEMA,
        "Unsupported final benchmark protocol",
    )
    locks = _object(protocol.get("locks"), "protocol locks")
    candidate_descriptor = _object(locks.get("candidate"), "candidate lock")
    candidate_path = _safe_child(
        protocol_root, candidate_descriptor.get("path"), "candidate lock"
    )
    _require(
        sha256_file(candidate_path) == candidate_descriptor.get("sha256"),
        "Candidate lock drift",
    )
    candidates = _object(
        _read_json(candidate_path, "candidate lock").get("models"), "candidate models"
    )
    _require(candidate_name in candidates, f"Candidate is not frozen: {candidate_name}")
    metrics_descriptor = _object(locks.get("metrics"), "metrics lock")
    metrics_path = _safe_child(protocol_root, metrics_descriptor.get("path"), "metrics lock")
    _require(
        sha256_file(metrics_path) == metrics_descriptor.get("sha256"),
        "Metrics lock drift",
    )
    metrics = _read_json(metrics_path, "metrics lock")
    uncertainty = _object(metrics.get("uncertainty"), "frozen uncertainty")
    iterations = uncertainty.get("bootstrap_iterations")
    seed = uncertainty.get("seed")
    _require(isinstance(iterations, int) and iterations >= 10_000, "Bad bootstrap count")
    _require(isinstance(seed, int), "Bad bootstrap seed")
    _require(metrics.get("post_access_model_selection_forbidden") is True, "Selection unlocked")
    _require(metrics.get("post_access_threshold_tuning_forbidden") is True, "Tuning unlocked")
    return state.access_id, iterations, seed


def _load_evaluation_inputs(
    *,
    inference_root: Path,
    expected_inference_report_sha256: str,
    answer_root: Path,
    expected_answer_report_sha256: str,
    protocol_sha256: str,
    access_id: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str],
    str,
    str,
]:
    inference_report_path = inference_root / "inference_bundle_report.json"
    answer_report_path = answer_root / "answer_key_report.json"
    _require(
        sha256_file(inference_report_path) == expected_inference_report_sha256,
        "Inference report drift",
    )
    _require(
        sha256_file(answer_report_path) == expected_answer_report_sha256,
        "Answer report drift",
    )
    inference = _read_json(inference_report_path, "final inference bundle")
    answers_report = _read_json(answer_report_path, "final answer report")
    _require(
        inference.get("schema_version") == FINAL_INFERENCE_BUNDLE_SCHEMA,
        "Bad final inference schema",
    )
    _require(
        answers_report.get("schema_version") == FINAL_ANSWER_REPORT_SCHEMA,
        "Bad final answer schema",
    )
    for report, label in ((inference, "inference"), (answers_report, "answer")):
        source = _object(report.get("source"), f"{label} source")
        _require(source.get("protocol_sha256") == protocol_sha256, f"{label} protocol drift")
        _require(source.get("access_id") == access_id, f"{label} access drift")
        _require(report.get("internal_test_accessed") is True, f"{label} is not test-bound")
    inference_source = _object(inference.get("source"), "inference source")
    answer_source = _object(answers_report.get("source"), "answer source")
    dataset_sha256 = str(inference_source.get("dataset_report_sha256") or "")
    _require(
        bool(_HEX_64.fullmatch(dataset_sha256))
        and answer_source.get("dataset_report_sha256") == dataset_sha256,
        "Prompt/answer Dataset drift",
    )
    prompt_path, prompt_descriptor = _artifact(
        inference_root, inference, "inference_prompts", "inference"
    )
    answer_path, answer_descriptor = _artifact(answer_root, answers_report, "answers", "answer")
    manifest_path, manifest_descriptor = _artifact(
        answer_root, answers_report, "instruction_manifest", "answer"
    )
    prompts = _read_jsonl(prompt_path, "final prompts")
    answers = _read_jsonl(answer_path, "final answers")
    manifest = _read_jsonl(manifest_path, "final instruction manifest")
    _require(prompt_descriptor.get("records") == len(prompts), "Prompt count drift")
    _require(answer_descriptor.get("records") == len(answers), "Answer count drift")
    _require(manifest_descriptor.get("records") == len(manifest), "Manifest count drift")
    prompt_by_id = _unique(prompts, "prompts")
    answer_by_id = _unique(answers, "answers")
    manifest_by_id = _unique(manifest, "manifest")
    _require(
        set(prompt_by_id) == set(answer_by_id) == set(manifest_by_id),
        "Final prompt/answer/manifest IDs disagree",
    )
    ordered_ids = [str(row["instruction_id"]) for row in prompts]
    for instruction_id in ordered_ids:
        prompt = prompt_by_id[instruction_id]
        answer = answer_by_id[instruction_id]
        row = manifest_by_id[instruction_id]
        _require(prompt.get("schema_version") == BUNDLE_PROMPT_SCHEMA, "Bad prompt schema")
        _require(answer.get("schema_version") == FINAL_ANSWER_RECORD_SCHEMA, "Bad answer schema")
        _require(
            row.get("schema_version") == FINAL_INSTRUCTION_MANIFEST_SCHEMA,
            "Bad final manifest schema",
        )
        _require(row.get("split") == "internal_test", "Manifest is not internal-test-only")
        for name in ("task", "pair_id", "language"):
            _require(
                prompt.get(name) == answer.get(name) == row.get(name),
                f"Final artifact mismatch: {name}",
            )
        _require(prompt.get("image") == row.get("image_path"), "Prompt image drift")
        expected_response = answer.get("expected_response")
        _require(isinstance(expected_response, str), "Final answer is not a string")
        _require(
            hashlib.sha256(expected_response.encode("utf-8")).hexdigest()
            == row.get("response_sha256"),
            "Final answer digest drift",
        )
        _require("expected_response" not in prompt, "Final prompt leaks its answer")
    return (
        prompt_by_id,
        answer_by_id,
        manifest_by_id,
        ordered_ids,
        dataset_sha256,
        str(prompt_descriptor["sha256"]),
    )


def _verify_generation(
    *,
    generation_report_path: Path,
    expected_generation_report_sha256: str,
    predictions_path: Path,
    prompt_by_id: dict[str, dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    ordered_ids: list[str],
    protocol_sha256: str,
    access_id: str,
    candidate_name: str,
    dataset_report_sha256: str,
    prompt_artifact_sha256: str,
) -> dict[str, Any]:
    _require(
        sha256_file(generation_report_path) == expected_generation_report_sha256,
        "Generation report drift",
    )
    report = _read_json(generation_report_path, "final generation report")
    _require(report.get("schema_version") == GENERATION_REPORT_SCHEMA, "Bad generation schema")
    _require(report.get("internal_test_accessed") is True, "Generation is not test-bound")
    _require(report.get("final_benchmark_candidate") is True, "Generation is not final")
    _require(report.get("development_comparison_candidate") is False, "Generation is dev-only")
    source = _object(report.get("source"), "generation source")
    _require(source.get("protocol_sha256") == protocol_sha256, "Generation protocol drift")
    _require(source.get("access_id") == access_id, "Generation access drift")
    _require(
        source.get("source_dataset_report_sha256") == dataset_report_sha256,
        "Generation Dataset drift",
    )
    _require(
        source.get("prompt_artifact_sha256") == prompt_artifact_sha256,
        "Generation prompt drift",
    )
    scope = _object(report.get("scope"), "generation scope")
    _require(scope.get("complete_prompt_coverage") is True, "Generation is sample-capped")
    _require(scope.get("max_records") is None, "Generation declares a record cap")
    _require(
        scope.get("bundle_prompts") == scope.get("selected_prompts") == len(ordered_ids),
        "Generation prompt count drift",
    )
    contracts = _object(report.get("contracts"), "generation contracts")
    for name, expected in (
        ("input_is_prompt_only_bundle", True),
        ("answer_key_available_to_runner", False),
        ("answer_key_opened", False),
        ("instruction_manifest_available_to_runner", False),
        ("internal_test_accessed", True),
        ("predictions_preserve_prompt_order", True),
    ):
        _require(contracts.get(name) is expected, f"Generation contract failed: {name}")
    final_access = _object(contracts.get("final_benchmark_access"), "generation access")
    _require(final_access.get("candidate_name") == candidate_name, "Candidate drift")
    _require(final_access.get("access_id") == access_id, "Generation access ID drift")
    _require(final_access.get("protocol_sha256") == protocol_sha256, "Protocol drift")

    root = generation_report_path.parent
    prediction_artifact, prediction_descriptor = _artifact(
        root, report, "predictions", "generation"
    )
    _require(prediction_artifact == predictions_path, "Predictions are not report-bound")
    generation_path, generation_descriptor = _artifact(
        root, report, "generation_records", "generation"
    )
    run_config_path, _ = _artifact(root, report, "run_config", "generation")
    runtime_path, _ = _artifact(root, report, "runtime_metadata", "generation")
    _require(prediction_descriptor.get("records") == len(ordered_ids), "Prediction count drift")
    _require(generation_descriptor.get("records") == len(ordered_ids), "Journal count drift")
    run_config = _read_json(run_config_path, "generation config")
    _require(run_config.get("schema_version") == GENERATION_CONFIG_SCHEMA, "Bad config schema")
    config_contracts = _object(run_config.get("contracts"), "config contracts")
    _require(config_contracts.get("final_benchmark_access") == final_access, "Config access drift")
    _require(config_contracts.get("internal_test_accessed") is True, "Config is not test-bound")
    runtime = _read_json(runtime_path, "runtime metadata")
    _require(runtime == report.get("runtime"), "Runtime metadata drift")
    _require(runtime.get("backend") == "transformers", "Final generation is not Transformers")
    model = _object(report.get("model"), "generation model")
    resolved = model.get("resolved_revision")
    artifact_hash = model.get("artifact_sha256")
    immutable = bool(
        isinstance(artifact_hash, str)
        and _HEX_64.fullmatch(artifact_hash)
        or isinstance(resolved, str)
        and _HEX_40_TO_64.fullmatch(resolved)
    )
    _require(immutable and model.get("identity_immutable") is True, "Model is mutable")

    predictions = _read_jsonl(predictions_path, "final predictions")
    generation_records = _read_jsonl(generation_path, "final generation records")
    _require(
        len(predictions) == len(generation_records) == len(ordered_ids),
        "Evidence count drift",
    )
    for instruction_id, prediction, record in zip(
        ordered_ids, predictions, generation_records
    ):
        prompt = prompt_by_id[instruction_id]
        manifest = manifest_by_id[instruction_id]
        _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "Bad prediction schema")
        _require(prediction.get("instruction_id") == instruction_id, "Prediction order drift")
        response = prediction.get("response")
        _require(isinstance(response, str), "Prediction response is not a string")
        _require(record.get("schema_version") == GENERATION_RECORD_SCHEMA, "Bad journal schema")
        _require(record.get("instruction_id") == instruction_id, "Journal order drift")
        _require(record.get("response") == response, "Prediction/journal response drift")
        _require(record.get("task") == manifest.get("task"), "Journal task drift")
        _require(record.get("image") == manifest.get("image_path"), "Journal image drift")
        _require(record.get("image_sha256") == manifest.get("image_sha256"), "Image drift")
        _require(record.get("language") == manifest.get("language"), "Language drift")
        _require(record.get("pair_id") == manifest.get("pair_id"), "Pair drift")
        _require(
            record.get("prompt_sha256")
            == hashlib.sha256(str(prompt["prompt"]).encode("utf-8")).hexdigest(),
            "Journal prompt drift",
        )
        _require(
            record.get("response_sha256")
            == hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "Journal response digest drift",
        )
    return {
        "report_sha256": expected_generation_report_sha256,
        "backend": runtime["backend"],
        "model_identity_immutable": True,
        "candidate_name": candidate_name,
    }


def _score_row(
    *,
    instruction_id: str,
    prompt: dict[str, Any],
    answer: dict[str, Any],
    manifest: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    task = str(manifest["task"])
    expected, expected_json = _strict_json_object(answer["expected_response"])
    _require(expected_json and expected is not None, "Answer key is not a JSON object")
    parsed, valid_json = _strict_json_object(prediction["response"])
    row: dict[str, Any] = {
        "schema_version": FINAL_QWEN_EVALUATION_RECORD_SCHEMA,
        "instruction_id": instruction_id,
        "task": task,
        "asset_id": manifest["asset_id"],
        "group_id": manifest["group_id"],
        "language": manifest["language"],
        "pair_id": manifest["pair_id"],
        "valid_json": valid_json,
        "schema_valid": False,
        "exact_match": False,
    }
    if task in {"peak_presence", "peak_presence_metadata"}:
        _require(
            set(expected) == {"peak_present"} and isinstance(expected["peak_present"], bool),
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
        _require(set(expected) == {"bbox_2d"} and expected_box is not None, "Bad bbox answer")
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
            "Invalid QC answer",
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
                "predicted_qc_state": parsed["qc_state"] if schema_valid else None,
                "predicted_reason": parsed["reason"] if schema_valid else None,
            }
        )
    return row


def evaluate_final_qwen_predictions(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    candidate_name: str,
    inference_root: Path,
    expected_inference_report_sha256: str,
    answer_root: Path,
    expected_answer_report_sha256: str,
    predictions_path: Path,
    generation_report_path: Path,
    expected_generation_report_sha256: str,
    output_dir: Path,
) -> FinalQwenEvaluationResult:
    """Score one frozen Qwen candidate without permitting metric overrides."""

    protocol_root = protocol_root.resolve()
    ledger_dir = ledger_dir.resolve()
    inference_root = inference_root.resolve()
    answer_root = answer_root.resolve()
    predictions_path = predictions_path.resolve()
    generation_report_path = generation_report_path.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Final evaluation already exists: {output_dir}")
    access_id, bootstrap_iterations, bootstrap_seed = _protocol_context(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
        candidate_name=candidate_name,
    )
    (
        prompt_by_id,
        answer_by_id,
        manifest_by_id,
        ordered_ids,
        dataset_report_sha256,
        prompt_artifact_sha256,
    ) = _load_evaluation_inputs(
        inference_root=inference_root,
        expected_inference_report_sha256=expected_inference_report_sha256,
        answer_root=answer_root,
        expected_answer_report_sha256=expected_answer_report_sha256,
        protocol_sha256=expected_protocol_sha256,
        access_id=access_id,
    )
    generation = _verify_generation(
        generation_report_path=generation_report_path,
        expected_generation_report_sha256=expected_generation_report_sha256,
        predictions_path=predictions_path,
        prompt_by_id=prompt_by_id,
        manifest_by_id=manifest_by_id,
        ordered_ids=ordered_ids,
        protocol_sha256=expected_protocol_sha256,
        access_id=access_id,
        candidate_name=candidate_name,
        dataset_report_sha256=dataset_report_sha256,
        prompt_artifact_sha256=prompt_artifact_sha256,
    )
    predictions = _read_jsonl(predictions_path, "final predictions")
    prediction_by_id = _unique(predictions, "predictions")
    _require(set(prediction_by_id) == set(ordered_ids), "Prediction IDs do not match prompts")
    for prediction in predictions:
        _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "Bad prediction schema")
        _require(isinstance(prediction.get("response"), str), "Bad prediction response")

    evidence = []
    task_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for instruction_id in ordered_ids:
        row = _score_row(
            instruction_id=instruction_id,
            prompt=prompt_by_id[instruction_id],
            answer=answer_by_id[instruction_id],
            manifest=manifest_by_id[instruction_id],
            prediction=prediction_by_id[instruction_id],
        )
        evidence.append(row)
        task_rows[str(row["task"])].append(row)

    metrics = _metrics_for_tasks(
        task_rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    metrics_by_language = {}
    for language in LANGUAGES:
        metrics_by_language[language] = _metrics_for_tasks(
            {
                task: [row for row in rows if row["language"] == language]
                for task, rows in task_rows.items()
            },
            bootstrap_iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    cross_language = _cross_language_consistency(evidence)
    source_groups = {str(row["group_id"]) for row in evidence}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        evidence_path = staging / "evaluation_records.jsonl"
        _write_jsonl(evidence_path, evidence)
        report_path = staging / "qwen_evaluation_report.json"
        _write_json(
            report_path,
            {
                "schema_version": FINAL_QWEN_EVALUATION_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "candidate_name": candidate_name,
                "inputs": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": dataset_report_sha256,
                    "inference_report_sha256": expected_inference_report_sha256,
                    "answer_report_sha256": expected_answer_report_sha256,
                    "generation_report_sha256": expected_generation_report_sha256,
                    "predictions_sha256": sha256_file(predictions_path),
                    "prompt_artifact_sha256": prompt_artifact_sha256,
                },
                "counts": {
                    "predictions": len(evidence),
                    "valid_json": sum(bool(row["valid_json"]) for row in evidence),
                    "schema_valid": sum(bool(row["schema_valid"]) for row in evidence),
                    "internal_test_source_groups": len(source_groups),
                    "independent_internal_test_assets": len(
                        {str(row["asset_id"]) for row in evidence}
                    ),
                    "semantic_language_pairs": len(
                        {str(row["pair_id"]) for row in evidence}
                    ),
                    "by_task": dict(
                        sorted(Counter(str(row["task"]) for row in evidence).items())
                    ),
                    "by_language": dict(
                        sorted(Counter(str(row["language"]) for row in evidence).items())
                    ),
                },
                "metrics": metrics,
                "metrics_by_language": metrics_by_language,
                "cross_language_consistency": cross_language,
                "evaluation": {
                    "bootstrap_iterations": bootstrap_iterations,
                    "seed": bootstrap_seed,
                    "bootstrap_unit": "complete_source_mzml_group",
                    "invalid_or_schema_invalid_predictions_are_scored_as_failures": True,
                    "combined_cross_task_score_reported": False,
                    "threshold_selection_performed": False,
                },
                "generation_provenance": generation,
                "artifacts": {
                    "evaluation_records": {
                        "path": evidence_path.name,
                        "sha256": sha256_file(evidence_path),
                        "records": len(evidence),
                    }
                },
                "answer_key_file_separate_from_generation_root": True,
                "prediction_generation_provenance_verified": True,
                "development_comparison_eligible": False,
                "internal_test_accessed": True,
                "final_benchmark_eligible": True,
            },
        )
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (evidence_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    final_report = output_dir / "qwen_evaluation_report.json"
    return FinalQwenEvaluationResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        manifest_path=output_dir / "artifact_manifest.sha256",
        prediction_records=len(evidence),
        valid_json_records=sum(bool(row["valid_json"]) for row in evidence),
        schema_valid_records=sum(bool(row["schema_valid"]) for row in evidence),
        source_groups=len(source_groups),
        candidate_name=candidate_name,
        access_id=access_id,
    )
