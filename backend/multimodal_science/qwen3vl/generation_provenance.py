"""Independent structural verification of Qwen3-VL generation provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.inference import (
    GENERATION_CONFIG_SCHEMA,
    GENERATION_RECORD_SCHEMA,
    GENERATION_REPORT_SCHEMA,
)

_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_40_TO_64 = re.compile(r"^[0-9a-f]{40,64}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
PREDICTION_SCHEMA = "chrompeak-qwen3vl-prediction-v1"


@dataclass(frozen=True)
class VerifiedGenerationProvenance:
    report_path: Path
    report_sha256: str
    prediction_records: int
    generation_records: int
    backend: str
    model_identity_immutable: bool
    development_comparison_eligible: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON for {context}: {path}") from error


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {context} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{context} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {context} JSON at line {line_number}") from error
    _require(bool(records), f"{context} is empty")
    return records


def _safe_artifact(
    root: Path,
    artifact: Any,
    context: str,
    *,
    expected_records: int | None = None,
) -> Path:
    artifact_object = _object(artifact, context)
    relative = artifact_object.get("path")
    _require(isinstance(relative, str) and bool(relative), f"Invalid path for {context}")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Artifact path must be relative: {context}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact path escapes generation root: {context}") from error
    expected_hash = artifact_object.get("sha256")
    _require(
        isinstance(expected_hash, str) and bool(_HEX_64.fullmatch(expected_hash)),
        f"Invalid artifact SHA-256: {context}",
    )
    _require(path.is_file(), f"Missing generation artifact: {context}")
    _require(sha256_file(path) == expected_hash, f"Artifact hash mismatch: {context}")
    if expected_records is not None:
        _require(
            artifact_object.get("records") == expected_records,
            f"Artifact record count mismatch: {context}",
        )
    return path


def _version_tuple(value: Any) -> tuple[int, ...]:
    _require(isinstance(value, str) and bool(value), "Missing Transformers version")
    numbers = re.match(r"^(\d+(?:\.\d+)+)", value)
    _require(numbers is not None, "Invalid Transformers version")
    return tuple(int(part) for part in numbers.group(1).split("."))


def verify_generation_provenance(
    generation_report_path: Path,
    predictions_path: Path,
    *,
    expected_generation_report_sha256: str,
    instruction_report_sha256: str,
    validation_prompts_sha256: str,
    source_dataset_report_sha256: str,
    expected_prediction_records: int,
    expected_records_by_id: dict[str, dict[str, Any]],
) -> VerifiedGenerationProvenance:
    """Verify full prompt-only generation evidence before development comparison."""

    generation_report_path = generation_report_path.resolve()
    predictions_path = predictions_path.resolve()
    _require(
        bool(_HEX_64.fullmatch(expected_generation_report_sha256)),
        "Expected generation report SHA-256 must be lowercase hexadecimal",
    )
    _require(
        sha256_file(generation_report_path) == expected_generation_report_sha256,
        "Generation report hash mismatch",
    )
    report = _read_json(generation_report_path, "generation report")
    _require(report.get("schema_version") == GENERATION_REPORT_SCHEMA, "Invalid report schema")
    _require(report.get("internal_test_accessed") is False, "Generation accessed internal test")
    _require(report.get("final_benchmark_eligible") is False, "Invalid final benchmark claim")

    source = _object(report.get("source"), "generation source")
    _require(
        source.get("instruction_report_sha256") == instruction_report_sha256,
        "Generation instruction report hash mismatch",
    )
    _require(
        source.get("validation_prompts_sha256") == validation_prompts_sha256,
        "Generation validation prompt hash mismatch",
    )
    _require(
        source.get("source_dataset_report_sha256")
        == source_dataset_report_sha256,
        "Generation source dataset hash mismatch",
    )
    _require(
        isinstance(source.get("bundle_report_sha256"), str)
        and bool(_HEX_64.fullmatch(source["bundle_report_sha256"])),
        "Generation bundle report hash is invalid",
    )

    contracts = _object(report.get("contracts"), "generation contracts")
    required_contracts = {
        "input_is_prompt_only_bundle": True,
        "answer_key_available_to_runner": False,
        "answer_key_opened": False,
        "instruction_manifest_available_to_runner": False,
        "internal_test_accessed": False,
        "predictions_preserve_prompt_order": True,
    }
    for name, expected in required_contracts.items():
        _require(contracts.get(name) is expected, f"Generation contract failed: {name}")

    scope = _object(report.get("scope"), "generation scope")
    _require(scope.get("complete_prompt_coverage") is True, "Generation is sample-capped")
    _require(scope.get("max_records") is None, "Generation declares a sample cap")
    _require(
        scope.get("bundle_prompts")
        == scope.get("selected_prompts")
        == expected_prediction_records,
        "Generation scope does not match validation prompts",
    )
    counts = _object(report.get("counts"), "generation counts")
    _require(
        counts.get("predictions") == expected_prediction_records,
        "Generation prediction count mismatch",
    )

    root = generation_report_path.parent
    artifacts = _object(report.get("artifacts"), "generation artifacts")
    reported_predictions = _safe_artifact(
        root,
        artifacts.get("predictions"),
        "predictions",
        expected_records=expected_prediction_records,
    )
    _require(
        reported_predictions == predictions_path,
        "Evaluator predictions are not the generation-report artifact",
    )
    generation_records_path = _safe_artifact(
        root,
        artifacts.get("generation_records"),
        "generation records",
        expected_records=expected_prediction_records,
    )
    run_config_path = _safe_artifact(root, artifacts.get("run_config"), "run config")
    runtime_path = _safe_artifact(
        root,
        artifacts.get("runtime_metadata"),
        "runtime metadata",
    )

    run_config = _read_json(run_config_path, "generation run config")
    _require(
        run_config.get("schema_version") == GENERATION_CONFIG_SCHEMA,
        "Invalid generation config schema",
    )
    _require(
        run_config.get("bundle_report_sha256") == source["bundle_report_sha256"],
        "Generation config bundle hash mismatch",
    )
    _require(
        run_config.get("prompt_artifact_sha256")
        == source.get("prompt_artifact_sha256"),
        "Generation config prompt hash mismatch",
    )
    _require(run_config.get("scope") == scope, "Generation config scope mismatch")
    _require(
        run_config.get("generation") == report.get("generation"),
        "Generation settings mismatch",
    )
    config_contracts = _object(run_config.get("contracts"), "generation config contracts")
    for name in (
        "input_is_prompt_only_bundle",
        "instruction_root_not_accepted",
        "answer_key_path_not_accepted",
    ):
        _require(config_contracts.get(name) is True, f"Run config contract failed: {name}")
    _require(
        config_contracts.get("internal_test_accessed") is False,
        "Run config accessed internal test",
    )

    runtime = _read_json(runtime_path, "generation runtime metadata")
    _require(runtime == report.get("runtime"), "Generation runtime metadata mismatch")
    backend = runtime.get("backend")
    _require(backend == "transformers", "Development comparison requires Transformers")
    _require(
        _version_tuple(runtime.get("transformers_version")) >= (4, 57, 0),
        "Development comparison requires transformers>=4.57.0",
    )
    model = _object(report.get("model"), "generation model")
    config_model = _object(run_config.get("model"), "generation config model")
    _require(model.get("name_or_path") == config_model.get("name_or_path"), "Model mismatch")
    _require(
        model.get("requested_revision") == config_model.get("requested_revision"),
        "Requested model revision mismatch",
    )
    _require(
        model.get("artifact_sha256") == config_model.get("artifact_sha256"),
        "Model artifact hash mismatch",
    )
    resolved_revision = model.get("resolved_revision")
    artifact_hash = model.get("artifact_sha256")
    immutable = bool(
        (isinstance(resolved_revision, str) and _HEX_40_TO_64.fullmatch(resolved_revision))
        or (isinstance(artifact_hash, str) and _HEX_64.fullmatch(artifact_hash))
    )
    _require(immutable and model.get("identity_immutable") is True, "Model is not immutable")
    _require(
        report.get("development_comparison_candidate") is True,
        "Generation report is not a development comparison candidate",
    )

    predictions = _read_jsonl(predictions_path, "generation predictions")
    generation_records = _read_jsonl(
        generation_records_path,
        "generation records",
    )
    _require(
        len(predictions) == len(generation_records) == expected_prediction_records,
        "Generation evidence count mismatch",
    )
    _require(
        set(expected_records_by_id) == {
            str(prediction.get("instruction_id")) for prediction in predictions
        },
        "Generation evidence IDs do not match validation prompts",
    )
    seen_ids = set()
    forbidden = {"expected_response", "target", "label", "target_peak_present"}
    for prediction, record in zip(predictions, generation_records):
        instruction_id = prediction.get("instruction_id")
        _require(
            prediction.get("schema_version") == PREDICTION_SCHEMA
            and isinstance(instruction_id, str)
            and bool(_HEX_24.fullmatch(instruction_id)),
            "Invalid generation prediction",
        )
        _require(instruction_id not in seen_ids, "Duplicate generation prediction ID")
        seen_ids.add(instruction_id)
        response = prediction.get("response")
        _require(isinstance(response, str), "Generation prediction response is invalid")
        _require(
            record.get("schema_version") == GENERATION_RECORD_SCHEMA,
            "Invalid generation record schema",
        )
        _require(record.get("instruction_id") == instruction_id, "Generation order mismatch")
        _require(not (set(record) & forbidden), "Generation record contains answer evidence")
        _require(record.get("response") == response, "Generation response mismatch")
        _require(
            record.get("response_sha256")
            == hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "Generation response hash mismatch",
        )
        expected_record = _object(
            expected_records_by_id[instruction_id],
            f"expected generation record {instruction_id}",
        )
        for name in (
            "task",
            "image",
            "image_sha256",
            "prompt_sha256",
            "language",
            "pair_id",
        ):
            _require(
                record.get(name) == expected_record.get(name),
                f"Generation record {name} mismatch: {instruction_id}",
            )
        for name in ("image_sha256", "prompt_sha256"):
            value = record.get(name)
            _require(
                isinstance(value, str) and bool(_HEX_64.fullmatch(value)),
                f"Invalid generation record hash: {name}",
            )

    return VerifiedGenerationProvenance(
        report_path=generation_report_path,
        report_sha256=expected_generation_report_sha256,
        prediction_records=len(predictions),
        generation_records=len(generation_records),
        backend=str(backend),
        model_identity_immutable=immutable,
        development_comparison_eligible=True,
    )
