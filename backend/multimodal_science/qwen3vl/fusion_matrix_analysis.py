"""Provenance-bound development analysis for the formal fusion matrix."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA


REPORT_SCHEMA = "chrompeak-qwen3vl-xic-fusion-matrix-analysis-v1"
LANGUAGES = ("en", "zh-CN")
SCOPES = ("overall", *LANGUAGES)
MATRIX_CONFIGURATIONS = {
    "primary-seed17": (4, 17),
    "primary-seed29": (4, 29),
    "primary-seed43": (4, 43),
    "tokens1-seed17": (1, 17),
    "tokens8-seed17": (8, 17),
}
PRIMARY_LABELS = ("primary-seed17", "primary-seed29", "primary-seed43")
TOKEN_LABELS = ("tokens1-seed17", "primary-seed17", "tokens8-seed17")
CLASSIFICATION_TASKS = ("peak_presence", "peak_presence_metadata")
CLASSIFICATION_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "false_positive_rate",
)
GROUNDING_METRICS = ("mean_bbox_iou_all", "iou_at_0_5_rate_all")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FusionMatrixRun:
    training_root: Path
    generation_root: Path
    evaluation_root: Path


@dataclass(frozen=True)
class FusionMatrixAnalysisResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class _LoadedRun:
    label: str
    sensor_tokens: int
    training_seed: int
    training_root: Path
    generation_root: Path
    evaluation_root: Path
    training: dict[str, Any]
    generation: dict[str, Any]
    evaluation: dict[str, Any]
    training_sha256: str
    generation_sha256: str
    evaluation_sha256: str
    training_manifest_sha256: str
    generation_manifest_sha256: str
    evaluation_manifest_sha256: str
    evaluation_records_sha256: str
    evaluation_records: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), label)


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            rows.append(_object(json.loads(line), f"{label} line {line_number}"))
    _require(rows, f"{label} is empty")
    return tuple(rows)


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    _require(
        value
        and not pure.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"Unsafe {label}: {value}",
    )
    return Path(*pure.parts)


def _resolve_artifact(root: Path, relative: Path, label: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its artifact root") from error
    _require(path.is_file(), f"Missing {label}: {path}")
    return path


def _verify_manifest(root: Path, required: set[str], label: str) -> str:
    manifest_path = root / "artifact_manifest.sha256"
    _require(manifest_path.is_file(), f"Missing {label} manifest")
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed {label} manifest line {line_number}")
        digest, relative_text = parts
        relative_text = relative_text.lstrip(" *")
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad {label} manifest digest")
        relative = _safe_relative(relative_text, f"{label} manifest path")
        relative_posix = relative.as_posix()
        _require(relative_posix not in listed, f"Duplicate {label} manifest path")
        artifact = _resolve_artifact(
            root, relative, f"{label} artifact {relative_posix}"
        )
        _require(
            sha256_file(artifact) == digest,
            f"{label} artifact hash mismatch: {relative_posix}",
        )
        listed.add(relative_posix)
    _require(required <= listed, f"{label} manifest lacks {sorted(required - listed)}")
    return sha256_file(manifest_path)


def _scope_metrics(report: dict[str, Any], scope: str) -> dict[str, Any]:
    if scope == "overall":
        tasks = _object(report.get("metrics"), "overall evaluation metrics")
    else:
        tasks = _object(
            _object(report.get("metrics_by_language"), "metrics by language").get(scope),
            f"{scope} evaluation metrics",
        )
    result: dict[str, Any] = {}
    for task in CLASSIFICATION_TASKS:
        classification = _object(
            _object(tasks.get(task), f"{scope} {task}").get("classification"),
            f"{scope} {task} classification",
        )
        result[task] = {
            metric: _number(
                classification.get(metric), f"{scope} {task} {metric}"
            )
            for metric in CLASSIFICATION_METRICS
        }
    grounding = _object(
        _object(tasks.get("peak_grounding"), f"{scope} grounding").get("grounding"),
        f"{scope} grounding metrics",
    )
    result["peak_grounding"] = {
        metric: _number(grounding.get(metric), f"{scope} grounding {metric}")
        for metric in GROUNDING_METRICS
    }
    qc = _object(tasks.get("scientific_qc"), f"{scope} scientific QC")
    result["scientific_qc"] = {
        "exact_match_rate": _number(
            qc.get("exact_match_rate"), f"{scope} scientific QC exact match"
        )
    }
    return result


def _reported_selected_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {scope: _scope_metrics(report, scope) for scope in SCOPES}


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_from_counts(counts: Iterable[float]) -> dict[str, float]:
    tp, fp, tn, fn = (float(value) for value in counts)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    positive_f1 = _divide(2.0 * tp, 2.0 * tp + fp + fn)
    negative_f1 = _divide(2.0 * tn, 2.0 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "balanced_accuracy": (recall + specificity) / 2.0,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "mcc": _divide(tp * tn - fp * fn, denominator),
        "false_positive_rate": _divide(fp, fp + tn),
    }


def _recomputed_metrics(
    rows: Iterable[dict[str, Any]], scope: str
) -> dict[str, dict[str, float]]:
    scoped = list(rows) if scope == "overall" else [
        row for row in rows if row.get("language") == scope
    ]
    _require(scoped, f"No evaluation rows for scope {scope}")
    result: dict[str, dict[str, float]] = {}
    for task in CLASSIFICATION_TASKS:
        counts = [0.0] * 4
        task_rows = [row for row in scoped if row.get("task") == task]
        _require(task_rows, f"No {task} rows for scope {scope}")
        for row in task_rows:
            truth = bool(row.get("target_peak_present"))
            predicted = _number(
                row.get("classification_score"),
                f"{scope} {task} classification score",
            ) >= 0.5
            if truth and predicted:
                counts[0] += 1.0
            elif not truth and predicted:
                counts[1] += 1.0
            elif not truth and not predicted:
                counts[2] += 1.0
            else:
                counts[3] += 1.0
        result[task] = _classification_from_counts(counts)

    grounding = [row for row in scoped if row.get("task") == "peak_grounding"]
    _require(grounding, f"No peak_grounding rows for scope {scope}")
    result["peak_grounding"] = {
        "mean_bbox_iou_all": statistics.fmean(
            _number(row.get("bbox_iou"), f"{scope} grounding IoU")
            for row in grounding
        ),
        "iou_at_0_5_rate_all": statistics.fmean(
            float(bool(row.get("iou_at_0_5"))) for row in grounding
        ),
    }
    qc = [row for row in scoped if row.get("task") == "scientific_qc"]
    _require(qc, f"No scientific_qc rows for scope {scope}")
    result["scientific_qc"] = {
        "exact_match_rate": statistics.fmean(
            float(bool(row.get("exact_match"))) for row in qc
        )
    }
    return result


def _assert_metrics_match(
    recomputed: dict[str, dict[str, float]],
    reported: dict[str, dict[str, float]],
    label: str,
) -> None:
    for task, values in recomputed.items():
        for metric, value in values.items():
            _require(
                math.isclose(
                    value,
                    reported[task][metric],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{label} recomputed {task}/{metric} does not match its report",
            )


def _record_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("instruction_id"),
        row.get("pair_id"),
        row.get("asset_id"),
        row.get("group_id"),
        row.get("task"),
        row.get("language"),
        row.get("target_peak_present"),
        tuple(row.get("expected_bbox_2d") or ()),
    )


def _load_run(
    label: str,
    run: FusionMatrixRun,
    *,
    expected_tokens: int,
    expected_seed: int,
) -> _LoadedRun:
    training_root = run.training_root.resolve()
    generation_root = run.generation_root.resolve()
    evaluation_root = run.evaluation_root.resolve()
    training_path = training_root / "fusion_training_report.json"
    generation_path = generation_root / "generation_report.json"
    evaluation_path = evaluation_root / "qwen_evaluation_report.json"

    training_manifest_sha256 = _verify_manifest(
        training_root,
        {
            "fusion_training_report.json",
            "adapter/adapter_model.safetensors",
            "sensor_projector.safetensors",
        },
        f"{label} training",
    )
    generation_manifest_sha256 = _verify_manifest(
        generation_root,
        {"generation_report.json", "predictions.jsonl"},
        f"{label} generation",
    )
    evaluation_manifest_sha256 = _verify_manifest(
        evaluation_root,
        {"qwen_evaluation_report.json", "evaluation_records.jsonl"},
        f"{label} evaluation",
    )

    training = _read_object(training_path, f"{label} training report")
    generation = _read_object(generation_path, f"{label} generation report")
    evaluation = _read_object(evaluation_path, f"{label} evaluation report")
    training_sha256 = sha256_file(training_path)
    generation_sha256 = sha256_file(generation_path)
    evaluation_sha256 = sha256_file(evaluation_path)

    _require(
        training.get("schema_version") == FUSION_TRAINING_REPORT_SCHEMA,
        f"Unexpected {label} training schema",
    )
    training_settings = _object(training.get("training"), f"{label} training settings")
    projector = _object(
        _object(training.get("model"), f"{label} training model").get(
            "sensor_projector"
        ),
        f"{label} sensor projector",
    )
    gate = _object(
        _object(training.get("model"), f"{label} training model").get("sensor_gate"),
        f"{label} sensor gate",
    )
    initial_gate = _number(
        gate.get("initial_probability"), f"{label} initial gate probability"
    )
    final_gate = _number(
        gate.get("final_probability"), f"{label} final gate probability"
    )
    gate_change = _number(
        gate.get("absolute_probability_change"), f"{label} gate probability change"
    )
    _require(
        0.0 < initial_gate < 1.0
        and 0.0 < final_gate < 1.0
        and math.isclose(
            gate_change,
            final_gate - initial_gate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{label} sensor-gate audit is inconsistent",
    )
    _require(training_settings.get("seed") == expected_seed, f"{label} seed mismatch")
    _require(
        projector.get("sensor_tokens") == expected_tokens,
        f"{label} sensor-token mismatch",
    )
    _require(
        training_settings.get("optimizer_updates") == 3396,
        f"{label} optimizer-update count mismatch",
    )
    _require(
        training_settings.get("training_records") == 54335
        and training_settings.get("epochs") == 1
        and training_settings.get("batch_size") == 1
        and training_settings.get("gradient_accumulation_steps") == 16,
        f"{label} formal training protocol mismatch",
    )
    training_contracts = _object(
        training.get("contracts"), f"{label} training contracts"
    )
    for contract_name, expected in (
        ("train_split_only", True),
        ("validation_prompts_opened", False),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
        ("image_and_xic_forward", True),
        ("lora_and_projector_backward", True),
        ("parameter_updates_verified", True),
    ):
        _require(
            training_contracts.get(contract_name) is expected,
            f"{label} training contract failed: {contract_name}",
        )
    _require(
        training.get("development_training_complete") is True
        and training.get("development_comparison_eligible") is False
        and training.get("final_benchmark_eligible") is False
        and training.get("internal_test_accessed") is False,
        f"{label} training scope is invalid",
    )

    _require(
        generation.get("schema_version") == GENERATION_REPORT_SCHEMA,
        f"Unexpected {label} generation schema",
    )
    generation_model = _object(
        generation.get("model"), f"{label} generation model"
    )
    _require(
        generation_model.get("identity_immutable") is True,
        f"{label} base-model identity is mutable",
    )
    _require(
        generation_model.get("artifact_sha256")
        == _object(training.get("model"), f"{label} training model").get(
            "base_artifact_sha256"
        ),
        f"{label} generation base model differs from training",
    )
    adapter = _object(
        generation_model.get("adapter"),
        f"{label} generation adapter",
    )
    runtime = _object(generation.get("runtime"), f"{label} generation runtime")
    _require(
        runtime.get("backend") == "transformers"
        and runtime.get("manual_cached_greedy_decode") is True
        and runtime.get("explicit_fused_mrope_positions") is True,
        f"{label} fused inference runtime contract failed",
    )
    intervention = _object(runtime.get("xic_intervention"), f"{label} intervention")
    runtime_projector = _object(
        runtime.get("sensor_projector"), f"{label} runtime sensor projector"
    )
    runtime_gate = _object(runtime.get("sensor_gate"), f"{label} runtime sensor gate")
    _require(
        adapter.get("training_report_sha256") == training_sha256,
        f"{label} generation is not bound to its training report",
    )
    _require(
        adapter.get("manifest_sha256") == training_manifest_sha256,
        f"{label} generation is not bound to its training manifest",
    )
    _require(
        adapter.get("code_revision") == training.get("code_revision")
        and adapter.get("development_training_complete") is True
        and adapter.get("sensor_projector") == projector,
        f"{label} generation adapter metadata drift",
    )
    _require(
        runtime_projector == projector
        and runtime_projector.get("sensor_tokens") == expected_tokens,
        f"{label} runtime sensor-token mismatch",
    )
    _require(intervention.get("mode") == "aligned", f"{label} is not aligned")
    _require(intervention.get("answer_key_used") is False, f"{label} used answers")
    _require(
        adapter.get("xic_intervention") == intervention,
        f"{label} adapter does not bind the aligned intervention",
    )
    _require(
        math.isclose(
            _number(runtime_gate.get("probability"), f"{label} runtime gate"),
            _number(gate.get("final_probability"), f"{label} final gate"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{label} runtime gate differs from training",
    )
    _require(
        generation.get("development_comparison_candidate") is True
        and generation.get("final_benchmark_eligible") is False
        and generation.get("internal_test_accessed") is False,
        f"{label} generation scope is invalid",
    )
    generation_settings = _object(
        generation.get("generation"), f"{label} generation settings"
    )
    _require(
        generation_settings.get("batch_size") == 1
        and generation_settings.get("do_sample") is False
        and generation_settings.get("seed") == 17,
        f"{label} generation protocol drift",
    )

    _require(
        evaluation.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        f"Unexpected {label} evaluation schema",
    )
    inputs = _object(evaluation.get("inputs"), f"{label} evaluation inputs")
    counts = _object(evaluation.get("counts"), f"{label} evaluation counts")
    _require(
        inputs.get("generation_report_sha256") == generation_sha256,
        f"{label} evaluation is not bound to its generation report",
    )
    provenance = _object(
        evaluation.get("generation_provenance"),
        f"{label} generation provenance",
    )
    _require(
        provenance.get("report_sha256") == generation_sha256
        and evaluation.get("prediction_generation_provenance_verified") is True,
        f"{label} generation provenance is unverified",
    )
    _require(
        evaluation.get("development_comparison_eligible") is True
        and evaluation.get("final_benchmark_eligible") is False
        and evaluation.get("internal_test_accessed") is False,
        f"{label} evaluation scope is invalid",
    )
    _require(
        evaluation.get("language_variants_are_not_independent_source_assets") is True,
        f"{label} does not protect bilingual pairing",
    )
    _require(
        evaluation.get("answer_key_file_separate_from_prompts") is True,
        f"{label} answer-separation contract failed",
    )
    _require(counts.get("predictions") == 13708, f"{label} prediction count mismatch")
    for count_name in ("valid_json", "schema_valid"):
        value = counts.get(count_name)
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 13708,
            f"{label} {count_name} count is invalid",
        )
    _require(
        counts.get("independent_validation_assets") == 1815,
        f"{label} validation-asset count mismatch",
    )
    _require(
        counts.get("validation_source_groups") == 11,
        f"{label} validation source-group count mismatch",
    )
    by_language = _object(counts.get("by_language"), f"{label} language counts")
    _require(
        by_language == {"en": 6854, "zh-CN": 6854},
        f"{label} bilingual coverage mismatch",
    )

    generation_source = _object(
        generation.get("source"), f"{label} generation source"
    )
    for generation_field, evaluation_field in (
        ("instruction_report_sha256", "instruction_report_sha256"),
        ("validation_prompts_sha256", "validation_prompts_sha256"),
        ("source_dataset_report_sha256", "source_dataset_report_sha256"),
    ):
        _require(
            generation_source.get(generation_field) == inputs.get(evaluation_field),
            f"{label} generation/evaluation {generation_field} drift",
        )
    generation_counts = _object(
        generation.get("counts"), f"{label} generation counts"
    )
    _require(
        generation_counts.get("predictions") == counts.get("predictions"),
        f"{label} generation/evaluation count mismatch",
    )
    generation_artifacts = _object(
        generation.get("artifacts"), f"{label} generation artifacts"
    )
    predictions_artifact = _object(
        generation_artifacts.get("predictions"), f"{label} predictions artifact"
    )
    predictions_path = _resolve_artifact(
        generation_root,
        _safe_relative(
            str(predictions_artifact.get("path") or ""),
            f"{label} predictions path",
        ),
        f"{label} predictions",
    )
    predictions_sha256 = sha256_file(predictions_path)
    _require(
        predictions_sha256 == predictions_artifact.get("sha256")
        and predictions_sha256 == inputs.get("predictions_sha256")
        and predictions_artifact.get("records") == counts.get("predictions"),
        f"{label} prediction artifact provenance mismatch",
    )

    artifacts = _object(evaluation.get("artifacts"), f"{label} evaluation artifacts")
    records_artifact = _object(
        artifacts.get("evaluation_records"), f"{label} evaluation-record artifact"
    )
    records_relative = _safe_relative(
        str(records_artifact.get("path") or ""),
        f"{label} evaluation-record path",
    )
    records_path = _resolve_artifact(
        evaluation_root, records_relative, f"{label} evaluation records"
    )
    records_sha256 = sha256_file(records_path)
    _require(
        records_sha256 == records_artifact.get("sha256"),
        f"{label} evaluation-record hash mismatch",
    )
    records = _read_jsonl(records_path, f"{label} evaluation records")
    _require(
        len(records) == counts.get("predictions")
        and records_artifact.get("records") == len(records),
        f"{label} evaluation-record count mismatch",
    )
    _require(
        len({str(row.get("instruction_id")) for row in records}) == len(records),
        f"{label} instruction IDs are not unique",
    )
    _require(
        len({str(row.get("asset_id")) for row in records}) == 1815
        and len({str(row.get("group_id")) for row in records}) == 11,
        f"{label} evaluation-record scientific-unit count mismatch",
    )
    record_language_counts = {
        language: sum(row.get("language") == language for row in records)
        for language in LANGUAGES
    }
    _require(
        record_language_counts == by_language,
        f"{label} evaluation-record language count mismatch",
    )
    reported_metrics = _reported_selected_metrics(evaluation)
    for scope in SCOPES:
        _assert_metrics_match(
            _recomputed_metrics(records, scope),
            reported_metrics[scope],
            f"{label}/{scope}",
        )

    return _LoadedRun(
        label=label,
        sensor_tokens=expected_tokens,
        training_seed=expected_seed,
        training_root=training_root,
        generation_root=generation_root,
        evaluation_root=evaluation_root,
        training=training,
        generation=generation,
        evaluation=evaluation,
        training_sha256=training_sha256,
        generation_sha256=generation_sha256,
        evaluation_sha256=evaluation_sha256,
        training_manifest_sha256=training_manifest_sha256,
        generation_manifest_sha256=generation_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        evaluation_records_sha256=records_sha256,
        evaluation_records=records,
        metrics=reported_metrics,
    )


def _shared_value(loaded: dict[str, _LoadedRun], getter: Any, label: str) -> Any:
    values = {name: getter(run) for name, run in loaded.items()}
    first = next(iter(values.values()))
    _require(all(value == first for value in values.values()), f"Shared {label} drift: {values}")
    return first


def _metric_statistics(values_by_seed: dict[str, float]) -> dict[str, Any]:
    values = list(values_by_seed.values())
    _require(len(values) == 3, "Primary seed aggregate requires three runs")
    return {
        "values_by_seed": values_by_seed,
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "standard_deviation_ddof": 1,
        "minimum": min(values),
        "maximum": max(values),
    }


def _primary_aggregate(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in SCOPES:
        scope_result: dict[str, Any] = {}
        for task in (*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc"):
            metric_names = loaded[PRIMARY_LABELS[0]].metrics[scope][task]
            scope_result[task] = {
                metric: _metric_statistics(
                    {
                        str(loaded[label].training_seed): loaded[label].metrics[scope][task][
                            metric
                        ]
                        for label in PRIMARY_LABELS
                    }
                )
                for metric in metric_names
            }
        result[scope] = scope_result
    return result


def _language_gap(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in (*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc"):
        result[task] = {
            metric: metrics["en"][task][metric] - metrics["zh-CN"][task][metric]
            for metric in metrics["en"][task]
        }
    return result


def _primary_language_gap(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    gaps = {label: _language_gap(loaded[label].metrics) for label in PRIMARY_LABELS}
    result: dict[str, Any] = {}
    for task in (*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc"):
        result[task] = {
            metric: _metric_statistics(
                {
                    str(loaded[label].training_seed): gaps[label][task][metric]
                    for label in PRIMARY_LABELS
                }
            )
            for metric in gaps[PRIMARY_LABELS[0]][task]
        }
    return result


def _primary_output_quality(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for count_name in ("valid_json", "schema_valid"):
        result[f"{count_name}_rate"] = _metric_statistics(
            {
                str(loaded[label].training_seed): _divide(
                    float(loaded[label].evaluation["counts"][count_name]),
                    float(loaded[label].evaluation["counts"]["predictions"]),
                )
                for label in PRIMARY_LABELS
            }
        )
    return result


def _primary_cross_language(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    first = _object(
        loaded[PRIMARY_LABELS[0]].evaluation.get("cross_language_consistency"),
        "cross-language consistency",
    )
    for task, task_payload in first.items():
        task_values = _object(task_payload, f"{task} cross-language consistency")
        pairs = task_values.get("pairs")
        _require(isinstance(pairs, int) and pairs > 0, f"Bad {task} pair count")
        result[task] = {"pairs": pairs, "metrics": {}}
        for metric in task_values:
            if metric == "pairs":
                continue
            result[task]["metrics"][metric] = _metric_statistics(
                {
                    str(loaded[label].training_seed): _number(
                        _object(
                            loaded[label].evaluation.get(
                                "cross_language_consistency"
                            ),
                            "cross-language consistency",
                        )[task][metric],
                        f"{label} {task} {metric}",
                    )
                    for label in PRIMARY_LABELS
                }
            )
        for label in PRIMARY_LABELS:
            candidate = _object(
                _object(
                    loaded[label].evaluation.get("cross_language_consistency"),
                    "cross-language consistency",
                ).get(task),
                f"{task} cross-language consistency",
            )
            _require(candidate.get("pairs") == pairs, f"{task} pair-count drift")
    return result


def _metric_delta(candidate: _LoadedRun, baseline: _LoadedRun) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in SCOPES:
        result[scope] = {}
        for task in (*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc"):
            result[scope][task] = {}
            for metric, candidate_value in candidate.metrics[scope][task].items():
                baseline_value = baseline.metrics[scope][task][metric]
                raw_delta = candidate_value - baseline_value
                higher_is_better = metric != "false_positive_rate"
                result[scope][task][metric] = {
                    "candidate_minus_four_tokens": raw_delta,
                    "effect_toward_better": (
                        raw_delta if higher_is_better else -raw_delta
                    ),
                    "higher_is_better": higher_is_better,
                }
    return result


def _token_ablation(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    baseline = loaded["primary-seed17"]
    runs = {
        str(loaded[label].sensor_tokens): _run_summary(loaded[label])
        for label in TOKEN_LABELS
    }
    return {
        "training_seed": 17,
        "baseline_sensor_tokens": 4,
        "runs": runs,
        "comparisons_to_four_tokens": {
            str(loaded[label].sensor_tokens): _metric_delta(loaded[label], baseline)
            for label in ("tokens1-seed17", "tokens8-seed17")
        },
        "uncertainty_constraint": (
            "Each token count has one training seed; no cross-seed variance is claimed."
        ),
    }


def _run_summary(run: _LoadedRun) -> dict[str, Any]:
    model = _object(run.training.get("model"), f"{run.label} training model")
    gate = _object(model.get("sensor_gate"), f"{run.label} sensor gate")
    counts = _object(run.evaluation.get("counts"), f"{run.label} evaluation counts")
    return {
        "sensor_tokens": run.sensor_tokens,
        "training_seed": run.training_seed,
        "paths": {
            "training_root": str(run.training_root),
            "generation_root": str(run.generation_root),
            "evaluation_root": str(run.evaluation_root),
        },
        "sha256": {
            "training_report": run.training_sha256,
            "training_manifest": run.training_manifest_sha256,
            "generation_report": run.generation_sha256,
            "generation_manifest": run.generation_manifest_sha256,
            "evaluation_report": run.evaluation_sha256,
            "evaluation_manifest": run.evaluation_manifest_sha256,
            "evaluation_records": run.evaluation_records_sha256,
        },
        "sensor_gate": {
            "initial_probability": _number(
                gate.get("initial_probability"), f"{run.label} initial gate"
            ),
            "final_probability": _number(
                gate.get("final_probability"), f"{run.label} final gate"
            ),
            "absolute_probability_change": _number(
                gate.get("absolute_probability_change"), f"{run.label} gate change"
            ),
        },
        "counts": {
            "predictions": counts.get("predictions"),
            "valid_json": counts.get("valid_json"),
            "schema_valid": counts.get("schema_valid"),
            "valid_json_rate": _divide(
                float(counts.get("valid_json")), float(counts.get("predictions"))
            ),
            "schema_valid_rate": _divide(
                float(counts.get("schema_valid")), float(counts.get("predictions"))
            ),
        },
        "metrics": run.metrics,
        "cross_language_consistency": run.evaluation.get(
            "cross_language_consistency"
        ),
    }


def _gate_audit(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    primary_values = {
        str(loaded[label].training_seed): _number(
            _object(
                _object(loaded[label].training.get("model"), "training model").get(
                    "sensor_gate"
                ),
                "sensor gate",
            ).get("final_probability"),
            "final gate probability",
        )
        for label in PRIMARY_LABELS
    }
    return {
        "primary_four_token_final_probability": _metric_statistics(primary_values),
        "by_configuration": {
            label: _run_summary(loaded[label])["sensor_gate"] for label in loaded
        },
        "interpretation_constraint": (
            "Gate magnitude alone is not evidence of modality use; aligned XIC must also "
            "be compared with shuffled, zero, and availability-off interventions."
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL image + XIC development matrix",
        "",
        "All rows use the same leakage-safe bilingual validation set. English and Chinese "
        "prompts describe the same 1,815 scientific assets and are not independent samples.",
        "",
        "## Four-token reproducibility across three training seeds",
        "",
        "| Scope | Task | Metric | Mean | Sample SD | Minimum | Maximum |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope, tasks in report["primary_seed_aggregate"].items():
        for task, metrics in tasks.items():
            for metric, values in metrics.items():
                lines.append(
                    f"| {scope} | {task} | {metric} | {_fmt(values['mean'])} | "
                    f"{_fmt(values['sample_standard_deviation'])} | "
                    f"{_fmt(values['minimum'])} | {_fmt(values['maximum'])} |"
                )
    lines.extend(
        [
            "",
            "## Sensor-token ablation at training seed 17",
            "",
            "| Tokens | Presence Macro-F1 | Presence MCC | Metadata Macro-F1 | "
            "Grounding mean IoU | Grounding IoU@0.5 | QC exact match | Gate probability |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for token_text, run in sorted(
        report["token_ablation"]["runs"].items(), key=lambda item: int(item[0])
    ):
        metrics = run["metrics"]["overall"]
        lines.append(
            f"| {token_text} | {_fmt(metrics['peak_presence']['macro_f1'])} | "
            f"{_fmt(metrics['peak_presence']['mcc'])} | "
            f"{_fmt(metrics['peak_presence_metadata']['macro_f1'])} | "
            f"{_fmt(metrics['peak_grounding']['mean_bbox_iou_all'])} | "
            f"{_fmt(metrics['peak_grounding']['iou_at_0_5_rate_all'])} | "
            f"{_fmt(metrics['scientific_qc']['exact_match_rate'])} | "
            f"{_fmt(run['sensor_gate']['final_probability'])} |"
        )
    lines.extend(
        [
            "",
            "The sealed internal-test split was not accessed. This report is development "
            "evidence and is not a final benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_fusion_matrix(
    *,
    runs: dict[str, FusionMatrixRun],
    output_dir: Path,
) -> FusionMatrixAnalysisResult:
    """Validate and summarize the formal five-cell fusion development matrix."""

    _require(
        set(runs) == set(MATRIX_CONFIGURATIONS),
        f"Expected matrix labels {sorted(MATRIX_CONFIGURATIONS)}",
    )
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Analysis output already exists: {output_dir}")
    loaded = {
        label: _load_run(
            label,
            runs[label],
            expected_tokens=MATRIX_CONFIGURATIONS[label][0],
            expected_seed=MATRIX_CONFIGURATIONS[label][1],
        )
        for label in MATRIX_CONFIGURATIONS
    }

    shared = {
        "training_code_revision": _shared_value(
            loaded, lambda run: run.training.get("code_revision"), "training revision"
        ),
        "training_sources": _shared_value(
            loaded,
            lambda run: _object(run.training.get("sources"), "training sources"),
            "training sources",
        ),
        "base_model_artifact_sha256": _shared_value(
            loaded,
            lambda run: _object(run.training.get("model"), "training model").get(
                "base_artifact_sha256"
            ),
            "base-model artifact",
        ),
        "fusion_bundle_report_sha256": _shared_value(
            loaded,
            lambda run: _object(run.training.get("sources"), "training sources").get(
                "fusion_bundle_report_sha256"
            ),
            "fusion bundle",
        ),
        "dataset_report_sha256": _shared_value(
            loaded,
            lambda run: _object(run.training.get("sources"), "training sources").get(
                "dataset_report_sha256"
            ),
            "dataset report",
        ),
        "instruction_report_sha256": _shared_value(
            loaded,
            lambda run: _object(run.evaluation.get("inputs"), "evaluation inputs").get(
                "instruction_report_sha256"
            ),
            "instruction report",
        ),
        "validation_prompts_sha256": _shared_value(
            loaded,
            lambda run: _object(run.evaluation.get("inputs"), "evaluation inputs").get(
                "validation_prompts_sha256"
            ),
            "validation prompts",
        ),
        "validation_answers_sha256": _shared_value(
            loaded,
            lambda run: _object(run.evaluation.get("inputs"), "evaluation inputs").get(
                "validation_answers_sha256"
            ),
            "validation answers",
        ),
        "generation_source": _shared_value(
            loaded,
            lambda run: _object(run.generation.get("source"), "generation source"),
            "generation source",
        ),
        "generation_protocol": _shared_value(
            loaded,
            lambda run: _object(
                run.generation.get("generation"), "generation settings"
            ),
            "generation protocol",
        ),
        "evaluation_protocol": _shared_value(
            loaded,
            lambda run: _object(
                run.evaluation.get("evaluation"), "evaluation settings"
            ),
            "evaluation protocol",
        ),
        "evaluation_record_identity_sha256": None,
    }
    identities = _shared_value(
        loaded,
        lambda run: sorted(_record_identity(row) for row in run.evaluation_records),
        "evaluation-record identity",
    )
    shared["evaluation_record_identity_sha256"] = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix": {label: _run_summary(run) for label, run in loaded.items()},
        "primary_seed_aggregate": _primary_aggregate(loaded),
        "primary_output_quality": _primary_output_quality(loaded),
        "primary_en_minus_zh_cn": _primary_language_gap(loaded),
        "primary_cross_language_consistency": _primary_cross_language(loaded),
        "token_ablation": _token_ablation(loaded),
        "gate_audit": _gate_audit(loaded),
        "shared_provenance": shared,
        "contracts": {
            "same_leakage_safe_validation": True,
            "same_base_model": True,
            "same_training_data_and_initial_adapter": True,
            "three_independent_primary_training_seeds": True,
            "token_ablation_uses_seed17": True,
            "aligned_xic_only": True,
            "language_variants_are_not_independent_assets": True,
            "evaluation_metrics_recomputed_from_records": True,
            "evaluation_record_identities_match_across_runs": True,
            "internal_test_accessed": False,
        },
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    _require(not staging.exists(), f"Analysis staging exists: {staging}")
    staging.mkdir()
    try:
        report_path = staging / "fusion_matrix_analysis.json"
        markdown_path = staging / "fusion_matrix_analysis.md"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, markdown_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "fusion_matrix_analysis.json"
    return FusionMatrixAnalysisResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        markdown_path=output_dir / "fusion_matrix_analysis.md",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
