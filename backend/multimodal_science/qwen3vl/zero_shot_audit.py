"""Failure-mode audit for provenance-bound Qwen3-VL zero-shot results."""

from __future__ import annotations

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

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_RECORD_SCHEMA,
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
    _bbox_iou,
    _grouped_mean_bootstrap,
    _object,
    _read_json,
    _read_jsonl,
    _require,
    _strict_json_object,
    _valid_bbox,
    _write_json,
    _write_jsonl,
)
from multimodal_science.qwen3vl.inference import (
    GENERATION_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
)
from multimodal_science.qwen3vl.instruction_data import LANGUAGES

ZERO_SHOT_AUDIT_SCHEMA = "chrompeak-qwen3vl-zero-shot-audit-v1"
COORDINATE_AUDIT_RECORD_SCHEMA = "chrompeak-qwen3vl-coordinate-audit-record-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GROUNDING_TASK = "peak_grounding"
_PRESENCE_TASKS = ("peak_presence", "peak_presence_metadata")


@dataclass(frozen=True)
class ZeroShotAuditResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    coordinate_records_path: Path
    coordinate_records: int
    source_pixel_invalid_records: int
    internal_test_accessed: bool = False


def _safe_report_artifact(
    report_path: Path,
    report: dict[str, Any],
    name: str,
) -> tuple[Path, list[dict[str, Any]]]:
    artifacts = _object(report.get("artifacts"), f"{report_path.name} artifacts")
    artifact = _object(artifacts.get(name), f"{report_path.name} artifact {name}")
    relative = artifact.get("path")
    _require(isinstance(relative, str) and bool(relative), f"Invalid {name} path")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected relative {name} path")
    root = report_path.parent.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact path escapes report directory: {name}") from error
    expected_hash = artifact.get("sha256")
    _require(
        isinstance(expected_hash, str) and bool(_HEX_64.fullmatch(expected_hash)),
        f"Invalid {name} SHA-256",
    )
    _require(path.is_file(), f"Missing {name}: {path}")
    _require(sha256_file(path) == expected_hash, f"{name} SHA-256 mismatch")
    records = _read_jsonl(path, name)
    expected_records = artifact.get("records")
    _require(
        isinstance(expected_records, int) and expected_records == len(records),
        f"{name} record count mismatch",
    )
    return path, records


def _unique_by_instruction_id(
    records: Iterable[dict[str, Any]], context: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        instruction_id = record.get("instruction_id")
        _require(
            isinstance(instruction_id, str) and bool(instruction_id),
            f"Missing instruction_id in {context}",
        )
        _require(instruction_id not in result, f"Duplicate instruction_id in {context}")
        result[instruction_id] = record
    return result


def _prediction_value(
    task: str, response: str
) -> tuple[bool | tuple[str, str] | None, bool]:
    parsed, valid_json = _strict_json_object(response)
    if not valid_json or parsed is None:
        return None, False
    if task in _PRESENCE_TASKS:
        valid = set(parsed) == {"peak_present"} and isinstance(
            parsed["peak_present"], bool
        )
        return (bool(parsed["peak_present"]), True) if valid else (None, False)
    if task == "scientific_qc":
        valid = (
            set(parsed) == {"qc_state", "reason"}
            and all(isinstance(parsed[key], str) and bool(parsed[key]) for key in parsed)
        )
        return (
            (str(parsed["qc_state"]), str(parsed["reason"])),
            True,
        ) if valid else (None, False)
    return None, False


def _counter_payload(counter: Counter[Any]) -> list[dict[str, Any]]:
    rows = []
    for value, records in sorted(counter.items(), key=lambda item: str(item[0])):
        if isinstance(value, tuple):
            payload: dict[str, Any] = {
                "qc_state": value[0],
                "reason": value[1],
            }
        elif value is None:
            payload = {"value": "schema_invalid"}
        else:
            payload = {"value": str(value).lower()}
        payload["records"] = records
        rows.append(payload)
    return rows


def _distribution_summary(counter: Counter[Any]) -> dict[str, Any]:
    records = sum(counter.values())
    dominant = max(counter.values()) if counter else 0
    return {
        "records": records,
        "unique_outputs_including_invalid": len(counter),
        "dominant_output_rate": dominant / records if records else 0.0,
        "single_output": len(counter) == 1,
        "outputs": _counter_payload(counter),
    }


def _raw_bbox(response: str) -> list[float] | None:
    parsed, valid_json = _strict_json_object(response)
    if not valid_json or parsed is None or set(parsed) != {"bbox_2d"}:
        return None
    value = parsed["bbox_2d"]
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(item)
        for item in value
    ):
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (x1 < x2 and y1 < y2):
        return None
    return [x1, y1, x2, y2]


def _normalized_grid_bbox(
    raw: list[float] | None,
    width: int,
    height: int,
    *,
    scale_y: bool,
) -> list[float] | None:
    if raw is None or not all(0.0 <= value <= 1000.0 for value in raw):
        return None
    converted = [
        raw[0] * width / 1000.0,
        raw[1] * height / 1000.0 if scale_y else raw[1],
        raw[2] * width / 1000.0,
        raw[3] * height / 1000.0 if scale_y else raw[3],
    ]
    return _valid_bbox(converted, width, height)


def _mean(flags: Iterable[bool]) -> float:
    values = list(flags)
    return float(sum(values) / len(values)) if values else 0.0


def _coordinate_metrics(
    records: list[dict[str, Any]], interpretation: str, *, seed: int, iterations: int
) -> dict[str, Any]:
    field = f"{interpretation}_bbox_iou"
    valid_field = f"{interpretation}_schema_valid"
    ious = [float(record[field]) for record in records]
    groups = [str(record["group_id"]) for record in records]
    result: dict[str, Any] = {
        "records": len(records),
        "source_groups": len(set(groups)),
        "schema_valid_rate": _mean(bool(record[valid_field]) for record in records),
        "mean_bbox_iou_all": float(np.mean(ious)),
        "iou_at_0_5_rate_all": _mean(value >= 0.5 for value in ious),
    }
    if len(set(groups)) >= 2:
        result["grouped_bootstrap_mean_iou"] = _grouped_mean_bootstrap(
            ious,
            groups,
            iterations=iterations,
            seed=seed,
        )
    return result


def _coordinate_comparison(
    records: list[dict[str, Any]], *, seed: int, iterations: int
) -> dict[str, Any]:
    interpretations = (
        "source_pixel",
        "normalized_xy_0_1000",
        "normalized_x_0_1000_source_y",
    )
    metrics = {
        name: _coordinate_metrics(records, name, seed=seed, iterations=iterations)
        for name in interpretations
    }
    baseline = metrics["source_pixel"]["mean_bbox_iou_all"]
    for name in interpretations[1:]:
        metrics[name]["mean_iou_delta_vs_source_pixel"] = (
            metrics[name]["mean_bbox_iou_all"] - baseline
        )
        metrics[name]["source_pixel_invalid_rescued"] = sum(
            not bool(record["source_pixel_schema_valid"])
            and bool(record[f"{name}_schema_valid"])
            for record in records
        )
    return metrics


def audit_zero_shot_failures(
    generation_report_path: Path,
    evaluation_report_path: Path,
    output_dir: Path,
    *,
    expected_generation_report_sha256: str,
    expected_evaluation_report_sha256: str,
    bootstrap_iterations: int = 1000,
    seed: int = 17,
) -> ZeroShotAuditResult:
    """Audit output shortcuts and alternative grounding coordinate interpretations."""

    generation_report_path = generation_report_path.resolve()
    evaluation_report_path = evaluation_report_path.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Audit output already exists: {output_dir}")
    _require(bootstrap_iterations >= 2, "Bootstrap iterations must be at least two")
    for expected, name in (
        (expected_generation_report_sha256, "generation"),
        (expected_evaluation_report_sha256, "evaluation"),
    ):
        _require(
            isinstance(expected, str) and bool(_HEX_64.fullmatch(expected)),
            f"Expected {name} report SHA-256 must be lowercase hexadecimal",
        )

    generation_report = _read_json(generation_report_path, "generation report")
    evaluation_report = _read_json(evaluation_report_path, "evaluation report")
    _require(
        sha256_file(generation_report_path) == expected_generation_report_sha256,
        "Generation report SHA-256 mismatch",
    )
    _require(
        sha256_file(evaluation_report_path) == expected_evaluation_report_sha256,
        "Evaluation report SHA-256 mismatch",
    )
    _require(
        generation_report.get("schema_version") == GENERATION_REPORT_SCHEMA,
        "Unsupported generation report schema",
    )
    _require(
        evaluation_report.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        "Failure audit requires a bilingual evaluation report",
    )
    _require(
        generation_report.get("internal_test_accessed") is False
        and evaluation_report.get("internal_test_accessed") is False,
        "Failure audit may not consume internal test results",
    )
    _require(
        evaluation_report.get("prediction_generation_provenance_verified") is True,
        "Evaluation must verify generation provenance",
    )
    evaluation_inputs = _object(evaluation_report.get("inputs"), "evaluation inputs")
    _require(
        evaluation_inputs.get("generation_report_sha256")
        == expected_generation_report_sha256,
        "Evaluation is not bound to the supplied generation report",
    )

    predictions_path, predictions = _safe_report_artifact(
        generation_report_path, generation_report, "predictions"
    )
    records_path, evaluation_records = _safe_report_artifact(
        evaluation_report_path, evaluation_report, "evaluation_records"
    )
    predictions_sha256 = sha256_file(predictions_path)
    _require(
        evaluation_inputs.get("predictions_sha256") == predictions_sha256,
        "Evaluation is not bound to the supplied predictions",
    )
    prediction_by_id = _unique_by_instruction_id(predictions, "predictions")
    record_by_id = _unique_by_instruction_id(evaluation_records, "evaluation records")
    _require(
        set(prediction_by_id) == set(record_by_id),
        "Prediction and evaluation instruction IDs disagree",
    )

    distributions: dict[str, dict[str, Counter[Any]]] = {
        task: {language: Counter() for language in LANGUAGES}
        for task in (*_PRESENCE_TASKS, "scientific_qc")
    }
    coordinate_records = []
    for instruction_id, row in record_by_id.items():
        _require(
            row.get("schema_version") == BILINGUAL_EVALUATION_RECORD_SCHEMA,
            "Unsupported evaluation record schema",
        )
        task = row.get("task")
        language = row.get("language")
        _require(language in LANGUAGES, "Unsupported evaluation language")
        prediction = prediction_by_id[instruction_id]
        _require(
            prediction.get("schema_version") == PREDICTION_SCHEMA
            and isinstance(prediction.get("response"), str),
            "Invalid prediction record",
        )
        response = str(prediction["response"])
        if task in distributions:
            value, schema_valid = _prediction_value(str(task), response)
            distributions[str(task)][str(language)][
                value if schema_valid else None
            ] += 1
            continue
        if task != _GROUNDING_TASK:
            raise ValueError(f"Unsupported evaluation task: {task}")

        width = row.get("image_width")
        height = row.get("image_height")
        _require(
            isinstance(width, int)
            and width > 0
            and isinstance(height, int)
            and height > 0,
            "Invalid grounding image dimensions",
        )
        expected = _valid_bbox(row.get("expected_bbox_2d"), width, height)
        _require(expected is not None, "Invalid expected grounding box")
        raw = _raw_bbox(response)
        source_pixel = _valid_bbox(raw, width, height) if raw is not None else None
        normalized_xy = _normalized_grid_bbox(raw, width, height, scale_y=True)
        normalized_x = _normalized_grid_bbox(raw, width, height, scale_y=False)
        _require(
            bool(row.get("schema_valid")) == (source_pixel is not None),
            "Stored source-pixel schema result disagrees with raw prediction",
        )
        current_iou = _bbox_iou(expected, source_pixel) if source_pixel else 0.0
        _require(
            math.isclose(current_iou, float(row.get("bbox_iou")), abs_tol=1e-12),
            "Stored source-pixel IoU disagrees with raw prediction",
        )
        coordinate_records.append(
            {
                "schema_version": COORDINATE_AUDIT_RECORD_SCHEMA,
                "instruction_id": instruction_id,
                "asset_id": row.get("asset_id"),
                "group_id": row.get("group_id"),
                "pair_id": row.get("pair_id"),
                "language": language,
                "image_width": width,
                "image_height": height,
                "expected_bbox_2d": expected,
                "raw_bbox_2d": raw,
                "source_pixel_bbox_2d": source_pixel,
                "source_pixel_schema_valid": source_pixel is not None,
                "source_pixel_bbox_iou": current_iou,
                "normalized_xy_0_1000_bbox_2d": normalized_xy,
                "normalized_xy_0_1000_schema_valid": normalized_xy is not None,
                "normalized_xy_0_1000_bbox_iou": (
                    _bbox_iou(expected, normalized_xy) if normalized_xy else 0.0
                ),
                "normalized_x_0_1000_source_y_bbox_2d": normalized_x,
                "normalized_x_0_1000_source_y_schema_valid": normalized_x is not None,
                "normalized_x_0_1000_source_y_bbox_iou": (
                    _bbox_iou(expected, normalized_x) if normalized_x else 0.0
                ),
            }
        )

    distribution_report = {
        task: {
            language: _distribution_summary(counter)
            for language, counter in by_language.items()
        }
        for task, by_language in distributions.items()
    }
    grounding_by_language = {
        language: _coordinate_comparison(
            [record for record in coordinate_records if record["language"] == language],
            seed=seed,
            iterations=bootstrap_iterations,
        )
        for language in LANGUAGES
    }
    grounding_overall = _coordinate_comparison(
        coordinate_records,
        seed=seed,
        iterations=bootstrap_iterations,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        coordinate_path = staging / "coordinate_counterfactual_records.jsonl"
        _write_jsonl(coordinate_path, coordinate_records)
        report_path = staging / "zero_shot_failure_audit.json"
        report = {
            "schema_version": ZERO_SHOT_AUDIT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "generation_report_sha256": expected_generation_report_sha256,
                "evaluation_report_sha256": expected_evaluation_report_sha256,
                "predictions_sha256": predictions_sha256,
                "evaluation_records_sha256": sha256_file(records_path),
            },
            "counts": {
                "predictions": len(predictions),
                "evaluation_records": len(evaluation_records),
                "grounding_records": len(coordinate_records),
                "source_pixel_invalid_grounding_records": sum(
                    not bool(record["source_pixel_schema_valid"])
                    for record in coordinate_records
                ),
            },
            "prediction_distributions": distribution_report,
            "grounding_coordinate_counterfactual": {
                "overall": grounding_overall,
                "by_language": grounding_by_language,
            },
            "contracts": {
                "diagnostic_counterfactual_only": True,
                "does_not_replace_formal_zero_shot_metrics": True,
                "must_not_select_training_protocol_on_validation": True,
                "no_model_inference_performed": True,
                "generation_provenance_verified_upstream": True,
                "internal_test_accessed": False,
            },
            "artifacts": {
                "coordinate_counterfactual_records": {
                    "path": coordinate_path.name,
                    "sha256": sha256_file(coordinate_path),
                    "records": len(coordinate_records),
                }
            },
            "development_comparison_eligible": False,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "\n".join(
                [
                    f"{sha256_file(coordinate_path)}  {coordinate_path.name}",
                    f"{sha256_file(report_path)}  {report_path.name}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)

    final_report = output_dir / "zero_shot_failure_audit.json"
    final_coordinates = output_dir / "coordinate_counterfactual_records.jsonl"
    return ZeroShotAuditResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        coordinate_records_path=final_coordinates,
        coordinate_records=len(coordinate_records),
        source_pixel_invalid_records=sum(
            not bool(record["source_pixel_schema_valid"])
            for record in coordinate_records
        ),
    )
