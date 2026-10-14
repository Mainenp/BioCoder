"""Hash-bound localization analysis for the preregistered v2 candidates.

This module is deliberately development-only.  It accepts completed C0, C1 and
selected-RL *evaluation* artifacts, never prompts or answer keys, and rejects
any path or report that could refer to the sealed internal-test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_RECORD_SCHEMA,
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
    LANGUAGES,
    _cross_language_consistency,
)


REPORT_SCHEMA = "chrompeak-qwen3vl-v2-localization-analysis-v1"
FAILURE_SCHEMA = "chrompeak-qwen3vl-v2-localization-failure-v1"
METADATA_SCHEMA = "chrompeak-qwen3vl-v2-localization-metadata-v1"
METADATA_REPORT_SCHEMA = "chrompeak-qwen3vl-v2-localization-metadata-report-v1"
VALIDATION_SOURCE_GROUPS = 11
SCOPES = ("overall", *LANGUAGES)
MODELS = ("c0", "c1", "rl")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LocalizationAnalysisResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    failure_cases_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LocalizationMetadataResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    metadata_path: Path
    metadata_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class _Run:
    label: str
    root: Path
    report: dict[str, Any]
    report_sha256: str
    manifest_sha256: str
    records: tuple[dict[str, Any], ...]
    records_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), label)


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rows.append(_object(json.loads(line), f"{label} line {line_number}"))
    _require(rows, f"{label} is empty")
    return tuple(rows)


def _safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    _require(
        bool(value)
        and not pure.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"Unsafe {label}: {value}",
    )
    return Path(*pure.parts)


def _verify_manifest(root: Path, required: set[str], label: str) -> str:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Missing {label} manifest")
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed {label} manifest line {line_number}")
        digest, raw_relative = parts
        raw_relative = raw_relative.lstrip(" *")
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad {label} manifest digest")
        relative = _safe_relative(raw_relative, f"{label} manifest path")
        _require(relative.as_posix() not in listed, f"Duplicate {label} manifest path")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} manifest escapes its root") from error
        _require(artifact.is_file(), f"Missing {label} artifact {relative}")
        _require(sha256_file(artifact) == digest, f"{label} artifact hash mismatch")
        listed.add(relative.as_posix())
    _require(required <= listed, f"{label} manifest lacks {sorted(required - listed)}")
    return sha256_file(manifest)


def _assert_development_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix().casefold()
    _require(
        all(token not in normalized for token in ("internal_test", "internal-test", "internaltest")),
        f"{label} references internal test",
    )
    return resolved


def _record_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("instruction_id"), row.get("pair_id"), row.get("asset_id"),
        row.get("group_id"), row.get("task"), row.get("language"),
        row.get("target_peak_present"), tuple(row.get("expected_bbox_2d") or ()),
    )


def _load_run(label: str, root: Path) -> _Run:
    root = _assert_development_path(root, f"{label} evaluation root")
    manifest_sha = _verify_manifest(root, {"qwen_evaluation_report.json", "evaluation_records.jsonl"}, label)
    report_path = root / "qwen_evaluation_report.json"
    report = _read_object(report_path, f"{label} evaluation report")
    _require(report.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA, f"Bad {label} evaluation schema")
    _require(report.get("development_comparison_eligible") is True, f"{label} is not development eligible")
    _require(report.get("final_benchmark_eligible") is False and report.get("internal_test_accessed") is False, f"{label} has protected scope")
    _require(report.get("prediction_generation_provenance_verified") is True, f"{label} generation provenance is unverified")
    artifact = _object(_object(report.get("artifacts"), f"{label} artifacts").get("evaluation_records"), f"{label} records artifact")
    records_relative = _safe_relative(
        str(artifact.get("path")), f"{label} records artifact"
    )
    _require(
        records_relative.as_posix() == "evaluation_records.jsonl",
        f"{label} records artifact is not manifest-bound",
    )
    records_path = root / records_relative
    records_sha = sha256_file(records_path)
    _require(records_sha == artifact.get("sha256"), f"{label} records hash mismatch")
    rows = _read_jsonl(records_path, f"{label} evaluation records")
    _require(artifact.get("records") == len(rows), f"{label} record-count mismatch")
    _require(
        _object(report.get("counts"), f"{label} counts").get("predictions") == len(rows),
        f"{label} report prediction-count mismatch",
    )
    _require(len({_record_identity(row) for row in rows}) == len(rows), f"{label} duplicate record identity")
    for row in rows:
        _require(row.get("schema_version") == BILINGUAL_EVALUATION_RECORD_SCHEMA, f"{label} record schema drift")
        _require(row.get("task") in {"peak_presence", "peak_presence_metadata", "peak_grounding", "scientific_qc"}, f"{label} unknown task")
        _require(row.get("language") in LANGUAGES, f"{label} invalid language")
        _require(isinstance(row.get("group_id"), str) and bool(row["group_id"]), f"{label} missing group")
        _require(isinstance(row.get("pair_id"), str) and bool(row["pair_id"]), f"{label} missing pair")
    return _Run(label, root, report, sha256_file(report_path), manifest_sha, rows, records_sha)


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    _require(isinstance(value, list) and len(value) == 4, f"{label} must be a bbox")
    x1, y1, x2, y2 = (_number(item, label) for item in value)
    _require(x1 < x2 and y1 < y2, f"{label} has invalid bounds")
    return x1, y1, x2, y2


def _positive_grounding(rows: Iterable[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    _require(scope in SCOPES, f"Unsupported scope: {scope}")
    result = []
    for row in rows:
        if row.get("task") != "peak_grounding" or (scope != "overall" and row.get("language") != scope):
            continue
        target_present = row.get("target_peak_present")
        _require(
            isinstance(target_present, bool),
            "Grounding record lacks a boolean target_peak_present",
        )
        if not target_present:
            continue
        _bbox(row.get("expected_bbox_2d"), "Expected grounding bbox")
        result.append(row)
    _require(result, f"No positive grounding rows for {scope}")
    return result


def _metadata_rows(path: Path, expected_sha256: str) -> tuple[dict[str, Any], ...]:
    path = _assert_development_path(path, "validation metadata")
    _require(bool(_HEX_64.fullmatch(expected_sha256)), "Expected metadata SHA-256 is invalid")
    _require(path.is_file(), f"Missing validation metadata: {path}")
    _require(sha256_file(path) == expected_sha256, "Validation metadata hash mismatch")
    if path.suffix == ".jsonl":
        return _read_jsonl(path, "validation metadata")
    payload = _read_object(path, "validation metadata")
    rows = payload.get("records")
    _require(isinstance(rows, list), "Validation metadata records must be an array")
    return tuple(_object(row, "validation metadata record") for row in rows)


def _metadata_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        task, pair_id = row.get("task"), row.get("pair_id")
        _require(task == "peak_grounding", "Validation metadata exposes a non-grounding task")
        _require(isinstance(pair_id, str) and bool(pair_id), "Validation metadata missing pair_id")
        key = str(task), pair_id
        _require(key not in index, f"Duplicate validation metadata pair: {pair_id}")
        _require(_number(row.get("raw_max"), "Validation raw_max") >= 0.0, "Validation raw_max must be non-negative")
        roi = row.get("roi_window_minutes")
        if roi is not None:
            _require(isinstance(roi, list) and len(roi) == 2, "ROI window must have two bounds")
            _require(_number(roi[1], "ROI high") > _number(roi[0], "ROI low"), "ROI window must increase")
        index[key] = row
    _require(index, "Validation metadata is empty")
    return index


def _bound_report_artifact(
    root: Path,
    report: dict[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    descriptor = _object(
        _object(report.get("artifacts"), "source artifacts").get(name),
        f"source artifact {name}",
    )
    relative = _safe_relative(str(descriptor.get("path") or ""), name)
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), f"Source artifact escapes root: {name}")
    expected = descriptor.get("sha256")
    _require(
        isinstance(expected, str)
        and bool(_HEX_64.fullmatch(expected))
        and path.is_file()
        and sha256_file(path) == expected,
        f"Source artifact drift: {name}",
    )
    return path, descriptor


def build_localization_metadata(
    *,
    dataset_root: Path,
    expected_dataset_report_sha256: str,
    fusion_bundle_root: Path,
    expected_fusion_bundle_report_sha256: str,
    output_dir: Path,
) -> LocalizationMetadataResult:
    """Materialize answer-free validation covariates for the frozen analysis."""

    dataset_root = _assert_development_path(dataset_root, "Dataset root")
    fusion_bundle_root = _assert_development_path(fusion_bundle_root, "fusion bundle root")
    output_dir = _assert_development_path(output_dir, "localization metadata output")
    _require(not output_dir.exists(), f"Localization metadata output exists: {output_dir}")
    for digest, label in (
        (expected_dataset_report_sha256, "Dataset report"),
        (expected_fusion_bundle_report_sha256, "fusion bundle report"),
    ):
        _require(bool(_HEX_64.fullmatch(digest)), f"Invalid {label} SHA-256")
    dataset_report_path = dataset_root / "dataset_report.json"
    fusion_report_path = fusion_bundle_root / "fusion_bundle_report.json"
    _require(
        sha256_file(dataset_report_path) == expected_dataset_report_sha256,
        "Dataset report drift",
    )
    _require(
        sha256_file(fusion_report_path) == expected_fusion_bundle_report_sha256,
        "Fusion bundle report drift",
    )
    dataset = _read_object(dataset_report_path, "Dataset report")
    fusion = _read_object(fusion_report_path, "fusion bundle report")
    _require(dataset.get("internal_test_accessed") is False, "Dataset used internal test")
    _require(fusion.get("internal_test_accessed") is False, "Fusion bundle used internal test")
    validation_examples_path, examples_artifact = _bound_report_artifact(
        dataset_root, dataset, "validation_examples"
    )
    validation_links_path, links_artifact = _bound_report_artifact(
        fusion_bundle_root, fusion, "validation_xic_links"
    )
    examples = _read_jsonl(validation_examples_path, "validation examples")
    links = _read_jsonl(validation_links_path, "validation XIC links")
    _require(examples_artifact.get("records") == len(examples), "Validation example count drift")
    _require(links_artifact.get("records") == len(links), "Validation link count drift")
    examples_by_asset = {str(row.get("asset_id")): row for row in examples}
    _require(len(examples_by_asset) == len(examples), "Duplicate validation asset ID")
    rows_by_pair: dict[str, dict[str, Any]] = {}
    for link in links:
        if link.get("task") != "peak_grounding":
            continue
        pair_id = link.get("pair_id")
        asset_id = str(link.get("asset_id") or "")
        _require(isinstance(pair_id, str) and bool(pair_id), "Grounding link lacks pair ID")
        _require(asset_id in examples_by_asset, "Grounding link references an unknown asset")
        example = examples_by_asset[asset_id]
        metadata = _object(example.get("metadata"), "validation example metadata")
        log_max = _number(metadata.get("log1p_raw_maximum"), "log1p raw maximum")
        _require(log_max >= 0.0, "Validation log1p raw maximum is negative")
        group_id = link.get("group_id")
        _require(
            isinstance(group_id, str) and bool(group_id),
            "Grounding link lacks a source group",
        )
        roi = metadata.get("roi_window_minutes")
        _require(
            isinstance(roi, list)
            and len(roi) == 2
            and _number(roi[1], "ROI high") > _number(roi[0], "ROI low"),
            "Validation ROI window must increase",
        )
        row = {
            "schema_version": METADATA_SCHEMA,
            "task": "peak_grounding",
            "pair_id": pair_id,
            "asset_id": asset_id,
            "group_id": group_id,
            "raw_max": math.expm1(log_max),
            "roi_window_minutes": roi,
        }
        previous = rows_by_pair.setdefault(pair_id, row)
        _require(previous == row, f"Bilingual localization metadata drift: {pair_id}")
    rows = [rows_by_pair[key] for key in sorted(rows_by_pair)]
    _require(rows, "No grounding localization metadata was produced")
    _require(
        len({row["group_id"] for row in rows}) == VALIDATION_SOURCE_GROUPS,
        "Localization metadata must cover all validation source groups",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        metadata_path = staging / "validation_localization_metadata.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = {
            "schema_version": METADATA_REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "dataset_report_sha256": expected_dataset_report_sha256,
                "fusion_bundle_report_sha256": expected_fusion_bundle_report_sha256,
                "validation_examples_sha256": examples_artifact["sha256"],
                "validation_xic_links_sha256": links_artifact["sha256"],
            },
            "counts": {
                "records": len(rows),
                "source_groups": len({row["group_id"] for row in rows}),
            },
            "contracts": {
                "answer_free": True,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
                "bilingual_pairs_deduplicated": True,
            },
            "artifacts": {
                "validation_localization_metadata": {
                    "path": metadata_path.name,
                    "sha256": sha256_file(metadata_path),
                    "records": len(rows),
                }
            },
        }
        report_path = staging / "validation_localization_metadata_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (metadata_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    metadata_path = output_dir / "validation_localization_metadata.jsonl"
    report_path = output_dir / "validation_localization_metadata_report.json"
    return LocalizationMetadataResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
        manifest_path=output_dir / "artifact_manifest.sha256",
    )


def _merge_metadata(rows: Iterable[dict[str, Any]], metadata: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in rows:
        key = ("peak_grounding", str(row["pair_id"]))
        source = metadata.get(key)
        _require(source is not None, f"Missing validation metadata for {key[1]}")
        for field in ("asset_id", "group_id"):
            if field in source:
                _require(source[field] == row.get(field), f"Validation metadata {field} drift")
        joined = dict(row)
        joined["raw_max"] = _number(source.get("raw_max"), "Validation raw_max")
        joined["roi_window_minutes"] = source.get("roi_window_minutes")
        merged.append(joined)
    return merged


def _model_row(row: dict[str, Any]) -> dict[str, float | None]:
    expected = _bbox(row.get("expected_bbox_2d"), "Expected grounding bbox")
    width = _number(row.get("image_width"), "Image width")
    _require(width > 0.0, "Image width must be positive")
    predicted = row.get("predicted_bbox_2d")
    if predicted is None or not bool(row.get("schema_valid")):
        return {"iou": 0.0, "iou_at_0_5": 0.0, "iou_at_0_75": 0.0, "start_error_pixels": None, "end_error_pixels": None, "start_error_seconds": None, "end_error_seconds": None}
    px1, _, px2, _ = _bbox(predicted, "Predicted grounding bbox")
    x1, _, x2, _ = expected
    iou = _number(row.get("bbox_iou"), "Grounding IoU")
    _require(0.0 <= iou <= 1.0, "Grounding IoU is outside [0, 1]")
    start, end = abs(px1 - x1), abs(px2 - x2)
    roi = row.get("roi_window_minutes")
    factor = None
    if roi is not None:
        factor = 60.0 * (_number(roi[1], "ROI high") - _number(roi[0], "ROI low")) / width
    return {"iou": iou, "iou_at_0_5": float(iou >= 0.5), "iou_at_0_75": float(iou >= 0.75), "start_error_pixels": start, "end_error_pixels": end, "start_error_seconds": None if factor is None else start * factor, "end_error_seconds": None if factor is None else end * factor}


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(statistics.fmean(finite)) if finite else None


def _localization_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [_model_row(row) for row in rows]
    return {
        "positive_grounding_rows": len(values),
        "mean_iou": float(statistics.fmean(float(value["iou"]) for value in values)),
        "iou_at_0_5": float(statistics.fmean(float(value["iou_at_0_5"]) for value in values)),
        "iou_at_0_75": float(statistics.fmean(float(value["iou_at_0_75"]) for value in values)),
        "start_error_pixels": _mean(value["start_error_pixels"] for value in values),
        "end_error_pixels": _mean(value["end_error_pixels"] for value in values),
        "start_error_seconds": _mean(value["start_error_seconds"] for value in values),
        "end_error_seconds": _mean(value["end_error_seconds"] for value in values),
        "boundary_error_rows": sum(value["start_error_pixels"] is not None for value in values),
        "seconds_error_rows": sum(value["start_error_seconds"] is not None for value in values),
        "schema_invalid_or_unusable_rows": sum(value["start_error_pixels"] is None for value in values),
    }


def _tertile_boundaries(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = {
        "true_peak_width_pixels": [(_bbox(row["expected_bbox_2d"], "Expected bbox")[2] - _bbox(row["expected_bbox_2d"], "Expected bbox")[0]) for row in rows],
        "log1p_raw_max": [math.log1p(_number(row["raw_max"], "raw_max")) for row in rows],
        "min_edge_distance_pixels": [min(_bbox(row["expected_bbox_2d"], "Expected bbox")[0], _number(row["image_width"], "Image width") - _bbox(row["expected_bbox_2d"], "Expected bbox")[2]) for row in rows],
    }
    result: dict[str, Any] = {}
    for name, population in values.items():
        ordered = sorted(population)
        _require(len(ordered) >= 3, f"Need at least three positive rows for {name} tertiles")
        result[name] = {
            "lower_boundary": float(ordered[len(ordered) // 3 - 1]),
            "upper_boundary": float(ordered[(2 * len(ordered)) // 3 - 1]),
            "population_rows": len(ordered),
            "tie_policy": "low_le_lower; middle_gt_lower_and_le_upper; high_gt_upper",
        }
    return result


def _stratum(row: dict[str, Any], name: str, boundary: dict[str, Any]) -> str:
    expected = _bbox(row["expected_bbox_2d"], "Expected bbox")
    if name == "true_peak_width_pixels":
        value = expected[2] - expected[0]
    elif name == "log1p_raw_max":
        value = math.log1p(_number(row["raw_max"], "raw_max"))
    else:
        value = min(expected[0], _number(row["image_width"], "Image width") - expected[2])
    if value <= _number(boundary["lower_boundary"], "tertile lower boundary"):
        return "low"
    if value <= _number(boundary["upper_boundary"], "tertile upper boundary"):
        return "middle"
    return "high"


def _stratified_metrics(rows: Iterable[dict[str, Any]], boundaries: dict[str, Any]) -> dict[str, Any]:
    source = list(rows)
    result: dict[str, Any] = {}
    for variable, boundary in boundaries.items():
        result[variable] = {}
        for level in ("low", "middle", "high"):
            selected = [row for row in source if _stratum(row, variable, boundary) == level]
            result[variable][level] = None if not selected else _localization_metrics(selected)
    return result


def _group_stats(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        values = _model_row(row)
        group = str(row["group_id"])
        stats = result.setdefault(group, {"iou": 0.0, "iou50": 0.0, "iou75": 0.0, "start": 0.0, "end": 0.0, "n": 0.0, "boundary_n": 0.0})
        stats["iou"] += float(values["iou"])
        stats["iou50"] += float(values["iou_at_0_5"])
        stats["iou75"] += float(values["iou_at_0_75"])
        stats["n"] += 1.0
        if values["start_error_pixels"] is not None:
            stats["start"] += float(values["start_error_pixels"])
            stats["end"] += float(values["end_error_pixels"])
            stats["boundary_n"] += 1.0
    return result


def _ratio_metric(stats: Iterable[dict[str, float]], name: str) -> float:
    rows = list(stats)
    if name == "mean_iou":
        return sum(item["iou"] for item in rows) / sum(item["n"] for item in rows)
    if name == "iou_at_0_5":
        return sum(item["iou50"] for item in rows) / sum(item["n"] for item in rows)
    if name == "iou_at_0_75":
        return sum(item["iou75"] for item in rows) / sum(item["n"] for item in rows)
    denominator = sum(item["boundary_n"] for item in rows)
    return (sum(item["start" if name == "start_error_pixels" else "end"] for item in rows) / denominator) if denominator else math.nan


def _paired_bootstrap(reference: list[dict[str, Any]], rl: list[dict[str, Any]], *, comparison: str, scope: str, iterations: int, seed: int) -> dict[str, Any]:
    left, right = _group_stats(reference), _group_stats(rl)
    groups = sorted(left)
    _require(set(right) == set(groups), f"{comparison}/{scope} source-group drift")
    _require(len(groups) >= 2, f"{comparison}/{scope} requires at least two source groups")
    names = ("mean_iou", "iou_at_0_5", "iou_at_0_75", "start_error_pixels", "end_error_pixels")
    output: dict[str, Any] = {"source_groups": len(groups), "resampling": "ratio_of_sums"}
    for name in names:
        observed = _ratio_metric([right[group] for group in groups], name) - _ratio_metric([left[group] for group in groups], name)
        rng = random.Random(f"{seed}:{comparison}:{scope}:{name}")
        samples = []
        for _ in range(iterations):
            sampled = [rng.choice(groups) for _ in groups]
            samples.append(_ratio_metric([right[group] for group in sampled], name) - _ratio_metric([left[group] for group in sampled], name))
        finite = np.asarray([value for value in samples if math.isfinite(value)], dtype=np.float64)
        _require(finite.size == iterations, f"{comparison}/{scope}/{name} bootstrap has no boundary observations")
        low, high = np.quantile(finite, [0.025, 0.975]).tolist()
        higher_is_better = name not in {"start_error_pixels", "end_error_pixels"}
        benefit = finite if higher_is_better else -finite
        output[name] = {
            "rl_minus_reference": float(observed),
            "rl_minus_reference_ci_95": [float(low), float(high)],
            "higher_is_better": higher_is_better,
            "probability_rl_better": float(np.mean(benefit > 0.0) + 0.5 * np.mean(benefit == 0.0)),
        }
    return output


def _constraint_metrics(
    report: dict[str, Any], rows: Iterable[dict[str, Any]], scope: str
) -> dict[str, Any]:
    tasks = _object(report.get("metrics") if scope == "overall" else _object(report.get("metrics_by_language"), "metrics by language").get(scope), f"{scope} metrics")
    presence = _object(_object(tasks.get("peak_presence"), "presence task").get("classification"), "presence classification")
    metadata = _object(_object(tasks.get("peak_presence_metadata"), "metadata task").get("classification"), "metadata classification")
    qc = _object(tasks.get("scientific_qc"), "QC task")
    scoped = list(rows) if scope == "overall" else [
        row for row in rows if row.get("language") == scope
    ]
    _require(scoped, f"No records for {scope} JSON-valid metric")
    total = len(scoped)
    valid = sum(bool(row.get("valid_json")) for row in scoped)
    return {
        "presence_macro_f1": _number(presence.get("macro_f1"), "presence macro-F1"),
        "presence_mcc": _number(presence.get("mcc"), "presence MCC"),
        "presence_false_positive_rate": _number(presence.get("false_positive_rate"), "presence FPR"),
        "presence_recall": _number(presence.get("recall"), "presence recall"),
        "metadata_presence_macro_f1": _number(metadata.get("macro_f1"), "metadata macro-F1"),
        "scientific_qc_exact_match": _number(qc.get("exact_match_rate"), "QC exact match"),
        "json_valid_rate": valid / total,
    }


def _failures(c1_rows: Iterable[dict[str, Any]], rl_rows: Iterable[dict[str, Any]], *, seed: int, limit: int = 20) -> list[dict[str, Any]]:
    c1 = {str(row["instruction_id"]): row for row in c1_rows}
    candidates = []
    for row in rl_rows:
        reference = c1[str(row["instruction_id"])]
        delta = float(_model_row(row)["iou"]) - float(_model_row(reference)["iou"])
        if delta < 0.0:
            item = {
                "schema_version": FAILURE_SCHEMA,
                "instruction_id": row["instruction_id"], "pair_id": row["pair_id"],
                "asset_id": row["asset_id"], "group_id": row["group_id"], "language": row["language"],
                "c1_iou": _model_row(reference)["iou"], "rl_iou": _model_row(row)["iou"],
                "rl_minus_c1_iou": delta,
                "expected_bbox_2d": row["expected_bbox_2d"],
                "c1_predicted_bbox_2d": reference.get("predicted_bbox_2d"),
                "rl_predicted_bbox_2d": row.get("predicted_bbox_2d"),
            }
            digest = hashlib.sha256(f"{seed}:{item['instruction_id']}".encode()).hexdigest()
            candidates.append((digest, item))
    return [item for _, item in sorted(candidates)[:limit]]


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# V2 localization analysis", "", "Only positive grounding samples contribute to localization metrics. Stratification boundaries are fixed from shared truth/metadata before model comparison.", "", "| Model | Scope | Mean IoU | IoU @ 0.5 | IoU @ 0.75 | Start px error | End px error |", "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for scope in SCOPES:
            row = report["localization"][model][scope]
            lines.append(f"| {model} | {scope} | {row['mean_iou']:.4f} | {row['iou_at_0_5']:.4f} | {row['iou_at_0_75']:.4f} | {row['start_error_pixels'] if row['start_error_pixels'] is not None else '—'} | {row['end_error_pixels'] if row['end_error_pixels'] is not None else '—'} |")
    lines.extend(["", "## Paired source-group bootstrap", "", "Deltas are RL minus the named reference. Bootstrap resamples complete source groups (not language prompts) 10,000 times using ratio-of-sums estimators.", "", "| Comparison | Scope | Metric | Delta | 95% CI |", "|---|---|---|---:|---:|"])
    for comparison, scopes in report["paired_group_bootstrap"]["comparisons"].items():
        for scope, values in scopes.items():
            for metric in ("mean_iou", "iou_at_0_5", "iou_at_0_75"):
                cell = values[metric]
                lo, hi = cell["rl_minus_reference_ci_95"]
                lines.append(f"| {comparison} | {scope} | {metric} | {cell['rl_minus_reference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines.extend(["", "The sealed internal-test split was not accessed. Failure cases are a deterministic random sample of up to 20 positive grounding prompts where RL is worse than C1.", ""])
    return "\n".join(lines)


def analyze_localization(
    *,
    c0_evaluation_root: Path,
    c1_evaluation_root: Path,
    rl_evaluation_root: Path,
    validation_metadata_path: Path,
    expected_validation_metadata_sha256: str,
    output_dir: Path,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 17,
    failure_seed: int = 17,
) -> LocalizationAnalysisResult:
    """Compare the three frozen v2 candidates without opening any protected split."""

    _require(bootstrap_iterations == 10_000, "Localization analysis requires exactly 10,000 bootstrap iterations")
    _require(isinstance(bootstrap_seed, int) and isinstance(failure_seed, int), "Analysis seeds must be integers")
    output_dir = _assert_development_path(output_dir, "localization output")
    _require(not output_dir.exists(), f"Localization output already exists: {output_dir}")
    loaded = {"c0": _load_run("c0", c0_evaluation_root), "c1": _load_run("c1", c1_evaluation_root), "rl": _load_run("rl", rl_evaluation_root)}
    identities = {label: sorted(_record_identity(row) for row in run.records) for label, run in loaded.items()}
    _require(identities["c0"] == identities["c1"] == identities["rl"], "Evaluation record identity drift")
    metadata_sha = sha256_file(_assert_development_path(validation_metadata_path, "validation metadata"))
    metadata = _metadata_index(_metadata_rows(validation_metadata_path, expected_validation_metadata_sha256))
    merged = {label: _merge_metadata(_positive_grounding(run.records, "overall"), metadata) for label, run in loaded.items()}
    base_identity = sorted((row["instruction_id"], row["raw_max"], tuple(row["roi_window_minutes"] or ())) for row in merged["c0"])
    for label in ("c1", "rl"):
        _require(sorted((row["instruction_id"], row["raw_max"], tuple(row["roi_window_minutes"] or ())) for row in merged[label]) == base_identity, f"{label} localization metadata drift")
    boundaries = _tertile_boundaries(merged["c0"])
    by_scope = {label: {scope: [row for row in merged[label] if scope == "overall" or row["language"] == scope] for scope in SCOPES} for label in MODELS}
    for label, scopes in by_scope.items():
        for scope, rows in scopes.items():
            _require(
                len({str(row["group_id"]) for row in rows})
                == VALIDATION_SOURCE_GROUPS,
                f"{label}/{scope} must contain exactly {VALIDATION_SOURCE_GROUPS} validation source groups",
            )
    localization = {label: {scope: _localization_metrics(rows) for scope, rows in scopes.items()} for label, scopes in by_scope.items()}
    strata = {label: {scope: _stratified_metrics(rows, boundaries) for scope, rows in scopes.items()} for label, scopes in by_scope.items()}
    bootstrap = {comparison: {scope: _paired_bootstrap(by_scope[reference][scope], by_scope["rl"][scope], comparison=comparison, scope=scope, iterations=bootstrap_iterations, seed=bootstrap_seed) for scope in SCOPES} for comparison, reference in (("rl_minus_c0", "c0"), ("rl_minus_c1", "c1"))}
    failures = _failures(by_scope["c1"]["overall"], by_scope["rl"]["overall"], seed=failure_seed)
    report = {
        "schema_version": REPORT_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "development_validation", "development_comparison_eligible": True,
        "final_benchmark_eligible": False, "internal_test_accessed": False,
        "sources": {label: {"evaluation_root": str(run.root), "evaluation_report_sha256": run.report_sha256, "evaluation_manifest_sha256": run.manifest_sha256, "evaluation_records_sha256": run.records_sha256} for label, run in loaded.items()},
        "validation_metadata": {"path": str(validation_metadata_path.resolve()), "sha256": metadata_sha, "expected_sha256": expected_validation_metadata_sha256},
        "localization": localization, "constraints": {label: {scope: _constraint_metrics(run.report, run.records, scope) for scope in SCOPES} for label, run in loaded.items()},
        "cross_language_consistency": {label: _cross_language_consistency(list(run.records)) for label, run in loaded.items()},
        "stratification": {"boundaries": boundaries, "metrics": strata},
        "paired_group_bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed, "confidence_level": 0.95, "resampling_unit": "source_group", "estimator": "ratio_of_sums", "comparisons": bootstrap},
        "failure_cases": {"comparison": "rl_minus_c1", "sampling": "deterministic_sha256_order", "seed": failure_seed, "maximum_records": 20, "records": len(failures), "artifact": "localization_failure_cases.jsonl"},
        "contracts": {"positive_grounding_only": True, "shared_record_identity": True, "shared_fixed_empirical_tertiles": True, "paired_source_group_bootstrap": True, "internal_test_accessed": False, "validation_metadata_hash_bound": True},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        report_path = staging / "localization_analysis.json"
        markdown_path = staging / "localization_analysis.md"
        failures_path = staging / "localization_failure_cases.jsonl"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        failures_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in (report_path, markdown_path, failures_path)), encoding="utf-8")
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report_path = output_dir / "localization_analysis.json"
    return LocalizationAnalysisResult(output_dir, report_path, sha256_file(report_path), output_dir / "localization_analysis.md", output_dir / "localization_failure_cases.jsonl", output_dir / "artifact_manifest.sha256")
