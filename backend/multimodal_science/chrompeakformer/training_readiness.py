"""Audit a verified asset index before any multimodal training run."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TrainingReadinessResult:
    report_path: Path
    report_sha256: str
    asset_index_sha256: str
    asset_count: int
    splits: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, description: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{description} must be an object")
    return value


def _finite(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be numeric") from exc
    _require(math.isfinite(result), f"{description} must be finite")
    return result


def _sha256(value: Any, description: str) -> str:
    result = str(value or "")
    _require(bool(SHA256_PATTERN.fullmatch(result)), f"{description} must be SHA-256")
    return result


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, int | float]:
    _require(bool(values), "Cannot summarize an empty distribution")
    ordered = sorted(values)

    def rounded(value: float) -> float:
        return round(value, 6)

    return {
        "count": len(ordered),
        "min": rounded(ordered[0]),
        "mean": rounded(sum(ordered) / len(ordered)),
        "p50": rounded(_percentile(ordered, 0.50)),
        "p95": rounded(_percentile(ordered, 0.95)),
        "max": rounded(ordered[-1]),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), description)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc


def _rate(positive: int, total: int) -> float:
    return round(positive / total, 6) if total else 0.0


def build_training_readiness_report(
    asset_index_path: Path,
    asset_index_report_path: Path,
    output_path: Path,
    *,
    required_splits: frozenset[str] = frozenset({"train", "validation"}),
) -> TrainingReadinessResult:
    """Validate an asset index and write deterministic, training-facing summary evidence."""

    asset_index_path = asset_index_path.resolve()
    asset_index_report_path = asset_index_report_path.resolve()
    output_path = output_path.resolve()
    _require(bool(required_splits), "At least one required split must be configured")
    _require(
        output_path not in {asset_index_path, asset_index_report_path},
        "Readiness output cannot overwrite an input",
    )
    if not asset_index_path.is_file():
        raise FileNotFoundError(f"Asset index not found: {asset_index_path}")
    if not asset_index_report_path.is_file():
        raise FileNotFoundError(f"Asset-index report not found: {asset_index_report_path}")

    source_report = _read_json(asset_index_report_path, "asset-index report")
    _require(
        source_report.get("schema_version") == "chrompeak-asset-index-report-v1",
        "Unsupported asset-index report schema",
    )
    _require(source_report.get("partial") is False, "Partial asset indexes are not trainable")
    report_counts = _object(source_report.get("counts"), "asset-index report counts")
    actual_index_sha256 = sha256_file(asset_index_path)
    _require(
        source_report.get("asset_index_sha256") == actual_index_sha256,
        "Asset-index SHA-256 does not match its report",
    )
    report_plan_sha256 = _sha256(source_report.get("plan_sha256"), "report plan_sha256")

    asset_ids: set[str] = set()
    record_ids: set[str] = set()
    image_ids: set[int] = set()
    dataset_versions: set[str] = set()
    plan_hashes: set[str] = set()
    jobs_by_split: dict[str, set[str]] = defaultdict(set)
    components_by_split: dict[str, set[str]] = defaultdict(set)
    labels_by_split: dict[str, Counter[int]] = defaultdict(Counter)
    components: Counter[str] = Counter()
    positive_components: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    source_splits: dict[str, set[str]] = defaultdict(set)
    job_splits: dict[str, set[str]] = defaultdict(set)
    xic_point_counts: list[float] = []
    roi_widths: list[float] = []
    positive_peak_widths: list[float] = []
    bbox_widths: list[float] = []
    annotation_count = 0
    asset_count = 0

    with asset_index_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                asset = _object(json.loads(line), f"asset-index line {line_number}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at asset-index line {line_number}") from exc
            _require(
                asset.get("schema_version") == "chrompeak-asset-v1",
                f"Unsupported asset schema at line {line_number}",
            )

            asset_id = str(asset.get("asset_id") or "")
            record_id = str(asset.get("record_id") or "")
            _require(asset_id and asset_id not in asset_ids, f"Duplicate asset_id: {asset_id}")
            _require(
                record_id and record_id not in record_ids,
                f"Duplicate record_id: {record_id}",
            )
            asset_ids.add(asset_id)
            record_ids.add(record_id)

            split = str(asset.get("split") or "")
            job_id = str(asset.get("job_id") or "")
            dataset_version = str(asset.get("dataset_version") or "")
            _require(split != "", f"Missing split at line {line_number}")
            _require(job_id != "", f"Missing job_id at line {line_number}")
            _require(dataset_version != "", f"Missing dataset_version at line {line_number}")
            _require(
                asset.get("metrics_allowed") is True,
                f"Non-metric asset entered training index: {asset_id}",
            )
            dataset_versions.add(dataset_version)
            plan_hashes.add(_sha256(asset.get("plan_sha256"), "asset plan_sha256"))
            source_hash = _sha256(
                asset.get("source_artifact_hash"), "source_artifact_hash"
            )
            _sha256(asset.get("job_output_sha256"), "job_output_sha256")
            source_splits[source_hash].add(split)
            job_splits[job_id].add(split)
            jobs_by_split[split].add(job_id)

            image = _object(asset.get("image"), f"image for {asset_id}")
            image_id = image.get("id")
            _require(
                isinstance(image_id, int) and image_id not in image_ids,
                f"Duplicate or invalid image ID: {image_id}",
            )
            image_ids.add(image_id)
            _sha256(image.get("sha256"), "image sha256")
            _require(
                image.get("width") == 400 and image.get("height") == 300,
                f"Unexpected image dimensions for {asset_id}",
            )

            xic = _object(asset.get("xic"), f"XIC for {asset_id}")
            point_count = xic.get("point_count")
            _require(
                isinstance(point_count, int) and point_count >= 2,
                f"Invalid XIC point count for {asset_id}",
            )
            xic_point_counts.append(float(point_count))

            feature = _object(asset.get("feature"), f"feature for {asset_id}")
            roi_window = feature.get("roi_window")
            _require(
                isinstance(roi_window, list) and len(roi_window) == 2,
                f"Invalid ROI window for {asset_id}",
            )
            rt_lo = _finite(roi_window[0], "ROI lower bound")
            rt_hi = _finite(roi_window[1], "ROI upper bound")
            _require(rt_hi > rt_lo, f"Non-positive ROI width for {asset_id}")
            roi_widths.append(rt_hi - rt_lo)
            _finite(feature.get("q1"), "feature q1")
            _finite(feature.get("q3"), "feature q3")
            _finite(feature.get("rt"), "feature RT")

            label = _object(asset.get("label"), f"label for {asset_id}")
            peak_label = label.get("peak_label")
            _require(peak_label in {0, 1}, f"Invalid peak label for {asset_id}")
            component = str(label.get("component") or "").strip()
            channel = str(label.get("channel") or "").strip()
            _require(component != "", f"Missing component for {asset_id}")
            _require(channel != "", f"Missing channel for {asset_id}")
            _finite(label.get("expected_rt"), "label expected RT")
            boxes = label.get("coco_boxes")
            intervals = label.get("peak_intervals")
            _require(isinstance(boxes, list), f"Invalid COCO boxes for {asset_id}")
            _require(isinstance(intervals, list), f"Invalid peak intervals for {asset_id}")
            if peak_label == 1:
                _require(bool(boxes), f"Positive asset has no COCO box: {asset_id}")
                for interval in intervals:
                    _require(
                        isinstance(interval, list) and len(interval) == 2,
                        f"Invalid peak interval for {asset_id}",
                    )
                    start = _finite(interval[0], "peak interval start")
                    end = _finite(interval[1], "peak interval end")
                    _require(end > start, f"Non-positive peak width for {asset_id}")
                    positive_peak_widths.append(end - start)
                for box in boxes:
                    box_object = _object(box, f"COCO box for {asset_id}")
                    bbox = box_object.get("bbox")
                    _require(
                        isinstance(bbox, list) and len(bbox) == 4,
                        f"Invalid bbox for {asset_id}",
                    )
                    bbox_width = _finite(bbox[2], "bbox width")
                    _require(bbox_width > 0.0, f"Non-positive bbox width for {asset_id}")
                    bbox_widths.append(bbox_width)
            else:
                _require(not boxes, f"Negative asset has COCO boxes: {asset_id}")

            labels_by_split[split][int(peak_label)] += 1
            components[component] += 1
            positive_components[component] += int(peak_label)
            channels[channel] += 1
            components_by_split[split].add(component)
            annotation_count += len(boxes)
            asset_count += 1

    _require(asset_count > 0, "Asset index is empty")
    _require(len(dataset_versions) == 1, "Asset index mixes dataset versions")
    _require(plan_hashes == {report_plan_sha256}, "Asset plan hash differs from report")
    observed_splits = set(labels_by_split)
    missing_splits = sorted(required_splits - observed_splits)
    _require(not missing_splits, f"Required splits are missing: {missing_splits}")
    crossing_sources = sorted(key for key, splits in source_splits.items() if len(splits) > 1)
    crossing_jobs = sorted(key for key, splits in job_splits.items() if len(splits) > 1)
    _require(not crossing_sources, "Source artifacts cross protected splits")
    _require(not crossing_jobs, "Jobs cross protected splits")

    positive_count = sum(counter[1] for counter in labels_by_split.values())
    negative_count = sum(counter[0] for counter in labels_by_split.values())
    _require(positive_count > 0, "Training index contains no positive assets")
    _require(negative_count > 0, "Training index contains no negative assets")
    declared = {
        "assets": asset_count,
        "positive_assets": positive_count,
        "negative_assets": negative_count,
        "annotations": annotation_count,
    }
    for field, observed in declared.items():
        _require(
            report_counts.get(field) == observed,
            f"Declared {field} count does not match asset index",
        )

    split_summaries = {}
    for split in sorted(observed_splits):
        positive = labels_by_split[split][1]
        negative = labels_by_split[split][0]
        total = positive + negative
        split_summaries[split] = {
            "assets": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": _rate(positive, total),
            "jobs": len(jobs_by_split[split]),
            "components": len(components_by_split[split]),
        }

    train_components = components_by_split.get("train", set())
    validation_components = components_by_split.get("validation", set())
    validation_only = sorted(validation_components - train_components)
    warnings = []
    if validation_only:
        warnings.append(
            {
                "code": "validation_only_components",
                "count": len(validation_only),
                "components": validation_only,
            }
        )
    if "train" in split_summaries and "validation" in split_summaries:
        rate_gap = abs(
            split_summaries["train"]["positive_rate"]
            - split_summaries["validation"]["positive_rate"]
        )
    else:
        rate_gap = 0.0

    top_components = []
    ordered_components = sorted(
        components.items(), key=lambda item: (-item[1], item[0])
    )
    for component, total in ordered_components[:20]:
        top_components.append(
            {
                "component": component,
                "assets": total,
                "positive": positive_components[component],
                "positive_rate": _rate(positive_components[component], total),
            }
        )

    payload = {
        "schema_version": "chrompeak-training-readiness-v1",
        "dataset_version": next(iter(dataset_versions)),
        "plan_sha256": report_plan_sha256,
        "asset_index_sha256": actual_index_sha256,
        "quality_gate": {
            "passed": True,
            "checks": {
                "complete_index": True,
                "index_hash_matches": True,
                "declared_counts_match": True,
                "identifiers_unique": True,
                "source_groups_isolated": True,
                "required_splits_present": True,
            },
            "warnings": warnings,
        },
        "counts": {
            "assets": asset_count,
            "positive": positive_count,
            "negative": negative_count,
            "annotations": annotation_count,
            "jobs": len(job_splits),
            "components": len(components),
            "channels": dict(sorted(channels.items())),
            "splits": split_summaries,
        },
        "class_balance": {
            "overall_positive_rate": _rate(positive_count, asset_count),
            "train_validation_positive_rate_absolute_gap": round(rate_gap, 6),
        },
        "coverage": {
            "train_components": len(train_components),
            "validation_components": len(validation_components),
            "shared_train_validation_components": len(
                train_components & validation_components
            ),
            "validation_only_components": validation_only,
            "top_components": top_components,
        },
        "distributions": {
            "xic_point_count": _distribution(xic_point_counts),
            "roi_width_minutes": _distribution(roi_widths),
            "positive_peak_width_minutes": _distribution(positive_peak_widths),
            "positive_bbox_width_pixels": _distribution(bbox_widths),
        },
    }
    _write_json_atomic(output_path, payload)
    return TrainingReadinessResult(
        report_path=output_path,
        report_sha256=sha256_file(output_path),
        asset_index_sha256=actual_index_sha256,
        asset_count=asset_count,
        splits=tuple(sorted(observed_splits)),
    )
