"""Profile ROI-cropped numerical chromatograms before choosing a model input size."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multimodal_science.data.manifest import sha256_file


@dataclass(frozen=True)
class SequencePreflightResult:
    report_path: Path
    report_sha256: str
    asset_index_sha256: str
    asset_count: int
    matrix_count: int


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


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected a relative XIC path: {relative}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"XIC path escapes assets root: {relative}") from exc
    return candidate


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


def build_sequence_preflight_report(
    asset_index_path: Path,
    readiness_report_path: Path,
    assets_root: Path,
    output_path: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SequencePreflightResult:
    """Validate and profile each referenced XIC matrix exactly once."""

    import numpy as np

    asset_index_path = asset_index_path.resolve()
    readiness_report_path = readiness_report_path.resolve()
    assets_root = assets_root.resolve()
    output_path = output_path.resolve()
    if not asset_index_path.is_file():
        raise FileNotFoundError(f"Asset index not found: {asset_index_path}")
    if not readiness_report_path.is_file():
        raise FileNotFoundError(f"Readiness report not found: {readiness_report_path}")
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Assets root not found: {assets_root}")
    _require(
        output_path not in {asset_index_path, readiness_report_path},
        "Sequence preflight output cannot overwrite an input",
    )

    try:
        readiness = _object(
            json.loads(readiness_report_path.read_text(encoding="utf-8")),
            "readiness report",
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid readiness report JSON: {readiness_report_path}") from exc
    _require(
        readiness.get("schema_version") == "chrompeak-training-readiness-v1",
        "Unsupported readiness report schema",
    )
    quality_gate = _object(readiness.get("quality_gate"), "readiness quality gate")
    _require(quality_gate.get("passed") is True, "Training-readiness gate did not pass")
    actual_index_sha256 = sha256_file(asset_index_path)
    _require(
        readiness.get("asset_index_sha256") == actual_index_sha256,
        "Asset index does not match the readiness report",
    )
    readiness_counts = _object(readiness.get("counts"), "readiness counts")

    by_matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    asset_count = 0
    labels: Counter[int] = Counter()
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
            xic = _object(asset.get("xic"), f"XIC at line {line_number}")
            feature = _object(asset.get("feature"), f"feature at line {line_number}")
            label = _object(asset.get("label"), f"label at line {line_number}")
            xic_path = str(xic.get("path") or "")
            _require(xic_path != "", f"Missing XIC path at line {line_number}")
            signal_row = xic.get("signal_row")
            rt_row = xic.get("rt_row")
            point_count = xic.get("point_count")
            _require(
                isinstance(signal_row, int) and signal_row >= 1,
                f"Invalid XIC signal row at line {line_number}",
            )
            _require(rt_row == 0, f"Unsupported XIC RT row at line {line_number}")
            _require(
                isinstance(point_count, int) and point_count >= 2,
                f"Invalid XIC point count at line {line_number}",
            )
            roi_window = feature.get("roi_window")
            _require(
                isinstance(roi_window, list) and len(roi_window) == 2,
                f"Invalid ROI window at line {line_number}",
            )
            rt_lo = _finite(roi_window[0], "ROI lower bound")
            rt_hi = _finite(roi_window[1], "ROI upper bound")
            _require(rt_hi > rt_lo, f"Non-positive ROI width at line {line_number}")
            peak_label = label.get("peak_label")
            _require(peak_label in {0, 1}, f"Invalid peak label at line {line_number}")
            by_matrix[xic_path].append(
                {
                    "asset_id": str(asset.get("asset_id") or ""),
                    "signal_row": signal_row,
                    "point_count": point_count,
                    "rt_lo": rt_lo,
                    "rt_hi": rt_hi,
                    "peak_label": int(peak_label),
                }
            )
            labels[int(peak_label)] += 1
            asset_count += 1

    _require(asset_count > 0, "Asset index is empty")
    _require(
        readiness_counts.get("assets") == asset_count,
        "Readiness asset count does not match the asset index",
    )

    full_point_counts: list[float] = []
    full_trace_spans: list[float] = []
    cropped_point_counts: list[float] = []
    rt_step_seconds: list[float] = []
    effective_roi_step_seconds: list[float] = []
    crop_fractions: list[float] = []
    signal_dynamic_ranges: list[float] = []
    signal_maxima: list[float] = []
    constant_signal_by_label: Counter[int] = Counter()
    negative_value_by_label: Counter[int] = Counter()
    seen_rows: set[tuple[str, int]] = set()

    matrix_items = sorted(by_matrix.items())
    for matrix_number, (relative_path, records) in enumerate(matrix_items, start=1):
        matrix_path = _safe_child(assets_root, relative_path)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Referenced XIC matrix not found: {relative_path}")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        _require(matrix.ndim == 2, f"XIC matrix must be two-dimensional: {relative_path}")
        expected_points = {record["point_count"] for record in records}
        _require(
            expected_points == {matrix.shape[1]},
            f"XIC point-count mismatch: {relative_path}",
        )
        max_signal_row = max(record["signal_row"] for record in records)
        _require(
            max_signal_row < matrix.shape[0],
            f"XIC signal row exceeds matrix shape: {relative_path}",
        )
        rt = np.asarray(matrix[0], dtype=np.float64)
        _require(np.isfinite(rt).all(), f"Non-finite RT values: {relative_path}")
        intervals = np.diff(rt)
        _require(
            bool(intervals.size)
            and bool(np.isfinite(intervals).all())
            and bool(np.all(intervals > 0.0)),
            f"RT axis is not strictly increasing: {relative_path}",
        )
        full_point_counts.append(float(matrix.shape[1]))
        full_trace_spans.append(float(rt[-1] - rt[0]))
        rt_step_seconds.extend(float(value) for value in intervals * 60.0)

        for record in records:
            row_key = (relative_path, record["signal_row"])
            _require(row_key not in seen_rows, f"Duplicate XIC signal row: {row_key}")
            seen_rows.add(row_key)
            left = int(np.searchsorted(rt, record["rt_lo"], side="left"))
            right = int(np.searchsorted(rt, record["rt_hi"], side="right"))
            _require(
                right - left >= 2,
                f"ROI contains fewer than two XIC points: {record['asset_id']}",
            )
            signal = np.asarray(
                matrix[record["signal_row"], left:right], dtype=np.float64
            )
            _require(
                np.isfinite(signal).all(),
                f"Non-finite ROI signal: {record['asset_id']}",
            )
            cropped_point_counts.append(float(signal.size))
            crop_fractions.append(float(signal.size / matrix.shape[1]))
            effective_roi_step_seconds.append(
                float((rt[right - 1] - rt[left]) * 60.0 / (signal.size - 1))
            )
            signal_minimum = float(np.min(signal))
            signal_maximum = float(np.max(signal))
            dynamic_range = signal_maximum - signal_minimum
            _require(
                math.isfinite(dynamic_range),
                f"Non-finite ROI dynamic range: {record['asset_id']}",
            )
            signal_dynamic_ranges.append(dynamic_range)
            signal_maxima.append(signal_maximum)
            if dynamic_range <= 0.0:
                constant_signal_by_label[record["peak_label"]] += 1
            if signal_minimum < 0.0:
                negative_value_by_label[record["peak_label"]] += 1

        del intervals, rt, signal, matrix
        if progress_callback is not None:
            progress_callback(matrix_number, len(matrix_items), relative_path)

    warnings = []
    constant_total = sum(constant_signal_by_label.values())
    negative_total = sum(negative_value_by_label.values())
    if constant_total:
        warnings.append(
            {
                "code": "constant_roi_signals",
                "assets": constant_total,
                "by_label": {
                    str(label): count
                    for label, count in sorted(constant_signal_by_label.items())
                },
            }
        )
    if negative_total:
        warnings.append(
            {
                "code": "negative_roi_values",
                "assets": negative_total,
                "by_label": {
                    str(label): count
                    for label, count in sorted(negative_value_by_label.items())
                },
            }
        )

    payload = {
        "schema_version": "chrompeak-sequence-preflight-v2",
        "dataset_version": readiness.get("dataset_version"),
        "plan_sha256": readiness.get("plan_sha256"),
        "asset_index_sha256": actual_index_sha256,
        "readiness_report_sha256": sha256_file(readiness_report_path),
        "quality_gate": {
            "passed": True,
            "checks": {
                "readiness_gate_passed": True,
                "asset_index_hash_matches": True,
                "matrix_paths_confined": True,
                "matrix_shapes_match": True,
                "signal_rows_unique": True,
                "rt_axes_strictly_increasing": True,
                "roi_signals_finite": True,
                "roi_windows_have_multiple_points": True,
            },
            "warnings": warnings,
        },
        "counts": {
            "assets": asset_count,
            "matrices": len(matrix_items),
            "positive": labels[1],
            "negative": labels[0],
            "constant_signal_assets": constant_total,
            "negative_value_assets": negative_total,
        },
        "distributions": {
            "full_trace_point_count_per_matrix": _distribution(full_point_counts),
            "full_trace_span_minutes_per_matrix": _distribution(full_trace_spans),
            "rt_step_seconds_all_intervals": _distribution(rt_step_seconds),
            "cropped_roi_point_count_per_asset": _distribution(cropped_point_counts),
            "effective_roi_step_seconds_per_asset": _distribution(
                effective_roi_step_seconds
            ),
            "roi_fraction_of_full_trace": _distribution(crop_fractions),
            "roi_signal_dynamic_range": _distribution(signal_dynamic_ranges),
            "roi_signal_maximum": _distribution(signal_maxima),
        },
    }
    _write_json_atomic(output_path, payload)
    return SequencePreflightResult(
        report_path=output_path,
        report_sha256=sha256_file(output_path),
        asset_index_sha256=actual_index_sha256,
        asset_count=asset_count,
        matrix_count=len(matrix_items),
    )
