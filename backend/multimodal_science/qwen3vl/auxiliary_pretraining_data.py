"""Materialize strictly unlabeled auxiliary XIC traces for representation pretraining."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multimodal_science.data.manifest import sha256_file


AUXILIARY_PRETRAINING_DATASET_SCHEMA = (
    "chrompeak-auxiliary-signal-pretraining-dataset-v1"
)
AUXILIARY_PRETRAINING_EXAMPLE_SCHEMA = (
    "chrompeak-auxiliary-signal-pretraining-example-v1"
)


@dataclass(frozen=True)
class AuxiliaryPretrainingDatasetResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    manifest_sha256: str
    asset_count: int
    source_group_count: int
    target_points: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc


def _safe_child(root: Path, relative: str) -> Path:
    candidate_path = Path(relative)
    _require(not candidate_path.is_absolute(), f"Expected a relative path: {relative}")
    candidate = (root / candidate_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {relative}") from exc
    return candidate


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(_object(json.loads(line), f"{label} line {line_number}"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {label} line {line_number}") from exc
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _verify_source_manifest(root: Path, manifest_sha256: str) -> None:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), "Auxiliary index manifest is missing")
    _require(sha256_file(manifest) == manifest_sha256, "Auxiliary index manifest drift")
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        _require(len(parts) == 2, f"Malformed source manifest line {line_number}")
        digest, relative = parts
        path = _safe_child(root, relative)
        _require(path.is_file(), f"Auxiliary index artifact missing: {relative}")
        _require(sha256_file(path) == digest, f"Auxiliary index artifact drift: {relative}")
        listed.add(relative)
    _require(
        listed == {"auxiliary_asset_index.jsonl", "auxiliary_asset_index_report.json"},
        "Auxiliary index manifest has an unexpected artifact set",
    )


def _canonicalize_rt_axis(rt: Any, np: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Return a strict RT axis plus the stable permutation and duplicate groups.

    Vendor exports can contain repeated timestamps or concatenate scans out of RT order.
    Interpolation requires a strictly increasing coordinate, so normalization is explicit
    and its complete audit summary is persisted in the derived dataset report.
    """

    rt = np.asarray(rt, dtype=np.float64)
    _require(rt.ndim == 1 and rt.size >= 2, "Invalid auxiliary RT axis")
    _require(np.isfinite(rt).all(), "Non-finite auxiliary RT axis")
    deltas = np.diff(rt)
    adjacent_decreases = int(np.count_nonzero(deltas < 0.0))
    adjacent_duplicates = int(np.count_nonzero(deltas == 0.0))
    backward = deltas[deltas < 0.0]
    maximum_backward_step = float(-np.min(backward)) if backward.size else 0.0

    order = np.argsort(rt, kind="stable")
    sorted_rt = rt[order]
    canonical_rt, group_starts = np.unique(sorted_rt, return_index=True)
    _require(
        canonical_rt.size >= 2,
        "Auxiliary RT axis has fewer than two unique points",
    )
    _require(
        bool(np.all(np.diff(canonical_rt) > 0.0)),
        "Auxiliary RT axis normalization did not produce a strict axis",
    )
    duplicate_points_collapsed = int(rt.size - canonical_rt.size)
    metadata = {
        "input_point_count": int(rt.size),
        "canonical_point_count": int(canonical_rt.size),
        "adjacent_decreases": adjacent_decreases,
        "adjacent_duplicates": adjacent_duplicates,
        "duplicate_points_collapsed": duplicate_points_collapsed,
        "maximum_backward_step_minutes": maximum_backward_step,
        "was_reordered": adjacent_decreases > 0,
        "had_duplicate_rt": duplicate_points_collapsed > 0,
        "normalization_applied": adjacent_decreases > 0
        or duplicate_points_collapsed > 0,
    }
    return canonical_rt, order, group_starts, metadata


def _canonicalize_signal(
    signal: Any,
    order: Any,
    group_starts: Any,
    *,
    expected_points: int,
    np: Any,
) -> Any:
    """Apply the RT permutation and preserve peak amplitude at duplicate timestamps."""

    signal = np.asarray(signal, dtype=np.float64)
    _require(
        signal.ndim == 1 and signal.size == expected_points,
        "Auxiliary signal width does not match its RT axis",
    )
    _require(np.isfinite(signal).all(), "Non-finite auxiliary signal")
    ordered = signal[order]
    if group_starts.size == ordered.size:
        return ordered
    canonical = np.maximum.reduceat(ordered, group_starts)
    _require(np.isfinite(canonical).all(), "Non-finite canonical auxiliary signal")
    return canonical


def build_auxiliary_pretraining_dataset(
    auxiliary_index_root: Path,
    auxiliary_index_report_sha256: str,
    auxiliary_index_manifest_sha256: str,
    assets_root: Path,
    output_dir: Path,
    *,
    target_points: int = 160,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AuxiliaryPretrainingDatasetResult:
    """Create normalized train-only signals while preserving the zero-label contract."""

    import numpy as np

    index_root = auxiliary_index_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    _require(index_root.is_dir(), f"Auxiliary index root not found: {index_root}")
    _require(assets_root.is_dir(), f"Auxiliary asset root not found: {assets_root}")
    _require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    _require(target_points >= 32, "target_points must be at least 32")
    _verify_source_manifest(index_root, auxiliary_index_manifest_sha256)

    index_path = index_root / "auxiliary_asset_index.jsonl"
    source_report_path = index_root / "auxiliary_asset_index_report.json"
    source_report = _read_json(source_report_path, "auxiliary index report")
    _require(
        sha256_file(source_report_path) == auxiliary_index_report_sha256,
        "Auxiliary index report drift",
    )
    _require(
        source_report.get("schema_version")
        == "chrompeak-auxiliary-asset-index-report-v1",
        "Unsupported auxiliary index report schema",
    )
    _require(source_report.get("quality_gate_passed") is True, "Auxiliary index gate failed")
    contracts = _object(source_report.get("contracts"), "auxiliary index contracts")
    for name, expected in (
        ("complete_plan_coverage", True),
        ("labels_present", False),
        ("metrics_allowed", False),
        ("train_supervision_present", False),
        ("auxiliary_unlabeled_train_eligible", True),
        ("validation_membership_changed", False),
        ("benchmark_membership_changed", False),
        ("internal_test_accessed", False),
    ):
        _require(contracts.get(name) is expected, f"Auxiliary source contract failed: {name}")
    index_sha256 = sha256_file(index_path)
    _require(source_report.get("asset_index_sha256") == index_sha256, "Index digest drift")

    assets = _read_jsonl(index_path, "auxiliary asset index")
    declared_count = int(
        _object(source_report.get("counts"), "auxiliary index counts").get(
            "extracted_trace_assets", -1
        )
    )
    _require(len(assets) == declared_count and assets, "Auxiliary asset count mismatch")
    asset_ids = [str(asset.get("asset_id") or "") for asset in assets]
    _require(all(asset_ids), "Auxiliary asset ID is empty")
    _require(len(asset_ids) == len(set(asset_ids)), "Duplicate auxiliary asset IDs")

    signals = np.empty((len(assets), target_points), dtype=np.float32)
    examples: list[dict[str, Any]] = []
    availability = Counter()
    source_groups = Counter()
    verified_matrices: dict[str, Any] = {}
    verified_matrix_digests: dict[str, str] = {}
    canonical_axes: dict[str, tuple[Any, Any, Any, dict[str, Any]]] = {}

    for row, asset in enumerate(assets):
        _require(
            asset.get("schema_version") == "chrompeak-auxiliary-asset-v1",
            f"Unsupported auxiliary asset schema at row {row}",
        )
        _require(asset.get("split") == "auxiliary_unlabeled_train", "Bad auxiliary split")
        _require(asset.get("metrics_allowed") is False, "Metrics enabled on auxiliary row")
        supervision = _object(asset.get("supervision"), "auxiliary supervision")
        _require(supervision.get("label_status") == "unlabeled", "Auxiliary label drift")
        _require(
            supervision.get("supervised_train_eligible") is False,
            "Auxiliary row became supervised",
        )
        _require(
            supervision.get("auxiliary_unlabeled_train_eligible") is True,
            "Auxiliary row is not pretraining eligible",
        )
        _require(supervision.get("benchmark_eligible") is False, "Auxiliary benchmark leak")
        _require(supervision.get("internal_test_accessed") is False, "Internal test leak")

        image = _object(asset.get("image"), "auxiliary image")
        image_path = _safe_child(assets_root, str(image.get("path") or ""))
        _require(image_path.is_file(), f"Auxiliary image missing: {image_path}")
        _require(sha256_file(image_path) == image.get("sha256"), "Auxiliary image drift")

        xic = _object(asset.get("xic"), "auxiliary XIC")
        matrix_relative = str(xic.get("path") or "")
        matrix_path = _safe_child(assets_root, matrix_relative)
        expected_matrix_sha = str(xic.get("sha256") or "")
        if matrix_relative not in verified_matrices:
            _require(matrix_path.is_file(), f"Auxiliary XIC matrix missing: {matrix_relative}")
            _require(
                sha256_file(matrix_path) == expected_matrix_sha,
                f"Auxiliary XIC matrix drift: {matrix_relative}",
            )
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            _require(matrix.ndim == 2 and matrix.shape[0] >= 2, "Invalid XIC matrix shape")
            canonical_axes[matrix_relative] = _canonicalize_rt_axis(matrix[0], np)
            verified_matrices[matrix_relative] = matrix
            verified_matrix_digests[matrix_relative] = expected_matrix_sha
        else:
            _require(
                verified_matrix_digests[matrix_relative] == expected_matrix_sha,
                "Conflicting XIC matrix digest",
            )
        matrix = verified_matrices[matrix_relative]
        signal_row = int(xic.get("signal_row", -1))
        _require(1 <= signal_row < matrix.shape[0], "Auxiliary signal row is out of range")
        _require(int(xic.get("point_count", -1)) == matrix.shape[1], "XIC width drift")

        feature = _object(asset.get("feature"), "auxiliary feature")
        roi_window = feature.get("roi_window")
        _require(isinstance(roi_window, list) and len(roi_window) == 2, "Bad ROI window")
        rt_lo = _finite(roi_window[0], "ROI lower bound")
        rt_hi = _finite(roi_window[1], "ROI upper bound")
        _require(rt_hi > rt_lo, "ROI window is empty")
        target_rt = np.linspace(rt_lo, rt_hi, target_points, dtype=np.float64)
        canonical_rt, order, group_starts, axis_metadata = canonical_axes[matrix_relative]
        canonical_signal = _canonicalize_signal(
            matrix[signal_row],
            order,
            group_starts,
            expected_points=matrix.shape[1],
            np=np,
        )
        raw = np.interp(
            target_rt,
            canonical_rt,
            canonical_signal,
            left=0.0,
            right=0.0,
        )
        _require(np.isfinite(raw).all(), "Non-finite auxiliary signal")
        baseline = float(np.quantile(raw, 0.05))
        logged = np.log1p(np.maximum(raw - baseline, 0.0))
        maximum = float(np.max(logged))
        signal_available = maximum > 0.0
        if signal_available:
            signals[row] = (logged / maximum).astype(np.float32)
            availability["available"] += 1
        else:
            signals[row].fill(0.0)
            availability["unavailable"] += 1
        source_group = str(asset.get("source_group") or "")
        _require(bool(source_group), "Auxiliary source group is empty")
        source_groups[source_group] += 1
        examples.append(
            {
                "schema_version": AUXILIARY_PRETRAINING_EXAMPLE_SCHEMA,
                "row": row,
                "asset_id": asset["asset_id"],
                "source_group": source_group,
                "job_id": asset["job_id"],
                "image": {
                    "path": image["path"],
                    "sha256": image["sha256"],
                },
                "signal": {
                    "array": "signals.npy",
                    "row": row,
                    "length": target_points,
                    "available": signal_available,
                    "source_matrix_path": matrix_relative,
                    "source_matrix_sha256": expected_matrix_sha,
                    "source_signal_row": signal_row,
                    "rt_axis_normalization": axis_metadata,
                },
                "feature": {
                    "q1": _finite(feature.get("q1"), "feature q1"),
                    "q3": _finite(feature.get("q3"), "feature q3"),
                    "rt": _finite(feature.get("rt"), "feature RT"),
                    "roi_window": [rt_lo, rt_hi],
                },
                "supervision": {
                    "label_status": "unlabeled",
                    "metrics_allowed": False,
                    "benchmark_eligible": False,
                    "internal_test_accessed": False,
                },
            }
        )
        if progress_callback is not None:
            progress_callback(row + 1, len(assets), str(asset["asset_id"]))

    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=staging_parent))
    try:
        signals_path = staging / "signals.npy"
        with signals_path.open("wb") as stream:
            np.save(stream, signals, allow_pickle=False)
        examples_path = staging / "examples.jsonl"
        _write_jsonl(examples_path, examples)
        report_path = staging / "auxiliary_pretraining_dataset_report.json"
        axis_summaries = [axis[3] for axis in canonical_axes.values()]
        normalized_matrices = sum(
            int(summary["normalization_applied"]) for summary in axis_summaries
        )
        rt_axis_normalization = {
            "method": "stable_sort_then_collapse_exact_duplicates",
            "duplicate_intensity_reducer": "maximum",
            "matrices": len(axis_summaries),
            "matrices_already_strict": len(axis_summaries) - normalized_matrices,
            "matrices_normalized": normalized_matrices,
            "matrices_reordered": sum(
                int(summary["was_reordered"]) for summary in axis_summaries
            ),
            "matrices_with_duplicate_rt": sum(
                int(summary["had_duplicate_rt"]) for summary in axis_summaries
            ),
            "input_points": sum(
                int(summary["input_point_count"]) for summary in axis_summaries
            ),
            "canonical_points": sum(
                int(summary["canonical_point_count"]) for summary in axis_summaries
            ),
            "duplicate_points_collapsed": sum(
                int(summary["duplicate_points_collapsed"]) for summary in axis_summaries
            ),
            "adjacent_decreases": sum(
                int(summary["adjacent_decreases"]) for summary in axis_summaries
            ),
            "maximum_backward_step_minutes": max(
                (
                    float(summary["maximum_backward_step_minutes"])
                    for summary in axis_summaries
                ),
                default=0.0,
            ),
        }
        report = {
            "schema_version": AUXILIARY_PRETRAINING_DATASET_SCHEMA,
            "sources": {
                "auxiliary_index_report_sha256": auxiliary_index_report_sha256,
                "auxiliary_index_manifest_sha256": auxiliary_index_manifest_sha256,
                "auxiliary_asset_index_sha256": index_sha256,
            },
            "target_points": target_points,
            "counts": {
                "assets": len(assets),
                "signals_available": availability["available"],
                "signals_unavailable": availability["unavailable"],
                "independent_source_groups": len(source_groups),
                "labels": 0,
                "supervised_train_assets": 0,
            },
            "source_groups": dict(sorted(source_groups.items())),
            "rt_axis_normalization": rt_axis_normalization,
            "warnings": (
                [
                    {
                        "code": "rt_axis_normalized",
                        "matrices": normalized_matrices,
                        "duplicate_points_collapsed": rt_axis_normalization[
                            "duplicate_points_collapsed"
                        ],
                        "adjacent_decreases": rt_axis_normalization[
                            "adjacent_decreases"
                        ],
                    }
                ]
                if normalized_matrices
                else []
            ),
            "artifacts": {
                "signals": {
                    "path": signals_path.name,
                    "sha256": sha256_file(signals_path),
                    "dtype": "float32",
                    "shape": [len(assets), target_points],
                },
                "examples": {
                    "path": examples_path.name,
                    "sha256": sha256_file(examples_path),
                    "records": len(examples),
                },
            },
            "contracts": {
                "auxiliary_unlabeled_train_only": True,
                "labels_present": False,
                "metrics_allowed": False,
                "validation_opened": False,
                "internal_test_accessed": False,
                "benchmark_eligible": False,
                "image_and_signal_are_derived_views_of_the_same_trace": True,
                "rt_axis_normalization_is_explicit": True,
                "rt_axes_strictly_increasing_after_normalization": True,
                "duplicate_rt_intensity_reducer": "maximum",
            },
            "quality_gate_passed": True,
            "development_training_eligible": True,
            "development_comparison_eligible": False,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        artifact_paths = (signals_path, examples_path, report_path)
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in artifact_paths
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AuxiliaryPretrainingDatasetResult(
        output_dir=output_dir,
        report_path=output_dir / report_path.name,
        report_sha256=sha256_file(output_dir / report_path.name),
        manifest_path=output_dir / manifest_path.name,
        manifest_sha256=sha256_file(output_dir / manifest_path.name),
        asset_count=len(assets),
        source_group_count=len(source_groups),
        target_points=target_points,
    )
