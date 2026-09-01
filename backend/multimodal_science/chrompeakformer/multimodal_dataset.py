"""Materialize aligned image, numerical, metadata, and peak-boundary examples."""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multimodal_science.data.manifest import sha256_file

DATASET_SCHEMA = "chrompeak-multimodal-dataset-v1"
EXAMPLE_SCHEMA = "chrompeak-multimodal-example-v1"
SCALAR_FEATURE_NAMES = (
    "log1p_raw_maximum",
    "log1p_dynamic_range",
    "roi_width_minutes",
    "expected_rt_minutes",
    "q1",
    "q3",
    "signal_available",
)
STANDARDIZED_SCALAR_COUNT = len(SCALAR_FEATURE_NAMES) - 1
IMAGE_WIDTH = 400.0
BOUNDARY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class MultimodalDatasetResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    asset_index_sha256: str
    asset_count: int
    target_points: int
    cached: bool


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


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), description)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected a relative path: {relative}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {relative}") from exc
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _validate_report_chain(
    asset_index_path: Path,
    readiness_report_path: Path,
    sequence_preflight_path: Path,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    asset_index_sha256 = sha256_file(asset_index_path)
    readiness_sha256 = sha256_file(readiness_report_path)
    sequence_preflight_sha256 = sha256_file(sequence_preflight_path)
    readiness = _read_json(readiness_report_path, "training-readiness report")
    preflight = _read_json(sequence_preflight_path, "sequence-preflight report")
    _require(
        readiness.get("schema_version") == "chrompeak-training-readiness-v1",
        "Unsupported training-readiness schema",
    )
    _require(
        _object(readiness.get("quality_gate"), "training-readiness gate").get("passed")
        is True,
        "Training-readiness gate did not pass",
    )
    _require(
        readiness.get("asset_index_sha256") == asset_index_sha256,
        "Asset index does not match the training-readiness report",
    )
    _require(
        preflight.get("schema_version") == "chrompeak-sequence-preflight-v2",
        "Sequence-preflight v2 is required",
    )
    _require(
        _object(preflight.get("quality_gate"), "sequence-preflight gate").get("passed")
        is True,
        "Sequence-preflight gate did not pass",
    )
    _require(
        preflight.get("asset_index_sha256") == asset_index_sha256,
        "Asset index does not match the sequence-preflight report",
    )
    _require(
        preflight.get("readiness_report_sha256") == readiness_sha256,
        "Sequence preflight does not match the training-readiness report",
    )
    return (
        asset_index_sha256,
        readiness_sha256,
        sequence_preflight_sha256,
        readiness,
        preflight,
    )


def _read_assets(
    path: Path, include_splits: frozenset[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
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
            split = str(asset.get("split") or "")
            if split not in include_splits:
                continue
            xic = _object(asset.get("xic"), f"XIC at line {line_number}")
            matrix_path = str(xic.get("path") or "")
            _require(matrix_path != "", f"Missing XIC path at line {line_number}")
            asset["_dataset_row"] = len(by_split[split])
            by_split[split].append(asset)
            by_matrix[matrix_path].append(asset)
    missing = sorted(include_splits - set(by_split))
    _require(not missing, f"Selected splits contain no assets: {missing}")
    return dict(by_split), dict(by_matrix)


def _boundary_target(asset: dict[str, Any]) -> tuple[float, float, float]:
    label = _object(asset.get("label"), "asset label")
    feature = _object(asset.get("feature"), "asset feature")
    peak_label = label.get("peak_label")
    _require(peak_label in {0, 1}, f"Invalid peak label: {asset.get('asset_id')}")
    boxes = label.get("coco_boxes")
    _require(isinstance(boxes, list), f"Invalid COCO boxes: {asset.get('asset_id')}")
    if peak_label == 0:
        _require(not boxes, f"Negative asset has a COCO box: {asset.get('asset_id')}")
        return 0.0, -1.0, -1.0

    _require(
        len(boxes) == 1,
        f"Dataset v1 requires exactly one positive interval: {asset.get('asset_id')}",
    )
    box = _object(boxes[0], "positive COCO box")
    visible_interval = box.get("visible_rt_interval")
    bbox = box.get("bbox")
    roi_window = feature.get("roi_window")
    _require(
        isinstance(visible_interval, list) and len(visible_interval) == 2,
        f"Invalid visible RT interval: {asset.get('asset_id')}",
    )
    _require(
        isinstance(bbox, list) and len(bbox) == 4,
        f"Invalid COCO bbox: {asset.get('asset_id')}",
    )
    _require(
        isinstance(roi_window, list) and len(roi_window) == 2,
        f"Invalid ROI window: {asset.get('asset_id')}",
    )
    rt_lo = _finite(roi_window[0], "ROI lower bound")
    rt_hi = _finite(roi_window[1], "ROI upper bound")
    start = (_finite(visible_interval[0], "peak start") - rt_lo) / (rt_hi - rt_lo)
    end = (_finite(visible_interval[1], "peak end") - rt_lo) / (rt_hi - rt_lo)
    bbox_start = _finite(bbox[0], "bbox x") / IMAGE_WIDTH
    bbox_end = (_finite(bbox[0], "bbox x") + _finite(bbox[2], "bbox width")) / IMAGE_WIDTH
    _require(0.0 <= start < end <= 1.0, f"Invalid normalized boundary target")
    _require(
        abs(start - bbox_start) <= BOUNDARY_TOLERANCE
        and abs(end - bbox_end) <= BOUNDARY_TOLERANCE,
        f"Image and RT boundaries disagree: {asset.get('asset_id')}",
    )
    return 1.0, start, end


def _existing_result(
    output_dir: Path,
    *,
    asset_index_sha256: str,
    readiness_sha256: str,
    preflight_sha256: str,
    target_points: int,
    include_splits: frozenset[str],
) -> MultimodalDatasetResult | None:
    report_path = output_dir / "dataset_report.json"
    if not output_dir.exists():
        return None
    if not report_path.is_file():
        raise FileExistsError(f"Output directory exists without a dataset report: {output_dir}")
    report = _read_json(report_path, "multimodal dataset report")
    expected = {
        "schema_version": DATASET_SCHEMA,
        "asset_index_sha256": asset_index_sha256,
        "readiness_report_sha256": readiness_sha256,
        "sequence_preflight_sha256": preflight_sha256,
        "target_points": target_points,
        "splits": sorted(include_splits),
    }
    for field, value in expected.items():
        _require(report.get(field) == value, f"Existing dataset {field} mismatch")
    artifacts = _object(report.get("artifacts"), "dataset artifacts")
    for artifact in artifacts.values():
        value = _object(artifact, "dataset artifact")
        artifact_path = _safe_child(output_dir, str(value.get("path") or ""))
        _require(artifact_path.is_file(), f"Dataset artifact missing: {artifact_path.name}")
        _require(
            sha256_file(artifact_path) == value.get("sha256"),
            f"Dataset artifact hash mismatch: {artifact_path.name}",
        )
    counts = _object(report.get("counts"), "dataset counts")
    return MultimodalDatasetResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        asset_index_sha256=asset_index_sha256,
        asset_count=int(counts["assets"]),
        target_points=target_points,
        cached=True,
    )


def build_multimodal_dataset(
    asset_index_path: Path,
    readiness_report_path: Path,
    sequence_preflight_path: Path,
    assets_root: Path,
    output_dir: Path,
    *,
    include_splits: frozenset[str] = frozenset({"train", "validation"}),
    target_points: int = 160,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> MultimodalDatasetResult:
    """Build a model-ready, provenance-bound multimodal dataset atomically."""

    import numpy as np

    asset_index_path = asset_index_path.resolve()
    readiness_report_path = readiness_report_path.resolve()
    sequence_preflight_path = sequence_preflight_path.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    for path, description in (
        (asset_index_path, "asset index"),
        (readiness_report_path, "training-readiness report"),
        (sequence_preflight_path, "sequence-preflight report"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description.capitalize()} not found: {path}")
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Assets root not found: {assets_root}")
    _require(bool(include_splits), "At least one split must be selected")
    _require(target_points >= 32, "target_points must be at least 32")

    (
        asset_index_sha256,
        readiness_sha256,
        preflight_sha256,
        readiness,
        preflight,
    ) = _validate_report_chain(
        asset_index_path, readiness_report_path, sequence_preflight_path
    )
    cached = _existing_result(
        output_dir,
        asset_index_sha256=asset_index_sha256,
        readiness_sha256=readiness_sha256,
        preflight_sha256=preflight_sha256,
        target_points=target_points,
        include_splits=include_splits,
    )
    if cached is not None:
        return cached

    by_split, by_matrix = _read_assets(asset_index_path, include_splits)
    readiness_splits = _object(
        _object(readiness.get("counts"), "readiness counts").get("splits"),
        "readiness split counts",
    )
    for split, assets in by_split.items():
        declared = _object(readiness_splits.get(split), f"readiness split {split}")
        _require(
            declared.get("assets") == len(assets),
            f"Readiness count mismatch for split {split}",
        )
    preflight_counts = _object(preflight.get("counts"), "sequence-preflight counts")
    readiness_counts = _object(readiness.get("counts"), "readiness counts")
    _require(
        preflight_counts.get("assets") == readiness_counts.get("assets"),
        "Sequence-preflight asset count does not match readiness evidence",
    )

    signals = {
        split: np.empty((len(assets), target_points), dtype=np.float32)
        for split, assets in by_split.items()
    }
    raw_scalars = {
        split: np.empty((len(assets), len(SCALAR_FEATURE_NAMES)), dtype=np.float64)
        for split, assets in by_split.items()
    }
    targets = {
        split: np.empty((len(assets), 3), dtype=np.float32)
        for split, assets in by_split.items()
    }
    examples: dict[str, list[dict[str, Any] | None]] = {
        split: [None] * len(assets) for split, assets in by_split.items()
    }
    label_counts: dict[str, Counter[int]] = defaultdict(Counter)

    matrix_items = sorted(by_matrix.items())
    for matrix_number, (relative_path, assets) in enumerate(matrix_items, start=1):
        matrix_path = _safe_child(assets_root, relative_path)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Referenced XIC matrix not found: {relative_path}")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        _require(matrix.ndim == 2, f"XIC matrix must be two-dimensional: {relative_path}")
        rt = np.asarray(matrix[0], dtype=np.float64)
        _require(np.isfinite(rt).all(), f"Non-finite RT values: {relative_path}")
        _require(
            bool(np.all(np.diff(rt) > 0.0)),
            f"RT axis is not strictly increasing: {relative_path}",
        )

        for asset in assets:
            split = str(asset["split"])
            row = int(asset["_dataset_row"])
            xic = _object(asset.get("xic"), "asset XIC")
            feature = _object(asset.get("feature"), "asset feature")
            label = _object(asset.get("label"), "asset label")
            signal_row = int(xic["signal_row"])
            _require(signal_row < matrix.shape[0], f"XIC row out of range: {asset['asset_id']}")
            _require(
                int(xic["point_count"]) == matrix.shape[1],
                f"XIC point-count mismatch: {asset['asset_id']}",
            )
            roi_window = feature.get("roi_window")
            _require(isinstance(roi_window, list) and len(roi_window) == 2, "Invalid ROI window")
            rt_lo = _finite(roi_window[0], "ROI lower bound")
            rt_hi = _finite(roi_window[1], "ROI upper bound")
            target_rt = np.linspace(rt_lo, rt_hi, target_points, dtype=np.float64)
            raw_signal = np.interp(
                target_rt,
                rt,
                np.asarray(matrix[signal_row], dtype=np.float64),
                left=0.0,
                right=0.0,
            )
            _require(np.isfinite(raw_signal).all(), f"Non-finite signal: {asset['asset_id']}")
            raw_minimum = float(np.min(raw_signal))
            raw_maximum = float(np.max(raw_signal))
            dynamic_range = raw_maximum - raw_minimum
            _require(math.isfinite(dynamic_range), f"Non-finite range: {asset['asset_id']}")
            baseline = float(np.quantile(raw_signal, 0.05))
            corrected = np.maximum(raw_signal - baseline, 0.0)
            logged = np.log1p(corrected)
            logged_maximum = float(np.max(logged))
            signal_available = 1.0 if logged_maximum > 0.0 else 0.0
            if signal_available:
                signals[split][row] = (logged / logged_maximum).astype(np.float32)
            else:
                signals[split][row].fill(0.0)

            presence, start, end = _boundary_target(asset)
            targets[split][row] = (presence, start, end)
            expected_rt = _finite(label.get("expected_rt"), "expected RT")
            q1 = _finite(feature.get("q1"), "q1")
            q3 = _finite(feature.get("q3"), "q3")
            scalar_values = (
                math.log1p(max(raw_maximum, 0.0)),
                math.log1p(max(dynamic_range, 0.0)),
                rt_hi - rt_lo,
                expected_rt,
                q1,
                q3,
                signal_available,
            )
            raw_scalars[split][row] = scalar_values
            label_counts[split][int(presence)] += 1
            image = _object(asset.get("image"), "asset image")
            examples[split][row] = {
                "schema_version": EXAMPLE_SCHEMA,
                "row": row,
                "asset_id": asset["asset_id"],
                "record_id": asset["record_id"],
                "split": split,
                "group_id": asset["job_id"],
                "image": {
                    "path": image["path"],
                    "sha256": image["sha256"],
                    "width": image["width"],
                    "height": image["height"],
                },
                "sequence": {
                    "array": f"{split}/signals.npy",
                    "row": row,
                    "length": target_points,
                    "source_matrix": relative_path,
                    "source_row": signal_row,
                    "signal_available": bool(signal_available),
                    "resampling": "linear_interpolation_on_rt_minutes",
                },
                "scalar_features": {
                    "array": f"{split}/scalar_features.npy",
                    "row": row,
                    "names": list(SCALAR_FEATURE_NAMES),
                },
                "metadata": {
                    "component": label["component"],
                    "channel": label["channel"],
                    "q1": q1,
                    "q3": q3,
                    "expected_rt_minutes": expected_rt,
                    "roi_window_minutes": [rt_lo, rt_hi],
                    "log1p_raw_maximum": scalar_values[0],
                    "log1p_dynamic_range": scalar_values[1],
                },
                "target": {
                    "array": f"{split}/targets.npy",
                    "row": row,
                    "peak_present": bool(presence),
                    "start_normalized": start if presence else None,
                    "end_normalized": end if presence else None,
                    "coordinate_system": "roi_fraction_0_1",
                    "supervision_source": "human",
                },
                "provenance": {
                    "dataset_version": asset["dataset_version"],
                    "plan_sha256": asset["plan_sha256"],
                    "source_artifact_hash": asset["source_artifact_hash"],
                    "job_output_sha256": asset["job_output_sha256"],
                    "asset_index_sha256": asset_index_sha256,
                },
            }

        del raw_signal, corrected, logged, target_rt, rt, matrix
        if progress_callback is not None:
            progress_callback(matrix_number, len(matrix_items), relative_path)

    _require("train" in raw_scalars, "A train split is required for scalar normalization")
    train_values = raw_scalars["train"][:, :STANDARDIZED_SCALAR_COUNT]
    scalar_mean = np.mean(train_values, axis=0)
    scalar_std = np.std(train_values, axis=0)
    scalar_std = np.where(scalar_std > 1e-12, scalar_std, 1.0)
    standardized_scalars = {}
    for split, values in raw_scalars.items():
        standardized = values.astype(np.float32)
        standardized[:, :STANDARDIZED_SCALAR_COUNT] = (
            values[:, :STANDARDIZED_SCALAR_COUNT] - scalar_mean
        ) / scalar_std
        standardized_scalars[split] = standardized

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        normalization_path = staging / "scalar_normalization.json"
        _write_json(
            normalization_path,
            {
                "schema_version": "chrompeak-scalar-normalization-v1",
                "fit_split": "train",
                "feature_names": list(SCALAR_FEATURE_NAMES),
                "standardized_feature_count": STANDARDIZED_SCALAR_COUNT,
                "mean": [float(value) for value in scalar_mean],
                "std": [float(value) for value in scalar_std],
                "signal_available_standardized": False,
            },
        )
        artifacts: dict[str, dict[str, Any]] = {}
        for split in sorted(by_split):
            split_dir = staging / split
            split_dir.mkdir(parents=True)
            split_artifacts = {
                "signals": (split_dir / "signals.npy", signals[split]),
                "scalar_features": (
                    split_dir / "scalar_features.npy",
                    standardized_scalars[split],
                ),
                "targets": (split_dir / "targets.npy", targets[split]),
            }
            for name, (path, array) in split_artifacts.items():
                np.save(path, array, allow_pickle=False)
                relative = path.relative_to(staging).as_posix()
                artifacts[f"{split}_{name}"] = {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            example_records = examples[split]
            _require(all(record is not None for record in example_records), "Missing example rows")
            examples_path = split_dir / "examples.jsonl"
            _write_jsonl(examples_path, [record for record in example_records if record])
            artifacts[f"{split}_examples"] = {
                "path": examples_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(examples_path),
                "records": len(example_records),
            }
        artifacts["scalar_normalization"] = {
            "path": normalization_path.relative_to(staging).as_posix(),
            "sha256": sha256_file(normalization_path),
        }

        asset_count = sum(len(items) for items in by_split.values())
        report_path = staging / "dataset_report.json"
        _write_json(
            report_path,
            {
                "schema_version": DATASET_SCHEMA,
                "dataset_version": readiness.get("dataset_version"),
                "plan_sha256": readiness.get("plan_sha256"),
                "asset_index_sha256": asset_index_sha256,
                "readiness_report_sha256": readiness_sha256,
                "sequence_preflight_sha256": preflight_sha256,
                "target_points": target_points,
                "splits": sorted(by_split),
                "counts": {
                    "assets": asset_count,
                    "matrices": len(matrix_items),
                    "by_split": {
                        split: {
                            "assets": len(by_split[split]),
                            "positive": label_counts[split][1],
                            "negative": label_counts[split][0],
                            "signal_unavailable": int(
                                len(by_split[split])
                                - np.sum(raw_scalars[split][:, -1])
                            ),
                        }
                        for split in sorted(by_split)
                    },
                },
                "contracts": {
                    "signal_normalization": (
                        f"rt-linear-{target_points}|p05-baseline|clip-nonnegative|log1p|roi-max"
                    ),
                    "scalar_features": list(SCALAR_FEATURE_NAMES),
                    "scalar_normalization": "train-only-zscore-first-six",
                    "target_columns": ["peak_present", "start_normalized", "end_normalized"],
                    "negative_boundary_sentinel": -1.0,
                    "coordinate_system": "shared-image-sequence-roi-fraction-0-1",
                    "images_copied": False,
                },
                "artifacts": artifacts,
            },
        )
        staging.replace(output_dir)

    final_report_path = output_dir / "dataset_report.json"
    return MultimodalDatasetResult(
        output_dir=output_dir,
        report_path=final_report_path,
        report_sha256=sha256_file(final_report_path),
        asset_index_sha256=asset_index_sha256,
        asset_count=sum(len(items) for items in by_split.values()),
        target_points=target_points,
        cached=False,
    )
