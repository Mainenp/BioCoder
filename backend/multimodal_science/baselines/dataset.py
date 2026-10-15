"""Load sequence-baseline inputs only from a verified multimodal Dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file


@dataclass(frozen=True)
class SequenceSplit:
    split: str
    signals: Any
    scalar_features: Any
    targets: Any
    asset_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    roi_width_minutes: Any
    dataset_report_sha256: str
    asset_index_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, description: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{description} must be an object")
    return value


def _artifact_path(
    dataset_root: Path,
    artifacts: dict[str, Any],
    key: str,
    *,
    verify_hashes: bool,
) -> Path:
    artifact = _object(artifacts.get(key), f"artifact {key}")
    relative = Path(str(artifact.get("path") or ""))
    _require(not relative.is_absolute(), f"Artifact path must be relative: {key}")
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes Dataset root: {key}") from exc
    _require(path.is_file(), f"Dataset artifact is missing: {key}")
    if verify_hashes:
        _require(
            sha256_file(path) == artifact.get("sha256"),
            f"Dataset artifact hash mismatch: {key}",
        )
    return path


def load_sequence_split(
    dataset_root: Path,
    split: str,
    *,
    verify_hashes: bool = True,
) -> SequenceSplit:
    """Load arrays, group identities, and physical ROI widths for one split."""

    import numpy as np

    dataset_root = dataset_root.resolve()
    report_path = dataset_root / "dataset_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Dataset report not found: {report_path}")
    try:
        report = _object(json.loads(report_path.read_text(encoding="utf-8")), "report")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Dataset report JSON: {report_path}") from exc
    _require(
        report.get("schema_version") == "chrompeak-multimodal-dataset-v1",
        "Unsupported multimodal Dataset schema",
    )
    splits = report.get("splits")
    _require(isinstance(splits, list) and split in splits, f"Split is not available: {split}")
    artifacts = _object(report.get("artifacts"), "dataset artifacts")
    signals_path = _artifact_path(
        dataset_root, artifacts, f"{split}_signals", verify_hashes=verify_hashes
    )
    scalars_path = _artifact_path(
        dataset_root,
        artifacts,
        f"{split}_scalar_features",
        verify_hashes=verify_hashes,
    )
    targets_path = _artifact_path(
        dataset_root, artifacts, f"{split}_targets", verify_hashes=verify_hashes
    )
    examples_path = _artifact_path(
        dataset_root, artifacts, f"{split}_examples", verify_hashes=verify_hashes
    )
    signals = np.load(signals_path, allow_pickle=False)
    scalar_features = np.load(scalars_path, allow_pickle=False)
    targets = np.load(targets_path, allow_pickle=False)
    _require(signals.ndim == 2, f"Signals must be two-dimensional: {split}")
    _require(scalar_features.ndim == 2, f"Scalars must be two-dimensional: {split}")
    _require(targets.ndim == 2 and targets.shape[1] == 3, f"Invalid targets: {split}")
    target_points = report.get("target_points")
    contracts = _object(report.get("contracts"), "dataset contracts")
    scalar_names = contracts.get("scalar_features")
    _require(
        isinstance(target_points, int) and signals.shape[1] == target_points,
        f"Signal length does not match Dataset contract: {split}",
    )
    _require(
        isinstance(scalar_names, list)
        and scalar_names
        and scalar_features.shape[1] == len(scalar_names),
        f"Scalar width does not match Dataset contract: {split}",
    )
    row_count = signals.shape[0]
    _require(
        scalar_features.shape[0] == row_count and targets.shape[0] == row_count,
        f"Dataset array row counts disagree: {split}",
    )
    split_counts = _object(
        _object(report.get("counts"), "dataset counts").get("by_split"),
        "split counts",
    )
    declared_split = _object(
        split_counts.get(split),
        f"declared split {split}",
    )
    _require(declared_split.get("assets") == row_count, f"Declared split count mismatch: {split}")
    _require(np.isfinite(signals).all(), f"Signals contain non-finite values: {split}")
    _require(
        np.isfinite(scalar_features).all(),
        f"Scalar features contain non-finite values: {split}",
    )
    _require(np.isfinite(targets).all(), f"Targets contain non-finite values: {split}")
    presence_targets = targets[:, 0]
    _require(
        np.isin(presence_targets, [0.0, 1.0]).all(),
        f"Presence targets must be binary: {split}",
    )
    positive = presence_targets == 1.0
    _require(
        (
            (targets[positive, 1] >= 0.0)
            & (targets[positive, 1] < targets[positive, 2])
            & (targets[positive, 2] <= 1.0)
        ).all(),
        f"Positive boundary targets are invalid: {split}",
    )
    _require(
        (targets[~positive, 1:] == -1.0).all(),
        f"Negative boundary sentinel is invalid: {split}",
    )

    asset_ids = []
    group_ids = []
    roi_widths = []
    with examples_path.open(encoding="utf-8") as stream:
        for expected_row, line in enumerate(stream):
            try:
                example = _object(json.loads(line), f"example row {expected_row}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid example JSON at row {expected_row}") from exc
            _require(
                example.get("schema_version") == "chrompeak-multimodal-example-v1",
                f"Unsupported example schema at row {expected_row}",
            )
            _require(example.get("row") == expected_row, f"Non-contiguous example row: {split}")
            _require(example.get("split") == split, f"Example split mismatch at row {expected_row}")
            asset_id = str(example.get("asset_id") or "")
            group_id = str(example.get("group_id") or "")
            _require(asset_id != "" and group_id != "", f"Missing identity at row {expected_row}")
            metadata = _object(example.get("metadata"), f"metadata row {expected_row}")
            window = metadata.get("roi_window_minutes")
            _require(
                isinstance(window, list) and len(window) == 2,
                f"Invalid ROI window at row {expected_row}",
            )
            width = float(window[1]) - float(window[0])
            _require(width > 0.0, f"Non-positive ROI width at row {expected_row}")
            target = _object(example.get("target"), f"target row {expected_row}")
            presence = bool(target.get("peak_present"))
            _require(
                bool(targets[expected_row, 0]) == presence,
                f"JSON and array presence targets disagree at row {expected_row}",
            )
            if presence:
                _require(
                    target.get("start_normalized") is not None
                    and target.get("end_normalized") is not None,
                    f"Positive JSON boundary is missing at row {expected_row}",
                )
                expected_start = float(target["start_normalized"])
                expected_end = float(target["end_normalized"])
            else:
                _require(
                    target.get("start_normalized") is None
                    and target.get("end_normalized") is None,
                    f"Negative JSON boundary must be null at row {expected_row}",
                )
                expected_start = -1.0
                expected_end = -1.0
            _require(
                np.allclose(
                    targets[expected_row, 1:],
                    [expected_start, expected_end],
                    rtol=0.0,
                    atol=1e-6,
                ),
                f"JSON and array boundary targets disagree at row {expected_row}",
            )
            asset_ids.append(asset_id)
            group_ids.append(group_id)
            roi_widths.append(width)
    _require(len(asset_ids) == row_count, f"Example row count mismatch: {split}")
    _require(len(asset_ids) == len(set(asset_ids)), f"Duplicate asset IDs in split: {split}")

    return SequenceSplit(
        split=split,
        signals=signals,
        scalar_features=scalar_features,
        targets=targets,
        asset_ids=tuple(asset_ids),
        group_ids=tuple(group_ids),
        roi_width_minutes=np.asarray(roi_widths, dtype=np.float64),
        dataset_report_sha256=sha256_file(report_path),
        asset_index_sha256=str(report.get("asset_index_sha256") or ""),
    )
