"""Build a traceable ROI/XIC/COCO asset index from verified extraction outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multimodal_science.chrompeakformer.outputs import read_npy_metadata, validate_outputs
from multimodal_science.data.manifest import sha256_file

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
NAME_FIELD_PATTERN = re.compile(r'\bname=([^\"]*)$')
SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


@dataclass(frozen=True)
class AssetIndexResult:
    plan_sha256: str
    index_path: Path
    report_path: Path
    index_sha256: str
    selected_jobs: int
    indexed_jobs: int
    missing_jobs: int
    asset_count: int
    annotation_count: int
    coco_paths: tuple[Path, ...]


def _read_plan(path: Path) -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")
            job_id = value.get("job_id")
            if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
                raise ValueError(f"Invalid job_id at {path.name}:{line_number}")
            if job_id in seen:
                raise ValueError(f"Duplicate job_id in plan: {job_id}")
            seen.add(job_id)
            jobs.append(value)
    if not jobs:
        raise ValueError(f"Derivation plan is empty: {path}")
    return jobs


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {relative}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {relative}") from exc
    return candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _stable_hex(*parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _stable_int(*parts: str) -> int:
    return int(_stable_hex(*parts, length=13), 16)


def _finite_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG start marker: {path.name}")
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker_byte = stream.read(1)
            while marker_byte == b"\xff":
                marker_byte = stream.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length: {path.name}")
            if marker in SOF_MARKERS:
                frame = stream.read(5)
                if len(frame) != 5:
                    break
                height, width = struct.unpack(">HH", frame[1:])
                return width, height
            stream.seek(segment_length - 2, 1)
    raise ValueError(f"JPEG dimensions not found: {path.name}")


def _label_key(label: dict[str, Any], index: int) -> str:
    component = str(label.get("component") or "").strip()
    channel = str(label.get("channel") or "").strip()
    if not component:
        raise ValueError(f"labels[{index}].component must be non-empty")
    suffix = "1" if "定量" in channel else "2" if "定性" in channel else None
    if suffix is None:
        raise ValueError(f"labels[{index}].channel is unsupported: {channel}")
    return f"{component}-{suffix}"


def _label_maps(
    labels: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if not isinstance(labels, list) or not labels:
        raise ValueError("Label-driven job requires a non-empty labels list")
    by_key: dict[str, dict[str, Any]] = {}
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_records = set()
    for index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise TypeError(f"labels[{index}] must be an object")
        record_id = str(label.get("record_id") or "")
        if not JOB_ID_PATTERN.fullmatch(record_id):
            raise ValueError(f"labels[{index}].record_id is invalid")
        if record_id in seen_records:
            raise ValueError(f"Duplicate label record_id: {record_id}")
        seen_records.add(record_id)
        peak_label = label.get("peak_label")
        if peak_label not in {0, 1}:
            raise ValueError(f"labels[{index}].peak_label must be 0 or 1")
        key = _label_key(label, index)
        if key in by_key:
            raise ValueError(f"Duplicate component/channel label key: {key}")
        by_key[key] = label
        by_component[str(label["component"]).strip()].append(label)
    return by_key, by_component


def _match_label(
    native_id: str,
    by_key: dict[str, dict[str, Any]],
    by_component: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if native_id in by_key:
        return by_key[native_id]
    candidates = [native_id]
    match = NAME_FIELD_PATTERN.search(native_id)
    if match:
        candidates.insert(0, match.group(1).strip())
    for candidate in candidates:
        if candidate in by_key:
            return by_key[candidate]
        component_matches = by_component.get(candidate, [])
        if len(component_matches) == 1:
            return component_matches[0]
        if len(component_matches) > 1:
            raise ValueError(f"Ambiguous bare component native_id: {native_id}")
    raise ValueError(f"No label matches extracted native_id: {native_id}")


def _coco_boxes(
    label: dict[str, Any], rt_lo: float, rt_hi: float
) -> list[dict[str, Any]]:
    if label["peak_label"] == 0:
        return []
    intervals = label.get("peak_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"Positive label has no peak intervals: {label['record_id']}")
    boxes = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Invalid peak interval for {label['record_id']}")
        start = _finite_float(interval[0], "peak interval start")
        end = _finite_float(interval[1], "peak interval end")
        if end <= start:
            raise ValueError(f"Peak interval is not increasing: {label['record_id']}")
        if start > rt_hi or end < rt_lo:
            continue
        clipped_start = max(start, rt_lo)
        clipped_end = min(end, rt_hi)
        x1 = (clipped_start - rt_lo) / (rt_hi - rt_lo) * IMAGE_WIDTH
        x2 = (clipped_end - rt_lo) / (rt_hi - rt_lo) * IMAGE_WIDTH
        width = x2 - x1
        if width < 1.0:
            continue
        boxes.append(
            {
                "interval_index": index,
                "rt_interval": [start, end],
                "visible_rt_interval": [clipped_start, clipped_end],
                "bbox": [round(x1, 2), 0.0, round(width, 2), float(IMAGE_HEIGHT)],
                "area": round(width * IMAGE_HEIGHT, 2),
            }
        )
    if not boxes:
        raise ValueError(f"Positive label has no visible COCO box: {label['record_id']}")
    return boxes


def _verified_job_assets(
    job: dict[str, Any], job_dir: Path, assets_root: Path, plan_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provenance_path = job_dir / "derivation_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing provenance for completed job: {job['job_id']}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ok",
        "job_id": job["job_id"],
        "dataset_version": job["dataset_version"],
        "split": job["split"],
        "source_mzml": job["source_mzml"],
        "plan_sha256": plan_sha256,
        "source_artifact_hash": job.get("artifact_hash"),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"Provenance {field} mismatch for job {job['job_id']}")
    summary = validate_outputs(job_dir)
    if provenance.get("outputs", {}).get("output_sha256") != summary.output_sha256:
        raise ValueError(f"Output signature mismatch for job {job['job_id']}")

    features = _read_csv(job_dir / "feature.csv")
    windows = _read_csv(job_dir / "roi_windows.csv")
    matrix = read_npy_metadata(job_dir / "xic_matrix.npy")
    extractor_result = provenance.get("extractor_result")
    if not isinstance(extractor_result, dict):
        raise ValueError(f"Missing extractor result for job {job['job_id']}")
    by_key, by_component = _label_maps(job.get("labels"))
    used_records = set()
    assets = []
    coco_annotations = []
    for index, (feature, window) in enumerate(zip(features, windows, strict=True)):
        native_id = str(feature.get("native_id") or "").strip()
        label = _match_label(native_id, by_key, by_component)
        record_id = str(label["record_id"])
        if record_id in used_records:
            raise ValueError(f"Label matched more than one ROI: {record_id}")
        used_records.add(record_id)

        rt_lo = _finite_float(window.get("rt_lo"), "ROI rt_lo")
        rt_hi = _finite_float(window.get("rt_hi"), "ROI rt_hi")
        if rt_hi <= rt_lo:
            raise ValueError(f"Invalid ROI window for {record_id}")
        image_name = str(window.get("image") or "").strip().replace("\\", "/")
        image_path = _safe_child(job_dir, image_name)
        dimensions = _jpeg_dimensions(image_path)
        if dimensions != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(
                f"Unexpected ROI dimensions for {image_name}: {dimensions}"
            )

        boxes = _coco_boxes(label, rt_lo, rt_hi)
        asset_id = _stable_hex(str(job["dataset_version"]), str(job["job_id"]), record_id)
        image_id = _stable_int("image", asset_id)
        image_relative = image_path.relative_to(assets_root).as_posix()
        asset_boxes = []
        for box in boxes:
            annotation_id = _stable_int(
                "annotation", asset_id, str(box["interval_index"])
            )
            enriched = {**box, "annotation_id": annotation_id, "category_id": 0}
            asset_boxes.append(enriched)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                }
            )
        assets.append(
            {
                "schema_version": "chrompeak-asset-v1",
                "asset_id": asset_id,
                "dataset_version": job["dataset_version"],
                "plan_sha256": plan_sha256,
                "split": job["split"],
                "evaluation_tier": job.get("evaluation_tier"),
                "metrics_allowed": bool(job.get("metrics_allowed")),
                "job_id": job["job_id"],
                "record_id": record_id,
                "source_mzml": job["source_mzml"],
                "source_artifact_hash": job["artifact_hash"],
                "job_output_sha256": summary.output_sha256,
                "extractor": {
                    "adapter_version": extractor_result.get("adapter_version"),
                    "source_api": extractor_result.get("source_api"),
                    "private_code_sha256": extractor_result.get("private_code_sha256"),
                    "smooth_sigma": extractor_result.get("smooth_sigma"),
                },
                "image": {
                    "id": image_id,
                    "path": image_relative,
                    "sha256": sha256_file(image_path),
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                },
                "xic": {
                    "path": (Path(str(job["output_prefix"])) / "xic_matrix.npy").as_posix(),
                    "rt_row": 0,
                    "signal_row": index + 1,
                    "point_count": matrix.shape[1],
                },
                "feature": {
                    "native_id": native_id,
                    "q1": _finite_float(feature.get("mz"), "feature mz"),
                    "q3": _finite_float(feature.get("q3"), "feature q3"),
                    "rt": _finite_float(feature.get("RT"), "feature RT"),
                    "roi_window": [rt_lo, rt_hi],
                },
                "label": {
                    "component": label["component"],
                    "channel": label["channel"],
                    "expected_rt": _finite_float(label.get("rt"), "label RT"),
                    "peak_label": label["peak_label"],
                    "peak_intervals": label.get("peak_intervals", []),
                    "coco_boxes": asset_boxes,
                },
            }
        )
    if used_records != {str(label["record_id"]) for label in by_key.values()}:
        missing = sorted(
            {str(label["record_id"]) for label in by_key.values()} - used_records
        )
        raise ValueError(
            f"Labels did not produce exactly one ROI in job {job['job_id']}: {missing}"
        )
    return assets, coco_annotations


def build_asset_index(
    plan_path: Path,
    assets_root: Path,
    output_dir: Path,
    *,
    include_splits: frozenset[str] | None = None,
    allow_partial: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AssetIndexResult:
    plan_path = plan_path.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Derivation plan not found: {plan_path}")
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Asset root not found: {assets_root}")
    plan_sha256 = sha256_file(plan_path)
    jobs = [
        job
        for job in _read_plan(plan_path)
        if job.get("derivation_mode") == "label_driven"
        and (include_splits is None or job.get("split") in include_splits)
    ]
    if not jobs:
        raise ValueError("No label-driven jobs matched the asset-index selection")

    completed = []
    missing = []
    for job in jobs:
        job_dir = _safe_child(assets_root, str(job.get("output_prefix") or ""))
        if job_dir.is_dir():
            completed.append((job, job_dir))
        else:
            missing.append(str(job["job_id"]))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} selected jobs have no published assets; use allow_partial for pilots"
        )
    if not completed:
        raise FileNotFoundError("No published assets matched the asset-index selection")

    assets = []
    annotations_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for completed_count, (job, job_dir) in enumerate(completed, start=1):
        job_assets, annotations = _verified_job_assets(
            job, job_dir, assets_root, plan_sha256
        )
        assets.extend(job_assets)
        annotations_by_split[str(job["split"])].extend(annotations)
        if progress_callback is not None:
            progress_callback(completed_count, len(completed), str(job["job_id"]))
    assets.sort(
        key=lambda item: (
            str(item["split"]),
            str(item["job_id"]),
            item["xic"]["signal_row"],
        )
    )
    asset_ids = [str(asset["asset_id"]) for asset in assets]
    image_ids = [int(asset["image"]["id"]) for asset in assets]
    annotation_ids = [
        int(annotation["id"])
        for annotations in annotations_by_split.values()
        for annotation in annotations
    ]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Asset ID collision detected")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO image ID collision detected")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("COCO annotation ID collision detected")

    index_path = output_dir / "asset_index.jsonl"
    _write_jsonl_atomic(index_path, assets)
    index_sha256 = sha256_file(index_path)
    coco_paths = []
    for split in sorted({str(asset["split"]) for asset in assets}):
        split_assets = [asset for asset in assets if asset["split"] == split]
        coco_path = output_dir / f"{split}_coco.json"
        _write_json_atomic(
            coco_path,
            {
                "info": {
                    "description": "BioCoder chromatographic peak assets",
                    "plan_sha256": plan_sha256,
                    "asset_index_sha256": index_sha256,
                    "partial": bool(missing),
                },
                "images": [
                    {
                        "id": asset["image"]["id"],
                        "file_name": asset["image"]["path"],
                        "width": IMAGE_WIDTH,
                        "height": IMAGE_HEIGHT,
                    }
                    for asset in split_assets
                ],
                "annotations": sorted(
                    annotations_by_split[split], key=lambda item: item["id"]
                ),
                "categories": [
                    {
                        "id": 0,
                        "name": "peak",
                        "supercategory": "chromatographic_peak",
                    }
                ],
            },
        )
        coco_paths.append(coco_path)

    labels = Counter(int(asset["label"]["peak_label"]) for asset in assets)
    report_path = output_dir / "asset_index_report.json"
    _write_json_atomic(
        report_path,
        {
            "schema_version": "chrompeak-asset-index-report-v1",
            "plan_sha256": plan_sha256,
            "asset_index_path": str(index_path),
            "asset_index_sha256": index_sha256,
            "selection": {"splits": sorted(include_splits) if include_splits else None},
            "partial": bool(missing),
            "counts": {
                "selected_jobs": len(jobs),
                "indexed_jobs": len(completed),
                "missing_jobs": len(missing),
                "assets": len(assets),
                "positive_assets": labels[1],
                "negative_assets": labels[0],
                "annotations": sum(len(items) for items in annotations_by_split.values()),
            },
            "indexed_job_ids": [str(job["job_id"]) for job, _ in completed],
            "missing_job_ids": missing,
            "coco_paths": [str(path) for path in coco_paths],
        },
    )
    return AssetIndexResult(
        plan_sha256=plan_sha256,
        index_path=index_path,
        report_path=report_path,
        index_sha256=index_sha256,
        selected_jobs=len(jobs),
        indexed_jobs=len(completed),
        missing_jobs=len(missing),
        asset_count=len(assets),
        annotation_count=sum(len(items) for items in annotations_by_split.values()),
        coco_paths=tuple(coco_paths),
    )
