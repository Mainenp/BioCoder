"""Prepare a leakage-audited COCO layout for the ChromPeakFormer detector."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file


@dataclass(frozen=True)
class DetectorDatasetResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    coco_root: Path
    asset_index_sha256: str
    train_assets: int
    validation_assets: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    _require(bool(records), f"Asset index is empty: {path}")
    return records


def _safe_asset_path(assets_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Asset path must be relative: {relative}")
    candidate = (assets_root / relative_path).resolve()
    try:
        candidate.relative_to(assets_root)
    except ValueError as exc:
        raise ValueError(f"Asset path escapes assets root: {relative}") from exc
    return candidate


def _annotation_key(annotation: dict[str, Any]) -> tuple[int, int, int, tuple[float, ...]]:
    bbox = annotation.get("bbox")
    _require(isinstance(bbox, list) and len(bbox) == 4, "COCO bbox must contain four values")
    return (
        int(annotation["id"]),
        int(annotation["image_id"]),
        int(annotation["category_id"]),
        tuple(float(value) for value in bbox),
    )


def _prepare_split(
    *,
    split: str,
    source_coco: dict[str, Any],
    records: list[dict[str, Any]],
    assets_root: Path,
    asset_index_sha256: str,
    verify_image_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_records = [record for record in records if record.get("split") == split]
    _require(bool(split_records), f"No assets found for split: {split}")
    record_by_image_id = {int(record["image"]["id"]): record for record in split_records}
    _require(
        len(record_by_image_id) == len(split_records),
        f"Duplicate image IDs in {split} asset index",
    )
    info = source_coco.get("info")
    _require(isinstance(info, dict), f"{split} COCO info is missing")
    _require(
        info.get("asset_index_sha256") == asset_index_sha256,
        f"{split} COCO asset-index hash mismatch",
    )
    _require(not bool(info.get("partial")), f"{split} COCO dataset is partial")
    categories = source_coco.get("categories")
    _require(
        isinstance(categories, list)
        and len(categories) == 1
        and int(categories[0].get("id", -1)) == 0,
        f"{split} COCO dataset must contain the single peak category with id 0",
    )
    images = source_coco.get("images")
    annotations = source_coco.get("annotations")
    _require(isinstance(images, list), f"{split} COCO images must be a list")
    _require(isinstance(annotations, list), f"{split} COCO annotations must be a list")
    image_ids = [int(image["id"]) for image in images]
    _require(len(image_ids) == len(set(image_ids)), f"Duplicate image IDs in {split} COCO")
    _require(
        set(image_ids) == set(record_by_image_id),
        f"{split} COCO images do not exactly match the asset index",
    )

    prepared_images = []
    verified_bytes = 0
    for image in images:
        image_id = int(image["id"])
        record = record_by_image_id[image_id]
        indexed_image = record["image"]
        _require(
            str(image.get("file_name")) == str(indexed_image["path"]),
            f"COCO/index image path mismatch for image {image_id}",
        )
        _require(
            int(image.get("width", 0)) == int(indexed_image["width"])
            and int(image.get("height", 0)) == int(indexed_image["height"]),
            f"COCO/index image dimensions mismatch for image {image_id}",
        )
        source_image = _safe_asset_path(assets_root, str(indexed_image["path"]))
        _require(source_image.is_file(), f"Image asset is missing: {source_image}")
        if verify_image_hashes:
            _require(
                sha256_file(source_image) == indexed_image["sha256"],
                f"Image hash mismatch: {source_image}",
            )
            verified_bytes += source_image.stat().st_size
        prepared_images.append({**image, "file_name": str(source_image)})

    actual_annotations = {_annotation_key(annotation) for annotation in annotations}
    expected_annotations = {
        (
            int(box["annotation_id"]),
            int(record["image"]["id"]),
            int(box["category_id"]),
            tuple(float(value) for value in box["bbox"]),
        )
        for record in split_records
        for box in record["label"]["coco_boxes"]
    }
    _require(
        len(expected_annotations)
        == sum(len(record["label"]["coco_boxes"]) for record in split_records),
        f"Duplicate indexed annotations in {split}",
    )
    _require(
        len(actual_annotations) == len(annotations),
        f"Duplicate semantic annotations in {split} COCO",
    )
    _require(
        actual_annotations == expected_annotations,
        f"{split} COCO annotations do not exactly match indexed labels",
    )
    positive_assets = sum(int(record["label"]["peak_label"]) for record in split_records)
    return (
        {**source_coco, "images": prepared_images},
        {
            "assets": len(split_records),
            "positive_assets": positive_assets,
            "negative_assets": len(split_records) - positive_assets,
            "annotations": len(annotations),
            "source_groups": len({str(record["job_id"]) for record in split_records}),
            "image_hashes_verified": verify_image_hashes,
            "verified_image_bytes": verified_bytes,
        },
    )


def build_detector_dataset(
    *,
    asset_index_path: Path,
    asset_index_report_path: Path,
    assets_root: Path,
    train_coco_path: Path,
    validation_coco_path: Path,
    output_dir: Path,
    verify_image_hashes: bool = False,
) -> DetectorDatasetResult:
    """Build the exact train/val layout expected by the authorized detector source."""

    paths = (
        asset_index_path,
        asset_index_report_path,
        train_coco_path,
        validation_coco_path,
    )
    for path in paths:
        _require(path.is_file(), f"Required input does not exist: {path}")
    assets_root = assets_root.resolve()
    _require(assets_root.is_dir(), f"Assets root does not exist: {assets_root}")
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Detector dataset output already exists: {output_dir}")

    index_sha256 = sha256_file(asset_index_path)
    index_report = _read_json(asset_index_report_path)
    _require(
        index_report.get("asset_index_sha256") == index_sha256,
        "Asset-index report hash does not match the supplied index",
    )
    _require(not bool(index_report.get("partial")), "Asset index is partial")
    records = _read_jsonl(asset_index_path)
    _require(
        len(records) == len({str(record["asset_id"]) for record in records}),
        "Duplicate asset IDs",
    )
    _require(
        {str(record.get("split")) for record in records} == {"train", "validation"},
        "Detector contract requires exactly train and validation splits",
    )
    train_groups = {str(record["job_id"]) for record in records if record["split"] == "train"}
    validation_groups = {
        str(record["job_id"]) for record in records if record["split"] == "validation"
    }
    _require(
        not train_groups & validation_groups,
        "Source-group leakage across train and validation",
    )

    prepared = {}
    split_reports = {}
    for split, source_path in (
        ("train", train_coco_path),
        ("validation", validation_coco_path),
    ):
        prepared[split], split_reports[split] = _prepare_split(
            split=split,
            source_coco=_read_json(source_path),
            records=records,
            assets_root=assets_root,
            asset_index_sha256=index_sha256,
            verify_image_hashes=verify_image_hashes,
        )

    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        coco_root = staging / "coco"
        train_dir = coco_root / "train"
        validation_dir = coco_root / "val"
        train_dir.mkdir(parents=True)
        validation_dir.mkdir(parents=True)
        train_output = train_dir / "train_coco.json"
        validation_output = validation_dir / "val_coco.json"
        train_output.write_text(
            json.dumps(prepared["train"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validation_output.write_text(
            json.dumps(prepared["validation"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path = staging / "detector_dataset_report.json"
        report = {
            "schema_version": "chrompeak-detector-dataset-v1",
            "model_family": "ChromPeakFormer",
            "asset_index_sha256": index_sha256,
            "asset_index_report_sha256": sha256_file(asset_index_report_path),
            "source_coco_sha256": {
                "train": sha256_file(train_coco_path),
                "validation": sha256_file(validation_coco_path),
            },
            "prepared_coco_sha256": {
                "train": sha256_file(train_output),
                "validation": sha256_file(validation_output),
            },
            "splits": split_reports,
            "leakage_audit": {
                "group_field": "job_id",
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "overlap": [],
                "passed": True,
            },
            "quality_gate_passed": True,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "artifact_manifest.sha256").write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in (train_output, validation_output, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "detector_dataset_report.json"
    return DetectorDatasetResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        coco_root=output_dir / "coco",
        asset_index_sha256=index_sha256,
        train_assets=int(split_reports["train"]["assets"]),
        validation_assets=int(split_reports["validation"]["assets"]),
    )
