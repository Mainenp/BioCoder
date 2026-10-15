"""Index verified unlabeled channel-driven extraction outputs without creating labels."""

from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multimodal_science.chrompeakformer.asset_index import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    _finite_float,
    _jpeg_dimensions,
    _read_csv,
    _read_plan,
    _safe_child,
    _stable_hex,
    _stable_int,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from multimodal_science.chrompeakformer.outputs import read_npy_metadata, validate_outputs
from multimodal_science.data.manifest import sha256_file

AUXILIARY_SPLIT = "auxiliary_unlabeled_train"
AUXILIARY_TIER = "auxiliary_training_only"


@dataclass(frozen=True)
class AuxiliaryAssetIndexResult:
    plan_sha256: str
    index_path: Path
    report_path: Path
    manifest_path: Path
    index_sha256: str
    report_sha256: str
    manifest_sha256: str
    selected_jobs: int
    indexed_jobs: int
    missing_jobs: int
    asset_count: int
    source_group_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_job(job: dict[str, Any]) -> None:
    job_id = str(job["job_id"])
    _require(
        job.get("derivation_mode") == "channel_driven_inference",
        f"Auxiliary job has an invalid derivation mode: {job_id}",
    )
    _require(job.get("split") == AUXILIARY_SPLIT, f"Auxiliary split mismatch: {job_id}")
    _require(
        job.get("evaluation_tier") == AUXILIARY_TIER,
        f"Auxiliary evaluation tier mismatch: {job_id}",
    )
    _require(job.get("metrics_allowed") is False, f"Metrics must be disabled: {job_id}")
    _require(job.get("labels") == [], f"Auxiliary job unexpectedly contains labels: {job_id}")
    _require(
        isinstance(job.get("split_group"), str) and bool(job["split_group"].strip()),
        f"Auxiliary job has no source group: {job_id}",
    )
    _require(
        isinstance(job.get("artifact_hash"), str)
        and len(str(job["artifact_hash"])) == 64,
        f"Auxiliary job has an invalid source hash: {job_id}",
    )


def _verified_job_assets(
    job: dict[str, Any], job_dir: Path, assets_root: Path, plan_sha256: str
) -> list[dict[str, Any]]:
    job_id = str(job["job_id"])
    provenance_path = job_dir / "derivation_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing provenance for completed job: {job_id}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ok",
        "job_id": job_id,
        "dataset_version": job["dataset_version"],
        "split": AUXILIARY_SPLIT,
        "source_mzml": job["source_mzml"],
        "plan_sha256": plan_sha256,
        "source_artifact_hash": job["artifact_hash"],
    }
    for field, value in expected.items():
        _require(
            provenance.get(field) == value,
            f"Provenance {field} mismatch for job {job_id}",
        )

    summary = validate_outputs(job_dir)
    _require(
        provenance.get("outputs", {}).get("output_sha256") == summary.output_sha256,
        f"Output signature mismatch for job {job_id}",
    )
    features = _read_csv(job_dir / "feature.csv")
    windows = _read_csv(job_dir / "roi_windows.csv")
    matrix_path = job_dir / "xic_matrix.npy"
    matrix = read_npy_metadata(matrix_path)
    extractor_result = provenance.get("extractor_result")
    _require(isinstance(extractor_result, dict), f"Missing extractor result for job {job_id}")

    matrix_relative = (Path(str(job["output_prefix"])) / "xic_matrix.npy").as_posix()
    matrix_sha256 = sha256_file(matrix_path)
    source_group = str(job["split_group"]).strip()
    seen_images: set[str] = set()
    assets: list[dict[str, Any]] = []
    for index, (feature, window) in enumerate(zip(features, windows, strict=True)):
        native_id = str(feature.get("native_id") or "").strip()
        _require(bool(native_id), f"Empty native_id for job {job_id}, row {index + 2}")
        rt_lo = _finite_float(window.get("rt_lo"), "ROI rt_lo")
        rt_hi = _finite_float(window.get("rt_hi"), "ROI rt_hi")
        _require(rt_hi > rt_lo, f"Invalid ROI window for job {job_id}, row {index + 2}")
        image_name = str(window.get("image") or "").strip().replace("\\", "/")
        image_path = _safe_child(job_dir, image_name)
        _require(
            image_name not in seen_images,
            f"Duplicate ROI image in job {job_id}: {image_name}",
        )
        seen_images.add(image_name)
        dimensions = _jpeg_dimensions(image_path)
        _require(
            dimensions == (IMAGE_WIDTH, IMAGE_HEIGHT),
            f"Unexpected ROI dimensions for {image_name}: {dimensions}",
        )

        trace_id = _stable_hex(
            "auxiliary-trace",
            str(job["dataset_version"]),
            job_id,
            str(index),
            native_id,
        )
        image_id = _stable_int("auxiliary-image", trace_id)
        assets.append(
            {
                "schema_version": "chrompeak-auxiliary-asset-v1",
                "asset_id": trace_id,
                "record_id": trace_id,
                "dataset_version": job["dataset_version"],
                "plan_sha256": plan_sha256,
                "split": AUXILIARY_SPLIT,
                "evaluation_tier": AUXILIARY_TIER,
                "metrics_allowed": False,
                "job_id": job_id,
                "source_group": source_group,
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
                    "path": image_path.relative_to(assets_root).as_posix(),
                    "sha256": sha256_file(image_path),
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                },
                "xic": {
                    "path": matrix_relative,
                    "sha256": matrix_sha256,
                    "rt_row": 0,
                    "signal_row": index + 1,
                    "point_count": matrix.shape[1],
                },
                "feature": {
                    "trace_index": index,
                    "native_id": native_id,
                    "q1": _finite_float(feature.get("mz"), "feature mz"),
                    "q3": _finite_float(feature.get("q3"), "feature q3"),
                    "rt": _finite_float(feature.get("RT"), "feature RT"),
                    "roi_window": [rt_lo, rt_hi],
                },
                "supervision": {
                    "label_status": "unlabeled",
                    "supervised_train_eligible": False,
                    "auxiliary_unlabeled_train_eligible": True,
                    "benchmark_eligible": False,
                    "internal_test_accessed": False,
                },
            }
        )
    return assets


def _write_manifest(root: Path, relative_paths: tuple[str, ...]) -> Path:
    manifest_path = root / "artifact_manifest.sha256"
    manifest_path.write_text(
        "".join(f"{sha256_file(root / relative)}  {relative}\n" for relative in relative_paths),
        encoding="utf-8",
    )
    return manifest_path


def build_auxiliary_asset_index(
    plan_path: Path,
    assets_root: Path,
    output_dir: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AuxiliaryAssetIndexResult:
    """Build an immutable index for fully extracted, strictly unlabeled auxiliary jobs."""

    plan_path = plan_path.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Derivation plan not found: {plan_path}")
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Asset root not found: {assets_root}")
    if output_dir.exists():
        raise FileExistsError(f"Auxiliary index output already exists: {output_dir}")

    plan_sha256 = sha256_file(plan_path)
    jobs = [
        job
        for job in _read_plan(plan_path)
        if job.get("derivation_mode") == "channel_driven_inference"
    ]
    if not jobs:
        raise ValueError("No channel-driven jobs matched the auxiliary-index selection")
    for job in jobs:
        _validated_job(job)

    completed: list[tuple[dict[str, Any], Path]] = []
    missing: list[str] = []
    for job in jobs:
        job_dir = _safe_child(assets_root, str(job.get("output_prefix") or ""))
        if job_dir.is_dir():
            completed.append((job, job_dir))
        else:
            missing.append(str(job["job_id"]))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} auxiliary jobs have no published assets; "
            "partial auxiliary indexes are forbidden"
        )

    assets: list[dict[str, Any]] = []
    for completed_count, (job, job_dir) in enumerate(completed, start=1):
        assets.extend(_verified_job_assets(job, job_dir, assets_root, plan_sha256))
        if progress_callback is not None:
            progress_callback(completed_count, len(completed), str(job["job_id"]))
    assets.sort(key=lambda item: (str(item["job_id"]), int(item["xic"]["signal_row"])))
    asset_ids = [str(asset["asset_id"]) for asset in assets]
    image_ids = [int(asset["image"]["id"]) for asset in assets]
    _require(len(asset_ids) == len(set(asset_ids)), "Auxiliary asset ID collision detected")
    _require(len(image_ids) == len(set(image_ids)), "Auxiliary image ID collision detected")

    source_groups = Counter(str(asset["source_group"]) for asset in assets)
    job_counts = Counter(str(asset["job_id"]) for asset in assets)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        index_path = staging / "auxiliary_asset_index.jsonl"
        _write_jsonl_atomic(index_path, assets)
        index_sha256 = sha256_file(index_path)
        report_path = staging / "auxiliary_asset_index_report.json"
        _write_json_atomic(
            report_path,
            {
                "schema_version": "chrompeak-auxiliary-asset-index-report-v1",
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha256,
                "asset_index_file": index_path.name,
                "asset_index_sha256": index_sha256,
                "artifact_manifest_file": "artifact_manifest.sha256",
                "selection": {
                    "derivation_mode": "channel_driven_inference",
                    "split": AUXILIARY_SPLIT,
                    "evaluation_tier": AUXILIARY_TIER,
                },
                "counts": {
                    "selected_jobs": len(jobs),
                    "indexed_jobs": len(completed),
                    "missing_jobs": 0,
                    "extracted_trace_assets": len(assets),
                    "images": len(assets),
                    "xic_matrices": len(completed),
                    "independent_source_groups": len(source_groups),
                    "labels": 0,
                    "supervised_train_assets": 0,
                    "benchmark_assets": 0,
                },
                "source_groups": dict(sorted(source_groups.items())),
                "job_trace_counts": dict(sorted(job_counts.items())),
                "indexed_job_ids": [str(job["job_id"]) for job, _ in completed],
                "contracts": {
                    "complete_plan_coverage": True,
                    "labels_present": False,
                    "metrics_allowed": False,
                    "train_supervision_present": False,
                    "auxiliary_unlabeled_train_eligible": True,
                    "validation_membership_changed": False,
                    "benchmark_membership_changed": False,
                    "internal_test_accessed": False,
                },
                "quality_gate_passed": True,
                "claim_limits": [
                    "Indexed traces are unlabeled auxiliary-training inputs, "
                    "not supervised examples",
                    "They cannot contribute validation, benchmark, AP, IoU, "
                    "or classification metrics",
                    "Any later pseudo-label must remain explicitly weak and provenance-bound",
                ],
            },
        )
        manifest_path = _write_manifest(
            staging,
            (index_path.name, report_path.name),
        )
        report_sha256 = sha256_file(report_path)
        manifest_sha256 = sha256_file(manifest_path)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AuxiliaryAssetIndexResult(
        plan_sha256=plan_sha256,
        index_path=output_dir / index_path.name,
        report_path=output_dir / report_path.name,
        manifest_path=output_dir / manifest_path.name,
        index_sha256=index_sha256,
        report_sha256=report_sha256,
        manifest_sha256=manifest_sha256,
        selected_jobs=len(jobs),
        indexed_jobs=len(completed),
        missing_jobs=0,
        asset_count=len(assets),
        source_group_count=len(source_groups),
    )
