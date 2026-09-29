"""Bind converted vendor data to an auditable auxiliary-training contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import mzml_chromatogram_count, sha256_file


@dataclass(frozen=True)
class AuxiliaryMsdataResult:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    plan_path: Path
    report_sha256: str
    plan_sha256: str
    source_groups: int
    acquisition_frames: int
    transition_traces: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    _require(bool(records), f"Quarantine manifest is empty: {path}")
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _frame_job_id(dataset_version: str, frame_id: str, mzml_sha256: str) -> str:
    seed = f"{dataset_version}\0{frame_id}\0{mzml_sha256}".encode()
    return hashlib.sha256(seed).hexdigest()[:24]


def build_auxiliary_msdata_dataset(
    *,
    quarantine_manifest_path: Path,
    quarantine_report_path: Path,
    converted_root: Path,
    converter_path: Path,
    output_dir: Path,
) -> AuxiliaryMsdataResult:
    """Verify conversion completeness and emit an inference-only extraction plan."""

    for path in (quarantine_manifest_path, quarantine_report_path, converter_path):
        _require(path.is_file(), f"Required input does not exist: {path}")
    converted_root = converted_root.resolve()
    output_dir = output_dir.resolve()
    _require(converted_root.is_dir(), f"Converted root does not exist: {converted_root}")
    _require(not output_dir.exists(), f"Output already exists: {output_dir}")

    quarantine_report = _read_object(quarantine_report_path)
    quarantine_records = _read_jsonl(quarantine_manifest_path)
    _require(
        quarantine_report.get("schema_version")
        == "chrompeak-msdata-quarantine-report-v1",
        "Unsupported quarantine report schema",
    )
    _require(
        quarantine_report.get("manifest_sha256") == sha256_file(quarantine_manifest_path),
        "Quarantine manifest hash mismatch",
    )
    contracts = quarantine_report.get("contracts")
    _require(isinstance(contracts, dict), "Quarantine contracts must be an object")
    _require(contracts.get("labels_present") is False, "Labeled data must use the supervised path")
    _require(
        contracts.get("annotation_like_fields_detected") is False,
        "Quarantine report contains annotation-like fields",
    )
    _require(contracts.get("train_eligible") is False, "Quarantine sources cannot be supervised")
    _require(
        contracts.get("benchmark_eligible") is False,
        "Quarantine sources cannot enter benchmark splits",
    )

    dataset_version = str(quarantine_report.get("dataset_version"))
    _require(dataset_version.startswith("msdata-"), "Invalid auxiliary dataset version")
    report_counts = quarantine_report.get("counts")
    _require(isinstance(report_counts, dict), "Quarantine counts must be an object")
    _require(
        len(quarantine_records) == int(report_counts.get("source_files", -1)),
        "Quarantine source count mismatch",
    )

    expected_relative_paths: set[str] = set()
    auxiliary_records: list[dict[str, Any]] = []
    derivation_jobs: list[dict[str, Any]] = []
    total_transition_traces = 0
    total_chromatograms = 0
    recovered_mzml_files = 0

    for source in sorted(quarantine_records, key=lambda item: str(item.get("source_group"))):
        _require(source.get("schema_version") == "chrompeak-msdata-quarantine-record-v1", "Unsupported source schema")
        _require(source.get("label_status") == "unlabeled", "Auxiliary source unexpectedly has labels")
        _require(source.get("train_eligible") is False, "Auxiliary source cannot be supervised")
        _require(source.get("benchmark_eligible") is False, "Auxiliary source cannot be benchmark data")
        _require(not source.get("annotation_like_fields_detected"), "Source contains annotation-like fields")
        source_file = str(source.get("source_file"))
        source_sha256 = str(source.get("source_artifact_sha256"))
        _require(source_file == f"msdata-{source_sha256[:12]}.msdata", "Redacted source name/hash mismatch")
        source_stem = Path(source_file).stem
        frames = source.get("frames")
        _require(isinstance(frames, list) and frames, f"Source frames are missing: {source_stem}")
        _require(len(frames) == int(source.get("acquisition_frames", -1)), f"Frame count mismatch: {source_stem}")

        for expected_index, frame in enumerate(frames):
            _require(isinstance(frame, dict), f"Invalid frame record: {source_stem}")
            _require(int(frame.get("frame_index", -1)) == expected_index, f"Non-contiguous frame index: {source_stem}")
            transition_count = int(frame.get("transition_trace_candidates", -1))
            _require(transition_count > 0, f"Frame has no transition traces: {source_stem}:{expected_index}")
            relative_path = f"{source_stem}/{source_stem}_{expected_index + 1}.mzML"
            expected_relative_paths.add(relative_path)
            mzml_path = (converted_root / Path(relative_path)).resolve()
            try:
                mzml_path.relative_to(converted_root)
            except ValueError as exc:
                raise ValueError(f"Converted path escapes root: {relative_path}") from exc
            _require(mzml_path.is_file(), f"Converted frame is missing: {relative_path}")
            chromatogram_count, parse_warning = mzml_chromatogram_count(mzml_path)
            _require(
                chromatogram_count == transition_count + 1,
                f"Chromatogram count mismatch for {relative_path}: expected {transition_count + 1}, got {chromatogram_count}",
            )
            if parse_warning is None:
                mzml_sha256 = sha256_file(mzml_path)
                normalized_size = mzml_path.stat().st_size
                parse_status = "strict"
            else:
                _require(
                    parse_warning.startswith("recovered_invalid_utf8:"),
                    f"Converted mzML cannot be recovered: {relative_path}: {parse_warning}",
                )
                normalized_bytes = mzml_path.read_bytes().decode(
                    "utf-8", errors="replace"
                ).encode("utf-8")
                mzml_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
                normalized_size = len(normalized_bytes)
                parse_status = "normalized_invalid_utf8"
                recovered_mzml_files += 1
            frame_id = str(frame.get("frame_id"))
            job_id = _frame_job_id(dataset_version, frame_id, mzml_sha256)
            normalized_relative_path = f"mzml/{relative_path}"
            auxiliary_records.append(
                {
                    "schema_version": "chrompeak-auxiliary-msdata-frame-v1",
                    "dataset_version": dataset_version,
                    "record_id": frame_id,
                    "job_id": job_id,
                    "source_group": str(source.get("source_group")),
                    "source_artifact_sha256": source_sha256,
                    "frame_index": expected_index,
                    "source_mzml": normalized_relative_path,
                    "artifact_hash": mzml_sha256,
                    "file_size_bytes": normalized_size,
                    "chromatogram_count": chromatogram_count,
                    "transition_trace_count": transition_count,
                    "total_signal_trace_count": 1,
                    "source_parse_status": parse_status,
                    "source_conversion_artifact_sha256": sha256_file(mzml_path),
                    "label_status": "unlabeled",
                    "split": "auxiliary_unlabeled_train",
                    "evaluation_tier": "auxiliary_training_only",
                    "supervised_train_eligible": False,
                    "auxiliary_unlabeled_train_eligible": True,
                    "benchmark_eligible": False,
                    "internal_test_accessed": False,
                    "_converted_relative_path": relative_path,
                }
            )
            derivation_jobs.append(
                {
                    "job_id": job_id,
                    "dataset_version": dataset_version,
                    "derivation_mode": "channel_driven_inference",
                    "split": "auxiliary_unlabeled_train",
                    "evaluation_tier": "auxiliary_training_only",
                    "metrics_allowed": False,
                    "split_group": str(source.get("source_group")),
                    "source_mzml": normalized_relative_path,
                    "artifact_hash": mzml_sha256,
                    "record_count": 0,
                    "positive": None,
                    "negative": None,
                    "labels": [],
                    "output_prefix": f"jobs/auxiliary_unlabeled_train/{job_id}",
                    "expected_outputs": [
                        "feature.csv",
                        "roi_windows.csv",
                        "xic_matrix.npy",
                        "*.jpeg",
                        "derivation_provenance.json",
                    ],
                }
            )
            total_transition_traces += transition_count
            total_chromatograms += chromatogram_count

    actual_relative_paths = {
        path.relative_to(converted_root).as_posix()
        for path in converted_root.rglob("*.mzML")
        if path.is_file()
    }
    _require(
        actual_relative_paths == expected_relative_paths,
        "Converted mzML inventory differs from the quarantine frame inventory",
    )
    _require(len(auxiliary_records) == int(report_counts.get("acquisition_frames", -1)), "Total frame count mismatch")
    _require(total_transition_traces == int(report_counts.get("transition_trace_candidates", -1)), "Total transition count mismatch")
    source_groups = {str(record["source_group"]) for record in auxiliary_records}
    _require(len(source_groups) == int(report_counts.get("independent_source_groups", -1)), "Source-group count mismatch")

    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        for record in auxiliary_records:
            converted_relative_path = str(record.pop("_converted_relative_path"))
            source_path = converted_root / Path(converted_relative_path)
            destination = staging / Path(str(record["source_mzml"]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if record["source_parse_status"] == "strict":
                shutil.copyfile(source_path, destination)
            else:
                destination.write_bytes(
                    source_path.read_bytes().decode(
                        "utf-8", errors="replace"
                    ).encode("utf-8")
                )
            _require(
                sha256_file(destination) == record["artifact_hash"],
                f"Normalized mzML hash mismatch: {record['source_mzml']}",
            )
            normalized_count, normalized_warning = mzml_chromatogram_count(destination)
            _require(
                normalized_warning is None
                and normalized_count == record["chromatogram_count"],
                f"Normalized mzML verification failed: {record['source_mzml']}",
            )
        manifest_path = staging / "auxiliary_manifest.jsonl"
        plan_path = staging / "derivation_plan.jsonl"
        _write_jsonl(manifest_path, auxiliary_records)
        _write_jsonl(plan_path, derivation_jobs)
        report_path = staging / "auxiliary_report.json"
        report = {
            "schema_version": "chrompeak-auxiliary-msdata-report-v1",
            "dataset_version": dataset_version,
            "dataset_digest_sha256": quarantine_report.get("dataset_digest_sha256"),
            "source_quarantine_report_sha256": sha256_file(quarantine_report_path),
            "source_quarantine_manifest_sha256": sha256_file(quarantine_manifest_path),
            "converter": {
                "artifact_sha256": sha256_file(converter_path),
                "name": converter_path.name,
            },
            "auxiliary_manifest_file": manifest_path.name,
            "auxiliary_manifest_sha256": sha256_file(manifest_path),
            "derivation_plan_file": plan_path.name,
            "derivation_plan_sha256": sha256_file(plan_path),
            "counts": {
                "source_files": len(quarantine_records),
                "independent_source_groups": len(source_groups),
                "acquisition_frames": len(auxiliary_records),
                "transition_traces": total_transition_traces,
                "total_signal_traces": len(auxiliary_records),
                "chromatograms": total_chromatograms,
                "strict_mzml_files": len(auxiliary_records) - recovered_mzml_files,
                "normalized_invalid_utf8_mzml_files": recovered_mzml_files,
                "labels": 0,
                "supervised_train_assets": 0,
                "auxiliary_unlabeled_train_frames": len(auxiliary_records),
                "benchmark_assets": 0,
            },
            "contracts": {
                "split": "auxiliary_unlabeled_train",
                "source_groups_independent": True,
                "source_mzml_inventory_parseable": True,
                "training_mzml_normalized_utf8": True,
                "conversion_inventory_complete": True,
                "train_supervision_present": False,
                "validation_membership_changed": False,
                "benchmark_membership_changed": False,
                "metrics_allowed": False,
                "internal_test_accessed": False,
            },
            "quality_gate_passed": True,
            "claim_limits": [
                "These converted frames are unlabeled auxiliary-training inputs, not supervised examples",
                "They cannot contribute validation, benchmark, AP, IoU, or classification metrics",
                "Pseudo-labels, if produced later, must remain explicitly weak and provenance-bound",
            ],
        }
        _write_json(report_path, report)
        artifact_paths = (
            *sorted((staging / "mzml").rglob("*.mzML")),
            manifest_path,
            plan_path,
            report_path,
        )
        (staging / "artifact_manifest.sha256").write_text(
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

    final_report = output_dir / "auxiliary_report.json"
    final_plan = output_dir / "derivation_plan.jsonl"
    return AuxiliaryMsdataResult(
        output_dir=output_dir,
        manifest_path=output_dir / "auxiliary_manifest.jsonl",
        report_path=final_report,
        plan_path=final_plan,
        report_sha256=sha256_file(final_report),
        plan_sha256=sha256_file(final_plan),
        source_groups=len(source_groups),
        acquisition_frames=len(auxiliary_records),
        transition_traces=total_transition_traces,
    )
