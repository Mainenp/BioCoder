from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.contracts import (
    as_float,
    as_int,
    as_text,
    decide_eligibility,
    normalize_headers,
    peak_intervals,
)
from multimodal_science.data.xlsx import read_first_flat_worksheet


@dataclass(frozen=True)
class BuildResult:
    dataset_version: str
    manifest_path: Path
    report_path: Path
    record_count: int
    train_eligible_count: int
    benchmark_eligible_count: int
    audit_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mzml_chromatogram_count(path: Path) -> tuple[int, str | None]:
    def count_stream(stream: Any) -> int:
        count = 0
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] == "chromatogram":
                count += 1
            element.clear()
        return count

    try:
        return count_stream(path), None
    except (ET.ParseError, OSError) as exc:
        strict_error = str(exc)
    try:
        recovered = path.read_bytes().decode("utf-8", errors="replace").encode("utf-8")
        return count_stream(io.BytesIO(recovered)), f"recovered_invalid_utf8: {strict_error}"
    except (ET.ParseError, OSError) as exc:
        return 0, f"strict={strict_error}; recovery={exc}"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _dataset_name(workbook: Path) -> tuple[str, bool]:
    suffix = "_traditional"
    if workbook.stem.casefold().endswith(suffix):
        return workbook.stem[: -len(suffix)], True
    return workbook.stem, False


def _numeric_list(row: dict[str, Any], prefix: str) -> list[float]:
    values = []
    for index in range(1, 4):
        value = as_float(row.get(f"{prefix}{index}"))
        if value is not None:
            values.append(value)
    return values


def build_manifest(
    data_root: Path,
    output_dir: Path,
    *,
    source_archive_sha256: str,
) -> BuildResult:
    data_root = data_root.resolve()
    label_dir = data_root / "label"
    mzml_root = data_root / "mzml"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")
    if not mzml_root.is_dir():
        raise FileNotFoundError(f"Missing mzML directory: {mzml_root}")

    archive_hash = source_archive_sha256.strip().lower()
    invalid_hash_character = any(
        character not in "0123456789abcdef" for character in archive_hash
    )
    if len(archive_hash) != 64 or invalid_hash_character:
        raise ValueError("source_archive_sha256 must be a 64-character hexadecimal digest")
    dataset_version = f"raw-{archive_hash[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_mzml = sorted(mzml_root.rglob("*.mzML"), key=lambda path: path.as_posix().casefold())
    hash_cache = {path: sha256_file(path) for path in all_mzml}
    signal_cache = {path: mzml_chromatogram_count(path) for path in all_mzml}
    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for path, digest in hash_cache.items():
        hash_groups[digest].append(path)

    records: list[dict[str, Any]] = []
    matched_primary_sources: set[Path] = set()
    workbook_summaries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    workbooks = sorted(label_dir.glob("*.xlsx"), key=lambda path: path.name.casefold())
    for workbook_path in workbooks:
        dataset_name, alternate = _dataset_name(workbook_path)
        worksheet = read_first_flat_worksheet(workbook_path)
        dataset_mzml_dir = mzml_root / dataset_name
        dataset_sources = list(dataset_mzml_dir.glob("*.mzML")) if dataset_mzml_dir.is_dir() else []
        by_basename: dict[str, list[Path]] = defaultdict(list)
        for source_path in dataset_sources:
            by_basename[source_path.name.casefold()].append(source_path)

        workbook_counts: Counter[str] = Counter()
        for source_row, raw_row in enumerate(worksheet.rows, start=2):
            row = normalize_headers(raw_row)
            sample_id = as_text(row.get("sample_id"))
            candidates = by_basename.get(sample_id.casefold(), []) if sample_id else []
            matched_path = candidates[0] if len(candidates) == 1 else None
            decision = decide_eligibility(
                row,
                matched_source_count=len(candidates),
                alternate_label_variant=alternate,
                source_signal_valid=(
                    matched_path is None
                    or signal_cache[matched_path][0] > 0
                ),
            )
            if matched_path is not None and not alternate:
                matched_primary_sources.add(matched_path)
            intervals, _ = peak_intervals(row)
            record_id_seed = (
                f"{dataset_version}|{workbook_path.name}|{source_row}|{row.get('roi_id')}"
            )
            record = {
                "record_id": hashlib.sha256(record_id_seed.encode()).hexdigest()[:24],
                "dataset_version": dataset_version,
                "dataset_name": dataset_name,
                "label_variant": "alternate" if alternate else "primary",
                "label_workbook": workbook_path.name,
                "label_sheet": worksheet.name,
                "source_row": source_row,
                "roi_id": as_int(row.get("roi_id")),
                "sample_id": sample_id,
                "source_mzml": _relative(matched_path, data_root) if matched_path else None,
                "artifact_hash": hash_cache.get(matched_path),
                "chromatogram_count": signal_cache[matched_path][0] if matched_path else None,
                "source_signal_status": (
                    "recovered"
                    if matched_path
                    and signal_cache[matched_path][0] > 0
                    and signal_cache[matched_path][1]
                    else "strict"
                    if matched_path and signal_cache[matched_path][0] > 0
                    else "invalid"
                    if matched_path
                    else None
                ),
                "match_strategy": str(decision.match_strategy),
                "fallback_order": decision.fallback_order,
                "train_eligible": decision.train_eligible,
                "benchmark_eligible": decision.benchmark_eligible,
                "audit_bucket": decision.audit_bucket,
                "exclusion_reasons": list(decision.reasons),
                "split_group": _relative(matched_path, data_root) if matched_path else None,
                "component": as_text(row.get("component")),
                "channel": as_text(row.get("channel")),
                "expected_rt": as_float(row.get("rt")),
                "peak_label": as_int(row.get("peak_label")),
                "peak_count": as_int(row.get("peak_count")),
                "peak_intervals": intervals,
                "areas": _numeric_list(row, "area"),
                "snr": as_float(row.get("snr")),
                "instrument": as_text(row.get("instrument")),
                "raw_file": as_text(row.get("raw_file")),
                "product_id": as_text(row.get("product_id")),
            }
            records.append(record)
            workbook_counts["records"] += 1
            workbook_counts["train_eligible"] += int(decision.train_eligible)
            workbook_counts["benchmark_eligible"] += int(decision.benchmark_eligible)
            workbook_counts["audit"] += int(decision.audit_bucket)
            reason_counts.update(decision.reasons)

        workbook_summaries.append(
            {
                "workbook": workbook_path.name,
                "dataset_name": dataset_name,
                "label_variant": "alternate" if alternate else "primary",
                **workbook_counts,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, records)
    duplicate_groups = [
        {
            "sha256": digest,
            "files": [_relative(path, data_root) for path in paths],
        }
        for digest, paths in sorted(hash_groups.items())
        if len(paths) > 1
    ]
    extension_counts = Counter(
        path.suffix.casefold() or "<none>"
        for path in data_root.rglob("*")
        if path.is_file()
    )
    train_eligible_count = sum(int(record["train_eligible"]) for record in records)
    benchmark_eligible_count = sum(int(record["benchmark_eligible"]) for record in records)
    audit_count = sum(int(record["audit_bucket"]) for record in records)
    eligible_records = [record for record in records if record["train_eligible"]]
    dataset_label_summary = []
    for dataset_name in sorted({record["dataset_name"] for record in eligible_records}):
        dataset_records = [
            record for record in eligible_records if record["dataset_name"] == dataset_name
        ]
        dataset_label_summary.append(
            {
                "dataset_name": dataset_name,
                "records": len(dataset_records),
                "samples": len({record["sample_id"] for record in dataset_records}),
                "positive": sum(record["peak_label"] == 1 for record in dataset_records),
                "negative": sum(record["peak_label"] == 0 for record in dataset_records),
                "components": len(
                    {record["component"] for record in dataset_records if record["component"]}
                ),
            }
        )
    signal_dataset_summary = []
    for dataset_name in sorted({path.parent.name for path in all_mzml}):
        dataset_paths = [path for path in all_mzml if path.parent.name == dataset_name]
        counts = [signal_cache[path][0] for path in dataset_paths]
        signal_dataset_summary.append(
            {
                "dataset_name": dataset_name,
                "files": len(dataset_paths),
                "min_chromatograms": min(counts),
                "max_chromatograms": max(counts),
                "total_chromatograms": sum(counts),
                "invalid_files": sum(
                    signal_cache[path][0] == 0 for path in dataset_paths
                ),
                "recovered_files": sum(
                    signal_cache[path][0] > 0 and signal_cache[path][1] is not None
                    for path in dataset_paths
                ),
            }
        )
    report = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_archive_sha256": archive_hash,
        "manifest_file": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "counts": {
            "records": len(records),
            "train_eligible": train_eligible_count,
            "benchmark_eligible": benchmark_eligible_count,
            "audit": audit_count,
            "mzml_files": len(all_mzml),
            "labeled_primary_mzml": len(matched_primary_sources),
            "unlabeled_mzml": len(set(all_mzml) - matched_primary_sources),
        },
        "file_extensions": dict(sorted(extension_counts.items())),
        "workbooks": workbook_summaries,
        "label_summary": {
            "positive": sum(record["peak_label"] == 1 for record in eligible_records),
            "negative": sum(record["peak_label"] == 0 for record in eligible_records),
            "unique_components": len(
                {record["component"] for record in eligible_records if record["component"]}
            ),
            "channels": dict(
                sorted(Counter(record["channel"] for record in eligible_records).items())
            ),
            "instruments": dict(
                sorted(Counter(record["instrument"] for record in eligible_records).items())
            ),
            "datasets": dataset_label_summary,
        },
        "mzml_signal_summary": signal_dataset_summary,
        "invalid_mzml": [
            {
                "path": _relative(path, data_root),
                "chromatogram_count": signal_cache[path][0],
                "error": signal_cache[path][1],
            }
            for path in all_mzml
            if signal_cache[path][0] == 0
        ],
        "mzml_parse_warnings": [
            {
                "path": _relative(path, data_root),
                "chromatogram_count": signal_cache[path][0],
                "warning": signal_cache[path][1],
            }
            for path in all_mzml
            if signal_cache[path][0] > 0 and signal_cache[path][1] is not None
        ],
        "audit_reason_counts": dict(sorted(reason_counts.items())),
        "unlabeled_mzml": [
            _relative(path, data_root) for path in sorted(set(all_mzml) - matched_primary_sources)
        ],
        "duplicate_mzml_content": duplicate_groups,
    }
    report_path = output_dir / "audit_report.json"
    _write_json(report_path, report)
    return BuildResult(
        dataset_version=dataset_version,
        manifest_path=manifest_path,
        report_path=report_path,
        record_count=len(records),
        train_eligible_count=train_eligible_count,
        benchmark_eligible_count=benchmark_eligible_count,
        audit_count=audit_count,
    )
