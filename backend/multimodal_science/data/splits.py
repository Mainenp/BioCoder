from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file

PRIMARY_DATASET = "traindata3"
AUXILIARY_DATASETS = frozenset({"traindata1", "traindata2"})
LEGACY_EXTERNAL_DATASET = "test1"
PROTECTED_SPLITS = ("train", "validation", "internal_test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_SEED = "biocoder-chrompeak-group-split-v1"


@dataclass(frozen=True)
class SplitResult:
    dataset_version: str
    split_manifest_path: Path
    report_path: Path
    split_manifest_sha256: str
    record_count: int
    group_count: int


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def classify_group(records: list[dict[str, Any]]) -> str:
    sample_ids = {str(record.get("sample_id") or "").casefold() for record in records}
    if any("blank" in sample_id or "空白" in sample_id for sample_id in sample_ids):
        return "blank"
    if any("qc" in sample_id for sample_id in sample_ids):
        return "qc"
    return "sample"


def _allocation_counts(group_count: int) -> tuple[int, int, int]:
    exact = [group_count * ratio for ratio in SPLIT_RATIOS]
    counts = [int(value) for value in exact]
    remainder = group_count - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _stable_group_key(seed: str, dataset_version: str, stratum: str, group: str) -> str:
    payload = f"{seed}\0{dataset_version}\0{stratum}\0{group}".encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest(records: list[dict[str, Any]]) -> str:
    record_ids = [record.get("record_id") for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Every manifest record must have a non-empty record_id")
    duplicates = [key for key, count in Counter(record_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate record_id values: {duplicates[:3]}")

    dataset_versions = {record.get("dataset_version") for record in records}
    if len(dataset_versions) != 1 or None in dataset_versions:
        raise ValueError("A split manifest must contain exactly one dataset_version")

    for record in records:
        if record.get("train_eligible") and not record.get("split_group"):
            raise ValueError(
                f"Eligible record {record['record_id']} has no split_group; refusing to split"
            )
    return str(next(iter(dataset_versions)))


def _primary_assignments(
    records: list[dict[str, Any]], dataset_version: str, seed: str
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            record.get("dataset_name") == PRIMARY_DATASET
            and record.get("train_eligible")
            and not record.get("audit_bucket")
        ):
            groups[str(record["split_group"])].append(record)

    content_to_groups: dict[str, set[str]] = defaultdict(set)
    for group, group_records in groups.items():
        hashes = {record.get("artifact_hash") for record in group_records}
        if len(hashes) != 1 or None in hashes:
            raise ValueError(f"Primary group {group} must have exactly one artifact_hash")
        content_to_groups[str(next(iter(hashes)))].add(group)
    duplicate_content = {
        digest: sorted(paths) for digest, paths in content_to_groups.items() if len(paths) > 1
    }
    if duplicate_content:
        first_digest, paths = next(iter(sorted(duplicate_content.items())))
        raise ValueError(
            "Duplicate primary mzML content appears under multiple split groups: "
            f"{first_digest} -> {paths}. Deduplicate before splitting."
        )

    strata: dict[str, list[str]] = defaultdict(list)
    group_strata: dict[str, str] = {}
    for group, group_records in groups.items():
        stratum = classify_group(group_records)
        strata[stratum].append(group)
        group_strata[group] = stratum

    assignments: dict[str, str] = {}
    allocation: dict[str, dict[str, int]] = {}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda group: _stable_group_key(seed, dataset_version, stratum, group),
        )
        train_count, validation_count, internal_test_count = _allocation_counts(len(ordered))
        boundaries = (train_count, train_count + validation_count)
        for index, group in enumerate(ordered):
            if index < boundaries[0]:
                split = "train"
            elif index < boundaries[1]:
                split = "validation"
            else:
                split = "internal_test"
            assignments[group] = split
        allocation[stratum] = {
            "groups": len(ordered),
            "train": train_count,
            "validation": validation_count,
            "internal_test": internal_test_count,
        }
    return assignments, group_strata, allocation


def _record_assignment(
    record: dict[str, Any],
    primary_assignments: dict[str, str],
    primary_strata: dict[str, str],
) -> tuple[str, str, str | None, str]:
    if record.get("audit_bucket") or not record.get("train_eligible"):
        return "audit", "audit_only", None, "manifest_eligibility_gate"

    dataset_name = record.get("dataset_name")
    if dataset_name == PRIMARY_DATASET:
        group = str(record["split_group"])
        return (
            primary_assignments[group],
            "primary_internal_test"
            if primary_assignments[group] == "internal_test"
            else "primary_development",
            primary_strata[group],
            "stratified_source_mzml_group",
        )
    if dataset_name in AUXILIARY_DATASETS:
        return "auxiliary_train", "auxiliary_negative_only", "negative_only", "dataset_policy"
    if dataset_name == LEGACY_EXTERNAL_DATASET:
        return (
            "legacy_external",
            "legacy_external_non_pristine",
            classify_group([record]),
            "historical_evaluation_only",
        )
    return "audit", "audit_only", None, "unrecognized_dataset_fail_closed"


def _overlap_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)

    group_sets = {
        split: {record["split_group"] for record in by_split[split] if record.get("split_group")}
        for split in PROTECTED_SPLITS
    }
    hash_sets = {
        split: {
            record["artifact_hash"]
            for record in by_split[split]
            if record.get("artifact_hash")
        }
        for split in PROTECTED_SPLITS
    }
    group_overlap = []
    artifact_hash_overlap = []
    for left_index, left in enumerate(PROTECTED_SPLITS):
        for right in PROTECTED_SPLITS[left_index + 1 :]:
            group_overlap.extend(
                {"left": left, "right": right, "value": value}
                for value in sorted(group_sets[left] & group_sets[right])
            )
            artifact_hash_overlap.extend(
                {"left": left, "right": right, "value": value}
                for value in sorted(hash_sets[left] & hash_sets[right])
            )
    audit_contamination = [
        record["record_id"]
        for split in PROTECTED_SPLITS
        for record in by_split[split]
        if record.get("audit_bucket") or not record.get("train_eligible")
    ]
    return {
        "protected_splits": list(PROTECTED_SPLITS),
        "split_group_overlap": group_overlap,
        "artifact_hash_overlap": artifact_hash_overlap,
        "audit_contamination_record_ids": sorted(audit_contamination),
        "passed": not group_overlap and not artifact_hash_overlap and not audit_contamination,
    }


def _split_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for split in sorted({str(record["split"]) for record in records}):
        split_records = [record for record in records if record["split"] == split]
        summaries.append(
            {
                "split": split,
                "records": len(split_records),
                "groups": len(
                    {record["split_group"] for record in split_records if record.get("split_group")}
                ),
                "positive": sum(record.get("peak_label") == 1 for record in split_records),
                "negative": sum(record.get("peak_label") == 0 for record in split_records),
                "audit_records": sum(bool(record.get("audit_bucket")) for record in split_records),
                "strata": dict(
                    sorted(
                        Counter(
                            record["split_stratum"]
                            for record in split_records
                            if record.get("split_stratum")
                        ).items()
                    )
                ),
            }
        )
    return summaries


def build_splits(
    manifest_path: Path,
    output_dir: Path,
    *,
    audit_report_path: Path | None = None,
    seed: str = DEFAULT_SEED,
) -> SplitResult:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not seed.strip():
        raise ValueError("seed must not be empty")

    records = _read_jsonl(manifest_path)
    dataset_version = _validate_manifest(records)
    assignments, strata, allocation = _primary_assignments(records, dataset_version, seed)

    split_records = []
    for source in records:
        split, evaluation_tier, stratum, reason = _record_assignment(
            source, assignments, strata
        )
        split_records.append(
            {
                **source,
                "split": split,
                "evaluation_tier": evaluation_tier,
                "split_stratum": stratum,
                "split_assignment_reason": reason,
            }
        )
    split_records.sort(key=lambda record: str(record["record_id"]))

    leakage_audit = _overlap_report(split_records)
    if not leakage_audit["passed"]:
        raise ValueError(f"Leakage gate failed: {leakage_audit}")

    unlabeled_sources: list[str] = []
    if audit_report_path is not None:
        audit_report = json.loads(audit_report_path.resolve().read_text(encoding="utf-8"))
        if audit_report.get("dataset_version") != dataset_version:
            raise ValueError("Audit report dataset_version does not match manifest")
        unlabeled_sources = sorted(str(path) for path in audit_report.get("unlabeled_mzml", []))

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_dir / "split_manifest.jsonl"
    _write_jsonl(split_manifest_path, split_records)
    split_manifest_sha256 = sha256_file(split_manifest_path)
    group_count = len(
        {record["split_group"] for record in split_records if record.get("split_group")}
    )
    report = {
        "schema_version": "chrompeak-split-v1",
        "dataset_version": dataset_version,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": split_manifest_path.name,
        "split_manifest_sha256": split_manifest_sha256,
        "seed": seed,
        "policy": {
            "primary_dataset": PRIMARY_DATASET,
            "primary_group_key": "source_mzml via split_group",
            "primary_strata": ["blank", "qc", "sample"],
            "primary_ratios": dict(zip(PROTECTED_SPLITS, SPLIT_RATIOS, strict=True)),
            "auxiliary_datasets": sorted(AUXILIARY_DATASETS),
            "legacy_external_dataset": LEGACY_EXTERNAL_DATASET,
        },
        "counts": {"records": len(split_records), "groups": group_count},
        "primary_group_allocation": allocation,
        "splits": _split_summary(split_records),
        "leakage_audit": leakage_audit,
        "unlabeled_external": {
            "sources": unlabeled_sources,
            "metrics_allowed": False,
            "reason": "no_label_rows_in_manifest",
        },
        "claim_limits": [
            "legacy_external is historical and must not be described as a pristine unseen test set",
            "unlabeled_external may be used for inference demos but not supervised metrics",
            "auxiliary_train is negative-only and is excluded from primary benchmark metrics",
        ],
    }
    report_path = output_dir / "split_report.json"
    _write_json(report_path, report)
    return SplitResult(
        dataset_version=dataset_version,
        split_manifest_path=split_manifest_path,
        report_path=report_path,
        split_manifest_sha256=split_manifest_sha256,
        record_count=len(split_records),
        group_count=group_count,
    )
