from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file

CHROMPEAKFORMER_REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pyopenms",
)
DEFAULT_DERIVATION_SPLITS = frozenset(
    {"train", "validation", "internal_test", "auxiliary_train", "legacy_external"}
)


@dataclass(frozen=True)
class DerivationPlanResult:
    dataset_version: str
    plan_path: Path
    report_path: Path
    plan_sha256: str
    job_count: int
    source_count: int
    execution_ready: bool


def dependency_readiness(
    module_names: Iterable[str] = CHROMPEAKFORMER_REQUIRED_MODULES,
) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in module_names}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"Split manifest is empty: {path}")
    return records


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


def _job_id(dataset_version: str, source_mzml: str, artifact_hash: str) -> str:
    seed = f"{dataset_version}\0{source_mzml}\0{artifact_hash}".encode()
    return hashlib.sha256(seed).hexdigest()[:24]


def _label_input(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "component": record.get("component"),
        "channel": record.get("channel"),
        "rt": record.get("expected_rt"),
        "peak_label": record.get("peak_label"),
        "peak_intervals": record.get("peak_intervals", []),
    }


def _supervised_jobs(
    records: list[dict[str, Any]], data_root: Path, include_splits: frozenset[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") in include_splits:
            if record.get("audit_bucket") or not record.get("train_eligible"):
                raise ValueError(
                    f"Derivation split contains ineligible record: {record.get('record_id')}"
                )
            grouped[str(record["split_group"])].append(record)

    jobs = []
    source_checks = []
    for split_group, group_records in sorted(grouped.items()):
        fields = {
            "source_mzml": {record.get("source_mzml") for record in group_records},
            "artifact_hash": {record.get("artifact_hash") for record in group_records},
            "split": {record.get("split") for record in group_records},
            "evaluation_tier": {record.get("evaluation_tier") for record in group_records},
            "dataset_version": {record.get("dataset_version") for record in group_records},
        }
        inconsistent = [
            name for name, values in fields.items() if len(values) != 1 or None in values
        ]
        if inconsistent:
            raise ValueError(f"Inconsistent group {split_group}: {inconsistent}")

        source_mzml = str(next(iter(fields["source_mzml"])))
        expected_hash = str(next(iter(fields["artifact_hash"])))
        source_path = (data_root / Path(source_mzml)).resolve()
        try:
            source_path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"Source escapes data root: {source_mzml}") from exc
        if not source_path.is_file():
            raise FileNotFoundError(f"Source mzML not found: {source_mzml}")
        actual_hash = sha256_file(source_path)
        if actual_hash != expected_hash:
            message = (
                f"Source hash mismatch for {source_mzml}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
            raise ValueError(message)

        dataset_version = str(next(iter(fields["dataset_version"])))
        job_id = _job_id(dataset_version, source_mzml, actual_hash)
        label_inputs = sorted(
            (_label_input(record) for record in group_records),
            key=lambda item: str(item["record_id"]),
        )
        split = str(next(iter(fields["split"])))
        jobs.append(
            {
                "job_id": job_id,
                "dataset_version": dataset_version,
                "derivation_mode": "label_driven",
                "split": split,
                "evaluation_tier": str(next(iter(fields["evaluation_tier"]))),
                "metrics_allowed": True,
                "split_group": split_group,
                "source_mzml": source_mzml,
                "artifact_hash": actual_hash,
                "record_count": len(group_records),
                "positive": sum(record.get("peak_label") == 1 for record in group_records),
                "negative": sum(record.get("peak_label") == 0 for record in group_records),
                "labels": label_inputs,
                "output_prefix": f"jobs/{split}/{job_id}",
                "expected_outputs": [
                    "feature.csv",
                    "roi_windows.csv",
                    "xic_matrix.npy",
                    "roi_images/*.jpeg",
                    "derivation_provenance.json",
                ],
            }
        )
        source_checks.append(
            {"source_mzml": source_mzml, "artifact_hash": actual_hash, "status": "verified"}
        )
    return jobs, source_checks


def _unlabeled_jobs(
    audit_report_path: Path | None,
    data_root: Path,
    dataset_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if audit_report_path is None:
        return [], []
    audit_report = json.loads(audit_report_path.resolve().read_text(encoding="utf-8"))
    if audit_report.get("dataset_version") != dataset_version:
        raise ValueError("Audit report dataset_version does not match split manifest")

    jobs = []
    source_checks = []
    for source_mzml in sorted(str(path) for path in audit_report.get("unlabeled_mzml", [])):
        source_path = (data_root / Path(source_mzml)).resolve()
        try:
            source_path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"Source escapes data root: {source_mzml}") from exc
        if not source_path.is_file():
            raise FileNotFoundError(f"Unlabeled source mzML not found: {source_mzml}")
        actual_hash = sha256_file(source_path)
        job_id = _job_id(dataset_version, source_mzml, actual_hash)
        jobs.append(
            {
                "job_id": job_id,
                "dataset_version": dataset_version,
                "derivation_mode": "channel_driven_inference",
                "split": "unlabeled_external",
                "evaluation_tier": "inference_only",
                "metrics_allowed": False,
                "split_group": source_mzml,
                "source_mzml": source_mzml,
                "artifact_hash": actual_hash,
                "record_count": 0,
                "positive": None,
                "negative": None,
                "labels": [],
                "output_prefix": f"jobs/unlabeled_external/{job_id}",
                "expected_outputs": [
                    "feature.csv",
                    "roi_windows.csv",
                    "xic_matrix.npy",
                    "roi_images/*.jpeg",
                    "derivation_provenance.json",
                ],
            }
        )
        source_checks.append(
            {"source_mzml": source_mzml, "artifact_hash": actual_hash, "status": "verified"}
        )
    return jobs, source_checks


def build_derivation_plan(
    split_manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    audit_report_path: Path | None = None,
    include_splits: frozenset[str] = DEFAULT_DERIVATION_SPLITS,
) -> DerivationPlanResult:
    split_manifest_path = split_manifest_path.resolve()
    data_root = data_root.resolve()
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    records = _read_jsonl(split_manifest_path)
    versions = {record.get("dataset_version") for record in records}
    if len(versions) != 1 or None in versions:
        raise ValueError("Split manifest must contain exactly one dataset_version")
    dataset_version = str(next(iter(versions)))

    supervised_jobs, supervised_checks = _supervised_jobs(records, data_root, include_splits)
    unlabeled_jobs, unlabeled_checks = _unlabeled_jobs(
        audit_report_path, data_root, dataset_version
    )
    jobs = sorted(supervised_jobs + unlabeled_jobs, key=lambda job: str(job["job_id"]))
    if not jobs:
        raise ValueError("No derivation jobs were selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "derivation_plan.jsonl"
    _write_jsonl(plan_path, jobs)
    plan_sha256 = sha256_file(plan_path)
    readiness = dependency_readiness()
    missing_modules = sorted(name for name, available in readiness.items() if not available)
    execution_ready = not missing_modules
    source_checks = sorted(
        supervised_checks + unlabeled_checks, key=lambda item: str(item["source_mzml"])
    )
    report = {
        "schema_version": "chrompeak-derivation-v1",
        "dataset_version": dataset_version,
        "source_split_manifest": split_manifest_path.name,
        "source_split_manifest_sha256": sha256_file(split_manifest_path),
        "derivation_plan": plan_path.name,
        "derivation_plan_sha256": plan_sha256,
        "counts": {
            "jobs": len(jobs),
            "supervised_jobs": len(supervised_jobs),
            "unlabeled_inference_jobs": len(unlabeled_jobs),
            "sources_verified": len(source_checks),
            "label_records": sum(int(job["record_count"]) for job in jobs),
        },
        "dependency_gate": {
            "required_modules": readiness,
            "missing_modules": missing_modules,
            "execution_ready": execution_ready,
            "installation_attempted": False,
        },
        "extractor_contract": {
            "public_name": "ChromPeakFormer",
            "input_modes": ["label_driven", "channel_driven_inference"],
            "roi_shape": [300, 400],
            "rt_window_minutes": 2.0,
            "numerical_sequence": "xic_matrix.npy row 0 is RT; following rows are intensities",
            "required_outputs": ["feature.csv", "roi_windows.csv", "xic_matrix.npy"],
        },
        "source_verification": source_checks,
        "claim_limits": [
            "A ready plan is not evidence that ROI extraction or model training completed",
            "unlabeled_external jobs are inference-only and cannot produce supervised metrics",
        ],
    }
    report_path = output_dir / "derivation_report.json"
    _write_json(report_path, report)
    return DerivationPlanResult(
        dataset_version=dataset_version,
        plan_path=plan_path,
        report_path=report_path,
        plan_sha256=plan_sha256,
        job_count=len(jobs),
        source_count=len(source_checks),
        execution_ready=execution_ready,
    )
