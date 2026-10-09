"""Verify sealed benchmark evidence and publish a path-free release capsule.

This surface is deliberately read-only. It verifies the completed one-time access
ledger and hash-bound evaluation reports, but never opens predictions, answer keys,
raw chromatograms, images, model weights, or private detector sources.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FINAL_SEQUENCE_EVALUATION_SCHEMA = "chrompeak-sequence-final-evaluation-v1"
FINAL_DETECTOR_EVALUATION_SCHEMA = "chrompeak-detector-final-evaluation-v1"
FINAL_QWEN_EVALUATION_SCHEMA = "chrompeak-qwen3vl-final-evaluation-v1"
FINAL_BENCHMARK_PROTOCOL_SCHEMA = "chrompeak-final-benchmark-protocol-v1"
FINAL_BENCHMARK_REPORT_SCHEMA = "chrompeak-final-benchmark-report-v1"
FINAL_BENCHMARK_REGISTRY_SCHEMA = "chrompeak-final-benchmark-registry-v1"
ACCESS_EVENT_SCHEMA = "chrompeak-final-benchmark-access-v1"
ACCESS_COMPLETION_SCHEMA = "chrompeak-final-benchmark-access-completion-v1"
FINAL_BENCHMARK_RELEASE_SCHEMA = "chrompeak-final-benchmark-release-v1"
FINAL_BENCHMARK_CHECKSUMS_SCHEMA = "chrompeak-final-benchmark-release-checksums-v1"

_QWEN_CANDIDATES = (
    "qwen3vl_zero_shot",
    "qwen3vl_image_lora",
    "qwen3vl_image_xic_fusion",
)
_PRIMARY_CANDIDATES = (*_QWEN_CANDIDATES, "sequence_peak_net", "chrompeakformer")
_EVALUATION_FILES = {
    "qwen3vl_zero_shot": ("qwen_evaluation_report.json", FINAL_QWEN_EVALUATION_SCHEMA),
    "qwen3vl_image_lora": ("qwen_evaluation_report.json", FINAL_QWEN_EVALUATION_SCHEMA),
    "qwen3vl_image_xic_fusion": (
        "qwen_evaluation_report.json",
        FINAL_QWEN_EVALUATION_SCHEMA,
    ),
    "sequence_peak_net": (
        "sequence_evaluation_report.json",
        FINAL_SEQUENCE_EVALUATION_SCHEMA,
    ),
    "chrompeakformer": (
        "detector_evaluation_report.json",
        FINAL_DETECTOR_EVALUATION_SCHEMA,
    ),
}
_FINAL_REPORT_FILES = {
    "final_benchmark_report.json",
    "final_benchmark_table.md",
    "final_evidence_registry.json",
}
_RELEASE_FILES = {
    "README.md",
    "benchmark_summary.json",
    "benchmark_table.md",
    "evidence_checksums.json",
}
_REQUIRED_CONTRACTS = {
    "all_primary_candidates_present": True,
    "same_one_time_access_event": True,
    "candidate_selection_performed_after_test": False,
    "threshold_selection_performed_after_test": False,
    "combined_cross_task_score_reported": False,
    "qwen_languages_reported_separately": True,
    "additional_internal_test_access_authorized": False,
    "biocoder_agent_promotion_claimed": False,
}


@dataclass(frozen=True)
class FinalBenchmarkEvidenceVerification:
    protocol_sha256: str
    access_id: str
    access_completed: bool
    report_sha256: str
    report_manifest_sha256: str
    candidates: int
    rows: int
    internal_test_assets: int
    internal_test_source_groups: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "access_id": self.access_id,
            "access_completed": self.access_completed,
            "report_sha256": self.report_sha256,
            "report_manifest_sha256": self.report_manifest_sha256,
            "candidates": self.candidates,
            "rows": self.rows,
            "internal_test_assets": self.internal_test_assets,
            "internal_test_source_groups": self.internal_test_source_groups,
        }


@dataclass(frozen=True)
class FinalBenchmarkReleaseResult:
    output_dir: Path
    archive_path: Path
    archive_sha256: str
    manifest_path: Path
    manifest_sha256: str
    summary_path: Path
    protocol_sha256: str
    report_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "archive_path": str(self.archive_path),
            "archive_sha256": self.archive_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "summary_path": str(self.summary_path),
            "protocol_sha256": self.protocol_sha256,
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True)
class PublicReleaseVerification:
    release_root: Path
    manifest_sha256: str
    protocol_sha256: str
    report_sha256: str
    candidates: int
    rows: int
    archive_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_root": str(self.release_root),
            "manifest_sha256": self.manifest_sha256,
            "protocol_sha256": self.protocol_sha256,
            "report_sha256": self.report_sha256,
            "candidates": self.candidates,
            "rows": self.rows,
            "archive_sha256": self.archive_sha256,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex64(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid {label}",
    )
    return value


def _normalize_manifest_path(raw: str, label: str) -> str:
    normalized = raw.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    _require(
        bool(normalized)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "." not in pure.parts,
        f"Unsafe {label} manifest path: {raw}",
    )
    return pure.as_posix()


def _manifest_entries(
    root: Path,
    *,
    label: str,
    expected_files: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    root = root.resolve()
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Missing {label} manifest: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.strip().split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed {label} manifest line {line_number}")
        digest, raw_relative = parts
        relative = _normalize_manifest_path(raw_relative, label)
        digest = _hex64(digest, f"{label} digest")
        _require(relative not in entries, f"Duplicate {label} entry")
        path = (root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label.capitalize()} manifest path escapes its root") from error
        _require(path.is_file(), f"Missing {label} artifact: {relative}")
        _require(sha256_file(path) == digest, f"{label.capitalize()} artifact drift: {relative}")
        entries[relative] = digest
    _require(bool(entries), f"{label.capitalize()} manifest is empty")
    if expected_files is not None:
        _require(set(entries) == expected_files, f"Unexpected {label} artifact set")
    return sha256_file(manifest), entries


def _verify_frozen_protocol(
    protocol_root: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    expected_protocol_sha256 = _hex64(expected_protocol_sha256, "protocol SHA-256")
    _, entries = _manifest_entries(
        protocol_root,
        label="protocol",
        expected_files={
            "final_benchmark_protocol.json",
            "candidate_lock.json",
            "metrics_lock.json",
        },
    )
    report_path = protocol_root / "final_benchmark_protocol.json"
    _require(
        entries[report_path.name] == expected_protocol_sha256,
        "Protocol hash mismatch",
    )
    protocol = _read_json(report_path, "final benchmark protocol")
    _require(
        protocol.get("schema_version") == FINAL_BENCHMARK_PROTOCOL_SCHEMA,
        "Bad protocol schema",
    )
    _require(protocol.get("pre_internal_test_ready") is True, "Protocol is not ready")
    _require(protocol.get("internal_test_accessed") is False, "Protocol was not pre-access")
    return protocol


def _verify_completed_access(
    ledger_dir: Path, expected_protocol_sha256: str
) -> tuple[str, str, dict[str, Any]]:
    manifest_sha256, _ = _manifest_entries(
        ledger_dir,
        label="access ledger",
        expected_files={"access_started.json", "access_completed.json"},
    )
    started_path = ledger_dir / "access_started.json"
    checksum_path = ledger_dir / "access_started.sha256"
    _require(checksum_path.is_file(), "Access ledger checksum is missing")
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    _require(len(checksum_parts) == 2, "Invalid access-ledger checksum")
    _require(
        _hex64(checksum_parts[0], "access-ledger checksum") == sha256_file(started_path)
        and _normalize_manifest_path(checksum_parts[1], "access ledger")
        == started_path.name,
        "Access ledger was modified",
    )
    started = _read_json(started_path, "access ledger")
    _require(started.get("schema_version") == ACCESS_EVENT_SCHEMA, "Bad access schema")
    _require(started.get("access_sequence") == 1, "Invalid access sequence")
    _require(started.get("protocol_sha256") == expected_protocol_sha256, "Access protocol drift")
    _require(started.get("internal_test_accessed") is True, "Access event is not open")
    access_id = started.get("access_id")
    _require(isinstance(access_id, str) and bool(access_id), "Access ID is missing")

    completion = _read_json(ledger_dir / "access_completed.json", "access completion")
    _require(completion.get("schema_version") == ACCESS_COMPLETION_SCHEMA, "Bad completion schema")
    _require(completion.get("access_id") == access_id, "Completion access drift")
    _require(
        completion.get("protocol_sha256") == expected_protocol_sha256,
        "Completion protocol drift",
    )
    _require(completion.get("access_sequence") == 1, "Bad completion access sequence")
    _require(completion.get("completed") is True, "Access completion is incomplete")
    _require(
        completion.get("internal_test_accessed") is True,
        "Completion does not record test access",
    )
    _require(
        completion.get("additional_test_access_authorized") is False,
        "Extra access allowed",
    )
    return access_id, manifest_sha256, completion


def _candidate_provenance(report: dict[str, Any], candidate: str) -> dict[str, Any]:
    field = "inputs" if candidate in _QWEN_CANDIDATES else "provenance"
    return _object(report.get(field), f"{candidate} provenance")


def _finite_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite or null",
    )
    return float(value)


def _public_row(value: Any) -> dict[str, Any]:
    row = _object(value, "final-table row")
    candidate = row.get("candidate_name")
    model = row.get("model")
    scope = row.get("scope")
    _require(isinstance(candidate, str) and candidate in _PRIMARY_CANDIDATES, "Bad candidate")
    _require(isinstance(model, str) and bool(model), "Missing model label")
    _require(isinstance(scope, str) and bool(scope), "Missing result scope")
    classification = _object(row.get("classification"), "classification metrics")
    localization = _object(row.get("localization"), "localization metrics")
    public = {
        "candidate_name": candidate,
        "model": model,
        "scope": scope,
        "classification": {
            field: _finite_or_none(classification.get(field), f"classification {field}")
            for field in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "localization": {
            field: _finite_or_none(localization.get(field), f"localization {field}")
            for field in ("mean_iou", "iou_at_0_5_rate")
        },
        "scientific_qc_exact_match": _finite_or_none(
            row.get("scientific_qc_exact_match"), "scientific QC exact match"
        ),
        "valid_json_rate": _finite_or_none(row.get("valid_json_rate"), "valid JSON rate"),
        "schema_valid_rate": _finite_or_none(
            row.get("schema_valid_rate"), "schema-valid rate"
        ),
        "coco": None,
    }
    coco = row.get("coco")
    if coco is not None:
        coco_object = _object(coco, "COCO metrics")
        public["coco"] = {
            field: _finite_or_none(coco_object.get(field), f"COCO {field}")
            for field in ("ap_50_95", "ap_50", "ap_75")
        }
    return public


def _verify_release_rows(rows: Any) -> list[dict[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == 8, "Unexpected final-table row count")
    typed_rows = [_public_row(row) for row in rows]
    expected = {
        *((candidate, language) for candidate in _QWEN_CANDIDATES for language in ("en", "zh-CN")),
        ("sequence_peak_net", "language-neutral"),
        ("chrompeakformer", "language-neutral"),
    }
    actual = {(row.get("candidate_name"), row.get("scope")) for row in typed_rows}
    _require(actual == expected, "Final-table candidate or scope drift")
    return typed_rows


def _verify_overall_rows(rows: Any) -> list[dict[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == 3, "Unexpected Qwen overall-row count")
    typed_rows = [_public_row(row) for row in rows]
    _require(
        {(row["candidate_name"], row["scope"]) for row in typed_rows}
        == {(candidate, "overall") for candidate in _QWEN_CANDIDATES},
        "Qwen overall candidate drift",
    )
    return typed_rows


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Sealed internal-test benchmark",
        "",
        "All candidates and operating thresholds were frozen before the sole test access.",
        "Qwen rows are reported separately by language; specialist rows are language-neutral.",
        "",
        "| Model | Scope | Bal. acc. | Macro-F1 | MCC | FPR | Mean IoU | "
        "IoU@0.5 | QC exact | JSON valid | Schema valid | AP | AP50 | AP75 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        classification = row["classification"]
        localization = row["localization"]
        coco = row["coco"] or {}
        lines.append(
            "| "
            + " | ".join(
                (
                    row["model"],
                    row["scope"],
                    _format_metric(classification["balanced_accuracy"]),
                    _format_metric(classification["macro_f1"]),
                    _format_metric(classification["mcc"]),
                    _format_metric(classification["false_positive_rate"]),
                    _format_metric(localization["mean_iou"]),
                    _format_metric(localization["iou_at_0_5_rate"]),
                    _format_metric(row["scientific_qc_exact_match"]),
                    _format_metric(row["valid_json_rate"]),
                    _format_metric(row["schema_valid_rate"]),
                    _format_metric(coco.get("ap_50_95")),
                    _format_metric(coco.get("ap_50")),
                    _format_metric(coco.get("ap_75")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "No combined cross-task score or post-test model selection is reported.",
            "Source-group bootstrap intervals remain in each hash-bound evaluation report.",
            "",
        )
    )
    return "\n".join(lines)


def _verify_registry_evaluations(
    registry: dict[str, Any],
    *,
    protocol_sha256: str,
    access_id: str,
) -> dict[str, dict[str, str]]:
    evaluations = _object(registry.get("evaluations"), "evidence registry evaluations")
    _require(set(evaluations) == set(_PRIMARY_CANDIDATES), "Registry candidate set drift")
    verified: dict[str, dict[str, str]] = {}
    for candidate in _PRIMARY_CANDIDATES:
        source = _object(evaluations[candidate], f"registry source {candidate}")
        root_value = source.get("root")
        _require(isinstance(root_value, str) and bool(root_value), "Evaluation root is missing")
        root = Path(root_value).resolve()
        filename, schema = _EVALUATION_FILES[candidate]
        report_path = root / filename
        manifest_path = root / "artifact_manifest.sha256"
        _require(
            Path(str(source.get("report_path"))).resolve() == report_path,
            f"{candidate} report path drift",
        )
        _require(
            Path(str(source.get("manifest_path"))).resolve() == manifest_path,
            f"{candidate} manifest path drift",
        )
        manifest_sha256, entries = _manifest_entries(root, label=f"{candidate} evaluation")
        report_sha256 = source.get("report_sha256")
        _require(
            isinstance(report_sha256, str)
            and entries.get(filename) == report_sha256
            and sha256_file(report_path) == report_sha256,
            f"{candidate} report is not manifest-bound",
        )
        _require(
            source.get("manifest_sha256") == manifest_sha256,
            f"{candidate} manifest hash drift",
        )
        report = _read_json(report_path, f"{candidate} evaluation report")
        _require(report.get("schema_version") == schema, f"{candidate} schema drift")
        _require(report.get("candidate_name") == candidate, f"{candidate} label drift")
        _require(report.get("final_benchmark_eligible") is True, f"{candidate} is ineligible")
        _require(report.get("internal_test_accessed") is True, f"{candidate} is not test-bound")
        provenance = _candidate_provenance(report, candidate)
        _require(provenance.get("protocol_sha256") == protocol_sha256, "Protocol drift")
        _require(provenance.get("access_id") == access_id, "Access event drift")
        verified[candidate] = {
            "report_sha256": report_sha256,
            "manifest_sha256": manifest_sha256,
        }
    return verified


def verify_final_benchmark_evidence(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    report_root: Path,
) -> tuple[FinalBenchmarkEvidenceVerification, dict[str, Any]]:
    """Verify sealed source evidence without reopening protected records."""

    protocol_root = protocol_root.resolve()
    ledger_dir = ledger_dir.resolve()
    report_root = report_root.resolve()
    protocol = _verify_frozen_protocol(protocol_root, expected_protocol_sha256)
    access_id, access_manifest_sha256, completion = _verify_completed_access(
        ledger_dir, expected_protocol_sha256
    )

    report_manifest_sha256, report_entries = _manifest_entries(
        report_root,
        label="final report",
        expected_files=_FINAL_REPORT_FILES,
    )
    _require(
        Path(str(completion.get("final_evidence_manifest_path"))).resolve()
        == report_root / "artifact_manifest.sha256",
        "Completed access evidence path drift",
    )
    _require(
        completion.get("final_evidence_manifest_sha256") == report_manifest_sha256,
        "Completed access evidence hash drift",
    )
    _require(completion.get("additional_test_access_authorized") is False, "Extra access allowed")

    report_path = report_root / "final_benchmark_report.json"
    registry_path = report_root / "final_evidence_registry.json"
    table_path = report_root / "final_benchmark_table.md"
    report = _read_json(report_path, "final benchmark report")
    registry = _read_json(registry_path, "final evidence registry")
    _require(report.get("schema_version") == FINAL_BENCHMARK_REPORT_SCHEMA, "Bad report schema")
    _require(
        registry.get("schema_version") == FINAL_BENCHMARK_REGISTRY_SCHEMA,
        "Bad registry schema",
    )
    for payload, label in ((report, "report"), (registry, "registry")):
        _require(payload.get("protocol_sha256") == expected_protocol_sha256, f"{label} drift")
        _require(payload.get("access_id") == access_id, f"{label} access drift")
        _require(tuple(payload.get("candidate_order") or ()) == _PRIMARY_CANDIDATES, "Order drift")
        _require(payload.get("internal_test_accessed") is True, f"{label} is not test-bound")
    _require(report.get("final_benchmark_eligible") is True, "Final benchmark is ineligible")
    _require(report.get("development_comparison_eligible") is False, "Report scope drift")
    contracts = _object(report.get("contracts"), "final report contracts")
    for field, expected in _REQUIRED_CONTRACTS.items():
        _require(contracts.get(field) is expected, f"Final report contract drift: {field}")
    artifacts = _object(report.get("artifacts"), "final report artifacts")
    main_table = _object(artifacts.get("main_table"), "main table")
    evidence_registry = _object(artifacts.get("evidence_registry"), "evidence registry")
    _require(main_table.get("path") == table_path.name, "Final table path drift")
    _require(evidence_registry.get("path") == registry_path.name, "Registry path drift")
    _require(
        main_table.get("sha256") == report_entries[table_path.name],
        "Final table descriptor drift",
    )
    _require(
        evidence_registry.get("sha256") == report_entries[registry_path.name],
        "Evidence registry descriptor drift",
    )
    rows = _verify_release_rows(report.get("language_separated_rows"))
    overall_rows = _verify_overall_rows(report.get("qwen_overall_rows"))
    table = table_path.read_text(encoding="utf-8")
    _require(table == _render_table(rows), "Final table does not match report rows")
    evaluation_hashes = _verify_registry_evaluations(
        registry,
        protocol_sha256=expected_protocol_sha256,
        access_id=access_id,
    )
    sealed_split = _object(protocol.get("sealed_split"), "sealed split")
    assets = sealed_split.get("expected_assets")
    groups = sealed_split.get("expected_source_groups")
    _require(isinstance(assets, int) and assets > 0, "Invalid internal-test asset count")
    _require(isinstance(groups, int) and groups > 0, "Invalid source-group count")

    verification = FinalBenchmarkEvidenceVerification(
        protocol_sha256=expected_protocol_sha256,
        access_id=access_id,
        access_completed=True,
        report_sha256=report_entries[report_path.name],
        report_manifest_sha256=report_manifest_sha256,
        candidates=len(_PRIMARY_CANDIDATES),
        rows=len(rows),
        internal_test_assets=assets,
        internal_test_source_groups=groups,
    )
    return verification, {
        "report": report,
        "public_rows": rows,
        "public_overall_rows": overall_rows,
        "table": table,
        "evaluation_hashes": evaluation_hashes,
        "protocol_manifest_sha256": sha256_file(protocol_root / "artifact_manifest.sha256"),
        "access_started_sha256": sha256_file(ledger_dir / "access_started.json"),
        "access_completed_sha256": sha256_file(ledger_dir / "access_completed.json"),
        "access_manifest_sha256": access_manifest_sha256,
        "table_sha256": report_entries[table_path.name],
        "registry_sha256": report_entries[registry_path.name],
    }


def _release_readme(summary: dict[str, Any], table: str) -> str:
    return "\n".join(
        (
            "# BioCoder multimodal benchmark v1",
            "",
            "This public-safe capsule records the completed, frozen LC-MS multimodal model",
            "benchmark. It contains aggregate metrics and cryptographic evidence identifiers only.",
            "It excludes",
            "raw chromatograms, ROI images, labels, per-record predictions, model weights, private",
            "detector source code, and machine-specific paths.",
            "",
            "## Evidence contract",
            "",
            f"- Frozen protocol SHA-256: `{summary['protocol_sha256']}`",
            f"- Final report SHA-256: `{summary['final_report_sha256']}`",
            f"- One-time access ID: `{summary['access']['access_id']}`",
            f"- Independent internal-test source groups: `{summary['dataset']['source_groups']}`",
            f"- Internal-test assets: `{summary['dataset']['assets']}`",
            "- Candidate or threshold selection after test access: `false`",
            "- Additional internal-test access authorized: `false`",
            "",
            "## Interpretation boundary",
            "",
            "The table is a frozen model benchmark, not a BioCoder agent promotion result. The",
            "specialist and Qwen rows have different task surfaces, so no combined cross-task",
            "score",
            "is reported. With only 11 independent source groups, grouped uncertainty in the bound",
            "evaluation reports matters more than the derived prompt count.",
            "",
            "## Results",
            "",
            *table.strip().splitlines()[1:],
            "",
        )
    )


def _write_deterministic_zip(release_root: Path, archive_path: Path) -> None:
    prefix = "biocoder-multimodal-v1"
    members = sorted((*_RELEASE_FILES, "artifact_manifest.sha256"))
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative in members:
            info = zipfile.ZipInfo(f"{prefix}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            # Stored entries avoid zlib-version variance and keep the small text capsule
            # byte-identical across supported platforms.
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (release_root / relative).read_bytes())


def build_final_benchmark_release(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    report_root: Path,
    output_dir: Path,
) -> FinalBenchmarkReleaseResult:
    """Build a deterministic, path-free public release directory and ZIP archive."""

    output_dir = output_dir.resolve()
    archive_path = output_dir.parent / f"{output_dir.name}.zip"
    _require(not output_dir.exists(), f"Release output already exists: {output_dir}")
    _require(not archive_path.exists(), f"Release archive already exists: {archive_path}")
    verification, source = verify_final_benchmark_evidence(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
        report_root=report_root,
    )
    report = source["report"]
    summary = {
        "schema_version": FINAL_BENCHMARK_RELEASE_SCHEMA,
        "release_name": "biocoder-multimodal-v1",
        "benchmark_scope": "frozen_multimodal_model_candidates",
        "protocol_sha256": verification.protocol_sha256,
        "final_report_sha256": verification.report_sha256,
        "final_report_manifest_sha256": verification.report_manifest_sha256,
        "access": {
            "access_id": verification.access_id,
            "access_sequence": 1,
            "completed": True,
            "additional_test_access_authorized": False,
        },
        "dataset": {
            "split": "internal_test",
            "assets": verification.internal_test_assets,
            "source_groups": verification.internal_test_source_groups,
        },
        "candidate_order": list(_PRIMARY_CANDIDATES),
        "language_separated_rows": source["public_rows"],
        "qwen_overall_rows": source["public_overall_rows"],
        "uncertainty": {
            field: _object(report.get("uncertainty"), "uncertainty").get(field)
            for field in ("bootstrap_iterations", "seed")
        },
        "contracts": {
            **_REQUIRED_CONTRACTS,
            "source_group_uncertainty_reported_in_bound_inputs": True,
        },
        "final_benchmark_eligible": True,
        "internal_test_accessed": True,
    }
    checksums = {
        "schema_version": FINAL_BENCHMARK_CHECKSUMS_SCHEMA,
        "protocol_sha256": verification.protocol_sha256,
        "protocol_manifest_sha256": source["protocol_manifest_sha256"],
        "access": {
            "started_sha256": source["access_started_sha256"],
            "completed_sha256": source["access_completed_sha256"],
            "manifest_sha256": source["access_manifest_sha256"],
        },
        "final_report": {
            "report_sha256": verification.report_sha256,
            "manifest_sha256": verification.report_manifest_sha256,
            "table_sha256": source["table_sha256"],
            "registry_sha256": source["registry_sha256"],
        },
        "evaluations": source["evaluation_hashes"],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
        ) as staging_name:
            staging = Path(staging_name)
            _write_json(staging / "benchmark_summary.json", summary)
            _write_json(staging / "evidence_checksums.json", checksums)
            (staging / "benchmark_table.md").write_text(source["table"], encoding="utf-8")
            (staging / "README.md").write_text(
                _release_readme(summary, source["table"]), encoding="utf-8"
            )
            manifest = staging / "artifact_manifest.sha256"
            manifest.write_text(
                "".join(
                    f"{sha256_file(staging / relative)}  {relative}\n"
                    for relative in sorted(_RELEASE_FILES)
                ),
                encoding="utf-8",
            )
            temporary_archive = output_dir.parent / f".{output_dir.name}.zip.staging"
            _require(not temporary_archive.exists(), "Temporary release archive already exists")
            _write_deterministic_zip(staging, temporary_archive)
            staging.replace(output_dir)
        temporary_archive.replace(archive_path)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
        raise

    verified = verify_public_release(release_root=output_dir, archive_path=archive_path)
    return FinalBenchmarkReleaseResult(
        output_dir=output_dir,
        archive_path=archive_path,
        archive_sha256=str(verified.archive_sha256),
        manifest_path=output_dir / "artifact_manifest.sha256",
        manifest_sha256=verified.manifest_sha256,
        summary_path=output_dir / "benchmark_summary.json",
        protocol_sha256=verified.protocol_sha256,
        report_sha256=verified.report_sha256,
    )


def verify_public_release(
    *, release_root: Path, archive_path: Path | None = None
) -> PublicReleaseVerification:
    """Verify a standalone public capsule without private source artifacts."""

    release_root = release_root.resolve()
    manifest_sha256, _ = _manifest_entries(
        release_root,
        label="public release",
        expected_files=_RELEASE_FILES,
    )
    summary = _read_json(release_root / "benchmark_summary.json", "release summary")
    checksums = _read_json(release_root / "evidence_checksums.json", "release checksums")
    _require(summary.get("schema_version") == FINAL_BENCHMARK_RELEASE_SCHEMA, "Bad release schema")
    _require(
        checksums.get("schema_version") == FINAL_BENCHMARK_CHECKSUMS_SCHEMA,
        "Bad release checksums schema",
    )
    _require(summary.get("protocol_sha256") == checksums.get("protocol_sha256"), "Protocol drift")
    report_sha256 = summary.get("final_report_sha256")
    _require(
        report_sha256
        == _object(checksums.get("final_report"), "report checksums").get("report_sha256"),
        "Final report hash drift",
    )
    _require(tuple(summary.get("candidate_order") or ()) == _PRIMARY_CANDIDATES, "Candidate drift")
    rows = _verify_release_rows(summary.get("language_separated_rows"))
    _verify_overall_rows(summary.get("qwen_overall_rows"))
    _require(
        (release_root / "benchmark_table.md").read_text(encoding="utf-8")
        == _render_table(rows),
        "Release table does not match summary rows",
    )
    _require(summary.get("final_benchmark_eligible") is True, "Release is ineligible")
    access = _object(summary.get("access"), "release access")
    _require(access.get("completed") is True, "Release access is incomplete")
    _require(access.get("additional_test_access_authorized") is False, "Extra access allowed")
    evaluations = _object(checksums.get("evaluations"), "release evaluation checksums")
    _require(set(evaluations) == set(_PRIMARY_CANDIDATES), "Evaluation checksum set drift")

    archive_sha256 = None
    if archive_path is not None:
        archive_path = archive_path.resolve()
        _require(archive_path.is_file(), f"Release archive not found: {archive_path}")
        expected_names = {
            f"biocoder-multimodal-v1/{relative}"
            for relative in (*_RELEASE_FILES, "artifact_manifest.sha256")
        }
        with zipfile.ZipFile(archive_path, "r") as archive:
            _require(set(archive.namelist()) == expected_names, "Release archive member drift")
            for name in expected_names:
                relative = name.split("/", 1)[1]
                _require(
                    archive.read(name) == (release_root / relative).read_bytes(),
                    f"Release archive content drift: {relative}",
                )
        archive_sha256 = sha256_file(archive_path)
    return PublicReleaseVerification(
        release_root=release_root,
        manifest_sha256=manifest_sha256,
        protocol_sha256=str(summary["protocol_sha256"]),
        report_sha256=str(report_sha256),
        candidates=len(_PRIMARY_CANDIDATES),
        rows=len(rows),
        archive_sha256=archive_sha256,
    )
