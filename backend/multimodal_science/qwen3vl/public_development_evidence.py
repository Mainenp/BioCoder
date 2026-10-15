"""Publish path-free development evidence from hash-bound experiment artifacts."""

from __future__ import annotations

import json
import math
import re
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.auxiliary_pretraining import (
    AUXILIARY_PRETRAINING_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.development_dossier import (
    REPORT_SCHEMA as DEVELOPMENT_DOSSIER_SCHEMA,
)
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.lora_training import LORA_TRAINING_REPORT_SCHEMA


PUBLIC_DEVELOPMENT_EVIDENCE_SCHEMA = "chrompeak-public-development-evidence-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SCOPES = ("overall", "en", "zh-CN")
_PUBLIC_DEVELOPMENT_FILES = {
    "public_development_evidence.json",
    "public_development_evidence.md",
}
_PUBLIC_DEVELOPMENT_ARCHIVE_PREFIX = "biocoder-multimodal-v1.1-development"
_REQUIRED_PUBLIC_CONTRACTS = {
    "three_training_seeds": [17, 29, 43],
    "sample_standard_deviation_ddof": 1,
    "token_ablation_training_seed": 17,
    "selected_checkpoint_interventions": True,
    "path_free_public_payload": True,
    "language_variants_are_paired_views": True,
    "sealed_internal_test_reopened": False,
    "post_seal_controls_complete": True,
    "internal_test_accessed": False,
}
_PRIMARY_METRICS = (
    ("peak_presence", "balanced_accuracy", "Presence balanced accuracy"),
    ("peak_presence", "macro_f1", "Presence Macro-F1"),
    ("peak_presence", "mcc", "Presence MCC"),
    ("peak_presence", "false_positive_rate", "Presence FPR"),
    ("peak_presence_metadata", "macro_f1", "Metadata Macro-F1"),
    ("peak_presence_metadata", "mcc", "Metadata MCC"),
    ("peak_grounding", "mean_bbox_iou_all", "Grounding mean IoU"),
    ("peak_grounding", "iou_at_0_5_rate_all", "Grounding IoU@0.5"),
    ("scientific_qc", "exact_match_rate", "QC exact match"),
)


@dataclass(frozen=True)
class PublicDevelopmentEvidenceResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PublicDevelopmentEvidenceVerification:
    evidence_root: Path
    report_sha256: str
    markdown_sha256: str
    manifest_sha256: str
    archive_path: Path | None
    archive_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_root": str(self.evidence_root),
            "report_sha256": self.report_sha256,
            "markdown_sha256": self.markdown_sha256,
            "manifest_sha256": self.manifest_sha256,
            "archive_path": (
                None if self.archive_path is None else str(self.archive_path)
            ),
            "archive_sha256": self.archive_sha256,
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _safe_artifact(root: Path, relative: str) -> Path:
    _require(bool(relative) and "\\" not in relative, "Unsafe manifest path")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute()
        and all(part not in {"", ".", ".."} for part in posix.parts),
        "Unsafe manifest path",
    )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Manifest artifact escapes its root") from error
    _require(path.is_file(), f"Missing manifest artifact: {relative}")
    return path


def _verify_root(root: Path, report_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    root = root.resolve()
    _require(root.is_dir(), f"Evidence root not found: {root}")
    manifest_path = root / "artifact_manifest.sha256"
    _require(manifest_path.is_file(), f"Artifact manifest not found: {manifest_path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split("  ", maxsplit=1)
        _require(len(fields) == 2, f"Malformed manifest line {line_number}")
        digest, relative = fields
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad digest on manifest line {line_number}")
        _require(relative not in entries, f"Duplicate manifest artifact: {relative}")
        artifact = _safe_artifact(root, relative)
        _require(sha256_file(artifact) == digest, f"Artifact hash mismatch: {relative}")
        entries[relative] = digest
    _require(report_name in entries, f"Manifest does not bind {report_name}")
    report_path = root / report_name
    _require(sha256_file(report_path) == entries[report_name], "Report hash mismatch")
    return _read_json(report_path, report_name), entries


def _development_only(report: dict[str, Any], label: str) -> None:
    _require(report.get("internal_test_accessed") is False, f"{label} accessed internal test")
    _require(report.get("final_benchmark_eligible") is False, f"{label} claims final status")


def _metric(tasks: dict[str, Any], task: str, metric: str, label: str) -> float:
    task_metrics = _object(tasks.get(task), f"{label}/{task}")
    value = task_metrics.get(metric)
    if value is None:
        section_by_task = {
            "peak_presence": "classification",
            "peak_presence_metadata": "classification",
            "peak_grounding": "grounding",
        }
        section = section_by_task.get(task)
        _require(section is not None, f"{label}/{task}/{metric} is missing")
        value = _object(
            task_metrics.get(section), f"{label}/{task}/{section}"
        ).get(metric)
    return _number(value, f"{label}/{task}/{metric}")


def _statistics(
    dossier: dict[str, Any], scope: str, task: str, metric: str
) -> dict[str, Any]:
    payload = _object(
        _object(
            _object(dossier.get("fusion_reproducibility"), "fusion reproducibility").get(
                scope
            ),
            f"fusion reproducibility/{scope}",
        ).get(task),
        f"fusion reproducibility/{scope}/{task}",
    ).get(metric)
    values = _object(payload, f"fusion reproducibility/{scope}/{task}/{metric}")
    by_seed = _object(values.get("values_by_seed"), "values by seed")
    _require(set(by_seed) == {"17", "29", "43"}, "Three-seed evidence is incomplete")
    return {
        "mean": _number(values.get("mean"), "three-seed mean"),
        "sample_standard_deviation": _number(
            values.get("sample_standard_deviation"), "three-seed sample SD"
        ),
        "values_by_seed": {
            seed: _number(by_seed[seed], f"seed {seed} metric")
            for seed in ("17", "29", "43")
        },
    }


def _training_summary(
    root: Path,
    *,
    report_name: str,
    schema: str,
    label: str,
) -> dict[str, Any]:
    report, entries = _verify_root(root, report_name)
    _require(report.get("schema_version") == schema, f"Unexpected {label} schema")
    _development_only(report, label)
    _require(report.get("development_training_complete") is True, f"{label} is incomplete")
    training = _object(report.get("training"), f"{label} training")
    _require(training.get("max_steps") is None, f"{label} was step-capped")
    records = training.get("training_records")
    updates = training.get("optimizer_updates")
    _require(isinstance(records, int) and records > 0, f"Bad {label} record count")
    _require(isinstance(updates, int) and updates > 0, f"Bad {label} update count")
    wall_time = _number(report.get("wall_time_seconds"), f"{label} wall time")
    _require(wall_time > 0.0, f"Bad {label} wall time")
    return {
        "report_sha256": entries[report_name],
        "manifest_sha256": sha256_file(root.resolve() / "artifact_manifest.sha256"),
        "wall_time_seconds": wall_time,
        "training_records": records,
        "optimizer_updates": updates,
        "effective_batch_size": training.get("effective_batch_size"),
        "resumed_from": training.get("resumed_from"),
        "complete_uncapped_training": True,
    }


def _evaluation_summary(root: Path, label: str) -> dict[str, Any]:
    report, entries = _verify_root(root, "qwen_evaluation_report.json")
    _require(
        report.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        f"Unexpected {label} evaluation schema",
    )
    _development_only(report, label)
    _require(report.get("development_comparison_eligible") is True, f"{label} is ineligible")
    metrics = _object(report.get("metrics"), f"{label} overall metrics")
    return {
        "report_sha256": entries["qwen_evaluation_report.json"],
        "manifest_sha256": sha256_file(root.resolve() / "artifact_manifest.sha256"),
        "metrics": {
            display: _metric(metrics, task, metric, label)
            for task, metric, display in _PRIMARY_METRICS
        },
    }


def _post_seal_control_summary(root: Path, label: str) -> dict[str, Any]:
    """Extract the fixed bilingual metric surface for a completed v1.1 control."""

    report, entries = _verify_root(root, "qwen_evaluation_report.json")
    _require(
        report.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        f"Unexpected {label} evaluation schema",
    )
    _development_only(report, label)
    _require(report.get("development_comparison_eligible") is True, f"{label} is ineligible")
    counts = _object(report.get("counts"), f"{label} counts")
    _require(counts.get("predictions") == 13708, f"{label} lacks complete prompt coverage")
    by_language = _object(report.get("metrics_by_language"), f"{label} language metrics")
    _require(set(by_language) == {"en", "zh-CN"}, f"{label} bilingual metrics are incomplete")
    scopes = {
        "overall": _object(report.get("metrics"), f"{label} overall metrics"),
        "en": _object(by_language.get("en"), f"{label} English metrics"),
        "zh-CN": _object(by_language.get("zh-CN"), f"{label} Chinese metrics"),
    }
    return {
        "report_sha256": entries["qwen_evaluation_report.json"],
        "manifest_sha256": sha256_file(root.resolve() / "artifact_manifest.sha256"),
        "prediction_records": 13708,
        "metrics_by_scope": {
            scope: {
                display: _metric(tasks, task, metric, f"{label}/{scope}")
                for task, metric, display in _PRIMARY_METRICS
            }
            for scope, tasks in scopes.items()
        },
    }


def _hms(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BioCoder multimodal development evidence",
        "",
        "This is leakage-safe development evidence. It is separate from the sealed v1 "
        "internal-test benchmark and did not reopen that test set.",
        "",
        "## Four-token fusion reproducibility across three training seeds",
        "",
        "Values are mean ± sample standard deviation (`ddof=1`) across training seeds "
        "17, 29, and 43.",
        "",
        "| Scope | Metric | Mean ± SD | Seed 17 | Seed 29 | Seed 43 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in _SCOPES:
        for _, _, label in _PRIMARY_METRICS:
            values = report["three_seed_statistics"][scope][label]
            by_seed = values["values_by_seed"]
            lines.append(
                f"| {scope} | {label} | {values['mean']:.4f} ± "
                f"{values['sample_standard_deviation']:.4f} | "
                f"{by_seed['17']:.4f} | {by_seed['29']:.4f} | {by_seed['43']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Sensor-token ablation at training seed 17",
            "",
            "| Tokens | Presence Macro-F1 | Presence MCC | Metadata Macro-F1 | "
            "Mean IoU | IoU@0.5 | QC exact | Final gate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for token in ("1", "4", "8"):
        row = report["sensor_token_ablation"][token]
        lines.append(
            f"| {token} | {row['presence_macro_f1']:.4f} | {row['presence_mcc']:.4f} | "
            f"{row['metadata_macro_f1']:.4f} | {row['grounding_mean_iou']:.4f} | "
            f"{row['grounding_iou_at_0_5']:.4f} | {row['qc_exact_match']:.4f} | "
            f"{row['final_gate_probability']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## XIC intervention metrics on the selected seed-17 checkpoint",
            "",
            "| Intervention | Presence Macro-F1 | Presence MCC | Presence FPR | "
            "Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for intervention in ("aligned", "shuffled", "zero", "availability-off"):
        row = report["xic_interventions"][intervention]
        lines.append(
            f"| {intervention} | {row['presence_macro_f1']:.4f} | "
            f"{row['presence_mcc']:.4f} | {row['presence_fpr']:.4f} | "
            f"{row['metadata_macro_f1']:.4f} | {row['grounding_mean_iou']:.4f} | "
            f"{row['grounding_iou_at_0_5']:.4f} | {row['qc_exact_match']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full training wall-clock measurements",
            "",
            "| Run | Records | Optimizer updates | Wall seconds | HH:MM:SS | Report SHA-256 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for key, label in (("image_lora", "Image-only LoRA"), ("image_xic", "Image + XIC fusion")):
        row = report["training_wall_clock"][key]
        lines.append(
            f"| {label} | {row['training_records']} | {row['optimizer_updates']} | "
            f"{row['wall_time_seconds']:.3f} | {_hms(row['wall_time_seconds'])} | "
            f"`{row['report_sha256']}` |"
        )

    auxiliary = report.get("auxiliary_projector_ablation")
    if auxiliary is not None:
        lines.extend(
            [
                "",
                "## Auxiliary-projector initialization",
                "",
                "The unlabeled morphology pretraining run completed, but its controlled "
                "downstream comparison did not improve the primary endpoints.",
                "",
                "| Initialization | Presence Macro-F1 | Metadata Macro-F1 | Mean IoU | "
                "IoU@0.5 | QC exact |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, label in (("random", "Random projector"), ("auxiliary", "Auxiliary pretrained")):
            metrics = auxiliary["downstream_evaluations"][key]["metrics"]
            lines.append(
                f"| {label} | {metrics['Presence Macro-F1']:.4f} | "
                f"{metrics['Metadata Macro-F1']:.4f} | {metrics['Grounding mean IoU']:.4f} | "
                f"{metrics['Grounding IoU@0.5']:.4f} | {metrics['QC exact match']:.4f} |"
            )

    controls = report.get("post_seal_controls")
    if controls is not None:
        lines.extend(
            [
                "",
                "## Post-seal v1.1 controls",
                "",
                "These rows use the leakage-safe validation split and never revise the sealed "
                "v1 benchmark.",
                "",
                "| Model | Scope | Presence balanced accuracy | Presence Macro-F1 | "
                "Presence MCC | Presence FPR | Metadata Macro-F1 | Mean IoU | "
                "IoU@0.5 | QC exact |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, label in (
            ("xic_only_qwen", "XIC-only Qwen"),
            ("image_lora_sequence_prompt", "Image LoRA + SequencePeakNet prompt"),
        ):
            for scope in _SCOPES:
                metrics = controls[key]["metrics_by_scope"][scope]
                lines.append(
                    f"| {label} | {scope} | {metrics['Presence balanced accuracy']:.4f} | "
                    f"{metrics['Presence Macro-F1']:.4f} | {metrics['Presence MCC']:.4f} | "
                    f"{metrics['Presence FPR']:.4f} | {metrics['Metadata Macro-F1']:.4f} | "
                    f"{metrics['Grounding mean IoU']:.4f} | "
                    f"{metrics['Grounding IoU@0.5']:.4f} | "
                    f"{metrics['QC exact match']:.4f} |"
                )

    lines.extend(
        [
            "",
            "All source reports and manifests are SHA-256 bound in the companion JSON. "
            "Language variants are paired views, not independent scientific samples.",
            "",
        ]
    )
    return "\n".join(lines)


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _verify_path_free_payload(report: dict[str, Any]) -> None:
    for value in _iter_strings(report):
        normalized = value.replace("\\", "/")
        looks_like_drive_path = bool(re.match(r"^[A-Za-z]:/", normalized))
        looks_like_machine_root = normalized.startswith(
            ("/home/", "/tmp/", "/Users/", "/var/", "/mnt/")
        )
        _require(
            not looks_like_drive_path and not looks_like_machine_root,
            "Public development evidence contains a machine-specific path",
        )


def verify_public_development_evidence(
    *,
    evidence_root: Path,
    expected_report_sha256: str,
    archive_path: Path | None = None,
) -> PublicDevelopmentEvidenceVerification:
    """Verify the standalone v1.1 evidence without opening any source dataset."""

    evidence_root = evidence_root.resolve()
    _require(
        bool(_HEX_64.fullmatch(expected_report_sha256)),
        "Expected development report SHA-256 must be lowercase hexadecimal",
    )
    report, entries = _verify_root(
        evidence_root, "public_development_evidence.json"
    )
    _require(
        set(entries) == _PUBLIC_DEVELOPMENT_FILES,
        "Public development evidence artifact set drift",
    )
    report_sha256 = entries["public_development_evidence.json"]
    _require(report_sha256 == expected_report_sha256, "Development report SHA-256 drift")
    _require(
        report.get("schema_version") == PUBLIC_DEVELOPMENT_EVIDENCE_SCHEMA,
        "Unexpected public development evidence schema",
    )
    _require(
        report.get("evaluation_scope")
        == "leakage_safe_validation_public_evidence",
        "Unexpected public development evidence scope",
    )
    _development_only(report, "public development evidence")
    _require(
        report.get("development_comparison_eligible") is True,
        "Public development evidence is ineligible",
    )
    contracts = _object(report.get("contracts"), "public development contracts")
    for field, expected in _REQUIRED_PUBLIC_CONTRACTS.items():
        _require(contracts.get(field) == expected, f"Development contract drift: {field}")
    _require(
        set(_object(report.get("three_seed_statistics"), "three-seed statistics"))
        == set(_SCOPES),
        "Three-seed scope drift",
    )
    _require(
        set(_object(report.get("sensor_token_ablation"), "sensor-token ablation"))
        == {"1", "4", "8"},
        "Sensor-token row drift",
    )
    _require(
        set(_object(report.get("xic_interventions"), "XIC interventions"))
        == {"aligned", "shuffled", "zero", "availability-off"},
        "XIC intervention row drift",
    )
    _object(report.get("auxiliary_projector_ablation"), "auxiliary-projector ablation")
    _require(
        set(_object(report.get("post_seal_controls"), "post-seal controls"))
        == {"xic_only_qwen", "image_lora_sequence_prompt"},
        "Post-seal control row drift",
    )
    sources = _object(report.get("sources"), "public development sources")
    for label, digest in sources.items():
        _require(
            isinstance(digest, str) and bool(_HEX_64.fullmatch(digest)),
            f"Invalid source digest: {label}",
        )
    _verify_path_free_payload(report)

    markdown_path = evidence_root / "public_development_evidence.md"
    _require(
        markdown_path.read_text(encoding="utf-8") == _markdown(report),
        "Public development Markdown does not match the report",
    )
    markdown_sha256 = entries["public_development_evidence.md"]
    manifest_path = evidence_root / "artifact_manifest.sha256"
    manifest_sha256 = sha256_file(manifest_path)

    resolved_archive = None
    archive_sha256 = None
    if archive_path is not None:
        resolved_archive = archive_path.resolve()
        _require(
            resolved_archive.is_file(),
            f"Public development archive not found: {resolved_archive}",
        )
        relatives = sorted((*_PUBLIC_DEVELOPMENT_FILES, "artifact_manifest.sha256"))
        expected_names = {
            f"{_PUBLIC_DEVELOPMENT_ARCHIVE_PREFIX}/{relative}"
            for relative in relatives
        }
        with zipfile.ZipFile(resolved_archive, "r") as archive:
            names = archive.namelist()
            _require(
                len(names) == len(expected_names) and set(names) == expected_names,
                "Public development archive member drift",
            )
            for name in expected_names:
                relative = name.split("/", 1)[1]
                _require(
                    archive.read(name) == (evidence_root / relative).read_bytes(),
                    f"Public development archive content drift: {relative}",
                )
        archive_sha256 = sha256_file(resolved_archive)

    return PublicDevelopmentEvidenceVerification(
        evidence_root=evidence_root,
        report_sha256=report_sha256,
        markdown_sha256=markdown_sha256,
        manifest_sha256=manifest_sha256,
        archive_path=resolved_archive,
        archive_sha256=archive_sha256,
    )


def build_public_development_evidence_archive(
    *,
    evidence_root: Path,
    expected_report_sha256: str,
    archive_path: Path,
) -> PublicDevelopmentEvidenceVerification:
    """Create a deterministic ZIP around an already verified v1.1 evidence root."""

    evidence_root = evidence_root.resolve()
    archive_path = archive_path.resolve()
    _require(not archive_path.exists(), f"Development archive exists: {archive_path}")
    verify_public_development_evidence(
        evidence_root=evidence_root,
        expected_report_sha256=expected_report_sha256,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive_path.parent,
        prefix=f".{archive_path.name}-",
        suffix=".staging",
        delete=False,
    ) as handle:
        staging_path = Path(handle.name)
    try:
        relatives = sorted((*_PUBLIC_DEVELOPMENT_FILES, "artifact_manifest.sha256"))
        with zipfile.ZipFile(staging_path, "w") as archive:
            for relative in relatives:
                info = zipfile.ZipInfo(
                    f"{_PUBLIC_DEVELOPMENT_ARCHIVE_PREFIX}/{relative}",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (evidence_root / relative).read_bytes())
        staging_path.replace(archive_path)
        verified = verify_public_development_evidence(
            evidence_root=evidence_root,
            expected_report_sha256=expected_report_sha256,
            archive_path=archive_path,
        )
    except BaseException:
        staging_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
        raise
    return verified


def build_public_development_evidence(
    *,
    development_dossier_root: Path,
    lora_training_root: Path,
    fusion_training_root: Path,
    output_dir: Path,
    auxiliary_pretraining_root: Path | None = None,
    random_projector_evaluation_root: Path | None = None,
    auxiliary_projector_evaluation_root: Path | None = None,
    xic_only_evaluation_root: Path | None = None,
    sequence_prompt_evaluation_root: Path | None = None,
) -> PublicDevelopmentEvidenceResult:
    """Build a path-free public report from already completed development evidence."""

    dossier, dossier_entries = _verify_root(
        development_dossier_root, "development_dossier.json"
    )
    _require(
        dossier.get("schema_version") == DEVELOPMENT_DOSSIER_SCHEMA,
        "Unexpected development dossier schema",
    )
    _development_only(dossier, "development dossier")
    readiness = _object(dossier.get("pre_internal_test_readiness"), "dossier readiness")
    _require(readiness.get("ready") is True, "Development dossier is not complete")

    three_seed: dict[str, Any] = {}
    for scope in _SCOPES:
        three_seed[scope] = {
            display: _statistics(dossier, scope, task, metric)
            for task, metric, display in _PRIMARY_METRICS
        }

    ablation = _object(dossier.get("sensor_token_ablation"), "sensor-token ablation")
    runs = _object(ablation.get("runs"), "sensor-token runs")
    _require(set(runs) == {"1", "4", "8"}, "Sensor-token ablation is incomplete")
    token_rows: dict[str, Any] = {}
    for token in ("1", "4", "8"):
        run = _object(runs[token], f"sensor-token row {token}")
        metrics = _object(
            _object(run.get("metrics"), f"sensor-token metrics {token}").get("overall"),
            f"sensor-token overall metrics {token}",
        )
        gate = _object(run.get("sensor_gate"), f"sensor-token gate {token}")
        token_rows[token] = {
            "presence_macro_f1": _metric(metrics, "peak_presence", "macro_f1", token),
            "presence_mcc": _metric(metrics, "peak_presence", "mcc", token),
            "metadata_macro_f1": _metric(
                metrics, "peak_presence_metadata", "macro_f1", token
            ),
            "grounding_mean_iou": _metric(
                metrics, "peak_grounding", "mean_bbox_iou_all", token
            ),
            "grounding_iou_at_0_5": _metric(
                metrics, "peak_grounding", "iou_at_0_5_rate_all", token
            ),
            "qc_exact_match": _metric(metrics, "scientific_qc", "exact_match_rate", token),
            "final_gate_probability": _number(
                gate.get("final_probability"), f"sensor-token gate {token}"
            ),
        }

    intervention = _object(
        dossier.get("causal_xic_intervention_evidence"), "XIC intervention evidence"
    )
    _require(
        intervention.get("selected_checkpoint_match") is True,
        "Interventions are not from the selected checkpoint",
    )
    observed = _object(intervention.get("observed_metrics"), "observed interventions")
    overall = _object(observed.get("overall"), "overall interventions")
    _require(
        set(overall) == {"aligned", "shuffled", "zero", "availability-off"},
        "XIC interventions are incomplete",
    )
    intervention_rows: dict[str, Any] = {}
    for name in ("aligned", "shuffled", "zero", "availability-off"):
        tasks = _object(overall[name], f"intervention {name}")
        intervention_rows[name] = {
            "presence_macro_f1": _metric(tasks, "peak_presence", "macro_f1", name),
            "presence_mcc": _metric(tasks, "peak_presence", "mcc", name),
            "presence_fpr": _metric(
                tasks, "peak_presence", "false_positive_rate", name
            ),
            "metadata_macro_f1": _metric(
                tasks, "peak_presence_metadata", "macro_f1", name
            ),
            "grounding_mean_iou": _metric(
                tasks, "peak_grounding", "mean_bbox_iou_all", name
            ),
            "grounding_iou_at_0_5": _metric(
                tasks, "peak_grounding", "iou_at_0_5_rate_all", name
            ),
            "qc_exact_match": _metric(
                tasks, "scientific_qc", "exact_match_rate", name
            ),
        }

    training = {
        "image_lora": _training_summary(
            lora_training_root,
            report_name="lora_training_report.json",
            schema=LORA_TRAINING_REPORT_SCHEMA,
            label="image-only LoRA training",
        ),
        "image_xic": _training_summary(
            fusion_training_root,
            report_name="fusion_training_report.json",
            schema=FUSION_TRAINING_REPORT_SCHEMA,
            label="image-XIC fusion training",
        ),
    }

    optional = (
        auxiliary_pretraining_root,
        random_projector_evaluation_root,
        auxiliary_projector_evaluation_root,
    )
    _require(
        all(value is None for value in optional) or all(value is not None for value in optional),
        "Auxiliary evidence inputs must be supplied together",
    )
    auxiliary_payload = None
    if auxiliary_pretraining_root is not None:
        auxiliary_report, auxiliary_entries = _verify_root(
            auxiliary_pretraining_root, "auxiliary_pretraining_report.json"
        )
        _require(
            auxiliary_report.get("schema_version") == AUXILIARY_PRETRAINING_REPORT_SCHEMA,
            "Unexpected auxiliary pretraining schema",
        )
        _development_only(auxiliary_report, "auxiliary pretraining")
        _require(
            auxiliary_report.get("development_training_complete") is True,
            "Auxiliary pretraining is incomplete",
        )
        random_summary = _evaluation_summary(
            random_projector_evaluation_root,  # type: ignore[arg-type]
            "random-projector evaluation",
        )
        pretrained_summary = _evaluation_summary(
            auxiliary_projector_evaluation_root,
            "auxiliary-projector evaluation",  # type: ignore[arg-type]
        )
        auxiliary_payload = {
            "pretraining_report_sha256": auxiliary_entries[
                "auxiliary_pretraining_report.json"
            ],
            "pretraining_manifest_sha256": sha256_file(
                auxiliary_pretraining_root.resolve() / "artifact_manifest.sha256"
            ),
            "pretraining_wall_time_seconds": _number(
                auxiliary_report.get("wall_time_seconds"), "auxiliary wall time"
            ),
            "downstream_evaluations": {
                "random": random_summary,
                "auxiliary": pretrained_summary,
            },
            "positive_downstream_gain_claimed": False,
        }

    control_inputs = (xic_only_evaluation_root, sequence_prompt_evaluation_root)
    _require(
        all(value is None for value in control_inputs)
        or all(value is not None for value in control_inputs),
        "Both post-seal control evaluations must be supplied together",
    )
    post_seal_controls = None
    if xic_only_evaluation_root is not None:
        post_seal_controls = {
            "xic_only_qwen": _post_seal_control_summary(
                xic_only_evaluation_root,  # type: ignore[arg-type]
                "XIC-only Qwen evaluation",
            ),
            "image_lora_sequence_prompt": _post_seal_control_summary(
                sequence_prompt_evaluation_root,  # type: ignore[arg-type]
                "image-LoRA sequence-prompt evaluation",
            ),
        }

    report = {
        "schema_version": PUBLIC_DEVELOPMENT_EVIDENCE_SCHEMA,
        "evaluation_scope": "leakage_safe_validation_public_evidence",
        "three_seed_statistics": three_seed,
        "sensor_token_ablation": token_rows,
        "xic_interventions": intervention_rows,
        "training_wall_clock": training,
        "auxiliary_projector_ablation": auxiliary_payload,
        "post_seal_controls": post_seal_controls,
        "sources": {
            "development_dossier_report_sha256": dossier_entries[
                "development_dossier.json"
            ],
            "development_dossier_manifest_sha256": sha256_file(
                development_dossier_root.resolve() / "artifact_manifest.sha256"
            ),
        },
        "contracts": {
            "three_training_seeds": [17, 29, 43],
            "sample_standard_deviation_ddof": 1,
            "token_ablation_training_seed": 17,
            "selected_checkpoint_interventions": True,
            "path_free_public_payload": True,
            "language_variants_are_paired_views": True,
            "sealed_internal_test_reopened": False,
            "post_seal_controls_complete": post_seal_controls is not None,
            "internal_test_accessed": False,
        },
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }

    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Public evidence output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        report_path = staging / "public_development_evidence.json"
        markdown_path = staging / "public_development_evidence.md"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, markdown_path)
            ),
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_dir)

    final_report = output_dir / "public_development_evidence.json"
    return PublicDevelopmentEvidenceResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        markdown_path=output_dir / "public_development_evidence.md",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
