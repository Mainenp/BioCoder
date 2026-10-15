"""Assemble the immutable five-candidate sealed-test evidence dossier."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.baselines.final_sequence_evaluation import (
    FINAL_SEQUENCE_EVALUATION_SCHEMA,
)
from multimodal_science.chrompeakformer.final_detector_evaluation import (
    FINAL_DETECTOR_EVALUATION_SCHEMA,
)
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.final_benchmark_evaluation import (
    FINAL_QWEN_EVALUATION_SCHEMA,
)
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FinalBenchmarkRuntimeContext,
    load_final_benchmark_context,
)


FINAL_BENCHMARK_REPORT_SCHEMA = "chrompeak-final-benchmark-report-v1"
FINAL_BENCHMARK_REGISTRY_SCHEMA = "chrompeak-final-benchmark-registry-v1"
_QWEN_CANDIDATES = (
    "qwen3vl_zero_shot",
    "qwen3vl_image_lora",
    "qwen3vl_image_xic_fusion",
)
_PRIMARY_CANDIDATES = (*_QWEN_CANDIDATES, "sequence_peak_net", "chrompeakformer")
_MODEL_LABELS = {
    "qwen3vl_zero_shot": "Qwen3-VL zero-shot",
    "qwen3vl_image_lora": "Qwen3-VL image-only LoRA",
    "qwen3vl_image_xic_fusion": "Qwen3-VL image + aligned XIC",
    "sequence_peak_net": "SequencePeakNet",
    "chrompeakformer": "ChromPeakFormer",
}


@dataclass(frozen=True)
class FinalBenchmarkReportResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    table_path: Path
    registry_path: Path
    manifest_path: Path
    access_id: str
    candidates: int


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


def _manifest_entries(root: Path) -> tuple[str, dict[str, str]]:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Missing evaluation manifest: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.strip().split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed manifest line {line_number}: {manifest}")
        digest, relative = parts
        relative = relative.strip().replace("\\", "/")
        _require(len(digest) == 64 and relative not in entries, "Bad manifest entry")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Evaluation manifest path escapes its root") from error
        _require(path.is_file() and sha256_file(path) == digest, f"Artifact drift: {path}")
        entries[relative] = digest
    _require(bool(entries), "Evaluation manifest is empty")
    return sha256_file(manifest), entries


def _load_bound_report(
    *,
    root: Path,
    filename: str,
    expected_sha256: str,
    schema: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest_sha, entries = _manifest_entries(root)
    report_path = root / filename
    _require(
        report_path.is_file() and sha256_file(report_path) == expected_sha256,
        f"Evaluation report drift: {report_path}",
    )
    _require(entries.get(filename) == expected_sha256, "Evaluation report is not manifest-bound")
    report = _read_json(report_path, filename)
    _require(report.get("schema_version") == schema, f"Unexpected schema: {filename}")
    _require(report.get("internal_test_accessed") is True, f"Not test-bound: {filename}")
    _require(report.get("final_benchmark_eligible") is True, f"Ineligible: {filename}")
    _require(
        report.get("development_comparison_eligible") is False,
        f"Final report is incorrectly development-eligible: {filename}",
    )
    return report, {
        "root": str(root),
        "report_path": str(report_path),
        "report_sha256": expected_sha256,
        "manifest_path": str(root / "artifact_manifest.sha256"),
        "manifest_sha256": manifest_sha,
    }


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _structured_rates(tasks: dict[str, Any]) -> tuple[float, float]:
    instructions = 0
    valid = 0.0
    schema = 0.0
    for task, value in tasks.items():
        row = _object(value, f"task metrics {task}")
        count = row.get("instructions")
        _require(isinstance(count, int) and count > 0, "Bad instruction count")
        instructions += count
        valid += count * _number(row.get("valid_json_rate"), "valid JSON rate")
        schema += count * _number(row.get("schema_valid_rate"), "schema-valid rate")
    _require(instructions > 0, "No Qwen instructions")
    return valid / instructions, schema / instructions


def _qwen_row(report: dict[str, Any], candidate: str, scope: str) -> dict[str, Any]:
    tasks = (
        _object(report.get("metrics"), "Qwen metrics")
        if scope == "overall"
        else _object(
            _object(report.get("metrics_by_language"), "language metrics").get(scope),
            f"metrics {scope}",
        )
    )
    presence = _object(
        _object(tasks.get("peak_presence"), "presence task").get("classification"),
        "presence classification",
    )
    grounding = _object(
        _object(tasks.get("peak_grounding"), "grounding task").get("grounding"),
        "grounding metrics",
    )
    qc = _object(tasks.get("scientific_qc"), "scientific QC")
    valid_rate, schema_rate = _structured_rates(tasks)
    return {
        "candidate_name": candidate,
        "model": _MODEL_LABELS[candidate],
        "scope": scope,
        "classification": {
            name: _number(presence.get(name), f"{candidate} {scope} {name}")
            for name in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "localization": {
            "mean_iou": _number(
                grounding.get("mean_bbox_iou_all"), f"{candidate} mean IoU"
            ),
            "iou_at_0_5_rate": _number(
                grounding.get("iou_at_0_5_rate_all"), f"{candidate} IoU@0.5"
            ),
        },
        "scientific_qc_exact_match": _number(
            qc.get("exact_match_rate"), f"{candidate} QC exact match"
        ),
        "valid_json_rate": valid_rate,
        "schema_valid_rate": schema_rate,
        "coco": None,
    }


def _sequence_row(report: dict[str, Any]) -> dict[str, Any]:
    classification = _object(
        _object(report.get("classification"), "sequence classification").get(
            "validation_frozen_threshold"
        ),
        "sequence frozen-threshold metrics",
    )
    localization = _object(report.get("localization"), "sequence localization")
    return {
        "candidate_name": "sequence_peak_net",
        "model": _MODEL_LABELS["sequence_peak_net"],
        "scope": "language-neutral",
        "classification": {
            name: _number(classification.get(name), f"sequence {name}")
            for name in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "localization": {
            "mean_iou": _number(localization.get("mean_iou"), "sequence mean IoU"),
            "iou_at_0_5_rate": _number(
                localization.get("iou_at_0_5_rate"), "sequence IoU@0.5"
            ),
        },
        "scientific_qc_exact_match": None,
        "valid_json_rate": None,
        "schema_valid_rate": None,
        "coco": None,
    }


def _detector_row(report: dict[str, Any]) -> dict[str, Any]:
    classification = _object(
        _object(report.get("classification"), "detector classification").get(
            "validation_frozen_threshold"
        ),
        "detector frozen-threshold metrics",
    )
    localization = _object(
        _object(report.get("localization"), "detector localization").get(
            "validation_frozen_threshold"
        ),
        "detector frozen-threshold localization",
    )
    coco = _object(report.get("coco"), "detector COCO")
    return {
        "candidate_name": "chrompeakformer",
        "model": _MODEL_LABELS["chrompeakformer"],
        "scope": "language-neutral",
        "classification": {
            name: _number(classification.get(name), f"detector {name}")
            for name in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "localization": {
            "mean_iou": _number(localization.get("mean_best_iou"), "detector mean IoU"),
            "iou_at_0_5_rate": _number(
                localization.get("iou_at_0_5_rate"), "detector IoU@0.5"
            ),
        },
        "scientific_qc_exact_match": None,
        "valid_json_rate": None,
        "schema_valid_rate": None,
        "coco": {
            name: _number(coco.get(name), f"detector {name}")
            for name in ("ap_50_95", "ap_50", "ap_75")
        },
    }


def _verify_common(
    context: FinalBenchmarkRuntimeContext,
    report: dict[str, Any],
    candidate: str,
) -> None:
    _require(report.get("candidate_name") == candidate, f"Candidate label drift: {candidate}")
    provenance = (
        _object(report.get("inputs"), "Qwen final inputs")
        if candidate in _QWEN_CANDIDATES
        else _object(report.get("provenance"), "specialist provenance")
    )
    _require(provenance.get("protocol_sha256") == context.protocol_sha256, "Protocol drift")
    _require(provenance.get("access_id") == context.access_id, "Access event drift")
    frozen_uncertainty = _object(
        context.metrics_lock.get("uncertainty"), "frozen uncertainty policy"
    )
    evaluation = _object(report.get("evaluation"), f"{candidate} evaluation policy")
    for field in ("bootstrap_iterations", "seed"):
        _require(
            evaluation.get(field) == frozen_uncertainty.get(field),
            f"{candidate} uncertainty drift: {field}",
        )
    _require(
        evaluation.get("bootstrap_unit") == "complete_source_mzml_group",
        f"{candidate} bootstrap-unit drift",
    )


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _table(rows: list[dict[str, Any]]) -> str:
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
                    str(row["model"]),
                    str(row["scope"]),
                    _format(classification["balanced_accuracy"]),
                    _format(classification["macro_f1"]),
                    _format(classification["mcc"]),
                    _format(classification["false_positive_rate"]),
                    _format(localization["mean_iou"]),
                    _format(localization["iou_at_0_5_rate"]),
                    _format(row["scientific_qc_exact_match"]),
                    _format(row["valid_json_rate"]),
                    _format(row["schema_valid_rate"]),
                    _format(coco.get("ap_50_95")),
                    _format(coco.get("ap_50")),
                    _format(coco.get("ap_75")),
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


def build_final_benchmark_report(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    qwen_evaluation_roots: dict[str, Path],
    qwen_evaluation_report_sha256: dict[str, str],
    sequence_evaluation_root: Path,
    expected_sequence_report_sha256: str,
    detector_evaluation_root: Path,
    expected_detector_report_sha256: str,
    output_dir: Path,
) -> FinalBenchmarkReportResult:
    """Publish the descriptive final table without post-test candidate selection."""

    _require(set(qwen_evaluation_roots) == set(_QWEN_CANDIDATES), "Qwen candidate set drift")
    _require(
        set(qwen_evaluation_report_sha256) == set(_QWEN_CANDIDATES),
        "Qwen report-hash set drift",
    )
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Final benchmark report exists: {output_dir}")
    context = load_final_benchmark_context(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    frozen_primary = tuple(context.candidate_lock.get("primary_models") or ())
    _require(frozen_primary == _PRIMARY_CANDIDATES, "Frozen primary candidate set drift")

    reports: dict[str, dict[str, Any]] = {}
    registry: dict[str, dict[str, Any]] = {}
    for candidate in _QWEN_CANDIDATES:
        report, source = _load_bound_report(
            root=qwen_evaluation_roots[candidate],
            filename="qwen_evaluation_report.json",
            expected_sha256=qwen_evaluation_report_sha256[candidate],
            schema=FINAL_QWEN_EVALUATION_SCHEMA,
        )
        _verify_common(context, report, candidate)
        reports[candidate] = report
        registry[candidate] = source
    sequence, sequence_source = _load_bound_report(
        root=sequence_evaluation_root,
        filename="sequence_evaluation_report.json",
        expected_sha256=expected_sequence_report_sha256,
        schema=FINAL_SEQUENCE_EVALUATION_SCHEMA,
    )
    _verify_common(context, sequence, "sequence_peak_net")
    reports["sequence_peak_net"] = sequence
    registry["sequence_peak_net"] = sequence_source
    detector, detector_source = _load_bound_report(
        root=detector_evaluation_root,
        filename="detector_evaluation_report.json",
        expected_sha256=expected_detector_report_sha256,
        schema=FINAL_DETECTOR_EVALUATION_SCHEMA,
    )
    _verify_common(context, detector, "chrompeakformer")
    reports["chrompeakformer"] = detector
    registry["chrompeakformer"] = detector_source

    rows = [
        _qwen_row(reports[candidate], candidate, language)
        for candidate in _QWEN_CANDIDATES
        for language in ("en", "zh-CN")
    ]
    rows.extend((_sequence_row(sequence), _detector_row(detector)))
    overall_rows = [
        _qwen_row(reports[candidate], candidate, "overall")
        for candidate in _QWEN_CANDIDATES
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        registry_path = staging / "final_evidence_registry.json"
        _write_json(
            registry_path,
            {
                "schema_version": FINAL_BENCHMARK_REGISTRY_SCHEMA,
                "protocol_sha256": expected_protocol_sha256,
                "access_id": context.access_id,
                "benchmark_scope": "frozen_multimodal_model_candidates",
                "candidate_order": list(_PRIMARY_CANDIDATES),
                "evaluations": registry,
                "internal_test_accessed": True,
            },
        )
        table_path = staging / "final_benchmark_table.md"
        table_path.write_text(_table(rows), encoding="utf-8")
        report_path = staging / "final_benchmark_report.json"
        _write_json(
            report_path,
            {
                "schema_version": FINAL_BENCHMARK_REPORT_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": expected_protocol_sha256,
                "access_id": context.access_id,
                "benchmark_scope": "frozen_multimodal_model_candidates",
                "candidate_order": list(_PRIMARY_CANDIDATES),
                "language_separated_rows": rows,
                "qwen_overall_rows": overall_rows,
                "uncertainty": _object(
                    context.metrics_lock.get("uncertainty"), "uncertainty lock"
                ),
                "artifacts": {
                    "main_table": {
                        "path": table_path.name,
                        "sha256": sha256_file(table_path),
                    },
                    "evidence_registry": {
                        "path": registry_path.name,
                        "sha256": sha256_file(registry_path),
                    },
                },
                "contracts": {
                    "all_primary_candidates_present": True,
                    "same_one_time_access_event": True,
                    "candidate_selection_performed_after_test": False,
                    "threshold_selection_performed_after_test": False,
                    "combined_cross_task_score_reported": False,
                    "qwen_languages_reported_separately": True,
                    "source_group_uncertainty_reported_in_bound_inputs": True,
                    "additional_internal_test_access_authorized": False,
                    "biocoder_agent_promotion_claimed": False,
                },
                "development_comparison_eligible": False,
                "final_benchmark_eligible": True,
                "internal_test_accessed": True,
            },
        )
        manifest = staging / "artifact_manifest.sha256"
        manifest.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, table_path, registry_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    report = output_dir / "final_benchmark_report.json"
    return FinalBenchmarkReportResult(
        output_dir=output_dir,
        report_path=report,
        report_sha256=sha256_file(report),
        table_path=output_dir / "final_benchmark_table.md",
        registry_path=output_dir / "final_evidence_registry.json",
        manifest_path=output_dir / "artifact_manifest.sha256",
        access_id=context.access_id,
        candidates=len(_PRIMARY_CANDIDATES),
    )
