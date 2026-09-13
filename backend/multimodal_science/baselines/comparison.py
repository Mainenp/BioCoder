"""Build a hash-bound development comparison across the specialist and XIC baselines."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file


REPORT_SCHEMA = "chrompeak-development-ablation-v1"
_CLASSIFICATION_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "macro_f1",
    "mcc",
    "auroc",
    "auprc",
    "false_positive_rate",
)


@dataclass(frozen=True)
class DevelopmentComparisonResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    manifest_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Report must contain a JSON object: {path}")
    return payload


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _validate_development_scope(report: dict[str, Any], label: str) -> None:
    _require(
        report.get("development_comparison_eligible") is True,
        f"{label} is not eligible for a development comparison",
    )
    _require(
        report.get("final_benchmark_eligible") is False,
        f"{label} must not claim final-benchmark eligibility",
    )
    _require(
        report.get("internal_test_accessed") is False,
        f"{label} accessed the sealed internal test",
    )


def _classification_row(metrics: dict[str, Any]) -> dict[str, float]:
    row = {}
    for field in _CLASSIFICATION_FIELDS:
        value = metrics.get(field)
        _require(isinstance(value, (int, float)), f"Missing classification metric: {field}")
        row[field] = float(value)
    threshold = metrics.get("threshold")
    _require(isinstance(threshold, (int, float)), "Missing classification threshold")
    row["threshold"] = float(threshold)
    return row


def _validate_sequence_verification(
    verification: dict[str, Any],
    *,
    report_sha256: str,
    dataset_report_sha256: str,
    asset_index_sha256: str,
    label: str,
) -> None:
    _require(
        verification.get("schema_version") == "chrompeak-sequence-run-verification-v1",
        f"Unexpected {label} verification schema",
    )
    _require(verification.get("quality_gate_passed") is True, f"{label} quality gate failed")
    _require(
        verification.get("sealed_internal_test_confirmed") is True,
        f"{label} did not confirm the sealed internal test",
    )
    source_report = _object(verification.get("source_report"), f"{label} source report")
    source_dataset = _object(verification.get("source_dataset"), f"{label} source dataset")
    _require(
        source_report.get("sha256") == report_sha256,
        f"{label} verification does not bind the supplied scientific report",
    )
    _require(
        source_dataset.get("report_sha256") == dataset_report_sha256,
        f"{label} verification Dataset hash mismatch",
    )
    _require(
        source_dataset.get("asset_index_sha256") == asset_index_sha256,
        f"{label} verification asset-index hash mismatch",
    )


def _sequence_rows(
    report: dict[str, Any], expected_modality: str
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    _require(
        report.get("schema_version") == "chrompeak-sequence-baseline-report-v1",
        f"Unexpected {expected_modality} report schema",
    )
    _validate_development_scope(report, expected_modality)
    _require(
        report.get("evaluation_scope") == "train_validation_baseline",
        f"{expected_modality} is not a full train/validation run",
    )
    config = _object(report.get("config"), f"{expected_modality} config")
    _require(
        config.get("modality") == expected_modality,
        f"Expected {expected_modality} report",
    )
    _require(config.get("smoke_test") is False, f"{expected_modality} is a smoke run")
    validation = _object(report.get("validation"), f"{expected_modality} validation")
    fixed = _classification_row(
        _object(validation.get("fixed_threshold_0_5"), f"{expected_modality} fixed metrics")
    )
    selected = _classification_row(
        _object(validation.get("selected_threshold"), f"{expected_modality} selected metrics")
    )
    boundaries = _object(
        validation.get("positive_boundary_metrics"),
        f"{expected_modality} boundary metrics",
    )
    localization = {}
    for field in ("mean_interval_iou", "boundary_mae_seconds", "valid_prediction_rate"):
        value = boundaries.get(field)
        _require(isinstance(value, (int, float)), f"Missing boundary metric: {field}")
        localization[field] = float(value)
    return fixed, selected, localization


def _metric_deltas(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    return {
        field: candidate[field] - baseline[field]
        for field in _CLASSIFICATION_FIELDS
    }


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _markdown_table(report: dict[str, Any]) -> str:
    fixed = report["fixed_threshold_0_5"]
    localization = report["localization"]
    rows = []
    for key, label in (
        ("chrompeakformer", "ChromPeakFormer image detector"),
        ("sequence", "SequencePeakNet sequence"),
        ("sequence_metadata", "SequencePeakNet sequence + metadata"),
    ):
        metrics = fixed[key]
        rows.append(
            "| {label} | {accuracy} | {balanced} | {macro_f1} | {mcc} | "
            "{auroc} | {fpr} | {iou} |".format(
                label=label,
                accuracy=_format_metric(metrics["accuracy"]),
                balanced=_format_metric(metrics["balanced_accuracy"]),
                macro_f1=_format_metric(metrics["macro_f1"]),
                mcc=_format_metric(metrics["mcc"]),
                auroc=_format_metric(metrics["auroc"]),
                fpr=_format_metric(metrics["false_positive_rate"]),
                iou=_format_metric(localization[key]["mean_iou"]),
            )
        )
    return "\n".join(
        (
            "# ChromPeakFormer development ablation",
            "",
            "All classification rows use the fixed 0.5 threshold on the same leakage-safe "
            "validation assets. Validation-selected thresholds remain development-only.",
            "",
            "| Model | Accuracy | Balanced accuracy | Macro-F1 | MCC | AUROC | FPR | Mean IoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "COCO AP applies only to the image detector and is not inferred for interval models.",
            "The sealed internal-test split was not accessed; this is not a final benchmark.",
            "",
        )
    )


def build_development_comparison(
    *,
    detector_evaluation_path: Path,
    detector_dataset_report_path: Path,
    sequence_report_path: Path,
    sequence_verification_path: Path,
    sequence_metadata_report_path: Path,
    sequence_metadata_verification_path: Path,
    output_dir: Path,
) -> DevelopmentComparisonResult:
    """Validate and compare three development runs without opening the internal test."""

    paths = {
        "detector_evaluation": detector_evaluation_path.resolve(),
        "detector_dataset": detector_dataset_report_path.resolve(),
        "sequence_report": sequence_report_path.resolve(),
        "sequence_verification": sequence_verification_path.resolve(),
        "sequence_metadata_report": sequence_metadata_report_path.resolve(),
        "sequence_metadata_verification": sequence_metadata_verification_path.resolve(),
    }
    payloads = {name: _read_object(path) for name, path in paths.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}

    detector = payloads["detector_evaluation"]
    detector_dataset = payloads["detector_dataset"]
    sequence = payloads["sequence_report"]
    sequence_metadata = payloads["sequence_metadata_report"]

    _require(
        detector.get("schema_version") == "chrompeak-detector-evaluation-v1",
        "Unexpected detector evaluation schema",
    )
    _validate_development_scope(detector, "ChromPeakFormer")
    _require(detector.get("evaluation_split") == "validation", "Detector split is not validation")
    _require(
        detector_dataset.get("schema_version") == "chrompeak-detector-dataset-v1",
        "Unexpected detector Dataset schema",
    )
    _require(detector_dataset.get("quality_gate_passed") is True, "Detector Dataset gate failed")
    leakage = _object(detector_dataset.get("leakage_audit"), "detector leakage audit")
    _require(leakage.get("passed") is True and leakage.get("overlap") == [], "Detector leakage")
    provenance = _object(detector.get("provenance"), "detector provenance")
    _require(
        provenance.get("detector_dataset_report_sha256") == hashes["detector_dataset"],
        "Detector evaluation does not bind the supplied detector Dataset",
    )

    sequence_dataset = _object(sequence.get("dataset"), "sequence Dataset")
    metadata_dataset = _object(sequence_metadata.get("dataset"), "sequence metadata Dataset")
    asset_index_sha256 = str(sequence_dataset.get("asset_index_sha256"))
    dataset_report_sha256 = str(sequence_dataset.get("dataset_report_sha256"))
    _require(len(asset_index_sha256) == 64, "Invalid sequence asset-index hash")
    _require(len(dataset_report_sha256) == 64, "Invalid sequence Dataset hash")
    _require(
        metadata_dataset.get("asset_index_sha256") == asset_index_sha256
        and metadata_dataset.get("dataset_report_sha256") == dataset_report_sha256,
        "Sequence modalities do not share one Dataset",
    )
    _require(
        detector_dataset.get("asset_index_sha256") == asset_index_sha256,
        "Detector and sequence runs do not share one asset index",
    )

    detector_splits = _object(detector_dataset.get("splits"), "detector splits")
    train_split = _object(detector_splits.get("train"), "detector train split")
    validation_split = _object(detector_splits.get("validation"), "detector validation split")
    for sequence_name, dataset in (("sequence", sequence_dataset), ("metadata", metadata_dataset)):
        _require(
            dataset.get("used_train_samples") == train_split.get("assets")
            and dataset.get("used_validation_samples") == validation_split.get("assets"),
            f"{sequence_name} sample counts do not match the detector Dataset",
        )
        _require(
            dataset.get("validation_source_groups") == validation_split.get("source_groups"),
            f"{sequence_name} validation source groups do not match",
        )

    _validate_sequence_verification(
        payloads["sequence_verification"],
        report_sha256=hashes["sequence_report"],
        dataset_report_sha256=dataset_report_sha256,
        asset_index_sha256=asset_index_sha256,
        label="sequence",
    )
    _validate_sequence_verification(
        payloads["sequence_metadata_verification"],
        report_sha256=hashes["sequence_metadata_report"],
        dataset_report_sha256=dataset_report_sha256,
        asset_index_sha256=asset_index_sha256,
        label="sequence metadata",
    )

    detector_classification = _object(detector.get("classification"), "detector classification")
    detector_fixed = _classification_row(
        _object(detector_classification.get("fixed_threshold"), "detector fixed metrics")
    )
    detector_selected_record = _object(
        detector_classification.get("validation_selected_threshold"),
        "detector selected threshold",
    )
    detector_selected = _classification_row(
        _object(detector_selected_record.get("metrics"), "detector selected metrics")
    )
    sequence_fixed, sequence_selected, sequence_localization = _sequence_rows(
        sequence, "sequence"
    )
    metadata_fixed, metadata_selected, metadata_localization = _sequence_rows(
        sequence_metadata, "sequence_metadata"
    )
    detector_localization = _object(detector.get("localization"), "detector localization")
    detector_fixed_localization = _object(
        detector_localization.get("fixed_threshold"), "detector fixed localization"
    )
    detector_mean_iou = detector_fixed_localization.get("mean_best_iou")
    _require(isinstance(detector_mean_iou, (int, float)), "Missing detector mean IoU")

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "validation_development_ablation",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "dataset": {
            "dataset_report_sha256": dataset_report_sha256,
            "asset_index_sha256": asset_index_sha256,
            "train_assets": int(train_split["assets"]),
            "validation_assets": int(validation_split["assets"]),
            "train_source_groups": int(train_split["source_groups"]),
            "validation_source_groups": int(validation_split["source_groups"]),
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "fixed_threshold_0_5": {
            "chrompeakformer": detector_fixed,
            "sequence": sequence_fixed,
            "sequence_metadata": metadata_fixed,
        },
        "validation_selected_threshold": {
            "chrompeakformer": detector_selected,
            "sequence": sequence_selected,
            "sequence_metadata": metadata_selected,
        },
        "localization": {
            "chrompeakformer": {
                "mean_iou": float(detector_mean_iou),
                "metric": "mean_best_iou",
            },
            "sequence": {
                "mean_iou": sequence_localization["mean_interval_iou"],
                "boundary_mae_seconds": sequence_localization["boundary_mae_seconds"],
                "valid_prediction_rate": sequence_localization["valid_prediction_rate"],
                "metric": "mean_interval_iou",
            },
            "sequence_metadata": {
                "mean_iou": metadata_localization["mean_interval_iou"],
                "boundary_mae_seconds": metadata_localization["boundary_mae_seconds"],
                "valid_prediction_rate": metadata_localization["valid_prediction_rate"],
                "metric": "mean_interval_iou",
            },
        },
        "detector_only_coco": detector.get("coco"),
        "sequence_metadata_minus_sequence": {
            "fixed_threshold_0_5": _metric_deltas(sequence_fixed, metadata_fixed),
            "validation_selected_threshold": _metric_deltas(
                sequence_selected, metadata_selected
            ),
            "mean_iou": (
                metadata_localization["mean_interval_iou"]
                - sequence_localization["mean_interval_iou"]
            ),
            "boundary_mae_seconds": (
                metadata_localization["boundary_mae_seconds"]
                - sequence_localization["boundary_mae_seconds"]
            ),
        },
        "interpretation_limits": [
            "Validation-selected thresholds are development-only and reported separately.",
            "COCO AP applies only to the image detector.",
            "Image best-box IoU and sequence interval IoU are related but not identical tasks.",
            "Single-seed differences require replicated runs before stability claims.",
            "The sealed internal-test split remains unopened.",
        ],
    }

    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Comparison output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        report_path = staging / "development_ablation_report.json"
        markdown_path = staging / "development_ablation_table.md"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown_table(report), encoding="utf-8")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, markdown_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "development_ablation_report.json"
    return DevelopmentComparisonResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        markdown_path=output_dir / "development_ablation_table.md",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
