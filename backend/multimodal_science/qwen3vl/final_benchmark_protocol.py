"""Freeze the sealed internal-test protocol before any test record is opened.

This module deliberately does not accept an internal-test examples path.  It binds the
already completed development evidence, immutable model artifacts, split allocation,
metric policy, and extraction plan.  A separate access ledger is created atomically
immediately before the first operation that may read internal-test records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.development_dossier import (
    REPORT_SCHEMA as DEVELOPMENT_DOSSIER_SCHEMA,
)
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.lora_training import LORA_TRAINING_REPORT_SCHEMA

FINAL_BENCHMARK_PROTOCOL_SCHEMA = "chrompeak-final-benchmark-protocol-v1"
CANDIDATE_LOCK_SCHEMA = "chrompeak-final-benchmark-candidate-lock-v1"
METRICS_LOCK_SCHEMA = "chrompeak-final-benchmark-metrics-lock-v1"
ACCESS_EVENT_SCHEMA = "chrompeak-final-benchmark-access-v1"
ACCESS_COMPLETION_SCHEMA = "chrompeak-final-benchmark-access-completion-v1"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_READINESS = (
    "bilingual_failure_analysis_complete",
    "five_cell_matrix_complete",
    "hash_bound_source_inputs_complete",
    "localization_failure_analysis_complete",
    "selected_checkpoint_intervention_complete",
    "three_seed_statistics_complete",
)
_PRIMARY_MODELS = (
    "qwen3vl_zero_shot",
    "qwen3vl_image_lora",
    "qwen3vl_image_xic_fusion",
    "sequence_peak_net",
    "chrompeakformer",
)


@dataclass(frozen=True)
class FinalBenchmarkProtocolResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    candidate_lock_path: Path
    metrics_lock_path: Path


@dataclass(frozen=True)
class FinalBenchmarkAccessResult:
    ledger_dir: Path
    access_path: Path
    access_sha256: str
    access_id: str
    protocol_sha256: str
    completed: bool


@dataclass(frozen=True)
class FinalBenchmarkRuntimeContext:
    protocol_root: Path
    protocol_sha256: str
    access_id: str
    protocol: dict[str, Any]
    candidate_lock: dict[str, Any]
    metrics_lock: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, description: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{description} must be an object")
    return value


def _read_object(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"{description} not found: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), description)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} JSON: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _hex64(value: Any, description: str) -> str:
    _require(
        isinstance(value, str) and bool(_HEX_64.fullmatch(value)),
        f"{description} must be a lowercase SHA-256",
    )
    return value


def _finite_probability(value: Any, description: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"Invalid {description}")
    return result


def _safe_relative(root: Path, text: str, description: str) -> Path:
    relative = Path(text)
    _require(text != "" and relative != Path("."), f"Missing {description} path")
    _require(not relative.is_absolute(), f"{description} path must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} path escapes its artifact root") from exc
    return path


def _manifest_entries(root: Path) -> tuple[str, dict[str, str]]:
    """Verify every artifact listed by ``root/artifact_manifest.sha256``."""

    root = root.resolve()
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Artifact manifest not found: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Invalid artifact manifest line {line_number}")
        digest = _hex64(parts[0], f"Manifest digest at line {line_number}")
        relative = parts[1].strip().lstrip("*")
        _require(relative not in entries, f"Duplicate artifact manifest path: {relative}")
        artifact = _safe_relative(root, relative, "artifact manifest")
        _require(artifact.is_file(), f"Manifest artifact not found: {artifact}")
        _require(
            sha256_file(artifact) == digest,
            f"Manifest artifact hash mismatch: {relative}",
        )
        entries[Path(relative).as_posix()] = digest
    _require(bool(entries), f"Artifact manifest is empty: {manifest}")
    return sha256_file(manifest), entries


def _required_manifest_hash(
    root: Path, entries: dict[str, str], relative: str, description: str
) -> str:
    relative = Path(relative).as_posix()
    _require(relative in entries, f"{description} is not bound by the artifact manifest")
    path = _safe_relative(root, relative, description)
    _require(entries[relative] == sha256_file(path), f"{description} hash mismatch")
    return entries[relative]


def _bound_report(descriptor: Any, description: str) -> tuple[Path, dict[str, Any], str]:
    value = _object(descriptor, f"{description} descriptor")
    path_text = value.get("path")
    _require(isinstance(path_text, str) and path_text, f"Missing {description} path")
    path = Path(path_text).resolve()
    expected = _hex64(value.get("sha256"), f"{description} SHA-256")
    _require(path.is_file(), f"{description} report not found: {path}")
    _require(sha256_file(path) == expected, f"{description} report hash mismatch")
    return path, _read_object(path, description), expected


def _dossier_source(
    dossier: dict[str, Any], name: str, report_name: str
) -> tuple[Path, dict[str, Any], str, str]:
    sources = _object(dossier.get("sources"), "development dossier sources")
    descriptor = _object(sources.get(name), f"development dossier {name}")
    path_text = descriptor.get("path")
    _require(isinstance(path_text, str) and path_text, f"Missing dossier {name} path")
    path = Path(path_text).resolve()
    expected = _hex64(descriptor.get("report_sha256"), f"dossier {name} report SHA-256")
    _require(path.is_file(), f"Dossier source report not found: {path}")
    _require(path.name == report_name, f"Unexpected dossier source filename for {name}")
    _require(sha256_file(path) == expected, f"Dossier source hash mismatch: {name}")
    manifest_expected = _hex64(
        descriptor.get("manifest_sha256"), f"dossier {name} manifest SHA-256"
    )
    manifest_actual, _ = _manifest_entries(path.parent)
    _require(manifest_actual == manifest_expected, f"Dossier source manifest drift: {name}")
    return path, _read_object(path, f"dossier source {name}"), expected, manifest_actual


def _artifact_descriptor(root: Path, report: dict[str, Any], name: str) -> tuple[Path, str]:
    artifacts = _object(report.get("artifacts"), "report artifacts")
    descriptor = _object(artifacts.get(name), f"artifact {name}")
    path = _safe_relative(root, str(descriptor.get("path") or ""), f"artifact {name}")
    expected = _hex64(descriptor.get("sha256"), f"artifact {name} SHA-256")
    _require(path.is_file() and sha256_file(path) == expected, f"Artifact drift: {name}")
    return path, expected


def _candidate_lock(
    *,
    dossier: dict[str, Any],
    matrix: dict[str, Any],
    cross_family: dict[str, Any],
    dataset_root: Path,
    instruction_root: Path,
    base_model_manifest_path: Path,
) -> dict[str, Any]:
    shared = _object(
        _object(dossier.get("sources"), "dossier sources").get("shared_evidence"),
        "dossier shared evidence",
    )
    dataset_sha256 = _hex64(
        shared.get("dataset_report_sha256"), "shared Dataset report SHA-256"
    )
    instruction_sha256 = _hex64(
        shared.get("instruction_report_sha256"), "shared instruction report SHA-256"
    )
    base_model_sha256 = _hex64(
        shared.get("base_model_artifact_sha256"), "shared base-model artifact SHA-256"
    )
    dataset_report_path = dataset_root / "dataset_report.json"
    instruction_report_path = instruction_root / "instruction_dataset_report.json"
    _require(
        dataset_report_path.is_file() and sha256_file(dataset_report_path) == dataset_sha256,
        "Frozen development Dataset report drift",
    )
    _require(
        instruction_report_path.is_file()
        and sha256_file(instruction_report_path) == instruction_sha256,
        "Frozen instruction report drift",
    )
    dataset_report = _read_object(dataset_report_path, "frozen development Dataset report")
    _require(
        dataset_report.get("schema_version") == "chrompeak-multimodal-dataset-v1",
        "Unexpected frozen development Dataset schema",
    )
    normalization_path, normalization_sha256 = _artifact_descriptor(
        dataset_root,
        dataset_report,
        "scalar_normalization",
    )
    normalization = _read_object(normalization_path, "frozen scalar normalization")
    _require(
        normalization.get("schema_version") == "chrompeak-scalar-normalization-v1"
        and normalization.get("fit_split") == "train",
        "Final benchmark normalization must be fitted on train",
    )

    primary = _object(
        _object(matrix.get("matrix"), "fusion matrix").get("primary-seed17"),
        "fusion primary seed-17 row",
    )
    _require(primary.get("sensor_tokens") == 4, "Selected fusion candidate must use four tokens")
    _require(primary.get("training_seed") == 17, "Selected fusion candidate must use seed 17")
    primary_paths = _object(primary.get("paths"), "selected fusion paths")
    fusion_root = Path(str(primary_paths.get("training_root") or "")).resolve()
    fusion_manifest, fusion_entries = _manifest_entries(fusion_root)
    fusion_report_path = fusion_root / "fusion_training_report.json"
    fusion_report = _read_object(fusion_report_path, "selected fusion training report")
    primary_hashes = _object(primary.get("sha256"), "selected fusion hashes")
    _require(
        sha256_file(fusion_report_path)
        == _hex64(primary_hashes.get("training_report"), "selected fusion report SHA-256"),
        "Selected fusion report drift",
    )
    _require(
        fusion_manifest
        == _hex64(primary_hashes.get("training_manifest"), "selected fusion manifest SHA-256"),
        "Selected fusion manifest drift",
    )
    _require(
        fusion_report.get("schema_version") == FUSION_TRAINING_REPORT_SCHEMA,
        "Unexpected selected fusion training schema",
    )
    _require(
        fusion_report.get("development_training_complete") is True,
        "Selected fusion training is incomplete",
    )
    _require(fusion_report.get("internal_test_accessed") is False, "Fusion already saw test")
    fusion_model = _object(fusion_report.get("model"), "fusion model")
    projector_spec = _object(fusion_model.get("sensor_projector"), "sensor projector")
    fusion_training = _object(fusion_report.get("training"), "fusion training")
    _require(projector_spec.get("sensor_tokens") == 4, "Fusion report token-count drift")
    _require(fusion_training.get("seed") == 17, "Fusion report seed drift")
    _require(
        fusion_model.get("base_artifact_sha256") == base_model_sha256,
        "Fusion base-model drift",
    )
    fusion_sources = _object(fusion_report.get("sources"), "fusion sources")
    _require(
        fusion_sources.get("dataset_report_sha256") == dataset_sha256,
        "Fusion Dataset drift",
    )
    fusion_adapter_sha = _required_manifest_hash(
        fusion_root, fusion_entries, "adapter/adapter_model.safetensors", "fusion adapter"
    )
    fusion_projector_sha = _required_manifest_hash(
        fusion_root, fusion_entries, "sensor_projector.safetensors", "fusion projector"
    )

    cross_sources = _object(cross_family.get("sources"), "cross-family sources")
    _, zero_generation, zero_generation_sha = _bound_report(
        cross_sources.get("zero_shot_generation"), "zero-shot generation"
    )
    _, lora_generation, lora_generation_sha = _bound_report(
        cross_sources.get("lora_generation"), "image-LoRA generation"
    )
    for label, generation in (
        ("zero-shot", zero_generation),
        ("image-LoRA", lora_generation),
    ):
        _require(generation.get("internal_test_accessed") is False, f"{label} saw test")
        model = _object(generation.get("model"), f"{label} generation model")
        _require(model.get("artifact_sha256") == base_model_sha256, f"{label} model drift")
    zero_model = _object(zero_generation.get("model"), "zero-shot generation model")
    lora_model = _object(lora_generation.get("model"), "LoRA generation model")
    model_revision = str(zero_model.get("revision") or "")
    _require(bool(model_revision), "Zero-shot model revision is missing")
    _require(lora_model.get("revision") == model_revision, "Qwen model revision drift")
    _require(
        fusion_model.get("revision") == model_revision,
        "Fusion base-model revision drift",
    )
    fusion_attention = fusion_training.get("attention_implementation")
    _require(
        fusion_attention in {"sdpa", "eager"},
        "Fusion attention implementation is not reproducible",
    )

    lora_adapter = _object(
        _object(lora_generation.get("model"), "LoRA generation model").get("adapter"),
        "LoRA generation adapter",
    )
    lora_root = Path(str(lora_adapter.get("root") or "")).resolve()
    lora_manifest, lora_entries = _manifest_entries(lora_root)
    _require(
        lora_manifest == _hex64(lora_adapter.get("manifest_sha256"), "LoRA manifest SHA-256"),
        "Image-LoRA manifest drift",
    )
    lora_report_path = lora_root / "lora_training_report.json"
    lora_report = _read_object(lora_report_path, "image-only LoRA training report")
    lora_report_sha = sha256_file(lora_report_path)
    _require(
        lora_report_sha
        == _hex64(lora_adapter.get("training_report_sha256"), "LoRA report SHA-256"),
        "Image-LoRA report drift",
    )
    _require(lora_report.get("schema_version") == LORA_TRAINING_REPORT_SCHEMA, "Bad LoRA schema")
    _require(lora_report.get("development_training_complete") is True, "LoRA is incomplete")
    _require(lora_report.get("internal_test_accessed") is False, "LoRA already saw test")
    _require(
        _object(lora_report.get("source"), "LoRA source").get("dataset_report_sha256")
        == dataset_sha256,
        "LoRA Dataset drift",
    )
    _require(
        _object(lora_report.get("model"), "LoRA model").get("artifact_sha256")
        == base_model_sha256,
        "LoRA base-model drift",
    )
    lora_weights_sha = _required_manifest_hash(
        lora_root, lora_entries, "adapter/adapter_model.safetensors", "LoRA adapter"
    )

    _, specialist, _ = _bound_report(
        cross_sources.get("specialist_comparison"), "specialist comparison"
    )
    specialist_sources = _object(specialist.get("sources"), "specialist sources")
    sequence_path, sequence_report, sequence_sha = _bound_report(
        specialist_sources.get("sequence_report"), "sequence report"
    )
    sequence_metadata_path, sequence_metadata_report, sequence_metadata_sha = _bound_report(
        specialist_sources.get("sequence_metadata_report"), "sequence metadata report"
    )
    detector_eval_path, detector_eval, detector_eval_sha = _bound_report(
        specialist_sources.get("detector_evaluation"), "detector evaluation"
    )

    sequence_candidates: dict[str, Any] = {}
    for key, report_path, report, report_sha in (
        ("sequence_peak_net", sequence_path, sequence_report, sequence_sha),
        (
            "sequence_peak_net_metadata",
            sequence_metadata_path,
            sequence_metadata_report,
            sequence_metadata_sha,
        ),
    ):
        root = report_path.parent
        manifest_sha, entries = _manifest_entries(root)
        _require(
            report.get("schema_version") == "chrompeak-sequence-baseline-report-v1",
            "Bad sequence schema",
        )
        _require(report.get("development_comparison_eligible") is True, "Sequence run ineligible")
        _require(report.get("internal_test_accessed") is False, "Sequence run already saw test")
        _require(
            _object(report.get("dataset"), "sequence Dataset").get("dataset_report_sha256")
            == dataset_sha256,
            "Sequence Dataset drift",
        )
        checkpoint_path, checkpoint_sha = _artifact_descriptor(root, report, "checkpoint")
        threshold_path, threshold_sha = _artifact_descriptor(root, report, "frozen_threshold")
        threshold = _read_object(threshold_path, "frozen sequence threshold")
        _require(
            threshold.get("selected_on_split") == "validation",
            "Threshold was not frozen on validation",
        )
        _require(threshold.get("internal_test_accessed") is False, "Threshold saw test")
        selected_threshold = _finite_probability(threshold.get("threshold"), "sequence threshold")
        sequence_candidates[key] = {
            "root": str(root),
            "report_sha256": report_sha,
            "manifest_sha256": manifest_sha,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
            },
            "frozen_threshold": {
                "path": str(threshold_path),
                "sha256": threshold_sha,
                "value": selected_threshold,
                "selected_on_split": "validation",
                "objective": threshold.get("objective"),
            },
        }

    detector_run_root = detector_eval_path.parent.parent
    detector_manifest, detector_entries = _manifest_entries(detector_run_root)
    detector_training_path = detector_run_root / "training" / "detector_training_report.json"
    detector_training = _read_object(detector_training_path, "detector training report")
    _require(
        detector_training.get("schema_version") == "chrompeak-detector-training-v1",
        "Unexpected detector training schema",
    )
    _require(
        detector_training.get("development_comparison_eligible") is True,
        "Detector ineligible",
    )
    _require(detector_training.get("internal_test_accessed") is False, "Detector already saw test")
    detector_checkpoint_sha = _required_manifest_hash(
        detector_run_root,
        detector_entries,
        "training/checkpoint.pth",
        "detector checkpoint",
    )
    _require(
        detector_training.get("checkpoint_sha256") == detector_checkpoint_sha,
        "Detector checkpoint drift",
    )
    _require(
        detector_eval.get("schema_version") == "chrompeak-detector-evaluation-v1",
        "Unexpected detector evaluation schema",
    )
    _require(
        detector_eval.get("development_comparison_eligible") is True,
        "Detector eval ineligible",
    )
    _require(detector_eval.get("internal_test_accessed") is False, "Detector eval saw test")
    detector_selected = _object(
        _object(detector_eval.get("classification"), "detector classification").get(
            "validation_selected_threshold"
        ),
        "detector validation threshold",
    )
    detector_threshold = _finite_probability(
        detector_selected.get("threshold"), "detector threshold"
    )

    model_manifest_sha = sha256_file(base_model_manifest_path)
    _require(
        model_manifest_sha
        == _hex64(
            fusion_model.get("verification_manifest_sha256"),
            "fusion model-manifest SHA-256",
        ),
        "Base-model manifest drift",
    )

    return {
        "schema_version": CANDIDATE_LOCK_SCHEMA,
        "primary_models": list(_PRIMARY_MODELS),
        "secondary_models": ["sequence_peak_net_metadata"],
        "shared": {
            "dataset_root": str(dataset_root),
            "dataset_report_sha256": dataset_sha256,
            "scalar_normalization": {
                "path": str(normalization_path),
                "sha256": normalization_sha256,
                "fit_split": "train",
            },
            "instruction_root": str(instruction_root),
            "instruction_report_sha256": instruction_sha256,
            "base_model": {
                "name_or_path": zero_model.get("name_or_path"),
                "revision": model_revision,
                "artifact_sha256": base_model_sha256,
                "verification_manifest_path": str(base_model_manifest_path),
                "verification_manifest_sha256": model_manifest_sha,
            },
            "inference_policy": {
                "batch_size": 1,
                "max_new_tokens": 64,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "seed": 17,
                "dtype": "bfloat16",
                "zero_and_lora_device_map": "auto",
                "zero_and_lora_attention_implementation": None,
                "fusion_device_map": "single_cuda",
                "fusion_attention_implementation": fusion_attention,
            },
        },
        "models": {
            "qwen3vl_zero_shot": {
                "generation_report_sha256": zero_generation_sha,
                "adapter": None,
                "decision_rule": "greedy_json_generation",
            },
            "qwen3vl_image_lora": {
                "generation_report_sha256": lora_generation_sha,
                "root": str(lora_root),
                "training_report_sha256": lora_report_sha,
                "manifest_sha256": lora_manifest,
                "adapter_weights_sha256": lora_weights_sha,
                "decision_rule": "greedy_json_generation",
            },
            "qwen3vl_image_xic_fusion": {
                "root": str(fusion_root),
                "training_report_sha256": sha256_file(fusion_report_path),
                "manifest_sha256": fusion_manifest,
                "adapter_weights_sha256": fusion_adapter_sha,
                "sensor_projector_sha256": fusion_projector_sha,
                "sensor_tokens": 4,
                "training_seed": 17,
                "decision_rule": "greedy_json_generation_with_aligned_xic",
            },
            **sequence_candidates,
            "chrompeakformer": {
                "root": str(detector_run_root),
                "training_report_sha256": sha256_file(detector_training_path),
                "evaluation_report_sha256": detector_eval_sha,
                "manifest_sha256": detector_manifest,
                "checkpoint_sha256": detector_checkpoint_sha,
                "primary_threshold": detector_threshold,
                "primary_threshold_selected_on_split": "validation",
                "secondary_fixed_threshold": 0.5,
            },
        },
    }


def _metrics_lock(*, bootstrap_iterations: int, bootstrap_seed: int) -> dict[str, Any]:
    _require(
        bootstrap_iterations >= 10_000,
        "Final benchmark requires at least 10,000 bootstrap iterations",
    )
    _require(isinstance(bootstrap_seed, int), "Bootstrap seed must be an integer")
    return {
        "schema_version": METRICS_LOCK_SCHEMA,
        "primary_models": list(_PRIMARY_MODELS),
        "primary_qwen_reporting": {
            "languages": ["en", "zh-CN"],
            "paired_language_variants_are_not_independent_samples": True,
            "presence": ["balanced_accuracy", "macro_f1", "mcc", "false_positive_rate"],
            "grounding": ["mean_bbox_iou_all", "iou_at_0_5_rate_all"],
            "scientific_qc": ["exact_match_rate"],
            "structured_output": ["valid_json_rate", "schema_valid_rate"],
        },
        "primary_specialist_reporting": {
            "classification": [
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            ],
            "localization": ["mean_iou", "iou_at_0_5_rate"],
            "detector_only": ["coco_ap_50_95", "coco_ap_50", "coco_ap_75"],
        },
        "threshold_policy": {
            "primary": "validation_frozen_operating_threshold",
            "secondary": "fixed_0.5",
            "internal_test_threshold_selection_forbidden": True,
        },
        "uncertainty": {
            "unit": "complete_source_mzml_group",
            "bootstrap_iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "prompt_rows_are_not_bootstrap_units": True,
        },
        "invalid_prediction_policy": "invalid_or_schema_invalid_predictions_score_as_failures",
        "post_access_model_selection_forbidden": True,
        "post_access_threshold_tuning_forbidden": True,
        "combined_cross_task_score_reported": False,
    }


def freeze_final_benchmark_protocol(
    *,
    development_dossier_root: Path,
    split_manifest_path: Path,
    split_report_path: Path,
    derivation_plan_path: Path,
    derivation_report_path: Path,
    dataset_root: Path,
    instruction_root: Path,
    base_model_manifest_path: Path,
    output_dir: Path,
    expected_internal_test_assets: int = 1815,
    expected_internal_test_source_groups: int = 11,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 17,
) -> FinalBenchmarkProtocolResult:
    """Create an immutable protocol without parsing the internal-test manifest."""

    development_dossier_root = development_dossier_root.resolve()
    split_manifest_path = split_manifest_path.resolve()
    split_report_path = split_report_path.resolve()
    derivation_plan_path = derivation_plan_path.resolve()
    derivation_report_path = derivation_report_path.resolve()
    dataset_root = dataset_root.resolve()
    instruction_root = instruction_root.resolve()
    base_model_manifest_path = base_model_manifest_path.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Final benchmark protocol already exists: {output_dir}")
    for path, label in (
        (split_manifest_path, "split manifest"),
        (split_report_path, "split report"),
        (derivation_plan_path, "derivation plan"),
        (derivation_report_path, "derivation report"),
        (base_model_manifest_path, "base-model manifest"),
    ):
        _require(path.is_file(), f"{label.capitalize()} not found: {path}")
    _require(dataset_root.is_dir(), f"Dataset root not found: {dataset_root}")
    _require(instruction_root.is_dir(), f"Instruction root not found: {instruction_root}")

    dossier_manifest, dossier_entries = _manifest_entries(development_dossier_root)
    dossier_report_path = development_dossier_root / "development_dossier.json"
    dossier_report_sha = _required_manifest_hash(
        development_dossier_root,
        dossier_entries,
        dossier_report_path.name,
        "development dossier report",
    )
    dossier = _read_object(dossier_report_path, "development dossier")
    _require(dossier.get("schema_version") == DEVELOPMENT_DOSSIER_SCHEMA, "Bad dossier schema")
    _require(dossier.get("development_comparison_eligible") is True, "Dossier is ineligible")
    _require(dossier.get("final_benchmark_eligible") is False, "Dossier claims final metrics")
    _require(dossier.get("internal_test_accessed") is False, "Internal test was already accessed")
    readiness = _object(dossier.get("pre_internal_test_readiness"), "pre-test readiness")
    for field in _REQUIRED_READINESS:
        _require(readiness.get(field) is True, f"Pre-test readiness failed: {field}")
    _require(readiness.get("ready") is True, "Pre-test dossier is not ready")

    _, matrix, matrix_sha, matrix_manifest = _dossier_source(
        dossier, "fusion_matrix", "fusion_matrix_analysis.json"
    )
    _, cross_family, cross_sha, cross_manifest = _dossier_source(
        dossier, "cross_family", "cross_family_development_report.json"
    )
    candidate_lock = _candidate_lock(
        dossier=dossier,
        matrix=matrix,
        cross_family=cross_family,
        dataset_root=dataset_root,
        instruction_root=instruction_root,
        base_model_manifest_path=base_model_manifest_path,
    )

    split_report = _read_object(split_report_path, "split report")
    _require(split_report.get("schema_version") == "chrompeak-split-v1", "Bad split schema")
    split_manifest_sha = sha256_file(split_manifest_path)
    _require(
        split_report.get("split_manifest_sha256") == split_manifest_sha,
        "Split-manifest hash drift",
    )
    _require(
        _object(split_report.get("leakage_audit"), "split leakage audit").get("passed")
        is True,
        "Split leakage audit failed",
    )
    split_rows = split_report.get("splits")
    _require(isinstance(split_rows, list), "Split report summaries must be a list")
    test_summaries = [
        row
        for row in split_rows
        if isinstance(row, dict) and row.get("split") == "internal_test"
    ]
    _require(len(test_summaries) == 1, "Split report must contain one internal_test summary")
    test_summary = test_summaries[0]
    _require(
        test_summary.get("records") == expected_internal_test_assets,
        "Frozen internal-test asset count mismatch",
    )
    _require(
        test_summary.get("groups") == expected_internal_test_source_groups,
        "Frozen internal-test source-group count mismatch",
    )
    _require(test_summary.get("audit_records") == 0, "Internal test contains audit records")

    derivation_report = _read_object(derivation_report_path, "derivation report")
    _require(
        derivation_report.get("schema_version") == "chrompeak-derivation-v1",
        "Bad derivation-report schema",
    )
    derivation_plan_sha = sha256_file(derivation_plan_path)
    _require(
        derivation_report.get("derivation_plan_sha256") == derivation_plan_sha,
        "Derivation-plan hash drift",
    )
    _require(
        derivation_report.get("source_split_manifest_sha256") == split_manifest_sha,
        "Derivation plan does not bind the frozen split manifest",
    )

    metrics_lock = _metrics_lock(
        bootstrap_iterations=bootstrap_iterations, bootstrap_seed=bootstrap_seed
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        candidate_path = staging / "candidate_lock.json"
        metrics_path = staging / "metrics_lock.json"
        _write_json(candidate_path, candidate_lock)
        _write_json(metrics_path, metrics_lock)
        report = {
            "schema_version": FINAL_BENCHMARK_PROTOCOL_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_state": "frozen_before_internal_test_access",
            "development_dossier": {
                "root": str(development_dossier_root),
                "report_sha256": dossier_report_sha,
                "manifest_sha256": dossier_manifest,
            },
            "development_sources": {
                "cross_family_report_sha256": cross_sha,
                "cross_family_manifest_sha256": cross_manifest,
                "fusion_matrix_report_sha256": matrix_sha,
                "fusion_matrix_manifest_sha256": matrix_manifest,
            },
            "sealed_split": {
                "name": "internal_test",
                "dataset_version": split_report.get("dataset_version"),
                "split_report_path": str(split_report_path),
                "split_report_sha256": sha256_file(split_report_path),
                "split_manifest_path": str(split_manifest_path),
                "split_manifest_sha256": split_manifest_sha,
                "expected_assets": expected_internal_test_assets,
                "expected_source_groups": expected_internal_test_source_groups,
                "labels_opened_while_freezing_protocol": False,
            },
            "derivation": {
                "plan_path": str(derivation_plan_path),
                "plan_sha256": derivation_plan_sha,
                "report_path": str(derivation_report_path),
                "report_sha256": sha256_file(derivation_report_path),
                "execution_split": "internal_test",
            },
            "locks": {
                "candidate": {
                    "path": candidate_path.name,
                    "sha256": sha256_file(candidate_path),
                },
                "metrics": {
                    "path": metrics_path.name,
                    "sha256": sha256_file(metrics_path),
                },
            },
            "one_time_access_policy": {
                "access_ledger_must_be_created_before_reading_test_records": True,
                "second_access_event_forbidden": True,
                "same_protocol_crash_resume_allowed": True,
                "candidate_or_threshold_change_after_access_forbidden": True,
                "development_metrics_remain_separate": True,
            },
            "required_final_evidence": {
                "five_candidate_model_benchmark_report": True,
                "qwen_structured_output_and_qc_metrics": True,
                "frozen_development_failure_analysis": True,
                "evidence_registry": True,
                "artifact_manifest": True,
            },
            "benchmark_scope": "frozen_multimodal_model_candidates",
            "biocoder_agent_promotion_claimed": False,
            "pre_internal_test_ready": True,
            "internal_test_accessed": False,
            "final_benchmark_eligible": False,
        }
        report_path = staging / "final_benchmark_protocol.json"
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, candidate_path, metrics_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "final_benchmark_protocol.json"
    return FinalBenchmarkProtocolResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        manifest_path=output_dir / "artifact_manifest.sha256",
        candidate_lock_path=output_dir / "candidate_lock.json",
        metrics_lock_path=output_dir / "metrics_lock.json",
    )


def _verify_protocol(protocol_root: Path, expected_protocol_sha256: str) -> dict[str, Any]:
    protocol_root = protocol_root.resolve()
    expected_protocol_sha256 = _hex64(expected_protocol_sha256, "protocol SHA-256")
    _, entries = _manifest_entries(protocol_root)
    report_path = protocol_root / "final_benchmark_protocol.json"
    _required_manifest_hash(protocol_root, entries, report_path.name, "benchmark protocol")
    _required_manifest_hash(protocol_root, entries, "candidate_lock.json", "candidate lock")
    _required_manifest_hash(protocol_root, entries, "metrics_lock.json", "metrics lock")
    _require(sha256_file(report_path) == expected_protocol_sha256, "Protocol hash mismatch")
    report = _read_object(report_path, "final benchmark protocol")
    _require(
        report.get("schema_version") == FINAL_BENCHMARK_PROTOCOL_SCHEMA,
        "Bad protocol schema",
    )
    _require(report.get("pre_internal_test_ready") is True, "Protocol is not ready")
    _require(report.get("internal_test_accessed") is False, "Protocol was not pre-access")
    return report


def open_final_benchmark_access(
    *, protocol_root: Path, expected_protocol_sha256: str, ledger_dir: Path
) -> FinalBenchmarkAccessResult:
    """Atomically record the sole access event before a protected record is read."""

    protocol_root = protocol_root.resolve()
    ledger_dir = ledger_dir.resolve()
    protocol = _verify_protocol(protocol_root, expected_protocol_sha256)
    _require(
        not ledger_dir.exists(),
        f"Final benchmark access ledger already exists: {ledger_dir}",
    )
    ledger_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = ledger_dir.parent / f".{ledger_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        opened_at = datetime.now(timezone.utc).isoformat()
        access_id = hashlib.sha256(
            f"{expected_protocol_sha256}\0{opened_at}\0{uuid.uuid4().hex}".encode("utf-8")
        ).hexdigest()[:24]
        payload = {
            "schema_version": ACCESS_EVENT_SCHEMA,
            "access_id": access_id,
            "access_sequence": 1,
            "opened_at": opened_at,
            "state": "opened",
            "protocol_root": str(protocol_root),
            "protocol_sha256": expected_protocol_sha256,
            "candidate_lock_sha256": _object(
                _object(protocol.get("locks"), "protocol locks").get("candidate"),
                "candidate lock",
            ).get("sha256"),
            "metrics_lock_sha256": _object(
                _object(protocol.get("locks"), "protocol locks").get("metrics"),
                "metrics lock",
            ).get("sha256"),
            "split_manifest_sha256": _object(
                protocol.get("sealed_split"), "sealed split"
            ).get("split_manifest_sha256"),
            "permitted_scope": [
                "materialize_frozen_internal_test",
                "run_frozen_candidate_inference",
                "compute_predeclared_metrics",
                "publish_final_evidence",
                "resume_same_protocol_after_infrastructure_failure",
            ],
            "forbidden_scope": [
                "candidate_selection",
                "threshold_selection",
                "hyperparameter_tuning",
                "training_on_internal_test",
                "second_independent_test_access",
            ],
            "internal_test_accessed": True,
            "completed": False,
        }
        access_path = staging / "access_started.json"
        _write_json(access_path, payload)
        (staging / "access_started.sha256").write_text(
            f"{sha256_file(access_path)}  {access_path.name}\n", encoding="utf-8"
        )
        staging.replace(ledger_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    access_path = ledger_dir / "access_started.json"
    return FinalBenchmarkAccessResult(
        ledger_dir=ledger_dir,
        access_path=access_path,
        access_sha256=sha256_file(access_path),
        access_id=access_id,
        protocol_sha256=expected_protocol_sha256,
        completed=False,
    )


def verify_final_benchmark_access(
    *, protocol_root: Path, expected_protocol_sha256: str, ledger_dir: Path
) -> FinalBenchmarkAccessResult:
    """Authorize only a crash-resume of the exact already-opened protocol."""

    _verify_protocol(protocol_root.resolve(), expected_protocol_sha256)
    ledger_dir = ledger_dir.resolve()
    access_path = ledger_dir / "access_started.json"
    _require(access_path.is_file(), f"Access ledger not found: {access_path}")
    checksum_path = ledger_dir / "access_started.sha256"
    _require(checksum_path.is_file(), "Access ledger checksum is missing")
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    _require(len(checksum_parts) == 2, "Invalid access-ledger checksum")
    _require(
        _hex64(checksum_parts[0], "access-ledger checksum") == sha256_file(access_path),
        "Access ledger was modified",
    )
    access = _read_object(access_path, "access ledger")
    _require(access.get("schema_version") == ACCESS_EVENT_SCHEMA, "Bad access schema")
    _require(access.get("access_sequence") == 1, "Invalid access sequence")
    _require(access.get("protocol_sha256") == expected_protocol_sha256, "Access protocol drift")
    _require(access.get("internal_test_accessed") is True, "Access event is not open")
    completion_path = ledger_dir / "access_completed.json"
    completed = completion_path.is_file()
    if completed:
        completion = _read_object(completion_path, "access completion")
        _require(
            completion.get("schema_version") == ACCESS_COMPLETION_SCHEMA,
            "Bad access completion schema",
        )
        _require(completion.get("access_id") == access.get("access_id"), "Completion access drift")
        _require(
            completion.get("protocol_sha256") == expected_protocol_sha256,
            "Completion protocol drift",
        )
        _require(completion.get("access_sequence") == 1, "Bad completion access sequence")
        _require(completion.get("completed") is True, "Access completion is not complete")
        _require(
            completion.get("internal_test_accessed") is True,
            "Access completion does not record test access",
        )
        _require(
            completion.get("additional_test_access_authorized") is False,
            "Access completion authorizes an extra test access",
        )
    return FinalBenchmarkAccessResult(
        ledger_dir=ledger_dir,
        access_path=access_path,
        access_sha256=sha256_file(access_path),
        access_id=str(access["access_id"]),
        protocol_sha256=expected_protocol_sha256,
        completed=completed,
    )


def load_final_benchmark_context(
    *, protocol_root: Path, expected_protocol_sha256: str, ledger_dir: Path
) -> FinalBenchmarkRuntimeContext:
    """Load the immutable candidate/metric locks for an open one-time run."""

    state = verify_final_benchmark_access(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    _require(not state.completed, "Final benchmark access is already completed")
    protocol_root = protocol_root.resolve()
    protocol = _verify_protocol(protocol_root, expected_protocol_sha256)
    locks = _object(protocol.get("locks"), "protocol locks")
    loaded: dict[str, dict[str, Any]] = {}
    for name, filename, schema in (
        ("candidate", "candidate_lock.json", CANDIDATE_LOCK_SCHEMA),
        ("metrics", "metrics_lock.json", METRICS_LOCK_SCHEMA),
    ):
        descriptor = _object(locks.get(name), f"{name} lock descriptor")
        _require(descriptor.get("path") == filename, f"Unexpected {name} lock path")
        path = protocol_root / filename
        _require(
            descriptor.get("sha256") == sha256_file(path),
            f"{name.capitalize()} lock drift",
        )
        value = _read_object(path, f"{name} lock")
        _require(value.get("schema_version") == schema, f"Bad {name} lock schema")
        loaded[name] = value
    return FinalBenchmarkRuntimeContext(
        protocol_root=protocol_root,
        protocol_sha256=expected_protocol_sha256,
        access_id=state.access_id,
        protocol=protocol,
        candidate_lock=loaded["candidate"],
        metrics_lock=loaded["metrics"],
    )


def _ensure_access_manifest(
    ledger_dir: Path,
    access_path: Path,
    completion_path: Path,
) -> Path:
    manifest_path = ledger_dir / "artifact_manifest.sha256"
    expected = "".join(
        f"{sha256_file(path)}  {path.name}\n"
        for path in (access_path, completion_path)
    )
    if manifest_path.exists():
        _require(
            manifest_path.read_text(encoding="utf-8") == expected,
            "Final access manifest drift",
        )
        return manifest_path

    temporary = manifest_path.with_name(
        f".{manifest_path.name}-{uuid.uuid4().hex}.staging"
    )
    temporary.write_text(expected, encoding="utf-8")
    try:
        os.link(temporary, manifest_path)
    except FileExistsError:
        _require(
            manifest_path.read_text(encoding="utf-8") == expected,
            "Final access manifest drift",
        )
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def complete_final_benchmark_access(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    final_evidence_manifest_path: Path,
) -> FinalBenchmarkAccessResult:
    """Seal a successful one-time run to one final evidence manifest."""

    state = verify_final_benchmark_access(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    final_evidence_manifest_path = final_evidence_manifest_path.resolve()
    _require(
        final_evidence_manifest_path.is_file(),
        f"Final evidence manifest not found: {final_evidence_manifest_path}",
    )
    final_evidence_sha256 = sha256_file(final_evidence_manifest_path)
    completion_path = state.ledger_dir / "access_completed.json"
    if state.completed:
        completion = _read_object(completion_path, "access completion")
        _require(
            completion.get("protocol_sha256") == expected_protocol_sha256,
            "Completed access protocol drift",
        )
        _require(
            completion.get("final_evidence_manifest_path")
            == str(final_evidence_manifest_path),
            "Completed access evidence path drift",
        )
        _require(
            completion.get("final_evidence_manifest_sha256")
            == final_evidence_sha256,
            "Completed access evidence hash drift",
        )
        _ensure_access_manifest(state.ledger_dir, state.access_path, completion_path)
        return state

    completion = {
        "schema_version": ACCESS_COMPLETION_SCHEMA,
        "access_id": state.access_id,
        "protocol_sha256": expected_protocol_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "final_evidence_manifest_path": str(final_evidence_manifest_path),
        "final_evidence_manifest_sha256": final_evidence_sha256,
        "access_sequence": 1,
        "additional_test_access_authorized": False,
        "completed": True,
        "internal_test_accessed": True,
    }
    temporary = completion_path.with_name(
        f".{completion_path.name}-{uuid.uuid4().hex}.staging"
    )
    _write_json(temporary, completion)
    try:
        os.link(temporary, completion_path)
    except FileExistsError as exc:
        raise ValueError("Final benchmark access was already completed") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _ensure_access_manifest(state.ledger_dir, state.access_path, completion_path)
    return FinalBenchmarkAccessResult(
        ledger_dir=state.ledger_dir,
        access_path=state.access_path,
        access_sha256=state.access_sha256,
        access_id=state.access_id,
        protocol_sha256=state.protocol_sha256,
        completed=True,
    )
