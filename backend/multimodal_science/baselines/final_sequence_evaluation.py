"""Run a frozen SequencePeakNet candidate on the sealed internal test split."""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multimodal_science.baselines.dataset import SequenceSplit, load_sequence_split
from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
)
from multimodal_science.baselines.sequence_model import (
    SequenceModelSpec,
    build_sequence_peak_net,
)
from multimodal_science.baselines.sequence_training import CHECKPOINT_SCHEMA, REPORT_SCHEMA
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FinalBenchmarkRuntimeContext,
    load_final_benchmark_context,
)


FINAL_SEQUENCE_EVALUATION_SCHEMA = "chrompeak-sequence-final-evaluation-v1"
FINAL_SEQUENCE_PREDICTION_SCHEMA = "chrompeak-sequence-final-prediction-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FinalSequenceEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    predictions_path: Path
    assets: int
    source_groups: int
    candidate_name: str
    access_id: str


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def _manifest_entries(root: Path, expected_manifest_sha256: str) -> dict[str, str]:
    root = root.resolve()
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Candidate manifest not found: {manifest}")
    _require(
        sha256_file(manifest) == expected_manifest_sha256,
        "Candidate artifact manifest drift",
    )
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed manifest line {line_number}")
        digest, relative = parts
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad manifest digest at line {line_number}")
        relative = relative.strip().lstrip("*").replace("\\", "/")
        relative_path = Path(relative)
        _require(
            relative != "" and relative_path != Path("."),
            f"Missing manifest path at line {line_number}",
        )
        _require(
            not relative_path.is_absolute(),
            f"Manifest path must be relative at line {line_number}",
        )
        path = (root / relative).resolve()
        try:
            canonical = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Candidate manifest path escapes its root") from error
        _require(
            canonical not in entries,
            f"Duplicate normalized sequence manifest path: {canonical}",
        )
        _require(path.is_file() and sha256_file(path) == digest, f"Artifact drift: {relative}")
        entries[canonical] = digest
    _require(bool(entries), "Candidate artifact manifest is empty")
    return entries


def _candidate(
    context: FinalBenchmarkRuntimeContext, candidate_name: str
) -> dict[str, Any]:
    _require(
        candidate_name in {"sequence_peak_net", "sequence_peak_net_metadata"},
        f"Unsupported sequence candidate: {candidate_name}",
    )
    models = _object(context.candidate_lock.get("models"), "candidate models")
    return _object(models.get(candidate_name), f"candidate {candidate_name}")


def _verify_candidate_artifacts(
    candidate: dict[str, Any], candidate_name: str
) -> tuple[Path, Path, Path, float]:
    root = Path(str(candidate.get("root") or "")).resolve()
    _require(root.is_dir(), f"Candidate root not found: {root}")
    entries = _manifest_entries(root, str(candidate.get("manifest_sha256") or ""))
    report_path = root / "scientific_report.json"
    _require(
        report_path.is_file()
        and sha256_file(report_path) == candidate.get("report_sha256"),
        "Sequence training report drift",
    )
    report = _read_json(report_path, "sequence training report")
    _require(report.get("schema_version") == REPORT_SCHEMA, "Bad sequence report schema")
    _require(report.get("development_comparison_eligible") is True, "Candidate ineligible")
    _require(report.get("internal_test_accessed") is False, "Candidate already saw test")
    expected_modality = (
        "sequence_metadata" if candidate_name == "sequence_peak_net_metadata" else "sequence"
    )
    _require(
        _object(report.get("config"), "sequence config").get("modality")
        == expected_modality,
        "Sequence candidate modality drift",
    )

    checkpoint = _object(candidate.get("checkpoint"), "frozen checkpoint")
    checkpoint_path = Path(str(checkpoint.get("path") or "")).resolve()
    threshold = _object(candidate.get("frozen_threshold"), "frozen threshold")
    threshold_path = Path(str(threshold.get("path") or "")).resolve()
    for path, descriptor, label in (
        (checkpoint_path, checkpoint, "checkpoint"),
        (threshold_path, threshold, "threshold"),
    ):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Frozen {label} escapes candidate root") from error
        _require(path.is_file(), f"Frozen {label} not found: {path}")
        _require(sha256_file(path) == descriptor.get("sha256"), f"Frozen {label} drift")
        _require(entries.get(relative) == descriptor.get("sha256"), f"Unbound {label}")
    threshold_payload = _read_json(threshold_path, "frozen sequence threshold")
    _require(threshold_payload.get("selected_on_split") == "validation", "Bad threshold split")
    _require(threshold_payload.get("internal_test_accessed") is False, "Threshold saw test")
    frozen = float(threshold.get("value"))
    _require(math.isfinite(frozen) and 0.0 <= frozen <= 1.0, "Invalid frozen threshold")
    _require(float(threshold_payload.get("threshold")) == frozen, "Threshold value drift")
    return checkpoint_path, report_path, threshold_path, frozen


def _verify_training_data_lineage(
    context: FinalBenchmarkRuntimeContext,
    *,
    checkpoint: dict[str, Any],
    candidate_report_path: Path,
    threshold_path: Path,
) -> str:
    shared = _object(context.candidate_lock.get("shared"), "shared candidate inputs")
    training_root = Path(str(shared.get("dataset_root") or "")).resolve()
    training_report_path = training_root / "dataset_report.json"
    expected_report_sha256 = shared.get("dataset_report_sha256")
    _require(
        isinstance(expected_report_sha256, str)
        and bool(_HEX_64.fullmatch(expected_report_sha256)),
        "Frozen training Dataset report hash is invalid",
    )
    _require(
        training_report_path.is_file()
        and sha256_file(training_report_path) == expected_report_sha256,
        "Frozen training Dataset report drift",
    )
    training_dataset = _read_json(training_report_path, "frozen training Dataset report")
    training_asset_index_sha256 = training_dataset.get("asset_index_sha256")
    _require(
        isinstance(training_asset_index_sha256, str)
        and bool(_HEX_64.fullmatch(training_asset_index_sha256)),
        "Frozen training asset-index hash is invalid",
    )

    candidate_report = _read_json(candidate_report_path, "sequence training report")
    candidate_dataset = _object(candidate_report.get("dataset"), "sequence training Dataset")
    threshold = _read_json(threshold_path, "frozen sequence threshold")
    for payload, label in (
        (checkpoint, "Checkpoint"),
        (candidate_dataset, "Sequence training report"),
        (threshold, "Frozen threshold"),
    ):
        _require(
            payload.get("dataset_report_sha256") == expected_report_sha256,
            f"{label} is not bound to the frozen training Dataset report",
        )
        _require(
            payload.get("asset_index_sha256") == training_asset_index_sha256,
            f"{label} is not bound to the frozen training asset index",
        )
    return training_asset_index_sha256


def _metrics_contract(context: FinalBenchmarkRuntimeContext) -> tuple[int, int]:
    lock = context.metrics_lock
    policy = _object(lock.get("threshold_policy"), "threshold policy")
    _require(
        policy.get("internal_test_threshold_selection_forbidden") is True,
        "Internal-test threshold selection is not forbidden",
    )
    _require(lock.get("post_access_model_selection_forbidden") is True, "Selection unlocked")
    _require(lock.get("post_access_threshold_tuning_forbidden") is True, "Tuning unlocked")
    uncertainty = _object(lock.get("uncertainty"), "uncertainty protocol")
    iterations = uncertainty.get("bootstrap_iterations")
    seed = uncertainty.get("seed")
    _require(isinstance(iterations, int) and iterations >= 10_000, "Bad bootstrap count")
    _require(isinstance(seed, int), "Bad bootstrap seed")
    return iterations, seed


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    return _object(value, "sequence checkpoint")


def _predict(
    split: SequenceSplit,
    checkpoint: dict[str, Any],
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _require(device_name in {"cpu", "cuda"}, f"Unsupported device: {device_name}")
    if device_name == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    spec = SequenceModelSpec(**_object(checkpoint.get("model_spec"), "model spec"))
    model = build_sequence_peak_net(spec)
    load = model.load_state_dict(checkpoint.get("model_state_dict"), strict=True)
    _require(not load.missing_keys and not load.unexpected_keys, "Checkpoint load was not strict")
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(split.signals, dtype=np.float32)),
            torch.from_numpy(np.asarray(split.scalar_features, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    probabilities = []
    boundaries = []
    with torch.inference_mode():
        for signals, scalars in loader:
            logits, predicted = model(signals.to(device), scalars.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            boundaries.append(predicted.cpu().numpy())
    return (
        np.concatenate(probabilities).astype(np.float64),
        np.concatenate(boundaries).astype(np.float64),
    )


def _interval_ious(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    clipped = np.clip(predicted, 0.0, 1.0)
    intersection = np.maximum(
        0.0,
        np.minimum(clipped[:, 1], truth[:, 1])
        - np.maximum(clipped[:, 0], truth[:, 0]),
    )
    union = np.maximum(clipped[:, 1], truth[:, 1]) - np.minimum(
        clipped[:, 0], truth[:, 0]
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _grouped_bootstrap_localization(
    ious: np.ndarray,
    group_ids: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.unique(group_ids)
    _require(groups.size >= 1, "Localization bootstrap requires a positive source group")
    by_group = {group: np.flatnonzero(group_ids == group) for group in groups}
    mean_samples = np.empty(iterations, dtype=np.float64)
    rate_samples = np.empty(iterations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        selected = rng.choice(groups, size=groups.size, replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        sampled = ious[indices]
        mean_samples[iteration] = float(np.mean(sampled))
        rate_samples[iteration] = float(np.mean(sampled >= 0.5))

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    return {
        "iterations": iterations,
        "seed": seed,
        "group_count": int(groups.size),
        "intervals": {
            "mean_iou": interval(mean_samples),
            "iou_at_0_5_rate": interval(rate_samples),
        },
    }


def evaluate_final_sequence_candidate(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    candidate_name: str,
    dataset_root: Path,
    expected_dataset_report_sha256: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
    checkpoint_loader: Callable[[Path], dict[str, Any]] | None = None,
    predictor: Callable[
        [SequenceSplit, dict[str, Any], str, int], tuple[np.ndarray, np.ndarray]
    ]
    | None = None,
) -> FinalSequenceEvaluationResult:
    """Evaluate one lock-bound sequence candidate without selecting on test."""

    _require(batch_size > 0, "Batch size must be positive")
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Final sequence output exists: {output_dir}")
    context = load_final_benchmark_context(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    candidate = _candidate(context, candidate_name)
    checkpoint_path, report_path, threshold_path, threshold = _verify_candidate_artifacts(
        candidate, candidate_name
    )
    iterations, seed = _metrics_contract(context)
    loader = checkpoint_loader or _load_checkpoint
    checkpoint = loader(checkpoint_path)
    _require(checkpoint.get("schema_version") == CHECKPOINT_SCHEMA, "Bad checkpoint schema")
    training_asset_index_sha256 = _verify_training_data_lineage(
        context,
        checkpoint=checkpoint,
        candidate_report_path=report_path,
        threshold_path=threshold_path,
    )

    dataset_root = dataset_root.resolve()
    dataset_report_path = dataset_root / "dataset_report.json"
    _require(
        dataset_report_path.is_file()
        and sha256_file(dataset_report_path) == expected_dataset_report_sha256,
        "Internal-test Dataset report drift",
    )
    dataset_report = _read_json(dataset_report_path, "internal-test Dataset report")
    _require(dataset_report.get("splits") == ["internal_test"], "Dataset is not test-only")
    shared = _object(context.candidate_lock.get("shared"), "shared candidate inputs")
    normalization = _object(shared.get("scalar_normalization"), "scalar normalization")
    _require(
        dataset_report.get("frozen_scalar_normalization_sha256")
        == normalization.get("sha256"),
        "Internal-test normalization is not the frozen train fit",
    )
    split = load_sequence_split(dataset_root, "internal_test")
    sealed = _object(context.protocol.get("sealed_split"), "sealed split")
    _require(
        len(split.asset_ids) == sealed.get("expected_assets"),
        "Internal-test asset count drift",
    )
    _require(
        len(set(split.group_ids)) == sealed.get("expected_source_groups"),
        "Internal-test source-group count drift",
    )

    _require(
        training_asset_index_sha256 != split.asset_index_sha256,
        "Training and internal-test asset indices must be distinct",
    )
    spec = SequenceModelSpec(**_object(checkpoint.get("model_spec"), "model spec"))
    expected_modality = (
        "sequence_metadata" if candidate_name == "sequence_peak_net_metadata" else "sequence"
    )
    _require(spec.modality == expected_modality, "Checkpoint modality drift")
    _require(spec.input_points == split.signals.shape[1], "Signal width drift")
    _require(spec.scalar_features == split.scalar_features.shape[1], "Scalar width drift")

    predict = predictor or _predict
    probabilities, boundaries = predict(split, checkpoint, device, batch_size)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    boundaries = np.asarray(boundaries, dtype=np.float64)
    count = len(split.asset_ids)
    _require(probabilities.shape == (count,), "Sequence probability shape drift")
    _require(boundaries.shape == (count, 2), "Sequence boundary shape drift")
    _require(np.isfinite(probabilities).all(), "Non-finite sequence probabilities")
    _require(np.isfinite(boundaries).all(), "Non-finite sequence boundaries")
    _require(
        bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0))),
        "Sequence probabilities leave [0, 1]",
    )

    labels = split.targets[:, 0].astype(np.int64)
    positive = labels == 1
    localization = boundary_metrics(
        split.targets[:, 1:],
        boundaries,
        positive,
        roi_width_minutes=split.roi_width_minutes,
    )
    positive_ious = _interval_ious(split.targets[positive, 1:], boundaries[positive])
    localization.update(
        {
            "mean_iou": float(np.mean(positive_ious)),
            "iou_at_0_5_rate": float(np.mean(positive_ious >= 0.5)),
        }
    )
    primary_metrics = binary_metrics(labels, probabilities, threshold=threshold)
    fixed_metrics = binary_metrics(labels, probabilities, threshold=0.5)
    classification_bootstrap = grouped_bootstrap_binary(
        labels,
        probabilities,
        split.group_ids,
        threshold=threshold,
        iterations=iterations,
        seed=seed,
    )
    positive_groups = np.asarray(split.group_ids, dtype=object)[positive]
    localization_bootstrap = _grouped_bootstrap_localization(
        positive_ious,
        positive_groups,
        iterations=iterations,
        seed=seed,
    )
    predictions = [
        {
            "schema_version": FINAL_SEQUENCE_PREDICTION_SCHEMA,
            "row": row,
            "asset_id": split.asset_ids[row],
            "group_id": split.group_ids[row],
            "presence_probability": float(probabilities[row]),
            "predicted_peak_present": bool(probabilities[row] >= threshold),
            "start_normalized": float(boundaries[row, 0]),
            "end_normalized": float(boundaries[row, 1]),
            "target_peak_present": bool(labels[row]),
            "target_start_normalized": (
                float(split.targets[row, 1]) if labels[row] else None
            ),
            "target_end_normalized": (
                float(split.targets[row, 2]) if labels[row] else None
            ),
        }
        for row in range(count)
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        predictions_path = staging / "sequence_predictions.jsonl"
        _write_jsonl(predictions_path, predictions)
        final_report_path = staging / "sequence_evaluation_report.json"
        _write_json(
            final_report_path,
            {
                "schema_version": FINAL_SEQUENCE_EVALUATION_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "candidate_name": candidate_name,
                "modality": expected_modality,
                "counts": {
                    "assets": count,
                    "positive_assets": int(np.sum(positive)),
                    "negative_assets": int(np.sum(~positive)),
                    "internal_test_source_groups": len(set(split.group_ids)),
                },
                "classification": {
                    "validation_frozen_threshold": primary_metrics,
                    "secondary_fixed_threshold_0_5": fixed_metrics,
                },
                "localization": localization,
                "source_grouped_bootstrap_95": {
                    "classification": classification_bootstrap,
                    "localization": localization_bootstrap,
                },
                "evaluation": {
                    "bootstrap_iterations": iterations,
                    "seed": seed,
                    "bootstrap_unit": "complete_source_mzml_group",
                    "threshold_selection_performed": False,
                    "primary_threshold": threshold,
                    "primary_threshold_selected_on_split": "validation",
                },
                "provenance": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": context.access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                    "internal_test_asset_index_sha256": split.asset_index_sha256,
                    "training_dataset_report_sha256": _object(
                        context.candidate_lock.get("shared"), "shared candidate inputs"
                    ).get("dataset_report_sha256"),
                    "training_asset_index_sha256": training_asset_index_sha256,
                    "candidate_report_sha256": candidate.get("report_sha256"),
                    "candidate_report_path": str(report_path),
                    "candidate_manifest_sha256": candidate.get("manifest_sha256"),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "frozen_threshold_sha256": _object(
                        candidate.get("frozen_threshold"), "threshold"
                    ).get("sha256"),
                },
                "artifacts": {
                    "predictions": {
                        "path": predictions_path.name,
                        "sha256": sha256_file(predictions_path),
                        "records": count,
                    }
                },
                "development_comparison_eligible": False,
                "final_benchmark_eligible": True,
                "internal_test_accessed": True,
            },
        )
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (predictions_path, final_report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    report = output_dir / "sequence_evaluation_report.json"
    return FinalSequenceEvaluationResult(
        output_dir=output_dir,
        report_path=report,
        report_sha256=sha256_file(report),
        manifest_path=output_dir / "artifact_manifest.sha256",
        predictions_path=output_dir / "sequence_predictions.jsonl",
        assets=count,
        source_groups=len(set(split.group_ids)),
        candidate_name=candidate_name,
        access_id=context.access_id,
    )
