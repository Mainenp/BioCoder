"""Reproducible train/validation runner for the sequence baseline."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multimodal_science.baselines.dataset import SequenceSplit, load_sequence_split
from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
    select_threshold,
)
from multimodal_science.baselines.sequence_model import (
    SequenceModelSpec,
    build_sequence_peak_net,
)
from multimodal_science.data.manifest import sha256_file

REPORT_SCHEMA = "chrompeak-sequence-baseline-report-v1"
CHECKPOINT_SCHEMA = "chrompeak-sequence-baseline-checkpoint-v1"


@dataclass(frozen=True)
class SequenceTrainConfig:
    modality: str = "sequence"
    seed: int = 17
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    boundary_weight: float = 2.0
    dropout: float = 0.15
    patience: int = 8
    device: str = "cpu"
    amp: bool = True
    num_workers: int = 0
    bootstrap_iterations: int = 1000
    smoke_test: bool = False
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def validate(self) -> None:
        if self.modality not in {"sequence", "sequence_metadata"}:
            raise ValueError(f"Unsupported sequence modality: {self.modality}")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not math.isfinite(self.boundary_weight) or self.boundary_weight <= 0.0:
            raise ValueError("boundary_weight must be positive and finite")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.patience < 1 or self.num_workers < 0:
            raise ValueError("patience must be positive and num_workers nonnegative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.bootstrap_iterations < 2:
            raise ValueError("bootstrap_iterations must be at least two")
        limits = (self.max_train_samples, self.max_validation_samples)
        if any(limit is not None and limit < 2 for limit in limits):
            raise ValueError("Sample limits must be at least two")
        if any(limit is not None for limit in limits) and not self.smoke_test:
            raise ValueError("Sample limits are allowed only for an explicit smoke test")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SequenceTrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    checkpoint_path: Path
    best_epoch: int
    selected_threshold: float
    development_comparison_eligible: bool
    final_benchmark_eligible: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _subset_split(split: SequenceSplit, maximum: int | None) -> SequenceSplit:
    if maximum is None or maximum >= split.signals.shape[0]:
        return split
    indices = np.linspace(0, split.signals.shape[0] - 1, maximum, dtype=np.int64)
    return SequenceSplit(
        split=split.split,
        signals=split.signals[indices],
        scalar_features=split.scalar_features[indices],
        targets=split.targets[indices],
        asset_ids=tuple(split.asset_ids[index] for index in indices),
        group_ids=tuple(split.group_ids[index] for index in indices),
        roi_width_minutes=split.roi_width_minutes[indices],
        dataset_report_sha256=split.dataset_report_sha256,
        asset_index_sha256=split.asset_index_sha256,
    )


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _loss_components(
    torch: Any,
    functional: Any,
    presence_logits: Any,
    predicted_boundaries: Any,
    targets: Any,
    *,
    positive_weight: Any,
    boundary_weight: float,
) -> tuple[Any, Any, Any]:
    presence = targets[:, 0]
    classification = functional.binary_cross_entropy_with_logits(
        presence_logits,
        presence,
        pos_weight=positive_weight,
    )
    positive = presence > 0.5
    if bool(positive.any()):
        truth = targets[positive, 1:]
        predicted = predicted_boundaries[positive]
        regression = functional.smooth_l1_loss(predicted, truth)
        intersection = torch.clamp(
            torch.minimum(predicted[:, 1], truth[:, 1])
            - torch.maximum(predicted[:, 0], truth[:, 0]),
            min=0.0,
        )
        union = torch.maximum(predicted[:, 1], truth[:, 1]) - torch.minimum(
            predicted[:, 0], truth[:, 0]
        )
        overlap = 1.0 - torch.mean(intersection / torch.clamp(union, min=1e-7))
        boundary = regression + 0.5 * overlap
    else:
        boundary = presence_logits.sum() * 0.0
    total = classification + boundary_weight * boundary
    return total, classification, boundary


def _epoch(
    torch: Any,
    functional: Any,
    model: Any,
    loader: Any,
    device: Any,
    *,
    positive_weight: Any,
    boundary_weight: float,
    optimizer: Any | None,
    scaler: Any,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "classification_loss": 0.0, "boundary_loss": 0.0}
    samples = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for signals, scalars, targets in loader:
            signals = signals.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with precision_context:
                logits, boundaries = model(signals, scalars)
                loss, classification, boundary = _loss_components(
                    torch,
                    functional,
                    logits,
                    boundaries,
                    targets,
                    positive_weight=positive_weight,
                    boundary_weight=boundary_weight,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = int(signals.shape[0])
            samples += batch_size
            totals["loss"] += float(loss.detach()) * batch_size
            totals["classification_loss"] += float(classification.detach()) * batch_size
            totals["boundary_loss"] += float(boundary.detach()) * batch_size
    _require(samples > 0, "Training loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _predict(torch: Any, model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    boundaries = []
    model.eval()
    with torch.inference_mode():
        for signals, scalars, _ in loader:
            logits, predicted_boundaries = model(
                signals.to(device, non_blocking=True),
                scalars.to(device, non_blocking=True),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            boundaries.append(predicted_boundaries.cpu().numpy())
    return (
        np.concatenate(probabilities).astype(np.float64),
        np.concatenate(boundaries).astype(np.float64),
    )


def _prediction_records(
    split: SequenceSplit,
    probabilities: np.ndarray,
    boundaries: np.ndarray,
) -> list[dict[str, Any]]:
    _require(probabilities.shape == (len(split.asset_ids),), "Probability row mismatch")
    _require(boundaries.shape == (len(split.asset_ids), 2), "Boundary row mismatch")
    records = []
    for row in range(len(split.asset_ids)):
        target_present = bool(split.targets[row, 0])
        records.append(
            {
                "schema_version": "chrompeak-sequence-prediction-v1",
                "row": row,
                "asset_id": split.asset_ids[row],
                "group_id": split.group_ids[row],
                "presence_probability": float(probabilities[row]),
                "start_normalized": float(boundaries[row, 0]),
                "end_normalized": float(boundaries[row, 1]),
                "roi_width_minutes": float(split.roi_width_minutes[row]),
                "target_peak_present": target_present,
                "target_start_normalized": (
                    float(split.targets[row, 1]) if target_present else None
                ),
                "target_end_normalized": (
                    float(split.targets[row, 2]) if target_present else None
                ),
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_sequence_training(
    dataset_root: Path,
    output_dir: Path,
    config: SequenceTrainConfig,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SequenceTrainingResult:
    """Train, select on validation, and atomically publish a baseline run."""

    config.validate()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output directory already exists: {output_dir}")
    if config.device == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in {":4096:8", ":16:8"}:
            raise ValueError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, TensorDataset

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    train_full = load_sequence_split(dataset_root, "train")
    validation_full = load_sequence_split(dataset_root, "validation")
    _require(
        train_full.dataset_report_sha256 == validation_full.dataset_report_sha256,
        "Train and validation Dataset reports do not match",
    )
    _require(
        train_full.asset_index_sha256 == validation_full.asset_index_sha256,
        "Train and validation asset indices do not match",
    )
    train = _subset_split(train_full, config.max_train_samples)
    validation = _subset_split(validation_full, config.max_validation_samples)
    _require(len(set(train.group_ids) & set(validation.group_ids)) == 0, "Source-group leakage")

    def tensors(split: SequenceSplit) -> Any:
        return TensorDataset(
            torch.from_numpy(np.asarray(split.signals, dtype=np.float32)),
            torch.from_numpy(np.asarray(split.scalar_features, dtype=np.float32)),
            torch.from_numpy(np.asarray(split.targets, dtype=np.float32)),
        )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        tensors(train),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        tensors(validation),
        shuffle=False,
        **loader_options,
    )

    model_spec = SequenceModelSpec(
        input_points=int(train.signals.shape[1]),
        scalar_features=int(train.scalar_features.shape[1]),
        modality=config.modality,
        dropout=config.dropout,
    )
    model = build_sequence_peak_net(model_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.05,
    )
    positives = int(np.sum(train.targets[:, 0] == 1.0))
    negatives = int(train.targets.shape[0] - positives)
    _require(positives > 0 and negatives > 0, "Training requires both detection classes")
    positive_weight_value = negatives / positives
    positive_weight = torch.tensor(positive_weight_value, device=device, dtype=torch.float32)
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_losses = _epoch(
            torch,
            functional,
            model,
            train_loader,
            device,
            positive_weight=positive_weight,
            boundary_weight=config.boundary_weight,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_losses = _epoch(
            torch,
            functional,
            model,
            validation_loader,
            device,
            positive_weight=positive_weight,
            boundary_weight=config.boundary_weight,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        probabilities, _ = _predict(torch, model, validation_loader, device)
        fixed_metrics = binary_metrics(
            validation.targets[:, 0],
            probabilities,
            threshold=0.5,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_losses,
            "validation": validation_losses,
            "validation_macro_f1_at_0_5": fixed_metrics["macro_f1"],
            "validation_mcc_at_0_5": fixed_metrics["mcc"],
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if validation_losses["loss"] < best_validation_loss - 1e-8:
            best_validation_loss = validation_losses["loss"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    _require(best_state is not None and best_epoch > 0, "No valid model checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    probabilities, predicted_boundaries = _predict(
        torch,
        model,
        validation_loader,
        device,
    )

    threshold_selection = select_threshold(
        validation.targets[:, 0],
        probabilities,
        objective="macro_f1",
    )
    selected_threshold = float(threshold_selection["threshold"])
    fixed_metrics = binary_metrics(validation.targets[:, 0], probabilities, threshold=0.5)
    selected_metrics = binary_metrics(
        validation.targets[:, 0],
        probabilities,
        threshold=selected_threshold,
    )
    positive_mask = validation.targets[:, 0] == 1.0
    boundaries = boundary_metrics(
        validation.targets[:, 1:],
        predicted_boundaries,
        positive_mask,
        roi_width_minutes=validation.roi_width_minutes,
    )
    bootstrap = grouped_bootstrap_binary(
        validation.targets[:, 0],
        probabilities,
        validation.group_ids,
        threshold=selected_threshold,
        iterations=config.bootstrap_iterations,
        seed=config.seed + 10_000,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}-staging-",
    ) as staging_name:
        staging = Path(staging_name)
        config_path = staging / "config.json"
        checkpoint_path = staging / "best_model.pt"
        history_path = staging / "history.jsonl"
        predictions_path = staging / "validation_predictions.jsonl"
        threshold_path = staging / "frozen_threshold.json"
        _write_json(config_path, config.as_dict())
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "model_spec": model_spec.as_dict(),
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "dataset_report_sha256": train.dataset_report_sha256,
                "asset_index_sha256": train.asset_index_sha256,
            },
            checkpoint_path,
        )
        _write_jsonl(history_path, history)
        _write_jsonl(
            predictions_path,
            _prediction_records(validation, probabilities, predicted_boundaries),
        )
        _write_json(
            threshold_path,
            {
                "schema_version": "chrompeak-frozen-threshold-v1",
                "selected_on_split": "validation",
                "objective": threshold_selection["objective"],
                "threshold": selected_threshold,
                "dataset_report_sha256": train.dataset_report_sha256,
                "asset_index_sha256": train.asset_index_sha256,
                "internal_test_accessed": False,
            },
        )
        artifacts = {
            "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
            },
            "history": {"path": history_path.name, "sha256": sha256_file(history_path)},
            "validation_predictions": {
                "path": predictions_path.name,
                "sha256": sha256_file(predictions_path),
                "records": len(validation.asset_ids),
            },
            "frozen_threshold": {
                "path": threshold_path.name,
                "sha256": sha256_file(threshold_path),
            },
        }
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        report_path = staging / "scientific_report.json"
        _write_json(
            report_path,
            {
                "schema_version": REPORT_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_scope": (
                    "smoke_test" if config.smoke_test else "train_validation_baseline"
                ),
                "development_comparison_eligible": not config.smoke_test,
                "final_benchmark_eligible": False,
                "promotion_eligible": False,
                "internal_test_accessed": False,
                "evidence_gate": {
                    "scientific_gate_complete": False,
                    "missing_evidence": [
                        "sealed_internal_test_metrics",
                        "blank_stratified_false_positive_rate",
                        "quantification_metrics",
                        "declared_multimodal_ablations",
                    ],
                },
                "dataset": {
                    "dataset_report_sha256": train.dataset_report_sha256,
                    "asset_index_sha256": train.asset_index_sha256,
                    "original_train_samples": int(train_full.signals.shape[0]),
                    "original_validation_samples": int(validation_full.signals.shape[0]),
                    "used_train_samples": int(train.signals.shape[0]),
                    "used_validation_samples": int(validation.signals.shape[0]),
                    "train_source_groups": len(set(train.group_ids)),
                    "validation_source_groups": len(set(validation.group_ids)),
                },
                "code": {"git_commit": _git_commit()},
                "runtime": runtime,
                "config": config.as_dict(),
                "model": {
                    "spec": model_spec.as_dict(),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                },
                "training": {
                    "epochs_completed": len(history),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "positive_weight": positive_weight_value,
                    "early_stopped": len(history) < config.epochs,
                },
                "threshold_selection": threshold_selection,
                "validation": {
                    "fixed_threshold_0_5": fixed_metrics,
                    "selected_threshold": selected_metrics,
                    "positive_boundary_metrics": boundaries,
                    "source_grouped_bootstrap_95": bootstrap,
                },
                "artifacts": artifacts,
            },
        )
        staging.replace(output_dir)

    final_report = output_dir / "scientific_report.json"
    return SequenceTrainingResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        checkpoint_path=output_dir / "best_model.pt",
        best_epoch=best_epoch,
        selected_threshold=selected_threshold,
        development_comparison_eligible=not config.smoke_test,
        final_benchmark_eligible=False,
    )
