"""Self-supervised morphology pretraining for the Qwen XIC sensor projector."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.auxiliary_pretraining_data import (
    AUXILIARY_PRETRAINING_DATASET_SCHEMA,
)
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)


AUXILIARY_PRETRAINING_REPORT_SCHEMA = "chrompeak-xic-auxiliary-pretraining-v1"


@dataclass(frozen=True)
class AuxiliaryPretrainingSettings:
    epochs: int = 20
    max_steps: int | None = None
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    temperature: float = 0.1
    projection_size: int = 256
    noise_std: float = 0.02
    mask_fraction: float = 0.10
    max_shift_points: int = 4
    log_steps: int = 25
    seed: int = 17
    hidden_size: int = 2560


@dataclass(frozen=True)
class AuxiliaryPretrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    manifest_sha256: str
    projector_path: Path
    global_steps: int
    development_training_complete: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _verify_manifest(root: Path, expected_sha256: str) -> None:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), "Auxiliary Dataset manifest is missing")
    _require(sha256_file(manifest) == expected_sha256, "Auxiliary Dataset manifest drift")
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        _require(len(parts) == 2, f"Malformed Dataset manifest line {line_number}")
        digest, relative = parts
        relative_path = Path(relative)
        _require(not relative_path.is_absolute(), "Dataset manifest path must be relative")
        artifact = (root / relative_path).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError("Dataset manifest path escapes its root") from exc
        _require(artifact.is_file(), f"Dataset artifact missing: {relative}")
        _require(sha256_file(artifact) == digest, f"Dataset artifact drift: {relative}")


def _validate_settings(settings: AuxiliaryPretrainingSettings) -> None:
    _require(settings.epochs >= 1, "epochs must be positive")
    if settings.max_steps is not None:
        _require(settings.max_steps >= 1, "max_steps must be positive")
    _require(settings.batch_size >= 2, "batch_size must be at least two")
    _require(
        math.isfinite(settings.learning_rate) and settings.learning_rate > 0.0,
        "learning_rate must be positive and finite",
    )
    _require(
        math.isfinite(settings.weight_decay) and settings.weight_decay >= 0.0,
        "weight_decay must be finite and nonnegative",
    )
    _require(
        math.isfinite(settings.temperature) and settings.temperature > 0.0,
        "temperature must be positive and finite",
    )
    _require(settings.projection_size >= 32, "projection_size is too small")
    _require(0.0 <= settings.noise_std <= 0.25, "noise_std is out of range")
    _require(0.0 <= settings.mask_fraction < 0.5, "mask_fraction is out of range")
    _require(settings.max_shift_points >= 0, "max_shift_points must be nonnegative")
    _require(settings.log_steps >= 1, "log_steps must be positive")
    _require(settings.hidden_size >= 64, "hidden_size is too small")


def _morphology_tokens(projector: Any, signals: Any, sensor_tokens: int) -> Any:
    encoded = projector.encoder(signals.unsqueeze(1))
    points = int(encoded.shape[-1])
    _require(points % sensor_tokens == 0, "Encoded signal width is not token aligned")
    pooled = encoded.reshape(
        encoded.shape[0], encoded.shape[1], sensor_tokens, points // sensor_tokens
    ).mean(dim=-1)
    return projector.projector(pooled.transpose(1, 2))


def _augment(signals: Any, *, generator: Any, settings: AuxiliaryPretrainingSettings) -> Any:
    import torch

    augmented = signals.clone()
    batch, points = augmented.shape
    if settings.max_shift_points:
        shifts = torch.randint(
            -settings.max_shift_points,
            settings.max_shift_points + 1,
            (batch,),
            generator=generator,
            device="cpu",
        ).tolist()
        augmented = torch.stack(
            [torch.roll(row, int(shift), dims=0) for row, shift in zip(augmented, shifts)]
        )
    scales = 0.8 + 0.4 * torch.rand(
        (batch, 1), generator=generator, device="cpu"
    ).to(device=augmented.device, dtype=augmented.dtype)
    augmented = augmented * scales
    if settings.noise_std:
        noise = torch.randn(
            augmented.shape, generator=generator, device="cpu", dtype=torch.float32
        ).to(device=augmented.device, dtype=augmented.dtype)
        augmented = augmented + noise * settings.noise_std
    if settings.mask_fraction:
        keep = torch.rand(
            (batch, points), generator=generator, device="cpu"
        ).to(device=augmented.device) >= settings.mask_fraction
        augmented = augmented * keep.to(dtype=augmented.dtype)
    return augmented.clamp_(0.0, 1.0)


def run_auxiliary_pretraining(
    *,
    dataset_root: Path,
    dataset_report_sha256: str,
    dataset_manifest_sha256: str,
    output_dir: Path,
    code_revision: str,
    settings: AuxiliaryPretrainingSettings = AuxiliaryPretrainingSettings(),
) -> AuxiliaryPretrainingResult:
    """Pretrain signal morphology without labels, validation, or benchmark access."""

    _validate_settings(settings)
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import save_file
    from torch import nn

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    _require(dataset_root.is_dir(), f"Auxiliary Dataset root not found: {dataset_root}")
    _require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    _require(len(code_revision) == 40, "code_revision must be a full Git commit")
    _verify_manifest(dataset_root, dataset_manifest_sha256)
    report_path = dataset_root / "auxiliary_pretraining_dataset_report.json"
    report = _read_json(report_path, "auxiliary pretraining Dataset report")
    _require(sha256_file(report_path) == dataset_report_sha256, "Dataset report drift")
    _require(
        report.get("schema_version") == AUXILIARY_PRETRAINING_DATASET_SCHEMA,
        "Unsupported auxiliary Dataset schema",
    )
    _require(report.get("quality_gate_passed") is True, "Auxiliary Dataset gate failed")
    contracts = _object(report.get("contracts"), "auxiliary Dataset contracts")
    for name, expected in (
        ("auxiliary_unlabeled_train_only", True),
        ("labels_present", False),
        ("metrics_allowed", False),
        ("validation_opened", False),
        ("internal_test_accessed", False),
        ("benchmark_eligible", False),
        ("rt_axis_normalization_is_explicit", True),
        ("rt_axes_strictly_increasing_after_normalization", True),
    ):
        _require(contracts.get(name) is expected, f"Auxiliary Dataset contract failed: {name}")
    _require(
        contracts.get("duplicate_rt_intensity_reducer") == "maximum",
        "Auxiliary Dataset contract failed: duplicate_rt_intensity_reducer",
    )
    _require(report.get("development_training_eligible") is True, "Dataset is not trainable")
    _require(report.get("development_comparison_eligible") is False, "Dataset metrics leak")

    artifacts = _object(report.get("artifacts"), "auxiliary Dataset artifacts")
    signal_artifact = _object(artifacts.get("signals"), "auxiliary signals")
    signals_path = dataset_root / str(signal_artifact.get("path") or "")
    _require(sha256_file(signals_path) == signal_artifact.get("sha256"), "Signal drift")
    signals = np.load(signals_path, mmap_mode="r", allow_pickle=False)
    _require(
        signals.ndim == 2 and signals.shape[1] == int(report.get("target_points", -1)),
        "Unexpected auxiliary signal shape",
    )
    _require(signals.dtype == np.float32, "Auxiliary signals must be float32")
    _require(bool(np.isfinite(signals).all()), "Auxiliary signals contain non-finite values")
    _require(float(signals.min()) >= 0.0 and float(signals.max()) <= 1.0, "Bad normalization")
    available_indices = np.flatnonzero(np.max(signals, axis=1) > 0.0).astype(np.int64)
    _require(len(available_indices) >= settings.batch_size, "Too few nonconstant signals")

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SensorProjectorSpec(
        input_points=int(signals.shape[1]), hidden_size=settings.hidden_size
    )
    projector = build_sensor_projector(spec).to(device).train()
    projector.gate_logit.requires_grad_(False)
    for parameter in projector.availability_embedding.parameters():
        parameter.requires_grad_(False)
    projection_head = nn.Sequential(
        nn.Linear(settings.hidden_size, settings.projection_size),
        nn.GELU(),
        nn.Linear(settings.projection_size, settings.projection_size),
    ).to(device)
    trainable = [
        parameter
        for module in (projector, projection_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    batches_per_epoch = math.ceil(len(available_indices) / settings.batch_size)
    requested_steps = settings.epochs * batches_per_epoch
    total_steps = min(settings.max_steps or requested_steps, requested_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    augmentation_generator = torch.Generator(device="cpu")
    augmentation_generator.manual_seed(settings.seed + 1)
    history: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    global_step = 0
    first_gradient_norm: float | None = None
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(settings.epochs):
        epoch_generator = np.random.default_rng(settings.seed + epoch)
        order = epoch_generator.permutation(available_indices)
        for batch_start in range(0, len(order), settings.batch_size):
            if global_step >= total_steps:
                break
            batch_indices = order[batch_start : batch_start + settings.batch_size]
            if len(batch_indices) < 2:
                continue
            batch = torch.from_numpy(np.array(signals[batch_indices], copy=True)).to(
                device=device, dtype=torch.float32
            )
            first = _augment(batch, generator=augmentation_generator, settings=settings)
            second = _augment(batch, generator=augmentation_generator, settings=settings)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                first_tokens = _morphology_tokens(projector, first, spec.sensor_tokens)
                second_tokens = _morphology_tokens(projector, second, spec.sensor_tokens)
                first_embedding = functional.normalize(
                    projection_head(first_tokens.mean(dim=1)).float(), dim=-1
                )
                second_embedding = functional.normalize(
                    projection_head(second_tokens.mean(dim=1)).float(), dim=-1
                )
                logits = first_embedding @ second_embedding.transpose(0, 1)
                logits = logits / settings.temperature
                labels = torch.arange(len(batch_indices), device=device)
                loss = 0.5 * (
                    functional.cross_entropy(logits, labels)
                    + functional.cross_entropy(logits.transpose(0, 1), labels)
                )
            loss.backward()
            squared_norm = sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in trainable
                if parameter.grad is not None
            )
            gradient_norm = math.sqrt(squared_norm)
            _require(math.isfinite(gradient_norm) and gradient_norm > 0.0, "No gradient")
            if first_gradient_norm is None:
                first_gradient_norm = gradient_norm
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            record = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "batch_records": len(batch_indices),
                "loss": float(loss.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
            }
            history.append(record)
            if global_step % settings.log_steps == 0 or global_step == total_steps:
                print(
                    f"[auxiliary step {global_step}/{total_steps}] "
                    f"loss={record['loss']:.6f} lr={record['learning_rate']:.3e}",
                    flush=True,
                )
        if global_step >= total_steps:
            break

    _require(global_step == total_steps, "Auxiliary pretraining stopped early")
    development_complete = settings.max_steps is None
    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=staging_parent))
    try:
        projector_path = staging / "sensor_projector.safetensors"
        save_file(
            {key: value.detach().cpu() for key, value in projector.state_dict().items()},
            str(projector_path),
        )
        head_path = staging / "contrastive_head.safetensors"
        save_file(
            {key: value.detach().cpu() for key, value in projection_head.state_dict().items()},
            str(head_path),
        )
        history_path = staging / "history.jsonl"
        history_path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                for record in history
            ),
            encoding="utf-8",
        )
        output_report_path = staging / "auxiliary_pretraining_report.json"
        output_report = {
            "schema_version": AUXILIARY_PRETRAINING_REPORT_SCHEMA,
            "code_revision": code_revision,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.monotonic() - wall_start,
            "sources": {
                "dataset_report_sha256": dataset_report_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "signals_sha256": signal_artifact["sha256"],
            },
            "model": {
                "sensor_projector": spec.as_dict(),
                "objective_projection_size": settings.projection_size,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "persisted_projector_sha256": sha256_file(projector_path),
            },
            "training": {
                **asdict(settings),
                "available_training_records": len(available_indices),
                "excluded_constant_records": int(signals.shape[0] - len(available_indices)),
                "optimizer_updates": global_step,
                "requested_full_updates": requested_steps,
                "first_gradient_norm": first_gradient_norm,
                "objective": "symmetric_augmented_xic_info_nce",
                "gate_and_availability_embedding_frozen": True,
            },
            "runtime": {
                "device": str(device),
                "cuda_version": torch.version.cuda,
                "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "contracts": {
                "unlabeled_auxiliary_only": True,
                "validation_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
                "metrics_allowed": False,
                "base_qwen_loaded": False,
                "vision_tower_loaded": False,
                "image_xic_independence_claimed": False,
                "source_views_claimed_independent": False,
                "projector_weights_compatible_with_formal_fusion": True,
                "seed_controlled": True,
                "bitwise_determinism_claimed": False,
            },
            "claim_limits": [
                "Training loss is an optimization diagnostic, not a scientific metric",
                "ROI images and XIC arrays are derived views of the same chromatographic trace",
                "This stage pretrains XIC morphology only and establishes no validation gain",
            ],
            "development_training_complete": development_complete,
            "development_comparison_eligible": False,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }
        _write_json(output_report_path, output_report)
        manifest_path = staging / "artifact_manifest.sha256"
        artifact_paths = (projector_path, head_path, history_path, output_report_path)
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in artifact_paths
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AuxiliaryPretrainingResult(
        output_dir=output_dir,
        report_path=output_dir / output_report_path.name,
        report_sha256=sha256_file(output_dir / output_report_path.name),
        manifest_path=output_dir / manifest_path.name,
        manifest_sha256=sha256_file(output_dir / manifest_path.name),
        projector_path=output_dir / projector_path.name,
        global_steps=global_step,
        development_training_complete=development_complete,
    )


def load_verified_pretrained_projector(
    root: Path,
    *,
    report_sha256: str,
    manifest_sha256: str,
    expected_spec: SensorProjectorSpec,
) -> tuple[Path, dict[str, Any]]:
    """Verify a complete auxiliary run before formal fusion consumes its projector."""

    root = root.resolve()
    _require(root.is_dir(), f"Auxiliary pretraining root not found: {root}")
    _verify_manifest(root, manifest_sha256)
    report_path = root / "auxiliary_pretraining_report.json"
    report = _read_json(report_path, "auxiliary pretraining report")
    _require(sha256_file(report_path) == report_sha256, "Auxiliary report drift")
    _require(
        report.get("schema_version") == AUXILIARY_PRETRAINING_REPORT_SCHEMA,
        "Unsupported auxiliary pretraining report schema",
    )
    _require(report.get("development_training_complete") is True, "Incomplete pretraining")
    _require(report.get("development_comparison_eligible") is False, "Bad pretraining scope")
    _require(report.get("final_benchmark_eligible") is False, "Bad benchmark scope")
    contracts = _object(report.get("contracts"), "auxiliary pretraining contracts")
    for name, expected in (
        ("unlabeled_auxiliary_only", True),
        ("validation_opened", False),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
        ("metrics_allowed", False),
        ("projector_weights_compatible_with_formal_fusion", True),
    ):
        _require(contracts.get(name) is expected, f"Auxiliary pretraining contract failed: {name}")
    model = _object(report.get("model"), "auxiliary pretraining model")
    _require(model.get("sensor_projector") == expected_spec.as_dict(), "Projector spec drift")
    projector_path = root / "sensor_projector.safetensors"
    _require(projector_path.is_file(), "Pretrained projector weights are missing")
    _require(
        sha256_file(projector_path) == model.get("persisted_projector_sha256"),
        "Pretrained projector drift",
    )
    return projector_path, report
