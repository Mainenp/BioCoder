"""Audited training launcher for an authorized external ChromPeakFormer source tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.chrompeakformer.detector_inference import _source_tree_sha256
from multimodal_science.data.manifest import sha256_file


_DISTRIBUTED_LAUNCHER_ENVIRONMENT_VARIABLES = frozenset(
    {
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PMI_RANK",
        "PMI_SIZE",
        "RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_PROCID",
        "WORLD_SIZE",
    }
)


@dataclass(frozen=True)
class DetectorTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    report_path: Path
    report_sha256: str
    final_epoch: int
    best_observed_epoch: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_history(path: Path) -> list[dict[str, Any]]:
    history = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            history.append(value)
    _require(bool(history), "Detector training history is empty")
    return history


def _coco_summary(record: dict[str, Any]) -> dict[str, float]:
    values = record.get("test_coco_eval_bbox")
    _require(isinstance(values, list) and len(values) >= 9, "Training log lacks COCO bbox metrics")
    return {
        "ap_50_95": float(values[0]),
        "ap_50": float(values[1]),
        "ap_75": float(values[2]),
        "ar_at_1": float(values[6]),
        "ar_at_10": float(values[7]),
        "ar_at_100": float(values[8]),
    }


def _training_mean_box_width(train_coco_path: Path) -> float:
    coco = _read_json(train_coco_path)
    images = coco.get("images")
    annotations = coco.get("annotations")
    _require(isinstance(images, list) and bool(images), "Training COCO images are missing")
    _require(isinstance(annotations, list) and bool(annotations), "Training COCO boxes are missing")
    width_by_image = {int(image["id"]): float(image["width"]) for image in images}
    normalized_widths = []
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        _require(image_id in width_by_image, "Training annotation references an unknown image")
        box = annotation.get("bbox")
        _require(isinstance(box, list) and len(box) == 4, "Training bbox is invalid")
        width = float(box[2])
        _require(0.0 < width <= width_by_image[image_id], "Training bbox width is invalid")
        normalized_widths.append(width / width_by_image[image_id])
    return sum(normalized_widths) / len(normalized_widths)


def _single_process_environment(source_root: Path) -> tuple[dict[str, str], list[str]]:
    """Build an environment that cannot be mistaken for a distributed launcher."""

    environment = os.environ.copy()
    removed_variables = sorted(
        key for key in _DISTRIBUTED_LAUNCHER_ENVIRONMENT_VARIABLES if key in environment
    )
    for key in removed_variables:
        environment.pop(key)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), environment.get("PYTHONPATH", "")) if value
    )
    return environment, removed_variables


def run_detector_training(
    *,
    source_root: Path,
    source_config_path: Path,
    detector_dataset_root: Path,
    detector_dataset_report_path: Path,
    expected_dataset_report_sha256: str,
    output_dir: Path,
    device: str = "cuda",
    num_workers: int = 2,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int = 17,
    smoke_test: bool = False,
    resume_path: Path | None = None,
    expected_resume_sha256: str | None = None,
) -> DetectorTrainingResult:
    """Run the existing detector while pinning data, source, and weight provenance."""

    source_root = source_root.resolve()
    source_config_path = source_config_path.resolve()
    detector_dataset_root = detector_dataset_root.resolve()
    detector_dataset_report_path = detector_dataset_report_path.resolve()
    output_dir = output_dir.resolve()
    _require((source_root / "train.py").is_file(), "Detector source root is missing train.py")
    _require(source_config_path.is_file(), "Detector source config is missing")
    _require(
        (detector_dataset_root / "train" / "train_coco.json").is_file(),
        "Prepared training COCO is missing",
    )
    _require(
        (detector_dataset_root / "val" / "val_coco.json").is_file(),
        "Prepared validation COCO is missing",
    )
    _require(
        sha256_file(detector_dataset_report_path) == expected_dataset_report_sha256,
        "Detector dataset report SHA-256 mismatch",
    )
    dataset_report = _read_json(detector_dataset_report_path)
    _require(dataset_report.get("quality_gate_passed") is True, "Detector dataset gate failed")
    prepared_hashes = dataset_report.get("prepared_coco_sha256")
    _require(isinstance(prepared_hashes, dict), "Prepared COCO provenance is missing")
    train_coco_path = detector_dataset_root / "train" / "train_coco.json"
    validation_coco_path = detector_dataset_root / "val" / "val_coco.json"
    _require(
        prepared_hashes.get("train") == sha256_file(train_coco_path),
        "Prepared training COCO SHA-256 mismatch",
    )
    _require(
        prepared_hashes.get("validation") == sha256_file(validation_coco_path),
        "Prepared validation COCO SHA-256 mismatch",
    )
    _require(not output_dir.exists(), f"Detector training output already exists: {output_dir}")
    _require(device in {"cpu", "cuda"}, f"Unsupported device: {device}")
    _require(num_workers >= 0, "Worker count cannot be negative")
    _require(epochs is None or epochs > 0, "Epoch count must be positive")
    _require(batch_size is None or batch_size > 0, "Batch size must be positive")
    _require((resume_path is None) == (expected_resume_sha256 is None), "Resume path/hash mismatch")

    source_config = _read_json(source_config_path)
    _require(
        int(source_config.get("dec_layers", 0)) >= 3
        and int(source_config.get("num_fdr_bins", 0)) > 1,
        "Source config does not describe the boundary-refinement ChromPeakFormer variant",
    )
    train_mean_box_width = _training_mean_box_width(train_coco_path)
    source_tree_sha256 = _source_tree_sha256(source_root)
    effective_config = {
        **source_config,
        "coco_path": str(detector_dataset_root),
        "output_dir": str(output_dir),
        "device": device,
        "num_workers": num_workers,
        "seed": seed,
        "resume": None,
        "start_epoch": 0,
        "reset_optimizer": False,
        "amp": device == "cuda",
        "tf32": device == "cuda",
        "cudnn_benchmark": device == "cuda",
        "pw_ciou_mean_width": train_mean_box_width,
    }
    resume_sha256 = None
    if resume_path is not None:
        resume_path = resume_path.resolve()
        _require(resume_path.is_file(), f"Resume checkpoint is missing: {resume_path}")
        resume_sha256 = sha256_file(resume_path)
        _require(resume_sha256 == expected_resume_sha256, "Resume checkpoint SHA-256 mismatch")
        effective_config["resume"] = str(resume_path)
        effective_config["reset_optimizer"] = True
    if epochs is not None:
        effective_config["epochs"] = epochs
        effective_config["lr_drop"] = max(1, round(epochs * 2 / 3))
    if batch_size is not None:
        effective_config["batch_size"] = batch_size

    output_dir.mkdir(parents=True)
    effective_path = output_dir / "effective_source_config.json"
    effective_path.write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    environment, removed_launcher_variables = _single_process_environment(source_root)
    contract_path = output_dir / "training_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-detector-training-contract-v1",
                "model_family": "ChromPeakFormer",
                "source_tree_sha256": source_tree_sha256,
                "source_config_sha256": sha256_file(source_config_path),
                "effective_config_sha256": sha256_file(effective_path),
                "detector_dataset_report_sha256": expected_dataset_report_sha256,
                "resume_checkpoint_sha256": resume_sha256,
                "train_only_mean_box_width": train_mean_box_width,
                "execution_mode": "single_process",
                "distributed_launcher_environment_removed": removed_launcher_variables,
                "smoke_test": smoke_test,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [sys.executable, "train.py", "--config", str(effective_path)]
    try:
        subprocess.run(command, cwd=source_root, env=environment, check=True)
    except subprocess.CalledProcessError as exc:
        failure_path = output_dir / "training_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "chrompeak-detector-training-failure-v1",
                    "model_family": "ChromPeakFormer",
                    "return_code": exc.returncode,
                    "training_contract_sha256": sha256_file(contract_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    checkpoint_path = output_dir / "checkpoint.pth"
    history_path = output_dir / "log.txt"
    _require(checkpoint_path.is_file(), "Training completed without checkpoint.pth")
    _require(history_path.is_file(), "Training completed without log.txt")
    history = _read_history(history_path)
    best_record = max(history, key=lambda record: float(record["test_coco_eval_bbox"][0]))
    final_record = history[-1]
    report_path = output_dir / "detector_training_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-detector-training-v1",
                "model_family": "ChromPeakFormer",
                "training_contract_sha256": sha256_file(contract_path),
                "source_tree_sha256": source_tree_sha256,
                "source_config_sha256": sha256_file(source_config_path),
                "effective_config_sha256": sha256_file(effective_path),
                "detector_dataset_report_sha256": expected_dataset_report_sha256,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "history_sha256": sha256_file(history_path),
                "epochs_completed": len(history),
                "final_epoch": int(final_record["epoch"]) + 1,
                "best_observed_epoch": int(best_record["epoch"]) + 1,
                "best_observed_validation_coco": _coco_summary(best_record),
                "final_checkpoint_validation_coco": _coco_summary(final_record),
                "checkpoint_selection": "final_epoch",
                "execution_mode": "single_process",
                "run_scope": "smoke" if smoke_test else "full_development",
                "smoke_test": smoke_test,
                "development_comparison_eligible": not smoke_test,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "artifact_manifest.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in (
                checkpoint_path,
                history_path,
                effective_path,
                contract_path,
                report_path,
            )
        ),
        encoding="utf-8",
    )
    return DetectorTrainingResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        final_epoch=int(final_record["epoch"]) + 1,
        best_observed_epoch=int(best_record["epoch"]) + 1,
    )
