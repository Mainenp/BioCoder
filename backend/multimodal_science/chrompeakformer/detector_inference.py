"""Run an authorized external ChromPeakFormer detector as a standard COCO producer."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file


@dataclass(frozen=True)
class DetectorInferenceResult:
    output_dir: Path
    predictions_path: Path
    report_path: Path
    report_sha256: str
    prediction_count: int
    image_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    _require(bool(files), f"No Python source files found under: {source_root}")
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def run_detector_inference(
    *,
    source_root: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    training_report_path: Path,
    expected_training_report_sha256: str,
    detector_dataset_root: Path,
    detector_dataset_report_path: Path,
    expected_dataset_report_sha256: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 2,
    amp: bool = True,
) -> DetectorInferenceResult:
    """Load a trusted detector checkpoint and emit validation COCO detections."""

    source_root = source_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    training_report_path = training_report_path.resolve()
    detector_dataset_root = detector_dataset_root.resolve()
    detector_dataset_report_path = detector_dataset_report_path.resolve()
    output_dir = output_dir.resolve()
    _require((source_root / "models").is_dir(), "Detector source root is missing models/")
    _require((source_root / "framework").is_dir(), "Detector source root is missing framework/")
    _require(checkpoint_path.is_file(), f"Checkpoint does not exist: {checkpoint_path}")
    _require(
        sha256_file(checkpoint_path) == expected_checkpoint_sha256,
        "Checkpoint SHA-256 does not match the expected immutable artifact",
    )
    _require(training_report_path.is_file(), "Detector training report is missing")
    _require(
        sha256_file(training_report_path) == expected_training_report_sha256,
        "Detector training report SHA-256 mismatch",
    )
    _require(detector_dataset_report_path.is_file(), "Detector dataset report is missing")
    _require(
        sha256_file(detector_dataset_report_path) == expected_dataset_report_sha256,
        "Detector dataset report SHA-256 mismatch",
    )
    dataset_report = json.loads(detector_dataset_report_path.read_text(encoding="utf-8"))
    _require(dataset_report.get("quality_gate_passed") is True, "Detector dataset gate failed")
    _require(
        (detector_dataset_root / "val" / "val_coco.json").is_file(),
        "Validation COCO is missing",
    )
    validation_coco_path = detector_dataset_root / "val" / "val_coco.json"
    validation_coco_sha256 = sha256_file(validation_coco_path)
    _require(
        dataset_report.get("prepared_coco_sha256", {}).get("validation") == validation_coco_sha256,
        "Prepared validation COCO SHA-256 mismatch",
    )
    source_tree_sha256 = _source_tree_sha256(source_root)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    _require(training_report.get("model_family") == "ChromPeakFormer", "Model family mismatch")
    _require(
        training_report.get("checkpoint_sha256") == expected_checkpoint_sha256,
        "Training report/checkpoint SHA-256 mismatch",
    )
    _require(
        training_report.get("source_tree_sha256") == source_tree_sha256,
        "Training and inference source trees differ",
    )
    _require(
        training_report.get("detector_dataset_report_sha256") == expected_dataset_report_sha256,
        "Training and inference dataset reports differ",
    )
    _require(not output_dir.exists(), f"Detector inference output already exists: {output_dir}")
    _require(batch_size > 0, "Batch size must be positive")
    _require(num_workers >= 0, "Worker count cannot be negative")
    _require(device in {"cpu", "cuda"}, f"Unsupported device: {device}")

    import torch
    from torch.utils.data import DataLoader

    if device == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    sys.path.insert(0, str(source_root))
    try:
        from framework.datasets import build_dataset  # type: ignore[import-not-found]
        from framework.util.misc import collate_fn  # type: ignore[import-not-found]
        from models import build_model  # type: ignore[import-not-found]

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(isinstance(checkpoint, dict), "Detector checkpoint must be a mapping")
        _require("model" in checkpoint and "args" in checkpoint, "Checkpoint lacks model or args")
        arguments = checkpoint["args"]
        arguments.coco_path = str(detector_dataset_root)
        arguments.device = device
        arguments.distributed = False
        arguments.world_size = 1
        arguments.rank = 0
        arguments.gpu = 0
        arguments.num_workers = num_workers
        model, _, postprocessors = build_model(arguments)
        load_report = model.load_state_dict(checkpoint["model"], strict=True)
        _require(
            not load_report.missing_keys and not load_report.unexpected_keys,
            "Checkpoint did not load strictly into its recorded architecture",
        )
        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()
        validation = build_dataset(image_set="val", args=arguments)
        loader = DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=device == "cuda",
            collate_fn=collate_fn,
        )
        predictions: list[dict[str, Any]] = []
        autocast_enabled = bool(amp and device == "cuda")
        with torch.inference_mode():
            for samples, targets in loader:
                samples = samples.to(torch_device)
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    outputs = model(samples)
                original_sizes = torch.stack([target["orig_size"] for target in targets], dim=0).to(
                    torch_device
                )
                results = postprocessors["bbox"](outputs, original_sizes)
                for target, result in zip(targets, results, strict=True):
                    image_id = int(target["image_id"].item())
                    for score, label, box in zip(
                        result["scores"].detach().cpu().tolist(),
                        result["labels"].detach().cpu().tolist(),
                        result["boxes"].detach().cpu().tolist(),
                        strict=True,
                    ):
                        x1, y1, x2, y2 = (float(value) for value in box)
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)

    validation_coco = json.loads(validation_coco_path.read_text(encoding="utf-8"))
    image_count = len(validation_coco["images"])
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        predictions_path = staging / "coco_predictions.json"
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        report_path = staging / "detector_inference_report.json"
        report = {
            "schema_version": "chrompeak-detector-inference-v1",
            "model_family": "ChromPeakFormer",
            "source_tree_sha256": source_tree_sha256,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "training_report_sha256": expected_training_report_sha256,
            "detector_dataset_report_sha256": expected_dataset_report_sha256,
            "validation_coco_sha256": validation_coco_sha256,
            "predictions_sha256": sha256_file(predictions_path),
            "counts": {"images": image_count, "predictions": len(predictions)},
            "runtime": {
                "device": device,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "amp": bool(amp),
            },
            "complete_validation_coverage": len(
                {int(prediction["image_id"]) for prediction in predictions}
            )
            == image_count,
            "development_comparison_candidate": training_report.get(
                "development_comparison_eligible"
            )
            is True,
            "internal_test_accessed": False,
        }
        _require(report["complete_validation_coverage"], "Inference missed validation images")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "artifact_manifest.sha256").write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in (predictions_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_predictions = output_dir / "coco_predictions.json"
    final_report = output_dir / "detector_inference_report.json"
    return DetectorInferenceResult(
        output_dir=output_dir,
        predictions_path=final_predictions,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        prediction_count=len(predictions),
        image_count=image_count,
    )
