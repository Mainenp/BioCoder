"""Run the frozen ChromPeakFormer candidate on the sealed internal test split."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multimodal_science.baselines.metrics import binary_metrics
from multimodal_science.chrompeakformer.detector_evaluation import (
    _box_iou_xywh,
    _localization_metrics,
    _official_coco_metrics,
    _validate_and_group,
)
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.final_benchmark_data import (
    FINAL_DETECTOR_DATASET_SCHEMA,
)
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FinalBenchmarkRuntimeContext,
    load_final_benchmark_context,
)


FINAL_DETECTOR_EVALUATION_SCHEMA = "chrompeak-detector-final-evaluation-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FinalDetectorEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    predictions_path: Path
    images: int
    predictions: int
    source_groups: int
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


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _source_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    _require(bool(files), f"No Python source under: {source_root}")
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _manifest_entries(root: Path, expected_sha256: str) -> dict[str, str]:
    root = root.resolve()
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), "Detector candidate manifest is missing")
    _require(sha256_file(manifest) == expected_sha256, "Detector manifest drift")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed detector manifest line {line_number}")
        digest, relative = parts
        _require(
            bool(_HEX_64.fullmatch(digest)),
            f"Bad detector manifest digest at line {line_number}",
        )
        relative = relative.strip().lstrip("*").replace("\\", "/")
        relative_path = Path(relative)
        _require(
            relative != "" and relative_path != Path("."),
            f"Missing detector manifest path at line {line_number}",
        )
        _require(
            not relative_path.is_absolute(),
            f"Detector manifest path must be relative at line {line_number}",
        )
        path = (root / relative).resolve()
        try:
            canonical = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Detector manifest path escapes its root") from error
        _require(
            canonical not in entries,
            f"Duplicate normalized detector manifest path: {canonical}",
        )
        _require(
            path.is_file() and sha256_file(path) == digest,
            f"Artifact drift: {relative}",
        )
        entries[canonical] = digest
    _require(bool(entries), "Detector candidate manifest is empty")
    return entries


def _candidate_context(
    context: FinalBenchmarkRuntimeContext,
) -> tuple[dict[str, Any], int, int]:
    candidate = _object(
        _object(context.candidate_lock.get("models"), "candidate models").get(
            "chrompeakformer"
        ),
        "ChromPeakFormer candidate",
    )
    metrics = context.metrics_lock
    policy = _object(metrics.get("threshold_policy"), "threshold policy")
    _require(
        policy.get("internal_test_threshold_selection_forbidden") is True,
        "Test threshold selection is unlocked",
    )
    _require(metrics.get("post_access_model_selection_forbidden") is True, "Selection unlocked")
    _require(metrics.get("post_access_threshold_tuning_forbidden") is True, "Tuning unlocked")
    uncertainty = _object(metrics.get("uncertainty"), "uncertainty protocol")
    iterations = uncertainty.get("bootstrap_iterations")
    seed = uncertainty.get("seed")
    _require(isinstance(iterations, int) and iterations >= 10_000, "Bad bootstrap count")
    _require(isinstance(seed, int), "Bad bootstrap seed")
    return candidate, iterations, seed


def _verify_candidate(
    candidate: dict[str, Any], source_root: Path
) -> tuple[Path, Path, float, str]:
    root = Path(str(candidate.get("root") or "")).resolve()
    _require(root.is_dir(), f"Detector candidate root not found: {root}")
    entries = _manifest_entries(root, str(candidate.get("manifest_sha256") or ""))
    checkpoint = root / "training" / "checkpoint.pth"
    training_report = root / "training" / "detector_training_report.json"
    for path, expected, label in (
        (checkpoint, candidate.get("checkpoint_sha256"), "checkpoint"),
        (training_report, candidate.get("training_report_sha256"), "training report"),
    ):
        relative = path.relative_to(root).as_posix()
        _require(path.is_file() and sha256_file(path) == expected, f"Detector {label} drift")
        _require(entries.get(relative) == expected, f"Detector {label} is not manifest-bound")
    report = _read_json(training_report, "detector training report")
    _require(report.get("schema_version") == "chrompeak-detector-training-v1", "Bad report")
    _require(report.get("model_family") == "ChromPeakFormer", "Detector family drift")
    _require(report.get("development_comparison_eligible") is True, "Detector ineligible")
    _require(report.get("internal_test_accessed") is False, "Detector already saw test")
    source_sha256 = _source_tree_sha256(source_root)
    _require(report.get("source_tree_sha256") == source_sha256, "Private source-tree drift")
    threshold = float(candidate.get("primary_threshold"))
    _require(math.isfinite(threshold) and 0.0 <= threshold <= 1.0, "Bad detector threshold")
    _require(
        candidate.get("primary_threshold_selected_on_split") == "validation",
        "Detector threshold was not validation-frozen",
    )
    return checkpoint, training_report, threshold, source_sha256


@contextmanager
def _checkpoint_only_backbone_initialization(
    torchvision_models: Any,
    backbone_name: str,
) -> Iterator[None]:
    """Prevent pretrained downloads while reconstructing a complete checkpoint."""

    _require(bool(backbone_name), "Detector checkpoint does not name its backbone")
    original = getattr(torchvision_models, backbone_name, None)
    _require(callable(original), f"Unsupported detector backbone: {backbone_name}")

    def build_without_pretrained_download(*args: Any, **kwargs: Any) -> Any:
        # The private source uses torchvision's legacy ``pretrained=True`` API.
        # The frozen checkpoint contains the complete model state, so downloading
        # an initialization here is both unnecessary and an unfrozen dependency.
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False
        if "weights" in kwargs:
            kwargs["weights"] = None
        return original(*args, **kwargs)

    setattr(torchvision_models, backbone_name, build_without_pretrained_download)
    try:
        yield
    finally:
        setattr(torchvision_models, backbone_name, original)


def _default_predictor(
    source_root: Path,
    checkpoint_path: Path,
    coco_root: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    import torch
    import torchvision.models as torchvision_models
    from torch.utils.data import DataLoader

    _require(device_name in {"cpu", "cuda"}, f"Unsupported device: {device_name}")
    if device_name == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    sys.path.insert(0, str(source_root))
    try:
        from framework.datasets import build_dataset  # type: ignore[import-not-found]
        from framework.util.misc import collate_fn  # type: ignore[import-not-found]
        from models import build_model  # type: ignore[import-not-found]

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(isinstance(checkpoint, dict), "Detector checkpoint is not a mapping")
        _require("model" in checkpoint and "args" in checkpoint, "Checkpoint is incomplete")
        arguments = checkpoint["args"]
        arguments.coco_path = str(coco_root)
        arguments.device = device_name
        arguments.distributed = False
        arguments.world_size = 1
        arguments.rank = 0
        arguments.gpu = 0
        arguments.num_workers = num_workers
        backbone_name = getattr(arguments, "backbone", "")
        _require(isinstance(backbone_name, str), "Detector backbone name is invalid")
        with _checkpoint_only_backbone_initialization(
            torchvision_models,
            backbone_name,
        ):
            model, _, postprocessors = build_model(arguments)
        loaded = model.load_state_dict(checkpoint["model"], strict=True)
        _require(
            not loaded.missing_keys and not loaded.unexpected_keys,
            "Detector checkpoint load was not strict",
        )
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        validation = build_dataset(image_set="val", args=arguments)
        loader = DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=device_name == "cuda",
            collate_fn=collate_fn,
        )
        predictions: list[dict[str, Any]] = []
        processed: list[int] = []
        with torch.inference_mode():
            for samples, targets in loader:
                samples = samples.to(device)
                with torch.autocast(
                    device_type=device_name,
                    dtype=torch.float16,
                    enabled=bool(amp and device_name == "cuda"),
                ):
                    outputs = model(samples)
                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                ).to(device)
                results = postprocessors["bbox"](outputs, original_sizes)
                for target, result in zip(targets, results, strict=True):
                    image_id = int(target["image_id"].item())
                    processed.append(image_id)
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
        return predictions, processed
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)


def _grouped_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    best_ious: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    _require(unique.size >= 2, "Detector bootstrap requires two source groups")
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    names = (
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "false_positive_rate",
        "mean_iou",
        "iou_at_0_5_rate",
    )
    values = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        classification = binary_metrics(labels[indices], scores[indices], threshold=threshold)
        for name in names[:4]:
            values[name].append(float(classification[name]))
        positive_ious = best_ious[indices][labels[indices] == 1]
        if positive_ious.size:
            values["mean_iou"].append(float(np.mean(positive_ious)))
            values["iou_at_0_5_rate"].append(float(np.mean(positive_ious >= 0.5)))
    intervals = {}
    for name, samples in values.items():
        _require(bool(samples), f"No bootstrap values for {name}")
        intervals[name] = {
            "lower_95": float(np.quantile(samples, 0.025)),
            "median": float(np.quantile(samples, 0.5)),
            "upper_95": float(np.quantile(samples, 0.975)),
        }
    return {
        "iterations": iterations,
        "seed": seed,
        "group_count": int(unique.size),
        "intervals": intervals,
    }


def evaluate_final_detector_candidate(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    source_root: Path,
    detector_data_root: Path,
    expected_detector_report_sha256: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 2,
    amp: bool = True,
    predictor: Callable[
        [Path, Path, Path, str, int, int, bool],
        tuple[list[dict[str, Any]], list[int]],
    ]
    | None = None,
    coco_metric_function: Callable[[Path, Path], dict[str, float]] | None = None,
) -> FinalDetectorEvaluationResult:
    """Infer and evaluate the exact lock-bound detector without test tuning."""

    _require(batch_size > 0 and num_workers >= 0, "Invalid detector loader settings")
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Final detector output exists: {output_dir}")
    context = load_final_benchmark_context(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    candidate, iterations, seed = _candidate_context(context)
    source_root = source_root.resolve()
    _require((source_root / "models").is_dir(), "Private detector source lacks models/")
    _require((source_root / "framework").is_dir(), "Private detector source lacks framework/")
    checkpoint, training_report, threshold, source_sha = _verify_candidate(
        candidate, source_root
    )

    detector_data_root = detector_data_root.resolve()
    data_report_path = detector_data_root / "detector_dataset_report.json"
    _require(
        data_report_path.is_file()
        and sha256_file(data_report_path) == expected_detector_report_sha256,
        "Final detector data report drift",
    )
    data_report = _read_json(data_report_path, "final detector data report")
    _require(
        data_report.get("schema_version") == FINAL_DETECTOR_DATASET_SCHEMA,
        "Bad final detector data schema",
    )
    source = _object(data_report.get("source"), "detector data source")
    _require(source.get("protocol_sha256") == expected_protocol_sha256, "Protocol drift")
    _require(source.get("access_id") == context.access_id, "Access drift")
    split = _object(data_report.get("split"), "detector split")
    sealed = _object(context.protocol.get("sealed_split"), "sealed split")
    _require(split.get("name") == "internal_test", "Detector data is not test-only")
    _require(split.get("assets") == sealed.get("expected_assets"), "Asset count drift")
    _require(
        split.get("source_groups") == sealed.get("expected_source_groups"),
        "Source-group count drift",
    )
    artifact = _object(
        _object(data_report.get("artifacts"), "detector artifacts").get(
            "internal_test_coco"
        ),
        "internal-test COCO",
    )
    coco_path = (detector_data_root / str(artifact.get("path") or "")).resolve()
    try:
        coco_path.relative_to(detector_data_root)
    except ValueError as error:
        raise ValueError("Internal-test COCO escapes its root") from error
    _require(
        coco_path.is_file() and sha256_file(coco_path) == artifact.get("sha256"),
        "COCO drift",
    )
    ground_truth = _read_json(coco_path, "internal-test COCO")
    images = ground_truth.get("images")
    _require(isinstance(images, list) and len(images) == split.get("assets"), "COCO count drift")
    image_ids = [int(image["id"]) for image in images]
    _require(len(image_ids) == len(set(image_ids)), "Duplicate COCO image IDs")
    groups_by_image = {int(image["id"]): str(image["group_id"]) for image in images}
    _require(
        len(set(groups_by_image.values())) == split.get("source_groups"),
        "COCO group count drift",
    )

    infer = predictor or _default_predictor
    predictions, processed = infer(
        source_root,
        checkpoint,
        detector_data_root / "coco",
        device,
        batch_size,
        num_workers,
        amp,
    )
    _require(processed == image_ids, "Detector did not process every test image in order")
    truth_by_image, predictions_by_image, dimensions = _validate_and_group(
        ground_truth, predictions
    )
    _require(sorted(dimensions) == sorted(image_ids), "Detector dimensions drift")
    labels = np.asarray(
        [int(bool(truth_by_image.get(image_id))) for image_id in image_ids],
        dtype=np.int64,
    )
    scores = np.asarray(
        [
            max(
                (float(row["score"]) for row in predictions_by_image.get(image_id, [])),
                default=0.0,
            )
            for image_id in image_ids
        ],
        dtype=np.float64,
    )
    best_ious = np.zeros(len(image_ids), dtype=np.float64)
    for index, image_id in enumerate(image_ids):
        if labels[index] == 0:
            continue
        eligible = [
            row
            for row in predictions_by_image.get(image_id, [])
            if float(row["score"]) >= threshold
        ]
        best_ious[index] = max(
            (
                _box_iou_xywh(row["bbox"], truth)
                for row in eligible
                for truth in truth_by_image[image_id]
            ),
            default=0.0,
        )
    groups = np.asarray([groups_by_image[image_id] for image_id in image_ids], dtype=object)
    bootstrap = _grouped_bootstrap(
        labels,
        scores,
        best_ious,
        groups,
        threshold=threshold,
        iterations=iterations,
        seed=seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        predictions_path = staging / "coco_predictions.json"
        _write_json(predictions_path, predictions, compact=True)
        coco_metrics = (coco_metric_function or _official_coco_metrics)(
            coco_path, predictions_path
        )
        report_path = staging / "detector_evaluation_report.json"
        _write_json(
            report_path,
            {
                "schema_version": FINAL_DETECTOR_EVALUATION_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "candidate_name": "chrompeakformer",
                "counts": {
                    "images": len(image_ids),
                    "positive_images": int(np.sum(labels)),
                    "negative_images": int(np.sum(labels == 0)),
                    "ground_truth_boxes": sum(len(rows) for rows in truth_by_image.values()),
                    "predictions": len(predictions),
                    "internal_test_source_groups": len(set(groups.tolist())),
                },
                "coco": coco_metrics,
                "classification": {
                    "score_definition": "maximum peak-query probability per image",
                    "validation_frozen_threshold": binary_metrics(
                        labels, scores, threshold=threshold
                    ),
                    "secondary_fixed_threshold_0_5": binary_metrics(
                        labels, scores, threshold=0.5
                    ),
                },
                "localization": {
                    "validation_frozen_threshold": {
                        "score_threshold": threshold,
                        **_localization_metrics(
                            truth_by_image=truth_by_image,
                            predictions_by_image=predictions_by_image,
                            image_ids=image_ids,
                            score_threshold=threshold,
                        ),
                    }
                },
                "source_grouped_bootstrap_95": bootstrap,
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
                    "detector_data_report_sha256": expected_detector_report_sha256,
                    "coco_sha256": sha256_file(coco_path),
                    "predictions_sha256": sha256_file(predictions_path),
                    "candidate_manifest_sha256": candidate.get("manifest_sha256"),
                    "training_report_path": str(training_report),
                    "training_report_sha256": sha256_file(training_report),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "source_tree_sha256": source_sha,
                },
                "artifacts": {
                    "predictions": {
                        "path": predictions_path.name,
                        "sha256": sha256_file(predictions_path),
                        "records": len(predictions),
                    }
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
                for path in (predictions_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    report = output_dir / "detector_evaluation_report.json"
    return FinalDetectorEvaluationResult(
        output_dir=output_dir,
        report_path=report,
        report_sha256=sha256_file(report),
        manifest_path=output_dir / "artifact_manifest.sha256",
        predictions_path=output_dir / "coco_predictions.json",
        images=len(image_ids),
        predictions=len(predictions),
        source_groups=len(set(groups.tolist())),
        access_id=context.access_id,
    )
