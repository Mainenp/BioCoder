"""Unified COCO, localization, and image-level metrics for ChromPeakFormer."""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_science.baselines.metrics import binary_metrics, select_threshold
from multimodal_science.data.manifest import sha256_file


@dataclass(frozen=True)
class DetectorEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    validation_images: int
    predictions: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("COCO predictions must be a JSON list")
    return value


def _box_iou_xywh(first: list[float], second: list[float]) -> float:
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def _official_coco_metrics(validation_coco_path: Path, predictions_path: Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "Official COCO AP requires pycocotools in the detector environment"
        ) from exc
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        # pycocotools opens JSON paths with the platform default encoding. Load
        # through BioCoder's explicit UTF-8 boundary so Unicode sample names are
        # portable across Linux servers and Windows verification hosts.
        ground_truth = COCO()
        ground_truth.dataset = _read_object(validation_coco_path)
        ground_truth.createIndex()
        detections = ground_truth.loadRes(_read_predictions(predictions_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats.tolist()
    return {
        "ap_50_95": float(stats[0]),
        "ap_50": float(stats[1]),
        "ap_75": float(stats[2]),
        "ar_at_1": float(stats[6]),
        "ar_at_10": float(stats[7]),
        "ar_at_100": float(stats[8]),
    }


def _validate_and_group(
    ground_truth: dict[str, Any], predictions: list[dict[str, Any]]
) -> tuple[
    dict[int, list[list[float]]],
    dict[int, list[dict[str, Any]]],
    dict[int, tuple[int, int]],
]:
    images = ground_truth.get("images")
    annotations = ground_truth.get("annotations")
    _require(isinstance(images, list) and bool(images), "Validation COCO images are missing")
    _require(isinstance(annotations, list), "Validation COCO annotations are missing")
    dimensions = {int(image["id"]): (int(image["width"]), int(image["height"])) for image in images}
    _require(len(dimensions) == len(images), "Validation COCO contains duplicate image IDs")
    truth_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        _require(image_id in dimensions, f"Annotation references unknown image: {image_id}")
        _require(int(annotation["category_id"]) == 0, "Only peak category id 0 is supported")
        box = [float(value) for value in annotation["bbox"]]
        _require(len(box) == 4 and all(math.isfinite(value) for value in box), "Invalid GT bbox")
        _require(box[2] > 0.0 and box[3] > 0.0, "GT bbox must have positive area")
        truth_by_image[image_id].append(box)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, prediction in enumerate(predictions):
        _require(isinstance(prediction, dict), f"Prediction {position} must be an object")
        image_id = int(prediction["image_id"])
        _require(image_id in dimensions, f"Prediction references unknown image: {image_id}")
        _require(int(prediction["category_id"]) == 0, "Only peak category id 0 is supported")
        score = float(prediction["score"])
        box = [float(value) for value in prediction["bbox"]]
        _require(math.isfinite(score) and 0.0 <= score <= 1.0, "Invalid prediction score")
        _require(len(box) == 4 and all(math.isfinite(value) for value in box), "Invalid bbox")
        _require(box[2] > 0.0 and box[3] > 0.0, "Predicted bbox must have positive area")
        predictions_by_image[image_id].append({"score": score, "bbox": box})
    for image_predictions in predictions_by_image.values():
        image_predictions.sort(key=lambda item: -float(item["score"]))
    return dict(truth_by_image), dict(predictions_by_image), dimensions


def _localization_metrics(
    *,
    truth_by_image: dict[int, list[list[float]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    score_threshold: float,
) -> dict[str, int | float]:
    best_ious = []
    predicted_positive = 0
    for image_id in image_ids:
        truth = truth_by_image.get(image_id, [])
        if not truth:
            continue
        eligible = [
            prediction
            for prediction in predictions_by_image.get(image_id, [])
            if float(prediction["score"]) >= score_threshold
        ]
        predicted_positive += int(bool(eligible))
        best_ious.append(
            max(
                (
                    _box_iou_xywh(prediction["bbox"], target)
                    for prediction in eligible
                    for target in truth
                ),
                default=0.0,
            )
        )
    values = np.asarray(best_ious, dtype=np.float64)
    _require(bool(values.size), "Localization metrics require positive validation images")
    return {
        "positive_images": int(values.size),
        "predicted_positive_images": predicted_positive,
        "mean_best_iou": float(np.mean(values)),
        "median_best_iou": float(np.median(values)),
        "iou_at_0_5_rate": float(np.mean(values >= 0.5)),
        "iou_at_0_75_rate": float(np.mean(values >= 0.75)),
    }


def evaluate_detector_predictions(
    *,
    validation_coco_path: Path,
    predictions_path: Path,
    inference_report_path: Path,
    expected_inference_report_sha256: str,
    detector_dataset_report_path: Path,
    expected_dataset_report_sha256: str,
    output_dir: Path,
    fixed_score_threshold: float = 0.5,
) -> DetectorEvaluationResult:
    """Evaluate a complete validation prediction artifact without test-set access."""

    required = (
        validation_coco_path,
        predictions_path,
        inference_report_path,
        detector_dataset_report_path,
    )
    for path in required:
        _require(path.is_file(), f"Required evaluation input is missing: {path}")
    _require(
        sha256_file(inference_report_path) == expected_inference_report_sha256,
        "Inference report SHA-256 mismatch",
    )
    _require(
        sha256_file(detector_dataset_report_path) == expected_dataset_report_sha256,
        "Detector dataset report SHA-256 mismatch",
    )
    inference_report = _read_object(inference_report_path)
    dataset_report = _read_object(detector_dataset_report_path)
    _require(inference_report.get("model_family") == "ChromPeakFormer", "Model family mismatch")
    _require(
        inference_report.get("complete_validation_coverage") is True,
        "Inference does not cover the full validation split",
    )
    _require(dataset_report.get("quality_gate_passed") is True, "Dataset quality gate failed")
    _require(
        inference_report.get("detector_dataset_report_sha256") == expected_dataset_report_sha256,
        "Inference and evaluation dataset reports differ",
    )
    validation_coco_sha256 = sha256_file(validation_coco_path)
    _require(
        inference_report.get("validation_coco_sha256") == validation_coco_sha256,
        "Inference and evaluation validation COCO files differ",
    )
    _require(
        dataset_report.get("prepared_coco_sha256", {}).get("validation") == validation_coco_sha256,
        "Validation COCO does not match dataset provenance",
    )
    _require(
        inference_report.get("predictions_sha256") == sha256_file(predictions_path),
        "Predictions hash does not match inference provenance",
    )
    _require(not output_dir.exists(), f"Evaluation output already exists: {output_dir}")
    _require(
        math.isfinite(fixed_score_threshold) and 0.0 <= fixed_score_threshold <= 1.0,
        "Score threshold must be in [0, 1]",
    )

    ground_truth = _read_object(validation_coco_path)
    predictions = _read_predictions(predictions_path)
    inference_counts = inference_report.get("counts")
    _require(isinstance(inference_counts, dict), "Inference counts are missing")
    _require(
        int(inference_counts.get("predictions", -1)) == len(predictions),
        "Inference prediction count mismatch",
    )
    truth_by_image, predictions_by_image, dimensions = _validate_and_group(
        ground_truth, predictions
    )
    image_ids = sorted(dimensions)
    _require(
        int(inference_counts.get("images", -1)) == len(image_ids),
        "Inference image count mismatch",
    )
    labels = np.asarray([int(bool(truth_by_image.get(image_id))) for image_id in image_ids])
    scores = np.asarray(
        [
            max(
                (float(item["score"]) for item in predictions_by_image.get(image_id, [])),
                default=0.0,
            )
            for image_id in image_ids
        ]
    )
    threshold_selection = select_threshold(labels, scores, objective="macro_f1")
    selected_threshold = float(threshold_selection["threshold"])
    official_coco = _official_coco_metrics(validation_coco_path, predictions_path)
    report = {
        "schema_version": "chrompeak-detector-evaluation-v1",
        "model_family": "ChromPeakFormer",
        "evaluation_split": "validation",
        "counts": {
            "images": len(image_ids),
            "positive_images": int(np.sum(labels)),
            "negative_images": int(np.sum(labels == 0)),
            "ground_truth_boxes": sum(len(value) for value in truth_by_image.values()),
            "predictions": len(predictions),
            "source_groups": int(dataset_report["splits"]["validation"]["source_groups"]),
        },
        "coco": official_coco,
        "classification": {
            "score_definition": "maximum peak-query probability per image",
            "fixed_threshold": binary_metrics(labels, scores, threshold=fixed_score_threshold),
            "validation_selected_threshold": {
                **threshold_selection,
                "metrics": binary_metrics(labels, scores, threshold=selected_threshold),
            },
        },
        "localization": {
            "fixed_threshold": {
                "score_threshold": fixed_score_threshold,
                **_localization_metrics(
                    truth_by_image=truth_by_image,
                    predictions_by_image=predictions_by_image,
                    image_ids=image_ids,
                    score_threshold=fixed_score_threshold,
                ),
            },
            "validation_selected_threshold": {
                "score_threshold": selected_threshold,
                **_localization_metrics(
                    truth_by_image=truth_by_image,
                    predictions_by_image=predictions_by_image,
                    image_ids=image_ids,
                    score_threshold=selected_threshold,
                ),
            },
        },
        "provenance": {
            "validation_coco_sha256": sha256_file(validation_coco_path),
            "predictions_sha256": sha256_file(predictions_path),
            "inference_report_sha256": expected_inference_report_sha256,
            "detector_dataset_report_sha256": expected_dataset_report_sha256,
        },
        "development_comparison_eligible": inference_report.get("development_comparison_candidate")
        is True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    output_dir = output_dir.resolve()
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        report_path = staging / "detector_evaluation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8"
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    final_report = output_dir / "detector_evaluation_report.json"
    return DetectorEvaluationResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        validation_images=len(image_ids),
        predictions=len(predictions),
    )
