"""Dependency-light binary and chromatographic-boundary metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _binary_inputs(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Binary labels and scores must be aligned one-dimensional arrays")
    if labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary labels must be a non-empty 0/1 array")
    if not np.isfinite(scores).all():
        raise ValueError("Binary scores must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Binary scores must be probabilities in [0, 1]")
    return labels, scores


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return float("nan")
    ranks = _average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    position = 0
    while position < labels.size:
        end = position + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[position]:
            end += 1
        group = sorted_labels[position:end]
        true_positives += int(np.sum(group == 1))
        false_positives += int(np.sum(group == 0))
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        position = end
    return float(average_precision)


def binary_metrics(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, int | float]:
    labels, scores = _binary_inputs(y_true, y_score)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    negative_precision = _divide(tn, tn + fn)
    positive_f1 = _divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _divide(2 * tn, 2 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _divide(tp * tn - fp * fn, denominator)
    return {
        "threshold": float(threshold),
        "samples": int(labels.size),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _divide(tp + tn, labels.size),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_precision,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "mcc": mcc,
        "auroc": _auroc(labels, scores),
        "auprc": _average_precision(labels, scores),
        "false_positive_rate": _divide(fp, fp + tn),
    }


def select_threshold(
    y_true: Any,
    y_score: Any,
    *,
    objective: str = "macro_f1",
) -> dict[str, float]:
    labels, scores = _binary_inputs(y_true, y_score)
    supported = {"macro_f1", "mcc", "balanced_accuracy"}
    if objective not in supported:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    unique_scores = np.unique(scores)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique_scores)))
    best_threshold = 0.5
    best_value = -float("inf")
    for threshold in candidates:
        predicted = scores >= threshold
        positive = labels == 1
        negative = ~positive
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & negative))
        tn = int(np.sum(~predicted & negative))
        fn = int(np.sum(~predicted & positive))
        recall = _divide(tp, tp + fn)
        specificity = _divide(tn, tn + fp)
        if objective == "macro_f1":
            positive_f1 = _divide(2 * tp, 2 * tp + fp + fn)
            negative_f1 = _divide(2 * tn, 2 * tn + fp + fn)
            value = (positive_f1 + negative_f1) / 2.0
        elif objective == "mcc":
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            value = _divide(tp * tn - fp * fn, denominator)
        else:
            value = (recall + specificity) / 2.0
        candidate_key = (value, -abs(float(threshold) - 0.5), -float(threshold))
        best_key = (best_value, -abs(best_threshold - 0.5), -best_threshold)
        if candidate_key > best_key:
            best_value = value
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "objective": objective, "value": best_value}


def boundary_metrics(
    true_boundaries: Any,
    predicted_boundaries: Any,
    positive_mask: Any,
    *,
    roi_width_minutes: Any | None = None,
) -> dict[str, int | float]:
    truth = np.asarray(true_boundaries, dtype=np.float64)
    predicted = np.asarray(predicted_boundaries, dtype=np.float64)
    mask = np.asarray(positive_mask, dtype=bool)
    if truth.ndim != 2 or truth.shape[1] != 2 or predicted.shape != truth.shape:
        raise ValueError("Boundary arrays must have aligned shape [N, 2]")
    if mask.ndim != 1 or mask.shape[0] != truth.shape[0]:
        raise ValueError("Positive mask must align with boundary arrays")
    if not np.any(mask):
        raise ValueError("Boundary metrics require at least one positive example")
    selected_truth = truth[mask]
    selected_predicted = predicted[mask]
    if not np.isfinite(selected_truth).all() or not np.isfinite(selected_predicted).all():
        raise ValueError("Positive boundaries must be finite")
    if not (
        (selected_truth[:, 0] >= 0.0)
        & (selected_truth[:, 0] < selected_truth[:, 1])
        & (selected_truth[:, 1] <= 1.0)
    ).all():
        raise ValueError("True positive boundaries must satisfy 0 <= start < end <= 1")
    valid_prediction = (
        (selected_predicted[:, 0] >= 0.0)
        & (selected_predicted[:, 1] <= 1.0)
        & (selected_predicted[:, 0] < selected_predicted[:, 1])
    )
    evaluated = np.clip(selected_predicted, 0.0, 1.0)
    start_error = np.abs(evaluated[:, 0] - selected_truth[:, 0])
    end_error = np.abs(evaluated[:, 1] - selected_truth[:, 1])
    width_error = np.abs(
        (evaluated[:, 1] - evaluated[:, 0])
        - (selected_truth[:, 1] - selected_truth[:, 0])
    )
    intersection = np.maximum(
        0.0,
        np.minimum(evaluated[:, 1], selected_truth[:, 1])
        - np.maximum(evaluated[:, 0], selected_truth[:, 0]),
    )
    union = np.maximum(evaluated[:, 1], selected_truth[:, 1]) - np.minimum(
        evaluated[:, 0], selected_truth[:, 0]
    )
    result: dict[str, int | float] = {
        "positive_samples": int(selected_truth.shape[0]),
        "valid_prediction_rate": float(np.mean(valid_prediction)),
        "start_mae_normalized": float(np.mean(start_error)),
        "end_mae_normalized": float(np.mean(end_error)),
        "boundary_mae_normalized": float(np.mean(np.concatenate([start_error, end_error]))),
        "width_mae_normalized": float(np.mean(width_error)),
        "mean_interval_iou": float(
            np.mean(
                np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0.0,
                )
            )
        ),
    }
    if roi_width_minutes is not None:
        widths = np.asarray(roi_width_minutes, dtype=np.float64)
        if widths.ndim != 1 or widths.shape[0] != truth.shape[0]:
            raise ValueError("ROI widths must align with boundary arrays")
        selected_widths_seconds = widths[mask] * 60.0
        result.update(
            {
                "start_mae_seconds": float(np.mean(start_error * selected_widths_seconds)),
                "end_mae_seconds": float(np.mean(end_error * selected_widths_seconds)),
                "boundary_mae_seconds": float(
                    np.mean(
                        np.concatenate(
                            [
                                start_error * selected_widths_seconds,
                                end_error * selected_widths_seconds,
                            ]
                        )
                    )
                ),
            }
        )
    return result


def grouped_bootstrap_binary(
    y_true: Any,
    y_score: Any,
    group_ids: Any,
    *,
    threshold: float,
    iterations: int = 1000,
    seed: int = 17,
) -> dict[str, dict[str, float] | int]:
    labels, scores = _binary_inputs(y_true, y_score)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.shape[0] != labels.shape[0]:
        raise ValueError("Group IDs must align with binary inputs")
    if iterations < 2:
        raise ValueError("Bootstrap iterations must be at least two")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Grouped bootstrap requires at least two groups")
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    metric_names = (
        "macro_f1",
        "mcc",
        "auroc",
        "auprc",
        "specificity",
        "recall",
        "false_positive_rate",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        selected_indices = np.concatenate([indices_by_group[group] for group in selected_groups])
        metrics = binary_metrics(
            labels[selected_indices], scores[selected_indices], threshold=threshold
        )
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        _require(bool(values), f"No finite bootstrap values for {name}")
        intervals[name] = {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return {
        "iterations": iterations,
        "seed": seed,
        "group_count": int(unique_groups.size),
        "intervals": intervals,
    }
