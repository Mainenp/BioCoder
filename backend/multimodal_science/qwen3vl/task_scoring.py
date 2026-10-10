"""Shared, side-effect-free scoring for Qwen3-VL scientific tasks.

The validation evaluator and the v2 RL reward path must agree on JSON and task
schema validity.  This module deliberately performs no file access so training
can reuse those rules without gaining an answer- or split-loading surface.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


_GROUNDING_SIZE = re.compile(r"for a ([1-9][0-9]*)x([1-9][0-9]*) image")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def strict_json_object(response: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse exactly one JSON value and report syntax validity separately."""

    try:
        parsed = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (parsed, True) if isinstance(parsed, dict) else (None, True)


def is_number(value: Any) -> bool:
    """Return whether ``value`` is a finite JSON-style number (not a bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def grounding_dimensions(
    prompt: dict[str, Any], manifest: dict[str, Any]
) -> tuple[int, int]:
    """Resolve grounding image dimensions exactly as the evaluator does."""

    width = manifest.get("image_width")
    height = manifest.get("image_height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    ):
        return width, height
    conversations = prompt.get("conversations")
    _require(
        isinstance(conversations, list) and len(conversations) == 1,
        "Validation prompt must contain exactly one human message",
    )
    message = _object(conversations[0], "validation conversation")
    value = message.get("value")
    _require(message.get("from") == "human" and isinstance(value, str), "Invalid prompt")
    match = _GROUNDING_SIZE.search(value)
    _require(match is not None, "Grounding prompt does not declare image dimensions")
    return int(match.group(1)), int(match.group(2))


def valid_bbox(value: Any, width: int, height: int) -> list[float] | None:
    """Validate and normalize a four-coordinate image-space bounding box."""

    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(is_number(coordinate) for coordinate in value)
    ):
        return None
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return [x1, y1, x2, y2]


def bbox_iou(first: list[float], second: list[float]) -> float:
    """Compute ordinary two-dimensional bounding-box IoU."""

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def score_task_response(
    *,
    task: str,
    prompt: dict[str, Any],
    manifest: dict[str, Any],
    expected_response: str,
    predicted_response: str,
    bilingual: bool = False,
) -> dict[str, Any]:
    """Score one response while preserving the established evaluator schema."""

    expected, expected_json = strict_json_object(expected_response)
    _require(expected_json and expected is not None, "Builder answer is not a JSON object")
    parsed, valid_json = strict_json_object(predicted_response)
    result: dict[str, Any] = {
        "valid_json": valid_json,
        "schema_valid": False,
        "exact_match": False,
    }

    if task in {"peak_presence", "peak_presence_metadata"}:
        _require(
            set(expected) == {"peak_present"}
            and isinstance(expected["peak_present"], bool),
            "Invalid presence answer",
        )
        schema_valid = (
            parsed is not None
            and set(parsed) == {"peak_present"}
            and isinstance(parsed["peak_present"], bool)
        )
        truth = bool(expected["peak_present"])
        predicted = bool(parsed["peak_present"]) if schema_valid else None
        result.update(
            {
                "schema_valid": schema_valid,
                "exact_match": schema_valid and predicted == truth,
                "target_peak_present": truth,
                "predicted_peak_present": predicted,
                "classification_score": (
                    float(predicted) if schema_valid else float(not truth)
                ),
            }
        )
        return result

    if task == "peak_grounding":
        width, height = grounding_dimensions(prompt, manifest)
        expected_box = valid_bbox(expected.get("bbox_2d"), width, height)
        _require(
            set(expected) == {"bbox_2d"} and expected_box is not None,
            "Invalid grounding answer",
        )
        predicted_box = (
            valid_bbox(parsed.get("bbox_2d"), width, height)
            if parsed is not None and set(parsed) == {"bbox_2d"}
            else None
        )
        schema_valid = predicted_box is not None
        iou = bbox_iou(expected_box, predicted_box) if predicted_box is not None else 0.0
        result.update(
            {
                "schema_valid": schema_valid,
                "exact_match": schema_valid and predicted_box == expected_box,
                "image_width": width,
                "image_height": height,
                "expected_bbox_2d": expected_box,
                "predicted_bbox_2d": predicted_box,
                "bbox_iou": iou,
                "iou_at_0_5": iou >= 0.5,
                "x_boundary_absolute_error_pixels": (
                    [
                        abs(predicted_box[0] - expected_box[0]),
                        abs(predicted_box[2] - expected_box[2]),
                    ]
                    if predicted_box is not None
                    else None
                ),
                "full_height": (
                    predicted_box is not None
                    and predicted_box[1] == 0.0
                    and predicted_box[3] == float(height)
                ),
            }
        )
        return result

    _require(
        set(expected) == {"qc_state", "reason"}
        and all(isinstance(expected[key], str) for key in expected),
        "Invalid scientific QC answer",
    )
    schema_valid = (
        parsed is not None
        and set(parsed) == {"qc_state", "reason"}
        and all(isinstance(parsed[key], str) and bool(parsed[key]) for key in parsed)
    )
    result.update(
        {
            "schema_valid": schema_valid,
            "exact_match": schema_valid and parsed == expected,
            "qc_state_correct": (
                schema_valid and parsed["qc_state"] == expected["qc_state"]
            ),
            "reason_correct": schema_valid and parsed["reason"] == expected["reason"],
        }
    )
    if bilingual:
        result["predicted_qc_state"] = parsed["qc_state"] if schema_valid else None
        result["predicted_reason"] = parsed["reason"] if schema_valid else None
    return result


# Backward-compatible names used by audit and sealed-benchmark modules.  They
# remain aliases, so every caller now executes the one shared implementation.
_strict_json_object = strict_json_object
_is_number = is_number
_grounding_dimensions = grounding_dimensions
_valid_bbox = valid_bbox
_bbox_iou = bbox_iou

