"""Pre-registered v2 GRPO rewards for the scientific task family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from multimodal_science.qwen3vl.task_scoring import score_task_response


GROUNDING_TASK = "peak_grounding"
PRESENCE_TASKS = frozenset({"peak_presence", "peak_presence_metadata"})
QC_TASK = "scientific_qc"
SUPPORTED_TASKS = PRESENCE_TASKS | {GROUNDING_TASK, QC_TASK}
INVALID_GROUNDING_REWARD = -1.5
INVALID_AUXILIARY_REWARD = -1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RLRewardResult:
    """One auditable reward decision."""

    task: str
    reward: float
    valid: bool
    terminated_with_eos: bool
    valid_json: bool
    schema_valid: bool
    exact_match: bool
    full_height: bool | None = None
    interval_iou: float | None = None
    interval_giou: float | None = None
    predicted_interval_width: float | None = None
    expected_interval_width: float | None = None
    interval_width_ratio: float | None = None
    iou_at_0_5_bonus: float = 0.0
    iou_at_0_75_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def interval_iou_giou(
    predicted: tuple[float, float] | list[float],
    expected: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Return one-dimensional IoU and GIoU for two proper intervals."""

    _require(len(predicted) == 2 and len(expected) == 2, "Intervals need two endpoints")
    p1, p2 = (float(value) for value in predicted)
    g1, g2 = (float(value) for value in expected)
    _require(p1 < p2 and g1 < g2, "Intervals must have positive width")

    intersection = max(0.0, min(p2, g2) - max(p1, g1))
    union = (p2 - p1) + (g2 - g1) - intersection
    enclosure = max(p2, g2) - min(p1, g1)
    _require(union > 0.0 and enclosure > 0.0, "Intervals have invalid geometry")
    iou = intersection / union
    giou = iou - (enclosure - union) / enclosure
    return iou, giou


def score_rl_response(
    *,
    task: str,
    prompt: dict[str, Any],
    manifest: dict[str, Any],
    expected_response: str,
    predicted_response: str,
    terminated_with_eos: bool,
) -> RLRewardResult:
    """Apply the frozen v2 task reward to one generated response.

    ``valid`` means strict JSON, exact task schema, and an EOS generated within
    the caller's 64-token cap.  The function has no file-loading surface.
    """

    _require(task in SUPPORTED_TASKS, f"Unsupported RL task: {task}")
    scored = score_task_response(
        task=task,
        prompt=prompt,
        manifest=manifest,
        expected_response=expected_response,
        predicted_response=predicted_response,
    )
    valid_json = bool(scored["valid_json"])
    schema_valid = bool(scored["schema_valid"])
    valid = valid_json and schema_valid and bool(terminated_with_eos)
    exact_match = bool(scored["exact_match"])

    if task == GROUNDING_TASK:
        full_height = bool(scored["full_height"])
        if not valid or not full_height:
            return RLRewardResult(
                task=task,
                reward=INVALID_GROUNDING_REWARD,
                valid=valid,
                terminated_with_eos=bool(terminated_with_eos),
                valid_json=valid_json,
                schema_valid=schema_valid,
                exact_match=exact_match,
                full_height=full_height,
            )

        predicted_box = scored["predicted_bbox_2d"]
        expected_box = scored["expected_bbox_2d"]
        iou, giou = interval_iou_giou(
            [predicted_box[0], predicted_box[2]],
            [expected_box[0], expected_box[2]],
        )
        predicted_width = float(predicted_box[2] - predicted_box[0])
        expected_width = float(expected_box[2] - expected_box[0])
        _require(expected_width > 0.0, "Expected grounding interval has no width")
        bonus_0_5 = 0.2 if iou >= 0.5 else 0.0
        bonus_0_75 = 0.2 if iou >= 0.75 else 0.0
        return RLRewardResult(
            task=task,
            reward=giou + bonus_0_5 + bonus_0_75,
            valid=True,
            terminated_with_eos=True,
            valid_json=True,
            schema_valid=True,
            exact_match=exact_match,
            full_height=True,
            interval_iou=iou,
            interval_giou=giou,
            predicted_interval_width=predicted_width,
            expected_interval_width=expected_width,
            interval_width_ratio=predicted_width / expected_width,
            iou_at_0_5_bonus=bonus_0_5,
            iou_at_0_75_bonus=bonus_0_75,
        )

    if not valid:
        reward = INVALID_AUXILIARY_REWARD
    else:
        reward = 1.0 if exact_match else 0.0
    return RLRewardResult(
        task=task,
        reward=reward,
        valid=valid,
        terminated_with_eos=bool(terminated_with_eos),
        valid_json=valid_json,
        schema_valid=schema_valid,
        exact_match=exact_match,
    )
