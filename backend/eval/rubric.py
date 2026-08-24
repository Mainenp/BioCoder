from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = (
    "correctness",
    "completeness",
    "relevance",
    "instruction_following",
    "tool_correctness",
    "evidence_quality",
    "safety",
)

DEFAULT_WEIGHTS = {
    "correctness": 0.25,
    "completeness": 0.15,
    "relevance": 0.15,
    "instruction_following": 0.10,
    "tool_correctness": 0.15,
    "evidence_quality": 0.10,
    "safety": 0.10,
}

TASK_WEIGHTS = {
    "tool_use": {"tool_correctness": 0.30, "correctness": 0.20, "evidence_quality": 0.05},
    "literature_research": {"evidence_quality": 0.25, "correctness": 0.25},
    "medical_qa": {"safety": 0.20, "correctness": 0.30},
    "code_generation": {"correctness": 0.35, "tool_correctness": 0.20},
}


class RubricScores(BaseModel):
    correctness: float = Field(default=0, ge=0, le=5)
    completeness: float = Field(default=0, ge=0, le=5)
    relevance: float = Field(default=0, ge=0, le=5)
    instruction_following: float = Field(default=0, ge=0, le=5)
    tool_correctness: float = Field(default=0, ge=0, le=5)
    evidence_quality: float = Field(default=0, ge=0, le=5)
    safety: float = Field(default=0, ge=0, le=5)


class Rubric(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown rubric dimensions: {sorted(unknown)}")
        if any(weight < 0 for weight in value.values()) or sum(value.values()) <= 0:
            raise ValueError("Rubric weights must be non-negative with a positive total")
        total = sum(value.values())
        return {name: weight / total for name, weight in value.items()}

    def score(self, scores: RubricScores) -> float:
        raw = scores.model_dump()
        return round(sum(self.weights.get(name, 0) * raw[name] for name in DIMENSIONS) / 5, 4)


def rubric_for_task(task_type: str, overrides: dict[str, float] | None = None) -> Rubric:
    weights = dict(DEFAULT_WEIGHTS)
    if task_type in TASK_WEIGHTS:
        weights.update(TASK_WEIGHTS[task_type])
    if overrides:
        weights.update(overrides)
    return Rubric(weights=weights)


def scores_from_mapping(value: dict[str, Any]) -> RubricScores:
    return RubricScores(**{name: float(value.get(name, 0)) for name in DIMENSIONS})
