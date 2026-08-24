from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected: dict[str, Any] = Field(default_factory=dict)
    required_tools: list[str] = Field(default_factory=list)
    rubric: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentPrediction(BaseModel):
    id: str
    answer: str = ""
    plan: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    relevant_sources: list[str] = Field(default_factory=list)
    success: bool | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    retries: int = 0
    loop_detected: bool = False
    step_count: int = 0
    latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0


def load_dataset(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = BenchmarkCase.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid benchmark case at {path}:{line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"Duplicate benchmark id at {path}:{line_number}: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def load_predictions(path: Path) -> dict[str, AgentPrediction]:
    predictions: dict[str, AgentPrediction] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            prediction = AgentPrediction.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid prediction at {path}:{line_number}: {exc}") from exc
        predictions[prediction.id] = prediction
    return predictions


def reference_prediction(case: BenchmarkCase) -> AgentPrediction:
    """Create an explicit reference-baseline prediction for offline pipeline smoke tests."""
    expected = case.expected
    answer = str(expected.get("answer", ""))
    if not answer and expected.get("contains"):
        answer = " ".join(str(item) for item in expected["contains"])
    return AgentPrediction(
        id=case.id,
        answer=answer,
        plan=["understand", "retrieve", "verify", "answer"],
        tools_used=list(case.required_tools),
        sources=[{"title": item} for item in expected.get("sources", [])],
        retrieved_sources=list(expected.get("sources", [])),
        relevant_sources=list(expected.get("sources", [])),
        success=True,
        step_count=max(1, len(case.required_tools) + 2),
    )


def write_jsonl(path: Path, rows: list[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        row.model_dump_json() if isinstance(row, BaseModel) else json.dumps(row, ensure_ascii=False)
        for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
