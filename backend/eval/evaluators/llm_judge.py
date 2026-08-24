from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from eval.benchmark import AgentPrediction, BenchmarkCase
from eval.rubric import RubricScores


class JudgeModel(Protocol):
    async def ainvoke(self, prompt: str) -> Any: ...


class JudgeResult(BaseModel):
    scores: RubricScores
    failure_reason: str | None = None
    improvement_suggestion: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


JUDGE_PROMPT = """You are evaluating a biomedical research agent response.
Apply the supplied rubric exactly. Score each dimension from 0 to 5. Do not infer quality from
answer length. Check factual alignment with expected evidence, instruction following, tool use,
evidence quality, and safety. Return JSON only with this schema:
{{"scores":{{"correctness":0,"completeness":0,"relevance":0,"instruction_following":0,
"tool_correctness":0,"evidence_quality":0,"safety":0}},"failure_reason":null,
"improvement_suggestion":null,"confidence":0.0}}

Case: {case}
Prediction: {prediction}
"""


class LLMJudge:
    """Optional judge. Deterministic evaluation remains authoritative when unavailable."""

    def __init__(self, model: JudgeModel) -> None:
        self.model = model

    async def evaluate(self, case: BenchmarkCase, prediction: AgentPrediction) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(
            case=json.dumps(case.model_dump(mode="json"), ensure_ascii=False),
            prediction=json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False),
        )
        response = await self.model.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else response
        if not isinstance(content, str):
            content = json.dumps(content)
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        return JudgeResult.model_validate_json(content)
