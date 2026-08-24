from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from biocoder.security.validation import contains_model_protocol_artifact
from biocoder.trajectory.schema import FailureType
from eval.benchmark import AgentPrediction, BenchmarkCase
from eval.rubric import RubricScores, rubric_for_task


class EvaluationResult(BaseModel):
    case_id: str
    task_type: str
    score: float = Field(ge=0, le=1)
    task_success: bool
    rubric_scores: RubricScores
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_accuracy: float = Field(ge=0, le=1)
    tool_success_rate: float = Field(ge=0, le=1)
    plan_success: bool
    retry_count: int = 0
    loop_detected: bool = False
    step_count: int = 0
    latency_ms: float = 0
    tool_latency_ms: float = 0
    llm_latency_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0
    retrieval_precision_at_k: float | None = None
    retrieval_recall_at_k: float | None = None
    retrieval_mrr: float | None = None
    failure_type: FailureType | None = None
    failure_reason: str | None = None
    improvement_suggestion: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _retrieval_metrics(retrieved: list[str], relevant: set[str]) -> tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0
    normalized_relevant = {_normalize(item) for item in relevant}
    normalized_retrieved = [_normalize(item) for item in retrieved]
    hits = [item in normalized_relevant for item in normalized_retrieved]
    precision = sum(hits) / len(hits) if hits else 0.0
    recall = len({item for item in normalized_retrieved if item in normalized_relevant}) / len(
        normalized_relevant
    )
    first = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    return precision, recall, 1 / first if first else 0.0


class RuleEvaluator:
    """Deterministic evaluator; no model or network is required."""

    def evaluate(self, case: BenchmarkCase, prediction: AgentPrediction) -> EvaluationResult:
        answer = _normalize(prediction.answer)
        protocol_error = contains_model_protocol_artifact(prediction.answer) or any(
            error.get("type") == "TOOL_PROTOCOL_ERROR" for error in prediction.errors
        )
        expected_answer = _normalize(str(case.expected.get("answer", "")))
        expected_contains = [_normalize(str(item)) for item in case.expected.get("contains", [])]
        expected_regex = case.expected.get("regex")

        checks: list[bool] = []
        if expected_answer:
            checks.append(answer == expected_answer or expected_answer in answer)
        checks.extend(item in answer for item in expected_contains)
        if expected_regex:
            checks.append(bool(re.search(str(expected_regex), prediction.answer, flags=re.IGNORECASE)))
        correctness_ratio = sum(checks) / len(checks) if checks else float(bool(answer))
        if protocol_error:
            correctness_ratio = 0.0

        required = set(case.required_tools)
        used = set(prediction.tools_used)
        tool_accuracy = len(required & used) / len(required) if required else 1.0
        unexpected = used - required if required else set()
        if unexpected:
            tool_accuracy *= max(0.0, 1 - len(unexpected) / max(1, len(used)))

        expected_arguments = case.expected.get("tool_arguments", {})
        argument_checks: list[bool] = []
        for tool_name, expected_values in expected_arguments.items():
            calls = [call for call in prediction.tool_calls if call.get("name") == tool_name]
            for key, value in expected_values.items():
                argument_checks.append(
                    any(_normalize(str(call.get("arguments", {}).get(key, ""))) == _normalize(str(value)) for call in calls)
                )
        argument_accuracy = sum(argument_checks) / len(argument_checks) if argument_checks else 1.0

        tool_errors = [error for error in prediction.errors if error.get("source") == "tool"]
        tool_calls_count = max(len(prediction.tool_calls), len(prediction.tools_used))
        tool_success_rate = (
            max(0.0, 1 - len(tool_errors) / tool_calls_count) if tool_calls_count else 1.0
        )
        evidence_expected = bool(case.expected.get("sources")) or case.task_type in {
            "literature_research",
            "multi_step_research",
        }
        evidence_score = 5.0 if prediction.sources else (1.0 if not evidence_expected else 0.0)
        completeness = min(5.0, correctness_ratio * 4 + (1.0 if prediction.answer else 0.0))
        if protocol_error:
            completeness = 0.0
        safety = 0.0 if any(error.get("type") == "SAFETY_ERROR" for error in prediction.errors) else 5.0
        scores = RubricScores(
            correctness=5 * correctness_ratio,
            completeness=completeness,
            relevance=5.0 if answer and not protocol_error else 0.0,
            instruction_following=(
                5.0 if answer and not prediction.loop_detected and not protocol_error else 0.0
            ),
            tool_correctness=5 * tool_accuracy * argument_accuracy * tool_success_rate,
            evidence_quality=evidence_score,
            safety=safety,
        )
        rubric = rubric_for_task(case.task_type, case.rubric)
        score = rubric.score(scores)
        task_success = bool(
            (prediction.success is not False)
            and correctness_ratio >= 0.8
            and tool_accuracy >= 1.0
            and tool_success_rate >= 1.0
            and not prediction.loop_detected
            and not protocol_error
        )

        failure_type = None
        failure_reason = None
        suggestion = None
        if not task_success:
            if protocol_error:
                failure_type = FailureType.FORMAT_ERROR
                failure_reason = "The answer exposed an internal model/tool protocol."
                suggestion = "Reject protocol artifacts and retry a tool-free final summary."
            elif prediction.loop_detected:
                failure_type = FailureType.LOOP
                failure_reason = "Agent exceeded or repeated its execution loop."
                suggestion = "Tighten step budgets and stop conditions."
            elif tool_errors:
                failure_type = FailureType.TOOL_EXECUTION_ERROR
                failure_reason = "At least one required tool failed."
                suggestion = "Add tool-specific retries and validate upstream availability."
            elif tool_accuracy < 1:
                failure_type = FailureType.TOOL_SELECTION_ERROR
                failure_reason = "Required tools were not selected."
                suggestion = "Improve tool descriptions and routing examples."
            elif argument_accuracy < 1:
                failure_type = FailureType.TOOL_ARGUMENT_ERROR
                failure_reason = "Tool arguments did not match the expected values."
                suggestion = "Validate arguments against the tool schema before execution."
            elif not answer:
                failure_type = FailureType.FORMAT_ERROR
                failure_reason = "No final answer was produced."
                suggestion = "Ensure every terminal graph path produces a final response."
            else:
                failure_type = FailureType.REASONING_ERROR
                failure_reason = "The answer did not satisfy deterministic expectations."
                suggestion = "Use retrieved evidence to address every expected answer element."

        relevant = set(prediction.relevant_sources or case.expected.get("sources", []))
        retrieval = prediction.retrieved_sources
        precision = recall = mrr = None
        if relevant:
            precision, recall, mrr = _retrieval_metrics(retrieval, relevant)
            if recall == 0 and not protocol_error:
                task_success = False
                failure_type = FailureType.RETRIEVAL_ERROR
                failure_reason = "No labeled relevant evidence was retrieved."
                suggestion = "Improve query formulation, indexing, retrieval depth, or reranking."

        return EvaluationResult(
            case_id=case.id,
            task_type=case.task_type,
            score=score,
            task_success=task_success,
            rubric_scores=scores,
            tool_selection_accuracy=round(tool_accuracy, 4),
            tool_argument_accuracy=round(argument_accuracy, 4),
            tool_success_rate=round(tool_success_rate, 4),
            plan_success=bool(prediction.plan),
            retry_count=prediction.retries,
            loop_detected=prediction.loop_detected,
            step_count=prediction.step_count,
            latency_ms=prediction.latency_ms,
            tool_latency_ms=prediction.tool_latency_ms,
            llm_latency_ms=prediction.llm_latency_ms,
            prompt_tokens=prediction.prompt_tokens,
            completion_tokens=prediction.completion_tokens,
            estimated_cost=prediction.estimated_cost,
            retrieval_precision_at_k=precision,
            retrieval_recall_at_k=recall,
            retrieval_mrr=mrr,
            failure_type=failure_type,
            failure_reason=failure_reason,
            improvement_suggestion=suggestion,
            details={
                "checks": checks,
                "required_tools": sorted(required),
                "tools_used": sorted(used),
                "tool_protocol_error": protocol_error,
            },
        )
