from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any

from eval.evaluators.rule_evaluator import EvaluationResult


def _mean(values: list[float]) -> float:
    return round(fmean(values), 4) if values else 0.0


def aggregate_metrics(results: list[EvaluationResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation result set")
    retrieval_precision = [row.retrieval_precision_at_k for row in results if row.retrieval_precision_at_k is not None]
    retrieval_recall = [row.retrieval_recall_at_k for row in results if row.retrieval_recall_at_k is not None]
    retrieval_mrr = [row.retrieval_mrr for row in results if row.retrieval_mrr is not None]
    total_tokens = [row.prompt_tokens + row.completion_tokens for row in results]
    failures = Counter(row.failure_type.value for row in results if row.failure_type)
    return {
        "summary": {
            "task_success_rate": _mean([float(row.task_success) for row in results]),
            "answer_correctness": _mean([row.rubric_scores.correctness / 5 for row in results]),
            "completeness": _mean([row.rubric_scores.completeness / 5 for row in results]),
            "relevance": _mean([row.rubric_scores.relevance / 5 for row in results]),
            "tool_accuracy": _mean([row.tool_selection_accuracy for row in results]),
            "average_steps": _mean([float(row.step_count) for row in results]),
            "average_latency_ms": _mean([row.latency_ms for row in results]),
            "average_tokens": _mean([float(value) for value in total_tokens]),
            "estimated_cost": round(sum(row.estimated_cost for row in results), 8),
            "average_score": _mean([row.score for row in results]),
        },
        "agent": {
            "tool_selection_accuracy": _mean([row.tool_selection_accuracy for row in results]),
            "tool_argument_accuracy": _mean([row.tool_argument_accuracy for row in results]),
            "tool_success_rate": _mean([row.tool_success_rate for row in results]),
            "plan_success_rate": _mean([float(row.plan_success) for row in results]),
            "average_retry_count": _mean([float(row.retry_count) for row in results]),
            "loop_rate": _mean([float(row.loop_detected) for row in results]),
            "average_step_count": _mean([float(row.step_count) for row in results]),
        },
        "rag": {
            "precision_at_k": _mean(retrieval_precision),
            "recall_at_k": _mean(retrieval_recall),
            "mrr": _mean(retrieval_mrr),
            "evaluated_cases": len(retrieval_recall),
        },
        "system": {
            "end_to_end_latency_ms": _mean([row.latency_ms for row in results]),
            "tool_latency_ms": _mean([row.tool_latency_ms for row in results]),
            "llm_latency_ms": _mean([row.llm_latency_ms for row in results]),
            "prompt_tokens": sum(row.prompt_tokens for row in results),
            "completion_tokens": sum(row.completion_tokens for row in results),
            "total_tokens": sum(total_tokens),
            "estimated_cost": round(sum(row.estimated_cost for row in results), 8),
        },
        "failure_distribution": dict(sorted(failures.items())),
    }
