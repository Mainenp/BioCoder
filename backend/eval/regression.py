from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from model_registry.registry import ModelRegistry
from model_registry.schema import ModelStatus, ModelVersion


class RegressionThresholds(BaseModel):
    minimum_task_success: float = Field(default=0.70, ge=0, le=1)
    minimum_tool_accuracy: float = Field(default=0.80, ge=0, le=1)
    maximum_latency_regression: float = Field(default=0.20, ge=0)
    maximum_cost_regression: float = Field(default=0.20, ge=0)


class RegressionDecision(BaseModel):
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]
    reasons: list[str]
    baseline_model: str
    candidate_model: str


def _relative_regression(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0 if candidate <= 0 else math.inf
    return (candidate - baseline) / baseline


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: RegressionThresholds | None = None,
) -> RegressionDecision:
    limits = thresholds or RegressionThresholds()
    base = baseline["metrics"]["summary"]
    cand = candidate["metrics"]["summary"]
    success = float(cand["task_success_rate"])
    tool_accuracy = float(cand["tool_accuracy"])
    latency_regression = _relative_regression(
        float(cand["average_latency_ms"]), float(base["average_latency_ms"])
    )
    cost_regression = _relative_regression(float(cand["estimated_cost"]), float(base["estimated_cost"]))
    checks = {
        "minimum_task_success": success >= limits.minimum_task_success,
        "minimum_tool_accuracy": tool_accuracy >= limits.minimum_tool_accuracy,
        "maximum_latency_regression": latency_regression <= limits.maximum_latency_regression,
        "maximum_cost_regression": cost_regression <= limits.maximum_cost_regression,
    }
    values = {
        "task_success_rate": success,
        "tool_accuracy": tool_accuracy,
        "latency_regression": latency_regression,
        "cost_regression": cost_regression,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return RegressionDecision(
        passed=all(checks.values()),
        checks=checks,
        values=values,
        reasons=reasons,
        baseline_model=str(baseline.get("model_version", "unknown")),
        candidate_model=str(candidate.get("model_version", "unknown")),
    )


def apply_gate(
    registry: ModelRegistry,
    model_id: str,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    candidate_report_path: Path | None = None,
    thresholds: RegressionThresholds | None = None,
) -> RegressionDecision:
    decision = compare_reports(baseline_report, candidate_report, thresholds)
    score = float(candidate_report["metrics"]["summary"].get("average_score", 0))
    registry.transition(
        model_id,
        ModelStatus.STAGING if decision.passed else ModelStatus.REJECTED,
        benchmark_score=score,
        benchmark_report=str(candidate_report_path) if candidate_report_path else None,
        evidence=decision.model_dump(mode="json"),
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the BioCoder model regression gate.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-model", default="unknown")
    parser.add_argument("--training-method", default="unknown")
    parser.add_argument("--dataset-version", default="unknown")
    parser.add_argument("--minimum-task-success", type=float, default=0.70)
    parser.add_argument("--minimum-tool-accuracy", type=float, default=0.80)
    parser.add_argument("--maximum-latency-regression", type=float, default=0.20)
    parser.add_argument("--maximum-cost-regression", type=float, default=0.20)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    registry = ModelRegistry(args.registry)
    if registry.get(args.model_id) is None:
        registry.register(
            ModelVersion(
                model_id=args.model_id,
                base_model=args.base_model,
                training_method=args.training_method,
                dataset_version=args.dataset_version,
            )
        )
    decision = apply_gate(
        registry,
        args.model_id,
        baseline,
        candidate,
        candidate_report_path=args.candidate,
        thresholds=RegressionThresholds(
            minimum_task_success=args.minimum_task_success,
            minimum_tool_accuracy=args.minimum_tool_accuracy,
            maximum_latency_regression=args.maximum_latency_regression,
            maximum_cost_regression=args.maximum_cost_regression,
        ),
    )
    print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
    raise SystemExit(0 if decision.passed else 1)


if __name__ == "__main__":
    main()
