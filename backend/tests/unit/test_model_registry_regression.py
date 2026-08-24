import pytest

from eval.regression import RegressionThresholds, apply_gate, compare_reports
from model_registry.registry import ModelRegistry
from model_registry.schema import ModelStatus, ModelVersion


def report(model: str, *, success: float, tools: float, latency: float, cost: float, score: float = 0.8) -> dict:
    return {
        "model_version": model,
        "metrics": {
            "summary": {
                "task_success_rate": success,
                "tool_accuracy": tools,
                "average_latency_ms": latency,
                "estimated_cost": cost,
                "average_score": score,
            }
        },
    }


def test_candidate_passes_gate_and_moves_to_staging(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(
        ModelVersion(
            model_id="candidate-1",
            base_model="base",
            training_method="sft",
            dataset_version="v1",
        )
    )
    baseline = report("baseline", success=0.8, tools=0.9, latency=100, cost=1)
    candidate = report("candidate-1", success=0.85, tools=0.92, latency=110, cost=1.1)

    decision = apply_gate(registry, "candidate-1", baseline, candidate)
    stored = registry.get("candidate-1")

    assert decision.passed is True
    assert stored is not None
    assert stored.status == ModelStatus.STAGING
    assert stored.metadata["regression_gate"]["passed"] is True
    assert registry.transition("candidate-1", ModelStatus.PRODUCTION).status == ModelStatus.PRODUCTION


def test_failed_gate_rejects_and_registry_prevents_invalid_transition(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(
        ModelVersion(
            model_id="candidate-2",
            base_model="base",
            training_method="dpo",
            dataset_version="v2",
        )
    )
    baseline = report("baseline", success=0.8, tools=0.9, latency=100, cost=1)
    candidate = report("candidate-2", success=0.6, tools=0.7, latency=140, cost=1.5)

    decision = apply_gate(
        registry,
        "candidate-2",
        baseline,
        candidate,
        thresholds=RegressionThresholds(),
    )

    assert decision.passed is False
    assert set(decision.reasons) == {
        "minimum_task_success",
        "minimum_tool_accuracy",
        "maximum_latency_regression",
        "maximum_cost_regression",
    }
    assert registry.get("candidate-2").status == ModelStatus.REJECTED  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        registry.transition("candidate-2", ModelStatus.STAGING, evidence={"passed": True})


def test_compare_reports_handles_zero_baseline_without_division_error() -> None:
    baseline = report("baseline", success=1, tools=1, latency=0, cost=0)
    candidate = report("candidate", success=1, tools=1, latency=0, cost=0)
    assert compare_reports(baseline, candidate).passed is True
