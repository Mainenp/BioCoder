import asyncio
import json

from biocoder.trajectory.schema import FailureType
from eval.benchmark import AgentPrediction, BenchmarkCase
from eval.evaluators.rule_evaluator import RuleEvaluator
from eval.metrics import aggregate_metrics
from eval.runner import run_benchmark, write_reports


def test_rule_evaluator_scores_tool_and_retrieval_behavior() -> None:
    case = BenchmarkCase(
        id="egfr-1",
        task_type="literature_research",
        query="Explain EGFR C797S",
        expected={"contains": ["covalent"], "sources": ["paper.md"]},
        required_tools=["search_pubmed"],
    )
    prediction = AgentPrediction(
        id=case.id,
        answer="C797S impairs covalent inhibitor binding.",
        plan=["search", "verify"],
        tools_used=["search_pubmed"],
        sources=[{"title": "paper.md"}],
        retrieved_sources=["paper.md", "other.md"],
        step_count=4,
        success=True,
    )

    result = RuleEvaluator().evaluate(case, prediction)

    assert result.task_success is True
    assert result.tool_selection_accuracy == 1
    assert result.retrieval_recall_at_k == 1
    assert result.retrieval_precision_at_k == 0.5
    assert result.failure_type is None


def test_rule_evaluator_classifies_missing_tool() -> None:
    case = BenchmarkCase(
        id="tool-1",
        task_type="tool_use",
        query="Find trials",
        expected={"contains": ["NCT"]},
        required_tools=["search_clinical_trials"],
    )
    result = RuleEvaluator().evaluate(
        case,
        AgentPrediction(id=case.id, answer="No result", success=False),
    )

    assert result.task_success is False
    assert result.failure_type == FailureType.TOOL_SELECTION_ERROR


def test_rule_evaluator_rejects_internal_tool_protocol() -> None:
    case = BenchmarkCase(
        id="protocol-1",
        task_type="multi_step_research",
        query="Find trials",
    )
    prediction = AgentPrediction(
        id=case.id,
        answer="<｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>",
        sources=[{"title": "Evidence"}],
        success=True,
    )

    result = RuleEvaluator().evaluate(case, prediction)

    assert result.task_success is False
    assert result.failure_type == FailureType.FORMAT_ERROR
    assert result.details["tool_protocol_error"] is True


def test_rule_evaluator_separates_retriever_failure_from_generator_quality() -> None:
    case = BenchmarkCase(
        id="rag-failure",
        task_type="literature_research",
        query="Find EGFR evidence",
        expected={"answer": "Correct answer", "sources": ["expected.md"]},
    )
    prediction = AgentPrediction(
        id=case.id,
        answer="Correct answer",
        retrieved_sources=["wrong.md"],
        sources=[{"title": "wrong.md"}],
        success=True,
    )

    result = RuleEvaluator().evaluate(case, prediction)

    assert result.task_success is False
    assert result.failure_type == FailureType.RETRIEVAL_ERROR
    assert result.retrieval_recall_at_k == 0


def test_metrics_and_runner_write_three_report_formats(tmp_path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "qa-1",
                "task_type": "medical_qa",
                "query": "What is PARP?",
                "expected": {"contains": ["DNA repair"]},
                "required_tools": [],
                "rubric": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(run_benchmark(dataset))
    metrics = aggregate_metrics(
        [RuleEvaluator().evaluate(
            BenchmarkCase(
                id="qa-1",
                task_type="medical_qa",
                query="What is PARP?",
                expected={"contains": ["DNA repair"]},
            ),
            AgentPrediction(id="qa-1", answer="DNA repair", success=True),
        )]
    )
    paths = write_reports(report, tmp_path / "results")

    assert report["case_count"] == 1
    assert metrics["summary"]["task_success_rate"] == 1
    assert set(paths) == {"json", "csv", "markdown"}
    assert all(path.exists() for path in paths.values())
