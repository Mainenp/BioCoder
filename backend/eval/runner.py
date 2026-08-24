from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.benchmark import load_dataset, load_predictions, reference_prediction
from eval.evaluators.composite_evaluator import CompositeEvaluator
from eval.metrics import aggregate_metrics

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "benchmark.jsonl"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["metrics"]["summary"]
    failures = report["metrics"]["failure_distribution"]
    lines = [
        "# BioCoder Evaluation Report",
        "",
        f"- Model: `{report['model_version']}`",
        f"- Agent Version: `{report['agent_version']}`",
        f"- Dataset Version: `{report['dataset_version']}`",
        f"- Execution Mode: `{report['execution_mode']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Task Success Rate | {summary['task_success_rate']:.1%} |",
        f"| Tool Accuracy | {summary['tool_accuracy']:.1%} |",
        f"| Average Score | {summary['average_score']:.3f} |",
        f"| Average Steps | {summary['average_steps']:.2f} |",
        f"| Average Latency | {summary['average_latency_ms']:.2f} ms |",
        f"| Average Token | {summary['average_tokens']:.2f} |",
        f"| Estimated Cost | ${summary['estimated_cost']:.6f} |",
        "",
        "## Failure Distribution",
        "",
        "| Failure | Count |",
        "|---|---:|",
    ]
    if failures:
        lines.extend(f"| {name} | {count} |" for name, count in failures.items())
    else:
        lines.append("| None | 0 |")
    lines.extend(
        [
            "",
            "## Regression Comparison",
            "",
            "A baseline/candidate comparison is produced by the regression gate when both reports are supplied.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_benchmark(
    dataset_path: Path,
    *,
    predictions_path: Path | None = None,
    model_version: str = "reference-baseline",
    agent_version: str = "2.0.0",
) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    supplied = load_predictions(predictions_path) if predictions_path else {}
    evaluator = CompositeEvaluator()
    results = []
    for case in cases:
        prediction = supplied.get(case.id) if predictions_path else reference_prediction(case)
        if prediction is None:
            prediction = reference_prediction(case).model_copy(
                update={"answer": "", "success": False, "errors": [{"type": "MISSING_PREDICTION"}]}
            )
        results.append(await evaluator.evaluate(case, prediction))
    return {
        "benchmark": "BioCoder Benchmark",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path.resolve()),
        "dataset_version": dataset_path.stem,
        "model_version": model_version,
        "agent_version": agent_version,
        "execution_mode": "predictions" if predictions_path else "reference_baseline",
        "case_count": len(cases),
        "metrics": aggregate_metrics(results),
        "results": [row.model_dump(mode="json") for row in results],
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"benchmark_{stamp}"
    paths = {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
        "markdown": output_dir / f"{stem}.md",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "case_id",
            "task_type",
            "score",
            "task_success",
            "tool_selection_accuracy",
            "step_count",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost",
            "failure_type",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({name: row.get(name) for name in fieldnames})
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioCoder agent benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, default=BACKEND_ROOT / "results")
    parser.add_argument("--model-version", default="reference-baseline")
    parser.add_argument("--agent-version", default="2.0.0")
    args = parser.parse_args()
    report = asyncio.run(
        run_benchmark(
            args.dataset,
            predictions_path=args.predictions,
            model_version=args.model_version,
            agent_version=args.agent_version,
        )
    )
    paths = write_reports(report, args.output)
    print(render_markdown(report))
    print(f"JSON: {paths['json']}")
    print(f"CSV: {paths['csv']}")
    print(f"Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
