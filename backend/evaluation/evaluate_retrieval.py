from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.rag.store import KnowledgeStore

EVALUATION_ROOT = Path(__file__).resolve().parent


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("query") or not case.get("expected_sources"):
            raise ValueError(f"Invalid evaluation case at {path}:{line_number}")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _relevant_rank(results: list[dict[str, Any]], expected_sources: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result["title"] in expected_sources:
            return rank
    return None


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate(corpus_dir: Path, dataset_path: Path, top_k: int = 5) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    with tempfile.TemporaryDirectory(prefix="bioagent-eval-") as upload_dir:
        settings = Settings(
            embedding_provider="local",
            knowledge_dir=corpus_dir,
            uploads_dir=Path(upload_dir),
        )
        store = KnowledgeStore(settings)
        index_status = store.rebuild()

        details: list[dict[str, Any]] = []
        category_ranks: dict[str, list[int | None]] = defaultdict(list)
        for case in cases:
            results = store.search(case["query"], k=top_k)
            rank = _relevant_rank(results, set(case["expected_sources"]))
            category = case.get("category", "uncategorized")
            category_ranks[category].append(rank)
            details.append(
                {
                    "id": case.get("id"),
                    "category": category,
                    "query": case["query"],
                    "expected_sources": case["expected_sources"],
                    "relevant_rank": rank,
                    "retrieved_sources": [result["title"] for result in results],
                    "scores": [result["metadata"]["score"] for result in results],
                }
            )

    total = len(details)

    def hit_count(k: int, rows: list[dict[str, Any]] = details) -> int:
        return sum(row["relevant_rank"] is not None and row["relevant_rank"] <= k for row in rows)

    top1_hits = hit_count(1)
    ci_low, ci_high = _wilson_interval(top1_hits, total)
    metrics = {
        "top1_accuracy": top1_hits / total,
        "recall_at_3": hit_count(3) / total,
        "recall_at_5": hit_count(5) / total,
        "mrr_at_5": sum(
            1 / row["relevant_rank"] if row["relevant_rank"] is not None else 0 for row in details
        )
        / total,
        "top1_accuracy_ci95": [ci_low, ci_high],
    }
    by_category = {}
    for category, ranks in sorted(category_ranks.items()):
        category_total = len(ranks)
        by_category[category] = {
            "queries": category_total,
            "top1_accuracy": sum(rank == 1 for rank in ranks) / category_total,
            "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / category_total,
        }

    return {
        "benchmark": "BioCoder curated biomedical retrieval benchmark v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "embedding": "LocalHashEmbeddings(dimensions=768)",
        "retriever": "LangChain InMemoryVectorStore cosine similarity",
        "dataset": str(dataset_path),
        "corpus": str(corpus_dir),
        "queries": total,
        "documents": index_status["documents"],
        "chunks": index_status["chunks"],
        "metrics": metrics,
        "by_category": by_category,
        "details": details,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    low, high = metrics["top1_accuracy_ci95"]
    lines = [
        "# BioAgent RAG Retrieval Evaluation",
        "",
        f"- Benchmark: {report['benchmark']}",
        f"- Queries: {report['queries']}",
        f"- Documents / chunks: {report['documents']} / {report['chunks']}",
        f"- Embedding: `{report['embedding']}`",
        "",
        "## Overall metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Top-1 retrieval accuracy | {metrics['top1_accuracy']:.1%} |",
        f"| Recall@3 | {metrics['recall_at_3']:.1%} |",
        f"| Recall@5 | {metrics['recall_at_5']:.1%} |",
        f"| MRR@5 | {metrics['mrr_at_5']:.3f} |",
        f"| Top-1 95% Wilson CI | {low:.1%}–{high:.1%} |",
        "",
        "## Category breakdown",
        "",
        "| Category | Queries | Top-1 accuracy | Recall@3 |",
        "|---|---:|---:|---:|",
    ]
    for category, values in report["by_category"].items():
        lines.append(
            f"| {category} | {values['queries']} | {values['top1_accuracy']:.1%} | "
            f"{values['recall_at_3']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This is a small, curated offline retrieval benchmark. It measures whether the labeled "
                "evidence document is retrieved, not the clinical correctness of an LLM-generated answer. "
                "The corpus is kept separate from the runtime knowledge base, and every query-level ranking "
                "is available in the JSON report for reproducibility."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioAgent local RAG retrieval.")
    parser.add_argument("--corpus", type=Path, default=EVALUATION_ROOT / "corpus")
    parser.add_argument("--dataset", type=Path, default=EVALUATION_ROOT / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=EVALUATION_ROOT / "results")
    args = parser.parse_args()

    report = evaluate(args.corpus.resolve(), args.dataset.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "retrieval_metrics.json"
    markdown_path = args.output / "retrieval_metrics.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"Detailed results: {json_path}")


if __name__ == "__main__":
    main()
