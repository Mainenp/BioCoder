from pathlib import Path

from evaluation.evaluate_retrieval import _relevant_rank, _wilson_interval, evaluate

EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "evaluation"


def test_metric_helpers() -> None:
    results = [{"title": "wrong.md"}, {"title": "expected.md"}]
    assert _relevant_rank(results, {"expected.md"}) == 2
    assert _relevant_rank(results, {"missing.md"}) is None
    low, high = _wilson_interval(9, 10)
    assert 0 < low < 0.9 < high < 1


def test_curated_retrieval_benchmark_meets_regression_threshold() -> None:
    report = evaluate(EVALUATION_ROOT / "corpus", EVALUATION_ROOT / "queries.jsonl")
    assert report["queries"] == 36
    assert report["documents"] == 12
    assert report["metrics"]["top1_accuracy"] >= 0.9
    assert report["metrics"]["recall_at_3"] >= 0.95
