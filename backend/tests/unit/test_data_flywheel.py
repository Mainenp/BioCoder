import json

from data_flywheel.dedup import deduplicate
from data_flywheel.filters import contains_secret, filter_records
from data_flywheel.pipeline import build_flywheel
from data_flywheel.quality import quality_score


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_filter_quality_and_deduplication() -> None:
    base = {
        "task_id": "task-1",
        "query": "Explain EGFR",
        "answer": "Evidence-based answer",
        "source_type": "trajectory",
        "success": True,
        "evaluation_score": 0.9,
        "trajectory": {"steps": []},
        "feedback": [],
    }
    low = {**base, "quality_score": 0.2}
    high = {**base, "quality_score": 0.9}

    assert quality_score(base) == 0.9
    assert deduplicate([low, high])[0]["quality_score"] == 0.9
    assert filter_records([low, high], min_quality=0.5) == [high]
    assert contains_secret("token sk-abcdefghijklmnopqrstuvwxyz")
    protocol = {
        **high,
        "task_id": "task-protocol",
        "answer": "<｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>",
    }
    assert filter_records([protocol], min_quality=0.5) == []


def test_pipeline_keeps_provenance_and_versions_records(tmp_path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    bad_cases = tmp_path / "bad_cases.jsonl"
    write_jsonl(
        trajectories,
        [
            {
                "task_id": "task-1",
                "query": "Explain C797S",
                "final_answer": "It affects covalent binding.",
                "success": True,
                "metrics": {"score": 0.9},
                "steps": [],
                "model_version": "model-a",
                "agent_version": "2.0.0",
            }
        ],
    )
    write_jsonl(
        feedback,
        [{"task_id": "task-1", "feedback_type": "thumbs_up", "feedback_id": "fb-1"}],
    )
    write_jsonl(bad_cases, [])

    result = build_flywheel(
        trajectories_path=trajectories,
        feedback_path=feedback,
        bad_cases_path=bad_cases,
        output_root=tmp_path / "datasets",
        min_quality=0.5,
    )
    version_dir = tmp_path / "datasets" / result["dataset_version"]
    rows = [json.loads(line) for line in (version_dir / "flywheel_records.jsonl").read_text().splitlines()]
    manifest = json.loads((version_dir / "manifest.json").read_text())

    assert result["deduplicated"] == 1
    assert rows[0]["provenance"]["task_id"] == "task-1"
    assert rows[0]["feedback"][0]["feedback_id"] == "fb-1"
    assert manifest["dataset_version"] == result["dataset_version"]
