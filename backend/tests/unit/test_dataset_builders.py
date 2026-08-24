import json

from training.data.build_preference import build_preference_dataset
from training.data.build_sft import build_sft_dataset, passes_quality_gate


def write_records(path, records) -> None:
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")


def test_sft_builder_rejects_failures_and_preserves_tool_calls(tmp_path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "sft.jsonl"
    good = {
        "record_id": "good",
        "task_id": "task-good",
        "query": "Find EGFR evidence",
        "answer": "Evidence answer",
        "success": True,
        "failure_type": None,
        "quality_score": 0.9,
        "feedback": [],
        "trajectory": {
            "steps": [
                {
                    "step": 1,
                    "action": {
                        "type": "tool_call",
                        "tool": "search_pubmed",
                        "arguments": {"query": "EGFR"},
                    },
                    "tool_result": {"results": [{"title": "Paper"}]},
                    "error": None,
                }
            ]
        },
        "provenance": {"source": "trajectory"},
    }
    bad = {
        **good,
        "record_id": "bad",
        "task_id": "task-bad",
        "success": False,
        "failure_type": "TOOL_EXECUTION_ERROR",
        "quality_score": 0.1,
    }
    write_records(input_path, [good, bad])

    manifest = build_sft_dataset(
        input_path,
        output_path,
        dataset_version="dataset-v1",
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert passes_quality_gate(good)
    assert not passes_quality_gate(bad)
    assert manifest["accepted_records"] == 1
    assert [message["role"] for message in rows[0]["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert rows[0]["provenance"]["dataset_version"] == "dataset-v1"


def test_preference_builder_uses_success_and_human_correction(tmp_path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "preference.jsonl"
    records = [
        {
            "record_id": "high",
            "task_id": "task-high",
            "query": "Explain PARP",
            "answer": "High quality answer",
            "success": True,
            "quality_score": 0.95,
            "feedback": [],
        },
        {
            "record_id": "low",
            "task_id": "task-low",
            "query": " Explain   PARP ",
            "answer": "Wrong answer",
            "success": False,
            "quality_score": 0.1,
            "feedback": [
                {
                    "feedback_id": "fb-1",
                    "feedback_type": "corrected_answer",
                    "corrected_answer": "Human correction",
                }
            ],
        },
    ]
    write_records(input_path, records)

    manifest = build_preference_dataset(
        input_path,
        output_path,
        dataset_version="dataset-v1",
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert manifest["preference_pairs"] == 2
    assert {row["preference_source"] for row in rows} == {
        "successful_vs_failed",
        "human_corrected_vs_original",
    }
    assert all(row["chosen"] != row["rejected"] for row in rows)
