from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def passes_quality_gate(record: dict[str, Any], min_quality: float = 0.75) -> bool:
    feedback = record.get("feedback", [])
    negative = any(
        item.get("feedback_type") == "thumbs_down"
        or (item.get("feedback_type") == "rating" and (item.get("rating") or 5) <= 2)
        for item in feedback
    )
    return bool(
        record.get("success")
        and not record.get("failure_type")
        and str(record.get("answer", "")).strip()
        and float(record.get("quality_score", 0)) >= min_quality
        and not negative
    )


def _tool_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in record.get("trajectory", {}).get("steps", []):
        action = step.get("action", {})
        if action.get("type") != "tool_call" or step.get("error"):
            continue
        tool_name = action.get("tool") or action.get("name") or "tool"
        call_id = f"call_{step.get('step', len(messages) + 1)}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(action.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(
                    step.get("tool_result", step.get("observation")),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return messages


def to_sft_record(record: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": record["query"]}]
    messages.extend(_tool_messages(record))
    messages.append({"role": "assistant", "content": record["answer"]})
    return {
        "messages": messages,
        "provenance": {
            **record.get("provenance", {}),
            "source_task_id": record["task_id"],
            "source_record_id": record.get("record_id"),
            "model_version": record.get("model_version", "unknown"),
            "agent_version": record.get("agent_version", "unknown"),
            "generation_method": "quality_gated_trajectory_to_sft",
            "quality_score": record.get("quality_score", 0),
            "dataset_version": dataset_version,
            "created_at": datetime.now(UTC).isoformat(),
        },
    }


def build_sft_dataset(
    input_path: Path,
    output_path: Path,
    *,
    dataset_version: str,
    min_quality: float = 0.75,
) -> dict[str, Any]:
    records = load_records(input_path)
    accepted = [
        to_sft_record(record, dataset_version)
        for record in records
        if passes_quality_gate(record, min_quality)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in accepted)
        + ("\n" if accepted else ""),
        encoding="utf-8",
    )
    manifest = {
        "dataset_type": "sft",
        "dataset_version": dataset_version,
        "source": str(input_path),
        "created_at": datetime.now(UTC).isoformat(),
        "input_records": len(records),
        "accepted_records": len(accepted),
        "rejected_records": len(records) - len(accepted),
        "min_quality": min_quality,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-gated BioCoder SFT data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--min-quality", type=float, default=0.75)
    args = parser.parse_args()
    manifest = build_sft_dataset(
        args.input,
        args.output,
        dataset_version=args.dataset_version,
        min_quality=args.min_quality,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
