from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from data_flywheel.dedup import deduplicate
from data_flywheel.filters import filter_records
from data_flywheel.quality import quality_score
from data_flywheel.versioning import write_version
from training.data.build_preference import build_preference_dataset
from training.data.build_sft import build_sft_dataset

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class FlywheelRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: str
    task_id: str
    query: str
    answer: str = ""
    trajectory: dict[str, Any] = Field(default_factory=dict)
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = False
    failure_type: str | None = None
    evaluation_score: float = 0.0
    quality_score: float = 0.0
    model_version: str = "unknown"
    agent_version: str = "unknown"
    generation_method: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    provenance: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_records(
    trajectories_path: Path,
    feedback_path: Path,
    bad_cases_path: Path,
) -> list[dict[str, Any]]:
    feedback_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(feedback_path):
        feedback_by_task[str(row.get("task_id"))].append(row)

    records: list[dict[str, Any]] = []
    for trajectory in _read_jsonl(trajectories_path):
        task_id = str(trajectory.get("task_id", ""))
        record = FlywheelRecord(
            source_type="trajectory",
            task_id=task_id,
            query=str(trajectory.get("query", "")),
            answer=str(trajectory.get("final_answer") or ""),
            trajectory=trajectory,
            feedback=feedback_by_task.get(task_id, []),
            success=bool(trajectory.get("success")),
            failure_type=trajectory.get("failure_type"),
            evaluation_score=float(trajectory.get("metrics", {}).get("score", 0) or 0),
            model_version=str(trajectory.get("model_version", "unknown")),
            agent_version=str(trajectory.get("agent_version", "unknown")),
            generation_method="agent_runtime",
            created_at=str(trajectory.get("completed_at") or trajectory.get("started_at") or ""),
            provenance={
                "source": str(trajectories_path),
                "task_id": task_id,
                "model_version": trajectory.get("model_version", "unknown"),
                "agent_version": trajectory.get("agent_version", "unknown"),
                "generation_method": "agent_runtime",
            },
        ).model_dump(mode="json")
        record["quality_score"] = quality_score(record)
        records.append(record)

    for bad_case in _read_jsonl(bad_cases_path):
        task_id = str(bad_case.get("task_id", ""))
        record = FlywheelRecord(
            source_type="bad_case",
            task_id=task_id,
            query=str(bad_case.get("query", "")),
            answer=str(bad_case.get("answer", "")),
            trajectory=bad_case.get("trajectory", {}),
            feedback=feedback_by_task.get(task_id, []),
            success=False,
            failure_type=bad_case.get("failure_type"),
            evaluation_score=float(bad_case.get("score", 0) or 0),
            model_version=str(bad_case.get("model_version", "unknown")),
            agent_version=str(bad_case.get("trajectory", {}).get("agent_version", "unknown")),
            generation_method="bad_case_store",
            created_at=str(bad_case.get("timestamp", "")),
            provenance={
                "source": str(bad_cases_path),
                "bad_case_id": bad_case.get("bad_case_id"),
                "task_id": task_id,
                "generation_method": "bad_case_store",
            },
        ).model_dump(mode="json")
        record["quality_score"] = quality_score(record)
        records.append(record)
    return records


def build_flywheel(
    *,
    trajectories_path: Path,
    feedback_path: Path,
    bad_cases_path: Path,
    output_root: Path,
    min_quality: float = 0.0,
) -> dict[str, Any]:
    collected = collect_records(trajectories_path, feedback_path, bad_cases_path)
    filtered = filter_records(collected, min_quality=min_quality, include_failures=True)
    records = deduplicate(filtered)
    source_counts = dict(Counter(str(row["source_type"]) for row in records))
    version, version_dir = write_version(records, output_root, source_counts=source_counts)
    records_path = version_dir / "flywheel_records.jsonl"
    sft_manifest = build_sft_dataset(
        records_path,
        version_dir / "sft.jsonl",
        dataset_version=version,
    )
    preference_manifest = build_preference_dataset(
        records_path,
        version_dir / "preference.jsonl",
        dataset_version=version,
    )
    return {
        "dataset_version": version,
        "version_dir": str(version_dir),
        "collected": len(collected),
        "filtered": len(filtered),
        "deduplicated": len(records),
        "source_counts": source_counts,
        "sft": sft_manifest,
        "preference": preference_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned BioCoder flywheel dataset.")
    parser.add_argument("--trajectories", type=Path, default=BACKEND_ROOT / "data" / "trajectories.jsonl")
    parser.add_argument("--feedback", type=Path, default=BACKEND_ROOT / "data" / "feedback" / "feedback.jsonl")
    parser.add_argument("--bad-cases", type=Path, default=BACKEND_ROOT / "data" / "bad_cases" / "bad_cases.jsonl")
    parser.add_argument("--output", type=Path, default=BACKEND_ROOT / "data" / "datasets")
    parser.add_argument("--min-quality", type=float, default=0.0)
    args = parser.parse_args()
    result = build_flywheel(
        trajectories_path=args.trajectories,
        feedback_path=args.feedback,
        bad_cases_path=args.bad_cases,
        output_root=args.output,
        min_quality=args.min_quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
