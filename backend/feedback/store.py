from __future__ import annotations

import json
import threading
from pathlib import Path

from feedback.schema import FeedbackRecord, FeedbackRequest


class FeedbackStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jsonl_path = root / "feedback.jsonl"
        self._lock = threading.RLock()

    def add(self, request: FeedbackRequest) -> FeedbackRecord:
        record = FeedbackRecord(**request.model_dump())
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{record.feedback_id}.json"
        payload = record.model_dump(mode="json")
        with self._lock:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record

    def for_task(self, task_id: str) -> list[FeedbackRecord]:
        if not self.jsonl_path.exists():
            return []
        records = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = FeedbackRecord.model_validate_json(line)
                if record.task_id == task_id:
                    records.append(record)
        return records

    def all(self) -> list[FeedbackRecord]:
        if not self.jsonl_path.exists():
            return []
        return [
            FeedbackRecord.model_validate_json(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def has_score(self, task_id: str) -> bool:
        return any(record.is_score for record in self.for_task(task_id))

    def positive_count(self) -> int:
        return sum(record.is_positive for record in self.all())
