from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from biocoder.bad_cases.schema import BadCase


def should_create_bad_case(
    *,
    task_success: bool,
    score: float,
    threshold: float,
    tool_failure: bool = False,
    human_negative: bool = False,
) -> bool:
    return not task_success or score < threshold or tool_failure or human_negative


class BadCaseStore:
    """Content-addressed store: repeated failure evidence does not create duplicate rows."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jsonl_path = root / "bad_cases.jsonl"
        self._lock = threading.RLock()

    @staticmethod
    def fingerprint(bad_case: BadCase) -> str:
        stable = {
            "task_id": bad_case.task_id,
            "failure_type": bad_case.failure_type.value,
            "answer": bad_case.answer.strip(),
            "feedback": bad_case.feedback.strip(),
        }
        encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def add(self, bad_case: BadCase) -> tuple[Path, bool]:
        self.root.mkdir(parents=True, exist_ok=True)
        digest = self.fingerprint(bad_case)
        destination = self.root / f"{digest}.json"
        payload = {"bad_case_id": digest, **bad_case.model_dump(mode="json")}
        with self._lock:
            if destination.exists():
                return destination, False
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        return destination, True

    def all(self) -> list[dict]:
        if not self.jsonl_path.exists():
            return []
        return [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
