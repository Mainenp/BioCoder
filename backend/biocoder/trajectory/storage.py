from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path

from biocoder.trajectory.schema import Trajectory


class TrajectoryStorage:
    """Durable JSON records plus an append-only JSONL training/export stream."""

    def __init__(self, root: Path, jsonl_path: Path | None = None) -> None:
        self.root = root
        self.jsonl_path = jsonl_path or root / "trajectories.jsonl"
        self._lock = threading.RLock()

    def save(self, trajectory: Trajectory) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = trajectory.model_dump(mode="json")
        destination = self.root / f"{trajectory.task_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
        return destination

    def load(self, task_id: str) -> Trajectory | None:
        path = self.root / f"{task_id}.json"
        if not path.exists():
            return None
        return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_trajectories(self) -> Iterable[Trajectory]:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield Trajectory.model_validate_json(line)

    def export_jsonl(self, destination: Path, task_ids: set[str] | None = None) -> int:
        rows = [
            item.model_dump_json()
            for item in self.iter_trajectories()
            if task_ids is None or item.task_id in task_ids
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        return len(rows)
