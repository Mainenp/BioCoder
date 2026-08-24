from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biocoder.memory.schema import MemoryRecord, MemoryType
from biocoder.security.validation import contains_model_protocol_artifact, redact_secrets


def _tokens(value: str) -> set[str]:
    normalized = value.casefold()
    latin = set(re.findall(r"[a-z0-9_-]{2,}", normalized))
    compact = re.sub(r"\s+", "", normalized)
    grams = {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}
    return latin | grams


class SemanticMemoryStore:
    """Small quality-gated semantic experience store; no Chroma dependency is implied."""

    def __init__(self, root: Path, *, minimum_write_quality: float = 0.75) -> None:
        self.root = root
        self.minimum_write_quality = minimum_write_quality
        self._lock = threading.RLock()

    @staticmethod
    def memory_id(query: str, content: str, owner_id: str | None = None) -> str:
        normalized = " ".join(f"{owner_id or ''}\0{query}\0{content}".casefold().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _path(self, memory_id: str) -> Path:
        return self.root / f"{memory_id}.json"

    def _write(self, record: MemoryRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(record.memory_id)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def add(
        self,
        *,
        query: str,
        content: str,
        quality_score: float,
        source_task: str,
        model_version: str,
        owner_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryRecord | None, bool]:
        query = redact_secrets(query)
        content = redact_secrets(content)
        if (
            quality_score < self.minimum_write_quality
            or not query.strip()
            or not content.strip()
            or contains_model_protocol_artifact(content)
        ):
            return None, False
        memory_id = self.memory_id(query, content, owner_id)
        with self._lock:
            existing = self.get(memory_id)
            if existing:
                existing.quality_score = max(existing.quality_score, quality_score)
                existing.success_count += 1
                existing.active = True
                self._write(existing)
                return existing, False
            record = MemoryRecord(
                memory_id=memory_id,
                memory_type=MemoryType.SEMANTIC,
                query=query,
                content=content,
                quality_score=quality_score,
                source_task=source_task,
                owner_id=owner_id,
                model_version=model_version,
                success_count=1,
                metadata=metadata or {},
            )
            self._write(record)
        return record, True

    def get(self, memory_id: str) -> MemoryRecord | None:
        path = self._path(memory_id)
        if not path.exists():
            return None
        return MemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[MemoryRecord]:
        if not self.root.exists():
            return []
        return [
            MemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.root.glob("*.json"))
            if not path.name.endswith(".tmp")
        ]

    def search(
        self, query: str, *, limit: int = 3, owner_id: str | None = None
    ) -> list[MemoryRecord]:
        query_tokens = _tokens(query)
        scored = []
        for record in self.all():
            if not record.active or record.owner_id != owner_id:
                continue
            memory_tokens = _tokens(f"{record.query} {record.content}")
            overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            score = overlap * 0.7 + record.quality_score * 0.3
            if overlap > 0:
                scored.append((score, record))
        selected = [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]
        now = datetime.now(UTC)
        with self._lock:
            for record in selected:
                record.last_used = now
                self._write(record)
        return selected

    def record_outcome(self, memory_id: str, *, success: bool) -> MemoryRecord:
        with self._lock:
            record = self.get(memory_id)
            if record is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            if success:
                record.success_count += 1
            else:
                record.failure_count += 1
                total = record.success_count + record.failure_count
                record.quality_score = max(0.0, record.quality_score * (record.success_count / max(1, total)))
            self._write(record)
        return record

    def deactivate_by_source_task(
        self, source_task: str, *, owner_id: str | None = None
    ) -> int:
        """Deactivate memories derived from an answer later rejected by the user."""
        deactivated = 0
        with self._lock:
            for record in self.all():
                if (
                    record.source_task != source_task
                    or record.owner_id != owner_id
                    or not record.active
                ):
                    continue
                record.active = False
                record.failure_count += 1
                self._write(record)
                deactivated += 1
        return deactivated

    def decay(self, factor: float = 0.98, *, deactivate_below: float = 0.4) -> int:
        if not 0 < factor <= 1:
            raise ValueError("Decay factor must be in (0, 1]")
        deactivated = 0
        with self._lock:
            for record in self.all():
                record.quality_score = round(record.quality_score * factor, 4)
                if record.quality_score < deactivate_below and record.active:
                    record.active = False
                    deactivated += 1
                self._write(record)
        return deactivated
