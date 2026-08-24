from __future__ import annotations

import hashlib
import re
from typing import Any


def content_fingerprint(record: dict[str, Any]) -> str:
    query = re.sub(r"\s+", " ", str(record.get("query", "")).strip().casefold())
    answer = re.sub(r"\s+", " ", str(record.get("answer", "")).strip().casefold())
    source_type = str(record.get("source_type", "trajectory"))
    return hashlib.sha256(f"{source_type}\0{query}\0{answer}".encode()).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-quality copy of semantically identical query/answer/source rows."""
    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        digest = content_fingerprint(record)
        record["content_hash"] = digest
        if digest not in selected:
            selected[digest] = record
            order.append(digest)
        elif float(record.get("quality_score", 0)) > float(selected[digest].get("quality_score", 0)):
            selected[digest] = record
    return [selected[digest] for digest in order]
