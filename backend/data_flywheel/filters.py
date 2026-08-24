from __future__ import annotations

import re
from typing import Any

from biocoder.security.validation import contains_model_protocol_artifact

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
)


def contains_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in SECRET_PATTERNS)


def is_usable_record(
    record: dict[str, Any],
    *,
    min_quality: float = 0.0,
    include_failures: bool = True,
) -> bool:
    if not str(record.get("task_id", "")).strip() or not str(record.get("query", "")).strip():
        return False
    if float(record.get("quality_score", 0)) < min_quality:
        return False
    if not include_failures and not record.get("success"):
        return False
    combined = f"{record.get('query', '')}\n{record.get('answer', '')}"
    return not contains_secret(combined) and not contains_model_protocol_artifact(combined)


def filter_records(
    records: list[dict[str, Any]],
    *,
    min_quality: float = 0.0,
    include_failures: bool = True,
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if is_usable_record(row, min_quality=min_quality, include_failures=include_failures)
    ]
