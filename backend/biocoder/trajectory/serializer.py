from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from biocoder.security.validation import redact_secrets


def to_serializable(value: Any) -> Any:
    """Convert runtime values, including LangChain messages, to stable JSON data."""
    if isinstance(value, str):
        return redact_secrets(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())
    if hasattr(value, "content"):
        return {
            "type": value.__class__.__name__,
            "content": to_serializable(value.content),
            "name": getattr(value, "name", None),
            "tool_calls": to_serializable(getattr(value, "tool_calls", [])),
        }
    return repr(value)


def dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(to_serializable(value), ensure_ascii=False, indent=indent, sort_keys=True)
