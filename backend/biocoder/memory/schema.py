from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from biocoder.state import utc_now


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryRecord(BaseModel):
    memory_id: str
    memory_type: MemoryType
    query: str
    content: str
    quality_score: float = Field(ge=0, le=1)
    source_task: str
    owner_id: str | None = None
    model_version: str = "unknown"
    created_at: datetime = Field(default_factory=utc_now)
    last_used: datetime | None = None
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
