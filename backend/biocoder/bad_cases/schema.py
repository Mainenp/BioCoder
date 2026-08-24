from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from biocoder.state import utc_now
from biocoder.trajectory.schema import FailureType


class BadCase(BaseModel):
    task_id: str
    query: str
    trajectory: dict[str, Any]
    answer: str = ""
    score: float = Field(default=0, ge=0, le=1)
    failure_type: FailureType = FailureType.UNKNOWN
    feedback: str = ""
    model_version: str = "unknown"
    timestamp: datetime = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(default_factory=dict)
