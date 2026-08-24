from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from biocoder.state import utc_now


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    REJECTED = "rejected"


class ModelVersion(BaseModel):
    model_id: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    training_method: str
    dataset_version: str
    training_config: dict[str, Any] = Field(default_factory=dict)
    benchmark_score: float | None = Field(default=None, ge=0, le=1)
    benchmark_report: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    git_commit: str = "unknown"
    status: ModelStatus = ModelStatus.CANDIDATE
    metadata: dict[str, Any] = Field(default_factory=dict)
