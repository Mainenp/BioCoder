from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from biocoder.security.permissions import ToolPermission


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: float = Field(default=0.0, ge=0, le=60)
    retryable_errors: list[str] = Field(default_factory=list)


class ToolMetadata(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    permission: ToolPermission = ToolPermission.READ_ONLY
    timeout_seconds: float = Field(default=20.0, gt=0, le=600)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    side_effect: bool = False
