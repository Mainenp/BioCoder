from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMED_OUT = "timed_out"


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class AgentBudget(BaseModel):
    max_steps: int = Field(default=20, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=90.0, gt=0)
    token_budget: int = Field(default=100_000, ge=1)
    cost_budget: float = Field(default=10.0, ge=0)


class AgentState(BaseModel):
    """Serializable task state shared across runtime, evaluation, and data builders."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_query: str
    task_type: str = "multi_step_research"
    plan: list[str] = Field(default_factory=list)
    current_step: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: list[dict[str, Any]] = Field(default_factory=list)
    selected_skill: str | None = None
    selected_tool: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    retries: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    status: AgentStatus = AgentStatus.PENDING
    start_time: datetime = Field(default_factory=utc_now)
    end_time: datetime | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    model_version: str = "unknown"
    prompt_version: str = "v1"
    agent_version: str = "2.0.0"
    budget: AgentBudget = Field(default_factory=AgentBudget)

    def finish(self, answer: str, *, success: bool = True) -> None:
        self.final_answer = answer
        self.status = AgentStatus.COMPLETED if success else AgentStatus.FAILED
        self.end_time = utc_now()
        self.latency_ms = max(0.0, (self.end_time - self.start_time).total_seconds() * 1000)

    def fail(self, error: str, *, status: AgentStatus = AgentStatus.FAILED) -> None:
        self.errors.append({"message": error, "step": self.current_step})
        self.status = status
        self.end_time = utc_now()
        self.latency_ms = max(0.0, (self.end_time - self.start_time).total_seconds() * 1000)
