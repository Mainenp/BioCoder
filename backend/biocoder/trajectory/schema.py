from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from biocoder.state import TokenUsage, utc_now


class FailureType(StrEnum):
    PLANNING_ERROR = "PLANNING_ERROR"
    TOOL_SELECTION_ERROR = "TOOL_SELECTION_ERROR"
    TOOL_ARGUMENT_ERROR = "TOOL_ARGUMENT_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    REASONING_ERROR = "REASONING_ERROR"
    HALLUCINATION = "HALLUCINATION"
    FORMAT_ERROR = "FORMAT_ERROR"
    TIMEOUT = "TIMEOUT"
    LOOP = "LOOP"
    SAFETY_ERROR = "SAFETY_ERROR"
    UNKNOWN = "UNKNOWN"


class ActionType(StrEnum):
    REQUEST = "request"
    PLAN = "plan"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    SKILL_ROUTE = "skill_route"
    TOOL_CALL = "tool_call"
    MODEL_GENERATION = "model_generation"
    RETRY = "retry"
    EVALUATION = "evaluation"
    FINAL_ANSWER = "final_answer"


class TrajectoryAction(BaseModel):
    type: ActionType
    name: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    step: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)
    state: dict[str, Any] = Field(default_factory=dict)
    action: TrajectoryAction
    observation: Any = None
    tool_result: Any = None
    latency_ms: float = Field(default=0.0, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost: float = Field(default=0.0, ge=0)
    error: str | None = None
    retry: int = Field(default=0, ge=0)


class Trajectory(BaseModel):
    task_id: str
    trace_id: str
    session_id: str
    query: str
    task_type: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str | None = None
    success: bool = False
    status: str = "running"
    failure_type: FailureType | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost: float = Field(default=0.0, ge=0)
    model_version: str = "unknown"
    prompt_version: str = "v1"
    agent_version: str = "2.0.0"
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
