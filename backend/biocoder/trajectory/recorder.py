from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from biocoder.state import AgentState, AgentStatus, TokenUsage
from biocoder.trajectory.schema import (
    ActionType,
    FailureType,
    Trajectory,
    TrajectoryAction,
    TrajectoryStep,
)
from biocoder.trajectory.serializer import dumps, to_serializable

_trace_logger = logging.getLogger("biocoder.trace")

_current_recorder: ContextVar[TrajectoryRecorder | None] = ContextVar(
    "biocoder_trajectory_recorder", default=None
)


class TrajectoryRecorder:
    def __init__(self, state: AgentState) -> None:
        self.state = state
        self.trajectory = Trajectory(
            task_id=state.task_id,
            trace_id=state.trace_id,
            session_id=state.session_id,
            query=to_serializable(state.user_query),
            task_type=state.task_type,
            model_version=state.model_version,
            prompt_version=state.prompt_version,
            agent_version=state.agent_version,
            started_at=state.start_time,
        )

    def snapshot(self, **updates: Any) -> dict[str, Any]:
        snapshot = {
            "status": self.state.status.value,
            "current_step": self.state.current_step,
            "plan": self.state.plan,
            "selected_tool": self.state.selected_tool,
            "retries": self.state.retries,
            "errors": self.state.errors,
        }
        snapshot.update(updates)
        return to_serializable(snapshot)

    def record(
        self,
        action_type: ActionType,
        *,
        name: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: Any = None,
        tool_result: Any = None,
        latency_ms: float = 0.0,
        token_usage: TokenUsage | None = None,
        estimated_cost: float = 0.0,
        error: str | None = None,
        retry: int = 0,
        state: dict[str, Any] | None = None,
        counts_toward_budget: bool | None = None,
    ) -> TrajectoryStep:
        if counts_toward_budget is None:
            counts_toward_budget = action_type in {
                ActionType.PLAN,
                ActionType.MODEL_GENERATION,
                ActionType.TOOL_CALL,
                ActionType.RETRY,
            }
        if counts_toward_budget:
            self.state.current_step += 1
        usage = token_usage or TokenUsage()
        self.state.token_usage = self.state.token_usage + usage
        self.state.estimated_cost += estimated_cost
        step = TrajectoryStep(
            step=len(self.trajectory.steps) + 1,
            state=state or self.snapshot(),
            action=TrajectoryAction(
                type=action_type,
                name=name,
                tool=tool,
                arguments=to_serializable(arguments or {}),
            ),
            observation=to_serializable(observation),
            tool_result=to_serializable(tool_result),
            latency_ms=latency_ms,
            token_usage=usage,
            estimated_cost=estimated_cost,
            error=error,
            retry=retry,
        )
        self.trajectory.steps.append(step)
        self.trajectory.token_usage = self.trajectory.token_usage + usage
        self.trajectory.estimated_cost += estimated_cost
        _trace_logger.info(
            dumps(
                {
                    "event": action_type.value,
                    "trace_id": self.state.trace_id,
                    "task_id": self.state.task_id,
                    "session_id": self.state.session_id,
                    "step": step.step,
                    "name": name,
                    "tool": tool,
                    "latency_ms": latency_ms,
                    "success": error is None,
                    "error": error,
                }
            )
        )
        return step

    def finalize(self, answer: str, *, success: bool, metrics: dict[str, Any] | None = None) -> Trajectory:
        terminal_failure = self.state.status in {
            AgentStatus.BUDGET_EXCEEDED,
            AgentStatus.FAILED,
            AgentStatus.TIMED_OUT,
        }
        if terminal_failure and not success:
            self.state.final_answer = answer
            self.state.end_time = self.state.end_time or datetime.now(UTC)
            self.state.latency_ms = max(
                0.0,
                (self.state.end_time - self.state.start_time).total_seconds() * 1000,
            )
        else:
            self.state.finish(answer, success=success)
        safe_answer = to_serializable(answer)
        self.trajectory.final_answer = safe_answer
        self.trajectory.success = success
        self.trajectory.status = self.state.status.value
        self.trajectory.metrics = to_serializable(metrics or {})
        self.trajectory.completed_at = self.state.end_time
        self.record(
            ActionType.FINAL_ANSWER,
            name="final_answer",
            observation={"answer": safe_answer, "success": success},
            latency_ms=self.state.latency_ms,
        )
        return self.trajectory

    def fail(self, error: str, failure_type: FailureType = FailureType.UNKNOWN) -> Trajectory:
        self.state.fail(error)
        self.trajectory.success = False
        self.trajectory.status = self.state.status.value
        self.trajectory.failure_type = failure_type
        self.trajectory.completed_at = self.state.end_time or datetime.now(UTC)
        self.record(
            ActionType.FINAL_ANSWER,
            name="failure",
            observation={"success": False},
            error=error,
            latency_ms=self.state.latency_ms,
        )
        return self.trajectory


def current_recorder() -> TrajectoryRecorder | None:
    return _current_recorder.get()


@contextmanager
def use_recorder(recorder: TrajectoryRecorder) -> Iterator[TrajectoryRecorder]:
    token: Token[TrajectoryRecorder | None] = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)
