from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TaskContext:
    trace_id: str
    task_id: str
    session_id: str

    @classmethod
    def create(cls, session_id: str) -> TaskContext:
        return cls(trace_id=str(uuid4()), task_id=str(uuid4()), session_id=session_id)


_current_context: ContextVar[TaskContext | None] = ContextVar("biocoder_task_context", default=None)


def current_context() -> TaskContext | None:
    return _current_context.get()


@contextmanager
def use_task_context(context: TaskContext) -> Iterator[TaskContext]:
    token: Token[TaskContext | None] = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)
