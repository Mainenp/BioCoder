from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from biocoder.trajectory.serializer import to_serializable
from observability.tracing import current_context


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one machine-parseable JSON event with correlation identifiers."""
    context = current_context()
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "trace_id": context.trace_id if context else fields.pop("trace_id", None),
        "task_id": context.task_id if context else fields.pop("task_id", None),
        "session_id": context.session_id if context else fields.pop("session_id", None),
        **fields,
    }
    logger.log(level, json.dumps(to_serializable(payload), ensure_ascii=False, separators=(",", ":")))
