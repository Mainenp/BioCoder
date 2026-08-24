import json
import logging

from biocoder.state import TokenUsage
from observability.cost import ModelPrice, estimate_cost
from observability.logging import log_event
from observability.tracing import TaskContext, use_task_context


def test_cost_estimate_uses_separate_input_output_prices() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000)
    price = ModelPrice(prompt_per_million=2.0, completion_per_million=8.0)
    assert estimate_cost(usage, price) == 6.0


def test_structured_log_carries_correlation_ids(caplog) -> None:
    logger = logging.getLogger("test.biocoder.observability")
    context = TaskContext(trace_id="trace", task_id="task", session_id="session")

    with caplog.at_level(logging.INFO), use_task_context(context):
        log_event(logger, "tool_call", tool="search_pubmed", success=True)

    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "tool_call"
    assert payload["trace_id"] == "trace"
    assert payload["task_id"] == "task"
    assert payload["session_id"] == "session"
