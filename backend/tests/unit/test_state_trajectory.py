import json

from biocoder.state import AgentState, AgentStatus, TokenUsage
from biocoder.trajectory.recorder import TrajectoryRecorder, current_recorder, use_recorder
from biocoder.trajectory.schema import ActionType
from biocoder.trajectory.storage import TrajectoryStorage


def build_state() -> AgentState:
    return AgentState(
        task_id="task-1",
        trace_id="trace-1",
        session_id="session-1",
        user_query="What is EGFR C797S?",
        status=AgentStatus.RUNNING,
        model_version="test-model",
    )


def test_agent_state_serialization_round_trip() -> None:
    state = build_state()
    state.plan = ["retrieve", "summarize"]
    state.token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    restored = AgentState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.budget.max_steps == 20


def test_trajectory_records_and_exports_jsonl(tmp_path) -> None:
    recorder = TrajectoryRecorder(build_state())
    with use_recorder(recorder):
        assert current_recorder() is recorder
        recorder.record(
            ActionType.TOOL_CALL,
            tool="search_pubmed",
            arguments={"query": "EGFR"},
            observation={"success": True},
            token_usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )
    assert current_recorder() is None
    trajectory = recorder.finalize("Answer", success=True, metrics={"score": 1.0})

    storage = TrajectoryStorage(tmp_path / "records", tmp_path / "all.jsonl")
    path = storage.save(trajectory)
    loaded = storage.load("task-1")

    assert path.exists()
    assert loaded is not None
    assert loaded.final_answer == "Answer"
    assert loaded.steps[0].action.tool == "search_pubmed"
    assert loaded.token_usage.total_tokens == 3
    assert json.loads((tmp_path / "all.jsonl").read_text().splitlines()[0])["task_id"] == "task-1"

    exported = tmp_path / "export" / "subset.jsonl"
    assert storage.export_jsonl(exported, {"task-1"}) == 1
    assert '"task_id":"task-1"' in exported.read_text()


def test_observability_events_do_not_consume_agent_step_budget() -> None:
    recorder = TrajectoryRecorder(build_state())

    recorder.record(ActionType.REQUEST, name="request")
    recorder.record(ActionType.MEMORY, name="memory_retrieval")
    recorder.record(ActionType.SKILL_ROUTE, name="router")
    recorder.record(ActionType.RETRIEVAL, name="retrieval")
    assert recorder.state.current_step == 0

    recorder.record(ActionType.PLAN, name="planner")
    recorder.record(ActionType.MODEL_GENERATION, name="researcher")
    recorder.record(ActionType.TOOL_CALL, tool="search_pubmed")
    assert recorder.state.current_step == 3
