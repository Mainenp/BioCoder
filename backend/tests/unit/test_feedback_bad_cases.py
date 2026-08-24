import pytest
from pydantic import ValidationError

from biocoder.bad_cases.schema import BadCase
from biocoder.bad_cases.store import BadCaseStore, should_create_bad_case
from biocoder.trajectory.schema import FailureType
from feedback.schema import FeedbackRequest, FeedbackType
from feedback.store import FeedbackStore


def test_feedback_schema_and_task_lookup(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback")
    request = FeedbackRequest(
        task_id="task-1",
        feedback_type=FeedbackType.THUMBS_DOWN,
        text_feedback="The cited evidence is wrong.",
    )
    record = store.add(request)

    assert record.is_negative is True
    assert record.is_positive is False
    assert store.for_task("task-1") == [record]
    assert store.has_score("task-1") is True
    assert store.positive_count() == 0
    assert store.for_task("other") == []

    with pytest.raises(ValidationError):
        FeedbackRequest(task_id="task-2", feedback_type=FeedbackType.RATING)

    positive = store.add(
        FeedbackRequest(task_id="task-2", feedback_type=FeedbackType.RATING, rating=5)
    )
    assert positive.is_positive is True
    assert store.positive_count() == 1


def test_bad_case_store_deduplicates_content(tmp_path) -> None:
    store = BadCaseStore(tmp_path / "bad_cases")
    bad_case = BadCase(
        task_id="task-1",
        query="Find a trial",
        trajectory={"steps": []},
        answer="",
        score=0.2,
        failure_type=FailureType.TOOL_SELECTION_ERROR,
        model_version="baseline",
    )

    first_path, first_created = store.add(bad_case)
    second_path, second_created = store.add(bad_case)

    assert first_path == second_path
    assert first_created is True
    assert second_created is False
    assert len(store.all()) == 1


def test_bad_case_trigger_is_any_failure_signal() -> None:
    assert should_create_bad_case(task_success=False, score=1, threshold=0.5)
    assert should_create_bad_case(task_success=True, score=0.4, threshold=0.5)
    assert should_create_bad_case(task_success=True, score=1, threshold=0.5, tool_failure=True)
    assert should_create_bad_case(task_success=True, score=1, threshold=0.5, human_negative=True)
    assert not should_create_bad_case(task_success=True, score=0.9, threshold=0.5)
