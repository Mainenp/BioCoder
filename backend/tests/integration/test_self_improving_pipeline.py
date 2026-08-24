import json

from biocoder.bad_cases.schema import BadCase
from biocoder.bad_cases.store import BadCaseStore
from biocoder.state import AgentState, AgentStatus
from biocoder.trajectory.recorder import TrajectoryRecorder
from biocoder.trajectory.storage import TrajectoryStorage
from data_flywheel.pipeline import build_flywheel
from feedback.schema import FeedbackRequest, FeedbackType
from feedback.store import FeedbackStore
from training.data.build_preference import build_preference_dataset
from training.data.build_sft import build_sft_dataset


def test_trajectory_feedback_bad_case_to_training_data(tmp_path) -> None:
    trajectory_store = TrajectoryStorage(tmp_path / "trajectories", tmp_path / "trajectories.jsonl")
    state = AgentState(
        task_id="good-task",
        trace_id="trace",
        session_id="session",
        user_query="Explain EGFR C797S",
        status=AgentStatus.RUNNING,
        model_version="baseline",
    )
    recorder = TrajectoryRecorder(state)
    good = recorder.finalize("C797S impairs covalent binding.", success=True, metrics={"score": 0.95})
    trajectory_store.save(good)

    feedback_store = FeedbackStore(tmp_path / "feedback")
    feedback_store.add(FeedbackRequest(task_id="good-task", feedback_type=FeedbackType.THUMBS_UP))
    bad_store = BadCaseStore(tmp_path / "bad_cases")
    bad_store.add(
        BadCase(
            task_id="bad-task",
            query="Explain EGFR C797S",
            trajectory={"task_id": "bad-task", "steps": [], "agent_version": "2.0.0"},
            answer="Unrelated answer",
            score=0.1,
            model_version="baseline",
        )
    )

    result = build_flywheel(
        trajectories_path=tmp_path / "trajectories.jsonl",
        feedback_path=tmp_path / "feedback" / "feedback.jsonl",
        bad_cases_path=tmp_path / "bad_cases" / "bad_cases.jsonl",
        output_root=tmp_path / "datasets",
    )
    version_dir = tmp_path / "datasets" / result["dataset_version"]
    records = version_dir / "flywheel_records.jsonl"
    sft = version_dir / "sft.jsonl"
    preference = version_dir / "preference.jsonl"
    sft_manifest = build_sft_dataset(records, sft, dataset_version=result["dataset_version"])
    pref_manifest = build_preference_dataset(
        records,
        preference,
        dataset_version=result["dataset_version"],
    )

    assert result["source_counts"] == {"trajectory": 1, "bad_case": 1}
    assert sft_manifest["accepted_records"] == 1
    assert pref_manifest["preference_pairs"] == 1
    assert json.loads(preference.read_text().splitlines()[0])["preference_source"] == "successful_vs_failed"
