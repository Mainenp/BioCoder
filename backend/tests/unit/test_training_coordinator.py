import json

from app.config import Settings
from app.services.training import LocalSFTCoordinator
from feedback.schema import FeedbackRequest, FeedbackType
from feedback.store import FeedbackStore


def test_positive_feedback_schedules_single_dry_run_after_threshold(tmp_path) -> None:
    feedback_store = FeedbackStore(tmp_path / "feedback")
    first = feedback_store.add(
        FeedbackRequest(task_id="task-1", feedback_type=FeedbackType.THUMBS_UP)
    )
    calls = []

    def fake_flywheel(**kwargs):
        version_dir = tmp_path / "datasets" / "version-1"
        version_dir.mkdir(parents=True)
        (version_dir / "sft.jsonl").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "dataset_version": "version-1",
            "version_dir": str(version_dir),
            "sft": {"accepted_records": 2},
        }

    def fake_trainer(config, *, execute):
        calls.append((config.dataset_version, execute))
        return {"status": "dry_run_validated", "run_dir": str(tmp_path / "run")}

    settings = Settings(
        feedback_dir=tmp_path / "feedback",
        dataset_dir=tmp_path / "datasets",
        training_job_state_path=tmp_path / "training-state.json",
        auto_sft_config_path=tmp_path / "mlx.yaml",
        auto_sft_enabled=True,
        auto_sft_execute=False,
        auto_sft_min_new_positive_feedback=2,
        auto_sft_min_records=2,
        auto_sft_cooldown_seconds=0,
    )
    settings.auto_sft_config_path.write_text(
        f"""
model_name: local/model
dataset_path: {tmp_path / 'unused.jsonl'}
output_root: {tmp_path / 'models'}
training: {{iterations: 1, batch_size: 1, max_seq_length: 128}}
""",
        encoding="utf-8",
    )
    coordinator = LocalSFTCoordinator(
        settings,
        feedback_store=feedback_store,
        flywheel_builder=fake_flywheel,
        trainer=fake_trainer,
    )

    state = coordinator.maybe_schedule(first)
    assert state is not None
    assert state.status == "idle"

    second = feedback_store.add(
        FeedbackRequest(task_id="task-2", feedback_type=FeedbackType.RATING, rating=5)
    )
    state = coordinator.maybe_schedule(second)
    assert state is not None
    state = coordinator.wait()

    assert state.status == "dry_run_validated"
    assert state.positive_feedback_count == 2
    assert state.eligible_sft_records == 2
    assert calls == [("version-1", False)]


def test_training_is_skipped_below_eligible_record_threshold(tmp_path) -> None:
    feedback_store = FeedbackStore(tmp_path / "feedback")
    feedback = feedback_store.add(
        FeedbackRequest(task_id="task-1", feedback_type=FeedbackType.THUMBS_UP)
    )

    def fake_flywheel(**kwargs):
        version_dir = tmp_path / "datasets" / "version-1"
        version_dir.mkdir(parents=True)
        (version_dir / "sft.jsonl").write_text("", encoding="utf-8")
        return {
            "dataset_version": "version-1",
            "version_dir": str(version_dir),
            "sft": {"accepted_records": 1},
        }

    settings = Settings(
        feedback_dir=tmp_path / "feedback",
        dataset_dir=tmp_path / "datasets",
        training_job_state_path=tmp_path / "training-state.json",
        auto_sft_enabled=True,
        auto_sft_min_new_positive_feedback=1,
        auto_sft_min_records=2,
        auto_sft_cooldown_seconds=0,
    )
    coordinator = LocalSFTCoordinator(
        settings,
        feedback_store=feedback_store,
        flywheel_builder=fake_flywheel,
    )

    coordinator.maybe_schedule(feedback)
    state = coordinator.wait()

    assert state.status == "skipped_insufficient_data"
    assert state.eligible_sft_records == 1
    assert "minimum is 2" in state.message
