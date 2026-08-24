from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from app.config import Settings
from data_flywheel.pipeline import build_flywheel
from feedback.schema import FeedbackRecord
from feedback.store import FeedbackStore
from training.sft.train_mlx import load_config as load_mlx_config
from training.sft.train_mlx import run as run_mlx_sft

JobStatus = Literal[
    "idle",
    "queued",
    "running",
    "skipped_insufficient_data",
    "dry_run_validated",
    "trained_unvalidated",
    "failed",
]


def _now() -> datetime:
    return datetime.now(UTC)


class TrainingJobState(BaseModel):
    job_id: str | None = None
    status: JobStatus = "idle"
    trigger: str | None = None
    execute: bool = False
    positive_feedback_count: int = 0
    eligible_sft_records: int = 0
    dataset_version: str | None = None
    dataset_path: str | None = None
    run_dir: str | None = None
    message: str = "No local training job has been scheduled."
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status in {"queued", "running"}


FlywheelBuilder = Callable[..., dict[str, Any]]
SFTTrainer = Callable[..., dict[str, Any]]


class TrainingJobBusyError(RuntimeError):
    pass


class LocalSFTCoordinator:
    """Single-process local training queue with persisted, inspectable state."""

    def __init__(
        self,
        settings: Settings,
        *,
        feedback_store: FeedbackStore | None = None,
        flywheel_builder: FlywheelBuilder = build_flywheel,
        trainer: SFTTrainer = run_mlx_sft,
    ) -> None:
        self.settings = settings
        self.feedback_store = feedback_store or FeedbackStore(settings.feedback_dir)
        self.flywheel_builder = flywheel_builder
        self.trainer = trainer
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = self._load_state()

    def _load_state(self) -> TrainingJobState:
        path = self.settings.training_job_state_path
        if not path.exists():
            return TrainingJobState()
        try:
            state = TrainingJobState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return TrainingJobState(message="Previous training state could not be read.")
        if state.active:
            state.status = "failed"
            state.completed_at = _now()
            state.error = "Training process stopped before the application restarted."
            state.message = "The previous in-process training job was interrupted."
        return state

    def _persist(self) -> None:
        path = self.settings.training_job_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(self._state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def snapshot(self) -> TrainingJobState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def _cooldown_elapsed(self) -> bool:
        completed_at = self._state.completed_at
        if completed_at is None:
            return True
        cooldown = timedelta(seconds=self.settings.auto_sft_cooldown_seconds)
        return _now() - completed_at >= cooldown

    def maybe_schedule(self, feedback: FeedbackRecord) -> TrainingJobState | None:
        if not self.settings.auto_sft_enabled or not feedback.is_positive:
            return None
        positive_count = self.feedback_store.positive_count()
        with self._lock:
            previous_count = self._state.positive_feedback_count
            enough_new_feedback = (
                positive_count - previous_count
                >= self.settings.auto_sft_min_new_positive_feedback
            )
            if self._state.active or not enough_new_feedback or not self._cooldown_elapsed():
                return self._state.model_copy(deep=True)
        execute = self.settings.auto_sft_execute and self.settings.training_api_allow_execute
        return self.schedule(execute=execute, trigger="automatic_positive_feedback")

    def schedule(self, *, execute: bool, trigger: str) -> TrainingJobState:
        with self._lock:
            if self._state.active:
                raise TrainingJobBusyError("A local SFT job is already running.")
            positive_count = self.feedback_store.positive_count()
            self._state = TrainingJobState(
                job_id=str(uuid4()),
                status="queued",
                trigger=trigger,
                execute=execute,
                positive_feedback_count=positive_count,
                message="Local SFT job queued.",
                created_at=_now(),
            )
            self._persist()
            self._thread = threading.Thread(
                target=self._run_job,
                name=f"local-sft-{self._state.job_id}",
                daemon=True,
            )
            self._thread.start()
            return self._state.model_copy(deep=True)

    def _set_state(self, **updates: Any) -> None:
        with self._lock:
            self._state = self._state.model_copy(update=updates)
            self._persist()

    def _run_job(self) -> None:
        self._set_state(status="running", started_at=_now(), message="Building training data.")
        try:
            flywheel = self.flywheel_builder(
                trajectories_path=self.settings.trajectory_jsonl_path,
                feedback_path=self.settings.feedback_dir / "feedback.jsonl",
                bad_cases_path=self.settings.bad_case_dir / "bad_cases.jsonl",
                output_root=self.settings.dataset_dir,
                min_quality=0.0,
            )
            version_dir = Path(str(flywheel["version_dir"]))
            dataset_path = version_dir / "sft.jsonl"
            eligible = int(flywheel["sft"]["accepted_records"])
            common = {
                "eligible_sft_records": eligible,
                "dataset_version": str(flywheel["dataset_version"]),
                "dataset_path": str(dataset_path),
            }
            if eligible < self.settings.auto_sft_min_records:
                self._set_state(
                    **common,
                    status="skipped_insufficient_data",
                    completed_at=_now(),
                    message=(
                        f"Training skipped: {eligible} eligible SFT records; "
                        f"minimum is {self.settings.auto_sft_min_records}."
                    ),
                )
                return

            self._set_state(**common, message="Starting MLX SFT." if self._state.execute else "Validating MLX SFT inputs.")
            config = load_mlx_config(self.settings.auto_sft_config_path, dataset_path)
            config.dataset_version = str(flywheel["dataset_version"])
            outcome = self.trainer(config, execute=self._state.execute)
            status = str(outcome["status"])
            if status not in {"dry_run_validated", "trained_unvalidated"}:
                raise RuntimeError(f"Unexpected MLX SFT status: {status}")
            self._set_state(
                status=status,
                run_dir=str(outcome["run_dir"]),
                completed_at=_now(),
                message=(
                    "MLX SFT completed; the adapter still requires regression validation."
                    if status == "trained_unvalidated"
                    else "MLX SFT dry-run validation completed."
                ),
            )
        except Exception as exc:
            self._set_state(
                status="failed",
                completed_at=_now(),
                error=str(exc),
                message="Local SFT job failed. Inspect the error before retrying.",
            )

    def wait(self, timeout: float = 10) -> TrainingJobState:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.snapshot()
