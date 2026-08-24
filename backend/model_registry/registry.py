from __future__ import annotations

import json
import threading
from pathlib import Path

from model_registry.schema import ModelStatus, ModelVersion

ALLOWED_TRANSITIONS = {
    ModelStatus.CANDIDATE: {ModelStatus.STAGING, ModelStatus.REJECTED},
    ModelStatus.STAGING: {ModelStatus.PRODUCTION, ModelStatus.REJECTED},
    ModelStatus.PRODUCTION: set(),
    ModelStatus.REJECTED: set(),
}


class ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _load(self) -> dict[str, ModelVersion]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {row["model_id"]: ModelVersion.model_validate(row) for row in payload.get("models", [])}

    def _save(self, models: dict[str, ModelVersion]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "schema_version": 1,
            "models": [model.model_dump(mode="json") for model in models.values()],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def register(self, model: ModelVersion) -> ModelVersion:
        with self._lock:
            models = self._load()
            if model.model_id in models:
                raise ValueError(f"Model already registered: {model.model_id}")
            if model.status != ModelStatus.CANDIDATE:
                raise ValueError("New models must enter the registry as candidate")
            models[model.model_id] = model
            self._save(models)
        return model

    def get(self, model_id: str) -> ModelVersion | None:
        with self._lock:
            return self._load().get(model_id)

    def list(self, status: ModelStatus | None = None) -> list[ModelVersion]:
        with self._lock:
            values = list(self._load().values())
        return [model for model in values if status is None or model.status == status]

    def transition(
        self,
        model_id: str,
        target: ModelStatus,
        *,
        benchmark_score: float | None = None,
        benchmark_report: str | None = None,
        evidence: dict | None = None,
    ) -> ModelVersion:
        with self._lock:
            models = self._load()
            model = models.get(model_id)
            if model is None:
                raise KeyError(f"Unknown model: {model_id}")
            if target not in ALLOWED_TRANSITIONS[model.status]:
                raise ValueError(f"Invalid model transition: {model.status.value} -> {target.value}")
            if target == ModelStatus.STAGING and not evidence:
                raise ValueError("Staging promotion requires regression-gate evidence")
            model.status = target
            if benchmark_score is not None:
                model.benchmark_score = benchmark_score
            if benchmark_report is not None:
                model.benchmark_report = benchmark_report
            if evidence:
                model.metadata["regression_gate"] = evidence
            models[model_id] = model
            self._save(models)
        return model
