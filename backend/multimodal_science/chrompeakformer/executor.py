from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.chrompeakformer.outputs import summary_payload, validate_outputs
from multimodal_science.data.derivation import (
    CHROMPEAKFORMER_REQUIRED_MODULES,
    dependency_readiness,
)
from multimodal_science.data.manifest import sha256_file

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
ARTIFACT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
Extractor = Callable[[dict[str, Any], Path, Path], Mapping[str, Any] | None]


class DependencyGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    job_id: str
    status: str
    output_dir: Path | None
    provenance_path: Path
    output_sha256: str | None


@dataclass(frozen=True)
class PlanExecutionResult:
    plan_sha256: str
    selected_jobs: int
    completed: int
    cached: int
    failed: int
    results: tuple[ExecutionResult, ...]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {relative}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {relative}") from exc
    return candidate


def _load_plan(path: Path) -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")
            job_id = value.get("job_id")
            if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
                raise ValueError(f"Invalid job_id at {path.name}:{line_number}: {job_id}")
            if job_id in seen:
                raise ValueError(f"Duplicate job_id in plan: {job_id}")
            seen.add(job_id)
            jobs.append(value)
    if not jobs:
        raise ValueError(f"Derivation plan is empty: {path}")
    return jobs


def load_extractor(specification: str) -> Extractor:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Extractor must use the form package.module:callable")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name, None)
    if not callable(value):
        raise TypeError(f"Extractor is not callable: {specification}")
    return value


def lazy_extractor(specification: str) -> Extractor:
    loaded: Extractor | None = None

    def invoke(
        job: dict[str, Any], source_path: Path, output_dir: Path
    ) -> Mapping[str, Any] | None:
        nonlocal loaded
        if loaded is None:
            loaded = load_extractor(specification)
        return loaded(job, source_path, output_dir)

    return invoke


def _existing_result(
    final_dir: Path,
    job: dict[str, Any],
    plan_sha256: str,
) -> ExecutionResult | None:
    provenance_path = final_dir / "derivation_provenance.json"
    if not provenance_path.is_file():
        return None
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    matches = (
        payload.get("status") == "ok"
        and payload.get("job_id") == job["job_id"]
        and payload.get("plan_sha256") == plan_sha256
        and payload.get("source_artifact_hash") == job.get("artifact_hash")
    )
    if not matches:
        return None
    summary = validate_outputs(final_dir)
    if payload.get("outputs", {}).get("output_sha256") != summary.output_sha256:
        return None
    return ExecutionResult(
        job_id=str(job["job_id"]),
        status="cached",
        output_dir=final_dir,
        provenance_path=provenance_path,
        output_sha256=summary.output_sha256,
    )


def _failure_status(stage: str, error: Exception) -> str:
    if isinstance(error, DependencyGateError):
        return "dependency_blocked"
    if stage in {"input_validation", "source_verification"}:
        return "source_error"
    if stage == "output_validation":
        return "validation_error"
    return "tool_error"


def _failure_payload(
    *,
    job: dict[str, Any],
    plan_sha256: str,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "chrompeak-execution-v1",
        "job_id": job.get("job_id"),
        "status": _failure_status(stage, error),
        "stage": stage,
        "dataset_version": job.get("dataset_version"),
        "split": job.get("split"),
        "source_mzml": job.get("source_mzml"),
        "source_artifact_hash": job.get("artifact_hash"),
        "plan_sha256": plan_sha256,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_job(
    job: dict[str, Any],
    *,
    plan_sha256: str,
    data_root: Path,
    output_root: Path,
    extractor: Extractor,
    enforce_dependencies: bool = True,
    dependency_status: Mapping[str, bool] | None = None,
) -> ExecutionResult:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage = "input_validation"
    staging_dir: Path | None = None
    job_id = str(job.get("job_id") or "")
    failure_name = (
        job_id
        if JOB_ID_PATTERN.fullmatch(job_id)
        else "invalid-" + hashlib.sha256(job_id.encode()).hexdigest()[:16]
    )
    failure_path = output_root / "failures" / f"{failure_name}.json"
    try:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError(f"Invalid job_id: {job_id}")
        output_prefix = job.get("output_prefix")
        source_mzml = job.get("source_mzml")
        artifact_hash = job.get("artifact_hash")
        if not all(isinstance(value, str) and value for value in (output_prefix, source_mzml)):
            raise ValueError("Job requires output_prefix and source_mzml")
        if not isinstance(artifact_hash, str) or not ARTIFACT_HASH_PATTERN.fullmatch(
            artifact_hash
        ):
            raise ValueError("Job requires a lowercase hexadecimal artifact_hash")
        if Path(str(output_prefix)).parts[0] in {".staging", "failures"}:
            raise ValueError("Job output_prefix uses a reserved execution directory")

        final_dir = _safe_child(output_root, str(output_prefix))
        if final_dir.exists():
            stage = "output_validation"
            existing = _existing_result(final_dir, job, plan_sha256)
            if existing is not None:
                return existing
            raise FileExistsError(
                f"Output directory exists but is not a matching verified result: {output_prefix}"
            )

        stage = "source_verification"
        source_path = _safe_child(data_root, str(source_mzml))
        if not source_path.is_file():
            raise FileNotFoundError(f"Source mzML not found: {source_mzml}")
        actual_hash = sha256_file(source_path)
        if actual_hash != artifact_hash:
            raise ValueError(
                f"Source hash mismatch: expected {artifact_hash}, got {actual_hash}"
            )

        stage = "dependency_gate"
        if enforce_dependencies:
            status = dict(
                dependency_status if dependency_status is not None else dependency_readiness()
            )
            missing = sorted(
                name
                for name in CHROMPEAKFORMER_REQUIRED_MODULES
                if not status.get(name, False)
            )
            if missing:
                raise DependencyGateError(
                    "ChromPeakFormer runtime is missing modules: " + ", ".join(missing)
                )

        staging_root = output_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_root / f"{job_id}-{uuid.uuid4().hex}"
        staging_dir.mkdir()

        stage = "extraction"
        extractor_result = extractor(job, source_path, staging_dir)
        if extractor_result is not None and not isinstance(extractor_result, Mapping):
            raise TypeError("Extractor result must be a mapping or None")

        stage = "output_validation"
        summary = validate_outputs(staging_dir)
        provenance = {
            "schema_version": "chrompeak-execution-v1",
            "job_id": job_id,
            "status": "ok",
            "dataset_version": job.get("dataset_version"),
            "split": job.get("split"),
            "evaluation_tier": job.get("evaluation_tier"),
            "derivation_mode": job.get("derivation_mode"),
            "source_mzml": source_mzml,
            "source_artifact_hash": artifact_hash,
            "plan_sha256": plan_sha256,
            "extractor_result": dict(extractor_result or {}),
            "outputs": summary_payload(summary),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(staging_dir / "derivation_provenance.json", provenance)

        stage = "commit"
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.replace(final_dir)
        staging_dir = None
        if failure_path.exists():
            failure_path.unlink()
        return ExecutionResult(
            job_id=job_id,
            status="completed",
            output_dir=final_dir,
            provenance_path=final_dir / "derivation_provenance.json",
            output_sha256=summary.output_sha256,
        )
    except Exception as error:
        if staging_dir is not None and staging_dir.is_dir():
            shutil.rmtree(staging_dir)
        _write_json_atomic(
            failure_path,
            _failure_payload(
                job=job,
                plan_sha256=plan_sha256,
                stage=stage,
                error=error,
            ),
        )
        raise


def execute_plan(
    plan_path: Path,
    *,
    data_root: Path,
    output_root: Path,
    extractor: Extractor,
    include_splits: frozenset[str] | None = None,
    max_jobs: int | None = None,
    continue_on_error: bool = False,
    enforce_dependencies: bool = True,
    dependency_status: Mapping[str, bool] | None = None,
) -> PlanExecutionResult:
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Derivation plan not found: {plan_path}")
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive")

    plan_sha256 = sha256_file(plan_path)
    jobs = _load_plan(plan_path)
    selected = [
        job for job in jobs if include_splits is None or job.get("split") in include_splits
    ]
    if max_jobs is not None:
        selected = selected[:max_jobs]
    if not selected:
        raise ValueError("No derivation jobs matched the selection")

    results = []
    failure_count = 0
    for job in selected:
        try:
            result = run_job(
                job,
                plan_sha256=plan_sha256,
                data_root=data_root,
                output_root=output_root,
                extractor=extractor,
                enforce_dependencies=enforce_dependencies,
                dependency_status=dependency_status,
            )
            results.append(result)
        except Exception:
            failure_count += 1
            if not continue_on_error:
                raise

    return PlanExecutionResult(
        plan_sha256=plan_sha256,
        selected_jobs=len(selected),
        completed=sum(result.status == "completed" for result in results),
        cached=sum(result.status == "cached" for result in results),
        failed=failure_count,
        results=tuple(results),
    )


def execution_payload(result: PlanExecutionResult) -> dict[str, Any]:
    return {
        "plan_sha256": result.plan_sha256,
        "selected_jobs": result.selected_jobs,
        "completed": result.completed,
        "cached": result.cached,
        "failed": result.failed,
        "results": [
            {
                **asdict(item),
                "output_dir": str(item.output_dir) if item.output_dir is not None else None,
                "provenance_path": str(item.provenance_path),
            }
            for item in result.results
        ],
    }
