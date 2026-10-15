"""Dynamic difficulty screening for the preregistered v2 GRPO pool."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_data import (
    GROUNDING_TASK,
    assert_training_pool_isolation,
    load_rl_training_pool,
)
from multimodal_science.qwen3vl.rl_rewards import score_rl_response
from multimodal_science.qwen3vl.rl_rollout import SampledCompletion


DIFFICULTY_REPORT_SCHEMA = "chrompeak-qwen3vl-rl-difficulty-v1"
DIFFICULTY_RECORD_SCHEMA = "chrompeak-qwen3vl-rl-difficulty-record-v1"
GROUP_SIZE = 8
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 64
GROUNDING_STD_THRESHOLD = 0.05
UNINFORMATIVE_REINJECTION_FRACTION = 0.10
DIFFICULTY_SEED = 17


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RolloutRuntime(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def sample_group(
        self,
        record: dict[str, Any],
        *,
        group_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> list[SampledCompletion]: ...


@dataclass(frozen=True)
class DifficultyResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    selected_pool_path: Path
    rollout_path: Path
    input_records: int
    selected_records: int
    informative_records: int
    development_screening_complete: bool


def _stable_reinjection_key(seed: int, instruction_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0rl-difficulty-reinject\0{instruction_id}".encode("utf-8")
    ).hexdigest()


def _difficulty_work_dir(output_dir: Path) -> Path:
    """Name the resumable journal beside either final or dot-prefixed outputs."""

    hidden_name = (
        output_dir.name if output_dir.name.startswith(".") else f".{output_dir.name}"
    )
    return output_dir.parent / f"{hidden_name}.work"


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "Cannot summarize an empty distribution")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return an explicit empty distribution instead of inventing statistics."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "records": 0,
            "mean": None,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "records": len(ordered),
        "mean": statistics.fmean(ordered),
        "minimum": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def select_difficulty_records(
    records: Sequence[dict[str, Any]],
    rollout_records: Sequence[dict[str, Any]],
    *,
    seed: int = DIFFICULTY_SEED,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Select informative groups and a fixed-seed 10% easy-sample reinjection."""

    _require(seed == DIFFICULTY_SEED, "The preregistered difficulty seed is 17")
    by_id = {str(row.get("instruction_id") or ""): row for row in rollout_records}
    _require(len(by_id) == len(rollout_records), "Duplicate difficulty instruction IDs")
    _require(len(records) == len(rollout_records), "Difficulty rollout coverage mismatch")

    informative_ids: set[str] = set()
    uninformative_ids: list[str] = []
    task_totals = Counter()
    task_informative = Counter()
    grounding_means = []
    grounding_stds = []
    for record in records:
        instruction_id = str(record.get("instruction_id") or "")
        _require(instruction_id in by_id, f"Missing difficulty rollout: {instruction_id}")
        rollout = by_id[instruction_id]
        _require(rollout.get("source_record_sha256") == _canonical_sha256(record), "Difficulty source drift")
        rewards = rollout.get("rewards")
        exact_matches = rollout.get("exact_matches")
        _require(
            isinstance(rewards, list)
            and len(rewards) == GROUP_SIZE
            and all(isinstance(value, (int, float)) for value in rewards),
            "Difficulty reward group is invalid",
        )
        _require(
            isinstance(exact_matches, list)
            and len(exact_matches) == GROUP_SIZE
            and all(isinstance(value, bool) for value in exact_matches),
            "Difficulty correctness group is invalid",
        )
        task = str(record.get("task") or "")
        task_totals[task] += 1
        if task == GROUNDING_TASK:
            reward_std = statistics.pstdev(float(value) for value in rewards)
            reward_mean = statistics.fmean(float(value) for value in rewards)
            grounding_means.append(reward_mean)
            grounding_stds.append(reward_std)
            informative = reward_std > GROUNDING_STD_THRESHOLD
        else:
            informative = any(exact_matches) and not all(exact_matches)
        if informative:
            informative_ids.add(instruction_id)
            task_informative[task] += 1
        else:
            uninformative_ids.append(instruction_id)

    reinjection_count = int(
        math.floor(len(uninformative_ids) * UNINFORMATIVE_REINJECTION_FRACTION + 0.5)
    )
    reinjected_ids = set(
        sorted(
            uninformative_ids,
            key=lambda instruction_id: _stable_reinjection_key(seed, instruction_id),
        )[:reinjection_count]
    )
    selected_ids = informative_ids | reinjected_ids
    selected = tuple(
        {
            **record,
            "difficulty": {
                "informative": str(record["instruction_id"]) in informative_ids,
                "fixed_seed_uninformative_reinjection": (
                    str(record["instruction_id"]) in reinjected_ids
                ),
            },
        }
        for record in records
        if str(record.get("instruction_id") or "") in selected_ids
    )

    summary = {
        "thresholds": {
            "grounding_reward_std_strictly_greater_than": GROUNDING_STD_THRESHOLD,
            "auxiliary_requires_both_correct_and_incorrect": True,
            "uninformative_reinjection_fraction": UNINFORMATIVE_REINJECTION_FRACTION,
            "seed": seed,
        },
        "counts": {
            "input_records": len(records),
            "informative_records": len(informative_ids),
            "uninformative_records": len(uninformative_ids),
            "reinjected_records": len(reinjected_ids),
            "selected_records": len(selected),
        },
        "informative_by_task": {
            task: {
                "records": task_totals[task],
                "informative": task_informative[task],
                "informative_fraction": task_informative[task] / task_totals[task],
            }
            for task in sorted(task_totals)
        },
        "grounding_reward_mean_distribution": _distribution_summary(grounding_means),
        "grounding_reward_std_distribution": {
            "records": len(grounding_stds),
            "mean": statistics.fmean(grounding_stds) if grounding_stds else None,
            "zero_variance_fraction": (
                sum(value == 0.0 for value in grounding_stds) / len(grounding_stds)
                if grounding_stds
                else None
            ),
        },
        "input_by_language": dict(
            sorted(Counter(str(record.get("language")) for record in records).items())
        ),
        "selected_by_language": dict(
            sorted(Counter(str(record.get("language")) for record in selected).items())
        ),
        "input_by_target_peak_present": dict(
            sorted(
                Counter(
                    str(bool(record.get("target_peak_present"))).lower()
                    for record in records
                ).items()
            )
        ),
        "selected_by_target_peak_present": dict(
            sorted(
                Counter(
                    str(bool(record.get("target_peak_present"))).lower()
                    for record in selected
                ).items()
            )
        ),
    }
    return selected, summary


def _read_resume_journal(
    journal_path: Path, records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not journal_path.exists():
        return []
    rows = []
    with journal_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank difficulty journal line: {line_number}")
            row = json.loads(line)
            _require(isinstance(row, dict), f"Bad difficulty journal row: {line_number}")
            rows.append(row)
    _require(len(rows) <= len(records), "Difficulty journal exceeds the input pool")
    for index, row in enumerate(rows):
        expected = records[index]
        _require(
            row.get("instruction_id") == expected.get("instruction_id")
            and row.get("source_record_sha256") == _canonical_sha256(expected),
            f"Difficulty resume prefix drift: {index}",
        )
    return rows


def run_difficulty_screening(
    *,
    code_revision: str,
    rl_pool_root: Path,
    rl_pool_report_sha256: str,
    runtime: RolloutRuntime,
    output_dir: Path,
    max_records: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DifficultyResult:
    """Sample G=8 from the frozen starting policy and materialize difficulty."""

    _require(len(code_revision) == 40, "Code revision must be a full Git SHA")
    if max_records is not None:
        _require(max_records > 0, "max_records must be positive")
    pool = load_rl_training_pool(rl_pool_root, rl_pool_report_sha256)
    records = list(pool.records[:max_records] if max_records is not None else pool.records)
    assert_training_pool_isolation(records, pool.selection_group_ids)
    _require(not output_dir.exists(), f"Difficulty output already exists: {output_dir}")
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = _difficulty_work_dir(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    journal_path = work_dir / "difficulty_rollouts.jsonl"
    rollouts = _read_resume_journal(journal_path, records)
    started = time.perf_counter()
    with journal_path.open("a", encoding="utf-8", buffering=1) as journal:
        for index in range(len(rollouts), len(records)):
            record = records[index]
            completions = runtime.sample_group(
                record,
                group_size=GROUP_SIZE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            _require(len(completions) == GROUP_SIZE, "Rollout runtime returned the wrong group size")
            scored = [
                score_rl_response(
                    task=str(record["task"]),
                    prompt={
                        "conversations": [
                            {"from": "human", "value": str(record["prompt"])}
                        ]
                    },
                    manifest={},
                    expected_response=str(record["expected_response"]),
                    predicted_response=completion.response,
                    terminated_with_eos=completion.terminated_with_eos,
                )
                for completion in completions
            ]
            rollout = {
                "schema_version": DIFFICULTY_RECORD_SCHEMA,
                "instruction_id": record["instruction_id"],
                "source_record_sha256": _canonical_sha256(record),
                "task": record["task"],
                "language": record["language"],
                "target_peak_present": record["target_peak_present"],
                "responses": [completion.response for completion in completions],
                "terminated_with_eos": [
                    completion.terminated_with_eos for completion in completions
                ],
                "sample_seeds": [completion.sample_seed for completion in completions],
                "rewards": [result.reward for result in scored],
                "valid": [result.valid for result in scored],
                "exact_matches": [result.exact_match for result in scored],
                "interval_ious": [result.interval_iou for result in scored],
                "interval_gious": [result.interval_giou for result in scored],
            }
            journal.write(json.dumps(rollout, ensure_ascii=False, sort_keys=True) + "\n")
            rollouts.append(rollout)
            if progress_callback is not None:
                progress_callback(index + 1, len(records))

    selected, summary = select_difficulty_records(records, rollouts)
    assert_training_pool_isolation(selected, pool.selection_group_ids)
    development_complete = max_records is None
    elapsed = time.perf_counter() - started
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        selected_path = staging / "difficulty_training_pool.jsonl"
        rollout_path = staging / "difficulty_rollouts.jsonl"
        report_path = staging / "rl_difficulty_report.json"
        _write_jsonl(selected_path, selected)
        _write_jsonl(rollout_path, rollouts)
        report = {
            "schema_version": DIFFICULTY_REPORT_SCHEMA,
            "code_revision": code_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "rl_pool_report_sha256": rl_pool_report_sha256,
                "rl_training_pool_sha256": _object(
                    _object(pool.report.get("artifacts"), "RL artifacts").get(
                        "training_pool"
                    ),
                    "training pool artifact",
                )["sha256"],
            },
            "sampling": {
                "group_size": GROUP_SIZE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "seed": DIFFICULTY_SEED,
                "runtime": runtime.metadata,
            },
            "difficulty": summary,
            "runtime": {
                "wall_time_seconds_this_invocation": elapsed,
                "max_records": max_records,
            },
            "contracts": {
                "train_pool_only": True,
                "selection_groups_trained": False,
                "validation_prompts_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
                "vision_forward_calls": runtime.metadata.get("visual_forward_calls"),
                "visual_tower_unused": runtime.metadata.get("visual_forward_calls") == 0,
            },
            "development_screening_complete": development_complete,
            "development_training_eligible": development_complete,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "artifacts": {
                "difficulty_training_pool": {
                    "path": selected_path.name,
                    "sha256": sha256_file(selected_path),
                    "records": len(selected),
                },
                "difficulty_rollouts": {
                    "path": rollout_path.name,
                    "sha256": sha256_file(rollout_path),
                    "records": len(rollouts),
                },
            },
        }
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, selected_path, rollout_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    final_report = output_dir / "rl_difficulty_report.json"
    return DifficultyResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        selected_pool_path=output_dir / "difficulty_training_pool.jsonl",
        rollout_path=output_dir / "difficulty_rollouts.jsonl",
        input_records=len(records),
        selected_records=len(selected),
        informative_records=summary["counts"]["informative_records"],
        development_screening_complete=development_complete,
    )


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value
