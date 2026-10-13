"""Hash-bound selection-set evaluation and preregistered v2 RL choice rules.

This module is deliberately CPU-only.  It neither generates completions nor accepts a
validation answer path: callers hand it already-generated, greedy selection-set
predictions.  Validation can only be opened through the small, crash-resumable gate at
the bottom of this file after a selection result has been sealed.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from multimodal_science.baselines.metrics import binary_metrics
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_data import (
    GROUNDING_TASK,
    assert_no_protected_input_paths,
)
from multimodal_science.qwen3vl.task_scoring import score_task_response


SELECTION_BUNDLE_SCHEMA = "chrompeak-qwen3vl-rl-selection-bundle-v2"
SELECTION_PROMPT_SCHEMA = "chrompeak-qwen3vl-rl-selection-prompt-v2"
SELECTION_ANSWER_SCHEMA = "chrompeak-qwen3vl-rl-selection-answer-v1"
SELECTION_MANIFEST_SCHEMA = "chrompeak-qwen3vl-rl-selection-manifest-v1"
SELECTION_EVALUATION_SCHEMA = "chrompeak-qwen3vl-rl-selection-evaluation-v1"
SELECTION_RECORD_SCHEMA = "chrompeak-qwen3vl-rl-selection-record-v1"
SELECTION_RESULT_SCHEMA = "chrompeak-qwen3vl-rl-selection-result-v1"
SELECTION_ACCESS_SCHEMA = "chrompeak-qwen3vl-rl-selection-access-v1"
SELECTION_COMPLETION_SCHEMA = "chrompeak-qwen3vl-rl-selection-completion-v1"

JSON_VALIDITY_THRESHOLD = 0.999
PRESENCE_MACRO_F1_TOLERANCE = 0.01
_HEX_40 = __import__("re").compile(r"^[0-9a-f]{40}$")
_HEX_64 = __import__("re").compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SelectionBundleResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    prompts_path: Path
    answers_path: Path
    manifest_path: Path
    records: int


@dataclass(frozen=True)
class SelectionEvaluationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    records_path: Path
    metrics: dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class ValidationAccess:
    ledger_dir: Path
    access_path: Path
    access_sha256: str
    access_id: str
    mode: str
    completed: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context} must be an object")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {context}: {path}") from error


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"Blank {context} line {number}")
        try:
            values.append(_object(json.loads(line), f"{context} line {number}"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid {context} line {number}") from error
    _require(bool(values), f"Empty {context}")
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _hex(value: Any, context: str, *, length: int = 64) -> str:
    regex = _HEX_64 if length == 64 else _HEX_40
    _require(isinstance(value, str) and bool(regex.fullmatch(value)), f"Invalid {context}")
    return value


def _safe_child(root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Invalid {context} path")
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} path escapes root") from error
    return candidate


def _manifest_entries(root: Path) -> tuple[str, dict[str, str]]:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Artifact manifest not found: {manifest}")
    entries: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, f"Invalid artifact manifest line {number}")
        digest = _hex(fields[0], f"manifest digest {number}")
        relative = fields[1].strip().lstrip("*")
        artifact = _safe_child(root, relative, "manifest")
        canonical = artifact.relative_to(root).as_posix()
        _require(canonical not in entries, f"Duplicate manifest artifact: {canonical}")
        _require(artifact.is_file(), f"Manifest artifact missing: {canonical}")
        _require(sha256_file(artifact) == digest, f"Manifest artifact drift: {canonical}")
        entries[canonical] = digest
    _require(bool(entries), "Artifact manifest is empty")
    return sha256_file(manifest), entries


def _artifact(root: Path, report: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    descriptor = _object(_object(report.get("artifacts"), "artifacts").get(name), f"artifact {name}")
    path = _safe_child(root, descriptor.get("path"), name)
    expected = _hex(descriptor.get("sha256"), f"artifact {name} SHA-256")
    _require(path.is_file() and sha256_file(path) == expected, f"Artifact drift: {name}")
    return path, descriptor


def _unique(records: Iterable[dict[str, Any]], context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        identifier = row.get("instruction_id")
        _require(isinstance(identifier, str) and bool(identifier), f"Missing instruction ID in {context}")
        _require(identifier not in result, f"Duplicate instruction ID in {context}")
        result[identifier] = row
    return result


def _verify_rl_pool(pool_root: Path, expected_report_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pool_root = pool_root.resolve()
    assert_no_protected_input_paths([pool_root])
    report_path = pool_root / "rl_pool_report.json"
    _require(sha256_file(report_path) == _hex(expected_report_sha256, "RL pool report SHA-256"), "RL pool report hash mismatch")
    _, manifest_entries = _manifest_entries(pool_root)
    _require(
        manifest_entries.get("rl_pool_report.json") == sha256_file(report_path),
        "RL pool report is not manifest-bound",
    )
    report = _read_json(report_path, "RL pool report")
    selection_path, selection_descriptor = _artifact(pool_root, report, "selection_set")
    training_path, training_descriptor = _artifact(pool_root, report, "training_pool")
    groups_path, groups_descriptor = _artifact(pool_root, report, "selection_groups")
    selection_records = _read_jsonl(selection_path, "RL selection set")
    training_records = _read_jsonl(training_path, "RL training pool")
    group_records = _read_jsonl(groups_path, "RL selection group lock")
    _require(selection_descriptor.get("records") == len(selection_records), "Selection record count drift")
    _require(training_descriptor.get("records") == len(training_records), "Training record count drift")
    _require(groups_descriptor.get("records") == len(group_records), "Selection group-lock count drift")
    lock = _object(report.get("selection"), "selection lock")
    groups = lock.get("group_ids")
    _require(isinstance(groups, list) and len(groups) == 8 and len(set(groups)) == 8, "Invalid selection group lock")
    selected_groups = {str(value) for value in groups}
    _require(
        {str(row.get("group_id")) for row in group_records} == selected_groups,
        "Selection group lock disagrees with its manifest",
    )
    _require({str(row.get("group_id")) for row in selection_records} <= selected_groups, "Selection row group drift")
    _require(not ({str(row.get("group_id")) for row in training_records} & selected_groups), "Training-selection source-group overlap")
    return report, selection_records, training_records


def build_selection_bundle(*, pool_root: Path, expected_pool_report_sha256: str, output_dir: Path, code_revision: str) -> SelectionBundleResult:
    """Materialize answer-separated held-out-train artifacts from the locked pool."""

    _require(bool(_HEX_40.fullmatch(code_revision)), "Code revision must be a full Git SHA")
    output_dir = output_dir.resolve()
    assert_no_protected_input_paths([pool_root, output_dir])
    _require(not output_dir.exists(), f"Selection bundle already exists: {output_dir}")
    pool_report, rows, _ = _verify_rl_pool(pool_root, expected_pool_report_sha256)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary)
        prompts, answers, manifest = [], [], []
        for index, row in enumerate(rows):
            instruction_id = str(row.get("instruction_id") or "")
            task = str(row.get("task") or "")
            prompt_text = row.get("prompt")
            expected = row.get("expected_response")
            _require(instruction_id and isinstance(prompt_text, str) and isinstance(expected, str), "Invalid selection row")
            signal = _object(row.get("signal"), "selection signal link")
            signal_array = signal.get("array")
            signal_row = signal.get("row")
            signal_available = signal.get("available")
            _require(
                isinstance(signal_array, str)
                and bool(signal_array)
                and isinstance(signal_row, int)
                and signal_row >= 0
                and isinstance(signal_available, bool),
                "Invalid selection signal link",
            )
            prompts.append(
                {
                    "schema_version": SELECTION_PROMPT_SCHEMA,
                    "instruction_id": instruction_id,
                    "task": task,
                    "language": row.get("language"),
                    "conversations": [{"from": "human", "value": prompt_text}],
                    "signal": {
                        "array": signal_array,
                        "row": signal_row,
                        "available": signal_available,
                    },
                }
            )
            answers.append({"schema_version": SELECTION_ANSWER_SCHEMA, "instruction_id": instruction_id, "task": task, "expected_response": expected})
            manifest.append({"schema_version": SELECTION_MANIFEST_SCHEMA, "instruction_id": instruction_id, "task": task, "language": row.get("language"), "group_id": row.get("group_id"), "source_path": row.get("source_path"), "source_stratum": row.get("source_stratum"), "target_peak_present": row.get("target_peak_present"), "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(), "response_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(), "source_selection_row": index})
        prompts_path, answers_path, manifest_path = staging / "selection_prompts.jsonl", staging / "selection_answers.jsonl", staging / "selection_manifest.jsonl"
        _write_jsonl(prompts_path, prompts); _write_jsonl(answers_path, answers); _write_jsonl(manifest_path, manifest)
        prompt_manifest_path = staging / "selection_prompt_manifest.sha256"
        prompt_manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (prompts_path, manifest_path)
            ),
            encoding="utf-8",
        )
        report = {"schema_version": SELECTION_BUNDLE_SCHEMA, "code_revision": code_revision, "created_at": datetime.now(timezone.utc).isoformat(), "sources": {"rl_pool_report_sha256": expected_pool_report_sha256, "rl_selection_set_sha256": _artifact(pool_root.resolve(), pool_report, "selection_set")[1]["sha256"]}, "counts": {"instructions": len(rows), "source_groups": len({row["group_id"] for row in rows})}, "contracts": {"selection_source_groups_only": True, "validation_answers_opened": False, "validation_prompts_opened": False, "internal_test_accessed": False, "training_selection_source_group_overlap": 0, "answers_separated_from_prompts": True, "answer_free_generation_view_available": True}, "artifacts": {}}
        for name, path, records in (("selection_prompts", prompts_path, len(prompts)), ("selection_answers", answers_path, len(answers)), ("selection_manifest", manifest_path, len(manifest))):
            report["artifacts"][name] = {"path": path.name, "sha256": sha256_file(path), "records": records}
        report["artifacts"]["selection_prompt_manifest"] = {
            "path": prompt_manifest_path.name,
            "sha256": sha256_file(prompt_manifest_path),
            "records": 2,
        }
        report_path = staging / "selection_bundle_report.json"; _write_json(report_path, report)
        manifest_hash = staging / "artifact_manifest.sha256"
        manifest_hash.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in (prompts_path, answers_path, manifest_path, prompt_manifest_path, report_path)), encoding="utf-8")
        staging.replace(output_dir)
    report_path = output_dir / "selection_bundle_report.json"
    return SelectionBundleResult(output_dir, report_path, sha256_file(report_path), output_dir / "selection_prompts.jsonl", output_dir / "selection_answers.jsonl", output_dir / "selection_manifest.jsonl", len(rows))


def _load_selection_bundle(root: Path, expected_report_sha256: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = root.resolve(); assert_no_protected_input_paths([root])
    report_path = root / "selection_bundle_report.json"
    _require(sha256_file(report_path) == _hex(expected_report_sha256, "selection bundle report SHA-256"), "Selection bundle report hash mismatch")
    _, manifest_entries = _manifest_entries(root)
    _require(
        manifest_entries.get("selection_bundle_report.json") == sha256_file(report_path),
        "Selection bundle report is not manifest-bound",
    )
    report = _read_json(report_path, "selection bundle report")
    _require(report.get("schema_version") == SELECTION_BUNDLE_SCHEMA, "Wrong selection bundle schema")
    contracts = _object(report.get("contracts"), "selection contracts")
    for key in ("validation_answers_opened", "validation_prompts_opened", "internal_test_accessed"):
        _require(contracts.get(key) is False, f"Selection contract failed: {key}")
    _, prompt_artifact = _artifact(root, report, "selection_prompts"); prompts = _read_jsonl(_safe_child(root, prompt_artifact["path"], "prompts"), "selection prompts")
    _, answer_artifact = _artifact(root, report, "selection_answers"); answers = _read_jsonl(_safe_child(root, answer_artifact["path"], "answers"), "selection answers")
    _, manifest_artifact = _artifact(root, report, "selection_manifest"); manifests = _read_jsonl(_safe_child(root, manifest_artifact["path"], "manifest"), "selection manifest")
    return report, _unique(prompts, "selection prompts"), _unique(answers, "selection answers"), _unique(manifests, "selection manifest")


def verify_candidate_descriptor(descriptor: Mapping[str, Any], *, expected_pool_report_sha256: str) -> dict[str, Any]:
    """Validate a provenance-complete checkpoint descriptor before evaluating it."""

    value = _object(dict(descriptor), "candidate descriptor")
    family = value.get("family")
    _require(isinstance(family, str) and family in {"C0", "C1", "R1", "R2", "R3"}, "Unknown candidate family")
    config = value.get("config")
    _require(
        (isinstance(config, str) and bool(config)) or (isinstance(config, dict) and bool(config)),
        "Missing candidate configuration",
    )
    checkpoint = _object(value.get("checkpoint"), "candidate checkpoint")
    step = checkpoint.get("step")
    _require(isinstance(step, int) and step >= 0, "Invalid checkpoint step")
    checkpoint_path = Path(str(checkpoint.get("path") or "")).resolve()
    assert_no_protected_input_paths([checkpoint_path])
    _require(checkpoint_path.is_file(), "Candidate checkpoint is missing")
    _require(sha256_file(checkpoint_path) == _hex(checkpoint.get("sha256"), "checkpoint SHA-256"), "Candidate checkpoint drift")
    _, checkpoint_entries = _manifest_entries(checkpoint_path.parent)
    _require(
        checkpoint_entries.get(checkpoint_path.name) == sha256_file(checkpoint_path),
        "Candidate checkpoint is not manifest-bound",
    )
    training = _object(value.get("training_report"), "candidate training report")
    report_path = Path(str(training.get("path") or "")).resolve()
    assert_no_protected_input_paths([report_path])
    _require(report_path.is_file(), "Candidate training report is missing")
    _require(sha256_file(report_path) == _hex(training.get("sha256"), "training report SHA-256"), "Candidate training report drift")
    _, training_entries = _manifest_entries(report_path.parent)
    _require(
        training_entries.get(report_path.name) == sha256_file(report_path),
        "Candidate training report is not manifest-bound",
    )
    training_report = _read_json(report_path, "candidate training report")
    sources = _object(value.get("sources"), "candidate sources")
    _require(sources.get("rl_pool_report_sha256") == expected_pool_report_sha256, "Candidate RL pool provenance drift")
    for key, digest in sources.items():
        _hex(digest, f"candidate source {key}")
    if family != "C0":
        report_sources = _object(
            training_report.get("sources"), "candidate training report sources"
        )
        _require(
            report_sources.get("rl_pool_report_sha256")
            == expected_pool_report_sha256,
            "Candidate training report RL pool provenance drift",
        )
    _require(value.get("decode") == {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0}, "Selection decoding must be greedy")
    config_id = config if isinstance(config, str) else json.dumps(config, sort_keys=True, separators=(",", ":"))
    result = dict(value); result["candidate_id"] = f"{family}:{config_id}:step-{step:08d}"
    return result


def _prediction_response(row: dict[str, Any]) -> str:
    for name in ("response", "generated_response", "prediction"):
        value = row.get(name)
        if isinstance(value, str):
            return value
    raise ValueError("Prediction response is missing")


def evaluate_selection_predictions(*, bundle_root: Path, expected_bundle_report_sha256: str, predictions_path: Path, candidate: Mapping[str, Any], output_dir: Path) -> SelectionEvaluationResult:
    """Score a complete hash-bound greedy selection prediction file."""

    assert_no_protected_input_paths([bundle_root, predictions_path, output_dir])
    output_dir = output_dir.resolve(); _require(not output_dir.exists(), f"Selection evaluation exists: {output_dir}")
    bundle, prompts, answers, manifests = _load_selection_bundle(bundle_root, expected_bundle_report_sha256)
    candidate_bound = verify_candidate_descriptor(candidate, expected_pool_report_sha256=str(_object(bundle.get("sources"), "bundle sources").get("rl_pool_report_sha256")))
    predictions = _unique(_read_jsonl(predictions_path.resolve(), "selection predictions"), "selection predictions")
    expected_ids = set(prompts)
    _require(expected_ids == set(answers) == set(manifests) == set(predictions), "Selection prediction coverage mismatch")
    rows: list[dict[str, Any]] = []
    for instruction_id in sorted(expected_ids):
        prompt, answer, manifest, prediction = prompts[instruction_id], answers[instruction_id], manifests[instruction_id], predictions[instruction_id]
        _require(prediction.get("instruction_id") == instruction_id, "Prediction instruction ID drift")
        score = score_task_response(task=str(prompt["task"]), prompt={"conversations": prompt["conversations"]}, manifest={}, expected_response=str(answer["expected_response"]), predicted_response=_prediction_response(prediction), bilingual=False)
        rows.append({"schema_version": SELECTION_RECORD_SCHEMA, "instruction_id": instruction_id, "task": prompt["task"], "language": prompt.get("language"), "group_id": manifest["group_id"], "candidate_id": candidate_bound["candidate_id"], "response_sha256": hashlib.sha256(_prediction_response(prediction).encode("utf-8")).hexdigest(), **score})
    grounding = [float(row["bbox_iou"]) for row in rows if row["task"] == GROUNDING_TASK]
    presence = [row for row in rows if row["task"] == "peak_presence"]
    _require(bool(grounding) and bool(presence), "Selection set lacks required task coverage")
    presence_metrics = binary_metrics([int(bool(row["target_peak_present"])) for row in presence], [float(row["classification_score"]) for row in presence])
    metrics = {"grounding_mean_iou": sum(grounding) / len(grounding), "json_valid_rate": sum(bool(row["valid_json"]) for row in rows) / len(rows), "peak_presence_macro_f1": float(presence_metrics["macro_f1"]), "records": float(len(rows))}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary); records_path = staging / "selection_evaluation_records.jsonl"; _write_jsonl(records_path, rows)
        report = {"schema_version": SELECTION_EVALUATION_SCHEMA, "candidate": candidate_bound, "sources": {"selection_bundle_report_sha256": expected_bundle_report_sha256, "predictions_sha256": sha256_file(predictions_path.resolve())}, "metrics": metrics, "contracts": {"greedy_decoding_verified": True, "validation_answers_opened": False, "internal_test_accessed": False, "selection_groups_only": True}, "artifacts": {"records": {"path": records_path.name, "sha256": sha256_file(records_path), "records": len(rows)}}}
        report_path = staging / "selection_evaluation_report.json"; _write_json(report_path, report)
        (staging / "artifact_manifest.sha256").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in (records_path, report_path)), encoding="utf-8")
        staging.replace(output_dir)
    report_path = output_dir / "selection_evaluation_report.json"
    return SelectionEvaluationResult(output_dir, report_path, sha256_file(report_path), output_dir / "selection_evaluation_records.jsonl", metrics)


def _evaluation_descriptor(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _, entries = _manifest_entries(path.parent)
    _require(
        entries.get(path.name) == sha256_file(path),
        "Selection evaluation report is not manifest-bound",
    )
    report = _read_json(path, "selection evaluation report")
    _require(report.get("schema_version") == SELECTION_EVALUATION_SCHEMA, "Wrong selection evaluation schema")
    contracts = _object(report.get("contracts"), "selection evaluation contracts")
    _require(contracts.get("validation_answers_opened") is False and contracts.get("internal_test_accessed") is False, "Selection evaluation protection contract failed")
    metrics = _object(report.get("metrics"), "selection metrics")
    for key in ("grounding_mean_iou", "json_valid_rate", "peak_presence_macro_f1"):
        value = metrics.get(key)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"Invalid metric {key}",
        )
    candidate = _object(report.get("candidate"), "selection candidate")
    bundle_sha = _hex(
        _object(report.get("sources"), "selection evaluation sources").get(
            "selection_bundle_report_sha256"
        ),
        "selection bundle report SHA-256",
    )
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "candidate": candidate, "selection_bundle_report_sha256": bundle_sha, "metrics": {key: float(metrics[key]) for key in ("grounding_mean_iou", "json_valid_rate", "peak_presence_macro_f1")}}


def choose_preregistered_candidate(*, baseline_evaluation: Path, candidate_evaluations: Sequence[Path], family_prefixes: Sequence[str]) -> dict[str, Any]:
    """Choose one family using the frozen capability constraints and tie-break."""

    baseline = _evaluation_descriptor(baseline_evaluation.resolve())
    baseline_f1 = baseline["metrics"]["peak_presence_macro_f1"]
    candidates = [_evaluation_descriptor(path.resolve()) for path in candidate_evaluations]
    _require(bool(candidates), "No candidate evaluations")
    _require(
        all(
            item["selection_bundle_report_sha256"]
            == baseline["selection_bundle_report_sha256"]
            for item in candidates
        ),
        "Selection evaluations use different bundles",
    )
    _require(all(any(item["candidate"]["family"] == prefix for prefix in family_prefixes) for item in candidates), "Candidate family is not allowed")
    format_eligible = [item for item in candidates if item["metrics"]["json_valid_rate"] >= JSON_VALIDITY_THRESHOLD]
    _require(bool(format_eligible), "No format-valid candidate exists")
    capability_eligible = [item for item in format_eligible if item["metrics"]["peak_presence_macro_f1"] >= baseline_f1 - PRESENCE_MACRO_F1_TOLERANCE]
    pool, fallback = (capability_eligible, False) if capability_eligible else (format_eligible, True)
    selected = sorted(pool, key=lambda item: (-item["metrics"]["grounding_mean_iou"], str(item["candidate"]["candidate_id"])))[0]
    return {"baseline": baseline, "selected": selected, "fallback_format_only": fallback, "capability_thresholds": {"json_valid_rate_at_least": JSON_VALIDITY_THRESHOLD, "peak_presence_macro_f1_at_least": baseline_f1 - PRESENCE_MACRO_F1_TOLERANCE}}


def seal_selection_result(*, bundle_root: Path, expected_bundle_report_sha256: str, c0_evaluation: Path, rl_evaluations: Sequence[Path], c1_evaluations: Sequence[Path], output_dir: Path, code_revision: str) -> SelectionResult:
    """Atomically seal the preregistered RL and C1 choices before validation opens."""

    _require(bool(_HEX_40.fullmatch(code_revision)), "Code revision must be a full Git SHA")
    output_dir = output_dir.resolve(); _require(not output_dir.exists(), f"Selection result exists: {output_dir}")
    bundle, _, _, _ = _load_selection_bundle(bundle_root, expected_bundle_report_sha256)
    c0 = _evaluation_descriptor(c0_evaluation.resolve())
    _require(c0["candidate"]["family"] == "C0", "Baseline must be C0")
    _require(
        c0["selection_bundle_report_sha256"] == expected_bundle_report_sha256,
        "Baseline selection bundle drift",
    )
    rl_choice = choose_preregistered_candidate(baseline_evaluation=c0_evaluation, candidate_evaluations=rl_evaluations, family_prefixes=("R1", "R2", "R3"))
    c1_choice = choose_preregistered_candidate(baseline_evaluation=c0_evaluation, candidate_evaluations=c1_evaluations, family_prefixes=("C1",))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary)
        result = {"schema_version": SELECTION_RESULT_SCHEMA, "code_revision": code_revision, "created_at": datetime.now(timezone.utc).isoformat(), "sources": {"selection_bundle_report_sha256": expected_bundle_report_sha256, "rl_pool_report_sha256": _object(bundle.get("sources"), "bundle sources").get("rl_pool_report_sha256")}, "baseline_c0": c0, "rl_selection": rl_choice, "c1_selection": c1_choice, "contracts": {"selection_written_before_validation": True, "validation_answers_opened": False, "internal_test_accessed": False, "selection_groups_only": True, "pre_registered_rule": True}}
        report_path = staging / "rl_selection_result.json"; _write_json(report_path, result)
        markdown = staging / "rl_selection_result.md"; markdown.write_text("# V2 preregistered RL selection\n\nSelection-set absolute scores are optimistically biased because C0 previously saw these source files. They are used only for relative configuration and checkpoint selection.\n\n" + "| Family | Candidate | Mean IoU | JSON valid | Presence Macro-F1 | Fallback |\n| --- | --- | ---: | ---: | ---: | --- |\n" + "\n".join(f"| {name} | {choice['selected']['candidate']['candidate_id']} | {choice['selected']['metrics']['grounding_mean_iou']:.6f} | {choice['selected']['metrics']['json_valid_rate']:.6f} | {choice['selected']['metrics']['peak_presence_macro_f1']:.6f} | {choice['fallback_format_only']} |" for name, choice in (("RL", rl_choice), ("C1", c1_choice))) + "\n", encoding="utf-8")
        manifest = staging / "artifact_manifest.sha256"; manifest.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in (report_path, markdown)), encoding="utf-8")
        staging.replace(output_dir)
    report_path = output_dir / "rl_selection_result.json"
    return SelectionResult(output_dir, report_path, sha256_file(report_path), output_dir / "artifact_manifest.sha256")


def begin_one_time_validation_access(*, selection_root: Path, expected_selection_result_sha256: str, ledger_root: Path) -> ValidationAccess:
    """Open the sole validation slot, or resume only its unfinished exact protocol."""

    selection_root = selection_root.resolve(); ledger_root = ledger_root.resolve()
    assert_no_protected_input_paths([selection_root, ledger_root])
    report_path = selection_root / "rl_selection_result.json"
    expected = _hex(expected_selection_result_sha256, "selection result SHA-256")
    _require(sha256_file(report_path) == expected, "Selection result hash mismatch")
    _, manifest_entries = _manifest_entries(selection_root)
    _require(
        manifest_entries.get("rl_selection_result.json") == expected,
        "Selection result is not manifest-bound",
    )
    report = _read_json(report_path, "selection result")
    _require(
        report.get("schema_version") == SELECTION_RESULT_SCHEMA,
        "Wrong selection result schema",
    )
    contracts = _object(report.get("contracts"), "selection result contracts")
    _require(contracts.get("selection_written_before_validation") is True, "Selection was not sealed")
    _require(
        contracts.get("validation_answers_opened") is False
        and contracts.get("internal_test_accessed") is False,
        "Selection result protection contract failed",
    )
    ledger_dir = ledger_root / f"selection-{expected[:12]}"; started = ledger_dir / "access_started.json"; completed = ledger_dir / "access_completed.json"
    if completed.exists():
        raise ValueError("Validation access already completed; a second run is forbidden")
    if started.exists():
        access = _read_json(started, "validation access")
        _require(access.get("selection_result_sha256") == expected, "Validation resume protocol mismatch")
        return ValidationAccess(ledger_dir, started, sha256_file(started), str(access["access_id"]), "exact_selection_crash_resume", False)
    ledger_dir.mkdir(parents=True, exist_ok=False)
    access = {"schema_version": SELECTION_ACCESS_SCHEMA, "access_id": hashlib.sha256((expected + "\0selection-validation").encode("utf-8")).hexdigest()[:24], "selection_result_sha256": expected, "opened_at": datetime.now(timezone.utc).isoformat(), "completed": False, "internal_test_accessed": False}
    _write_json(started, access)
    return ValidationAccess(ledger_dir, started, sha256_file(started), str(access["access_id"]), "first_and_only_open", False)


def complete_one_time_validation_access(*, access: ValidationAccess, selection_result_sha256: str) -> Path:
    """Irreversibly mark the exact validation protocol complete."""

    _require(access.access_path.is_file(), "Validation access is not open")
    started = _read_json(access.access_path, "validation access")
    expected = _hex(selection_result_sha256, "selection result SHA-256")
    _require(started.get("selection_result_sha256") == expected and str(started.get("access_id")) == access.access_id, "Validation completion protocol mismatch")
    completed = access.ledger_dir / "access_completed.json"; _require(not completed.exists(), "Validation access is already complete")
    _write_json(completed, {"schema_version": SELECTION_COMPLETION_SCHEMA, "access_id": access.access_id, "selection_result_sha256": expected, "completed_at": datetime.now(timezone.utc).isoformat(), "internal_test_accessed": False})
    return completed


# Explicit aliases keep the orchestration surface readable without multiplying
# implementations.  They intentionally preserve the same strict contracts.
build_rl_selection_bundle = build_selection_bundle
evaluate_rl_selection_predictions = evaluate_selection_predictions
select_preregistered_models = choose_preregistered_candidate
