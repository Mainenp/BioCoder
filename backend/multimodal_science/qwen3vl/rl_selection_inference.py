"""Answer-free, hash-bound greedy inference for the v2 RL selection set."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths
from multimodal_science.qwen3vl.rl_rollout import (
    VerifiedTrainSignals,
    VerifiedXicRuntimeWeights,
    XicSamplingRuntime,
    verify_train_signals,
)
from multimodal_science.qwen3vl.rl_selection import (
    SELECTION_BUNDLE_SCHEMA,
    SELECTION_MANIFEST_SCHEMA,
    SELECTION_PROMPT_SCHEMA,
    _artifact,
    _hex,
    _manifest_entries,
    _object,
    _read_json,
    _read_jsonl,
    verify_candidate_descriptor,
)


SELECTION_INFERENCE_CONFIG_SCHEMA = "chrompeak-qwen3vl-rl-selection-inference-config-v1"
SELECTION_PREDICTION_SCHEMA = "chrompeak-qwen3vl-rl-selection-prediction-v1"
SELECTION_INFERENCE_REPORT_SCHEMA = "chrompeak-qwen3vl-rl-selection-inference-v1"
GPU_ALLOCATION_MODE = "manual_physical_index_guard_no_slurm_gres"


@dataclass(frozen=True)
class AnswerFreeSelectionInputs:
    report: dict[str, Any]
    report_sha256: str
    prompt_manifest_sha256: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SelectionInferenceResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    predictions_path: Path
    prediction_records: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def load_answer_free_selection_inputs(
    root: Path, expected_report_sha256: str
) -> AnswerFreeSelectionInputs:
    """Load only prompt/XIC links; fail if an answer artifact entered this process view."""

    root = root.resolve()
    assert_no_protected_input_paths([root])
    _require(
        not (root / "selection_answers.jsonl").exists(),
        "Selection answers entered the generation view",
    )
    _require(
        not (root / "artifact_manifest.sha256").exists(),
        "Full selection manifest entered the generation view",
    )
    report_path = root / "selection_bundle_report.json"
    report_sha256 = sha256_file(report_path)
    _require(
        report_sha256 == _hex(expected_report_sha256, "selection bundle report SHA-256"),
        "Selection bundle report hash mismatch",
    )
    report = _read_json(report_path, "selection bundle report")
    _require(report.get("schema_version") == SELECTION_BUNDLE_SCHEMA, "Wrong selection bundle schema")
    contracts = _object(report.get("contracts"), "selection contracts")
    for key in ("validation_answers_opened", "validation_prompts_opened", "internal_test_accessed"):
        _require(contracts.get(key) is False, f"Selection contract failed: {key}")
    _require(
        contracts.get("answer_free_generation_view_available") is True,
        "Selection bundle has no answer-free generation contract",
    )
    prompts_path, prompts_artifact = _artifact(root, report, "selection_prompts")
    manifest_path, manifest_artifact = _artifact(root, report, "selection_manifest")
    prompt_manifest_path, prompt_manifest_artifact = _artifact(
        root, report, "selection_prompt_manifest"
    )
    _require(
        sha256_file(prompt_manifest_path) == prompt_manifest_artifact.get("sha256"),
        "Selection prompt manifest drift",
    )
    expected_entries = {
        prompts_path.name: str(prompts_artifact["sha256"]),
        manifest_path.name: str(manifest_artifact["sha256"]),
    }
    entries: dict[str, str] = {}
    for number, line in enumerate(
        prompt_manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, f"Invalid selection prompt manifest line {number}")
        entries[fields[1].strip().lstrip("*")] = _hex(
            fields[0], f"selection prompt manifest digest {number}"
        )
    _require(entries == expected_entries, "Selection prompt manifest scope drift")

    prompts = _read_jsonl(prompts_path, "selection prompts")
    manifests = _read_jsonl(manifest_path, "selection manifest")
    _require(
        prompts_artifact.get("records") == len(prompts)
        and manifest_artifact.get("records") == len(manifests),
        "Selection prompt count drift",
    )
    manifest_ids = [str(row.get("instruction_id") or "") for row in manifests]
    prompt_ids = [str(row.get("instruction_id") or "") for row in prompts]
    _require(
        len(set(prompt_ids)) == len(prompt_ids)
        and prompt_ids == manifest_ids,
        "Selection prompt/manifest order or coverage drift",
    )
    records: list[dict[str, Any]] = []
    for prompt, manifest in zip(prompts, manifests, strict=True):
        _require(prompt.get("schema_version") == SELECTION_PROMPT_SCHEMA, "Wrong selection prompt schema")
        _require(manifest.get("schema_version") == SELECTION_MANIFEST_SCHEMA, "Wrong selection manifest schema")
        _require(not _contains_key(prompt, "expected_response"), "Selection prompt exposes an answer")
        conversations = prompt.get("conversations")
        _require(
            isinstance(conversations, list)
            and len(conversations) == 1
            and isinstance(conversations[0], dict)
            and conversations[0].get("from") == "human"
            and isinstance(conversations[0].get("value"), str)
            and bool(conversations[0]["value"]),
            "Invalid selection conversation",
        )
        signal = _object(prompt.get("signal"), "selection signal link")
        _require(
            isinstance(signal.get("array"), str)
            and signal["array"].replace("\\", "/").startswith("train/")
            and isinstance(signal.get("row"), int)
            and signal["row"] >= 0
            and isinstance(signal.get("available"), bool),
            "Invalid selection signal link",
        )
        records.append(
            {
                "instruction_id": prompt["instruction_id"],
                "task": prompt["task"],
                "language": prompt.get("language"),
                "prompt": conversations[0]["value"],
                "signal": dict(signal),
                "group_id": manifest.get("group_id"),
            }
        )
    return AnswerFreeSelectionInputs(
        report=report,
        report_sha256=report_sha256,
        prompt_manifest_sha256=sha256_file(prompt_manifest_path),
        records=tuple(records),
    )


def verify_selection_candidate_runtime(
    candidate: Mapping[str, Any],
    *,
    expected_pool_report_sha256: str,
    starting_adapter: AdapterSpec,
) -> tuple[dict[str, Any], VerifiedXicRuntimeWeights]:
    """Bind the selected adapter/projector files to a complete candidate descriptor."""

    bound = verify_candidate_descriptor(
        candidate, expected_pool_report_sha256=expected_pool_report_sha256
    )
    checkpoint = _object(bound.get("checkpoint"), "candidate checkpoint")
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    checkpoint_root = checkpoint_path.parent
    checkpoint_manifest_sha256, checkpoint_entries = _manifest_entries(checkpoint_root)
    required = {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "sensor_projector.safetensors",
    }
    if int(checkpoint["step"]) > 0:
        required.add("checkpoint.json")
    _require(required <= set(checkpoint_entries), "Candidate runtime artifacts are not manifest-bound")
    if int(checkpoint["step"]) > 0:
        state = _read_json(checkpoint_root / "checkpoint.json", "candidate checkpoint state")
        _require(state.get("global_step") == checkpoint["step"], "Candidate checkpoint step drift")

    training = _object(bound.get("training_report"), "candidate training report")
    report_path = Path(str(training["path"])).resolve()
    training_report = _read_json(report_path, "candidate training report")
    if bound["family"] != "C0":
        report_sources = _object(training_report.get("sources"), "candidate training sources")
        descriptor_sources = _object(bound.get("sources"), "candidate sources")
        for key in (
            "rl_pool_report_sha256",
            "starting_adapter_report_sha256",
            "starting_adapter_manifest_sha256",
        ):
            _require(
                descriptor_sources.get(key) == report_sources.get(key),
                f"Candidate source drift: {key}",
            )
        _require(
            report_sources.get("starting_adapter_report_sha256")
            == starting_adapter.training_report_sha256
            and report_sources.get("starting_adapter_manifest_sha256")
            == starting_adapter.manifest_sha256,
            "Candidate does not descend from the frozen C0 adapter",
        )
    starting_config = _read_json(
        starting_adapter.root / "adapter" / "adapter_config.json",
        "starting adapter config",
    )
    candidate_config = _read_json(
        checkpoint_root / "adapter" / "adapter_config.json",
        "candidate adapter config",
    )
    _require(candidate_config == starting_config, "Candidate adapter architecture drift")
    return bound, VerifiedXicRuntimeWeights(
        adapter_dir=checkpoint_root / "adapter",
        projector_path=checkpoint_root / "sensor_projector.safetensors",
        candidate_id=str(bound["candidate_id"]),
        training_report_sha256=str(training["sha256"]),
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    )


def _verify_signal_links(
    records: tuple[dict[str, Any], ...],
    *,
    dataset_root: Path,
    train_signals: VerifiedTrainSignals,
) -> None:
    for record in records:
        signal = _object(record.get("signal"), "selection signal link")
        path = (dataset_root / str(signal["array"])).resolve()
        _require(path == train_signals.path, "Selection signal array disagrees with staged train tensor")
        _require(int(signal["row"]) < train_signals.shape[0], "Selection signal row is out of range")


RuntimeFactory = Callable[..., Any]


def run_selection_inference(
    *,
    code_revision: str,
    bundle_root: Path,
    bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    starting_adapter: AdapterSpec,
    candidate_path: Path,
    candidate_descriptor_sha256: str,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    model_manifest_path: Path,
    model_manifest_sha256: str,
    output_dir: Path,
    resume: bool = False,
    runtime_factory: RuntimeFactory | None = None,
) -> SelectionInferenceResult:
    """Generate complete greedy selection predictions without opening answer material."""

    _require(os.environ.get("BIOCODER_VERIFIED_CODE_REVISION") == code_revision, "Code revision was not verified by the immutable launcher")
    _require(os.environ.get("BIOCODER_SELECTION_INPUT_SCOPE") == "staged_selection_prompts_and_train_signals_only", "Selection inference requires an answer-free staged view")
    _require(os.environ.get("BIOCODER_GPU_ALLOCATION_MODE") == GPU_ALLOCATION_MODE, "GPU allocation mode was not declared")
    for digest, label in (
        (bundle_report_sha256, "selection bundle hash"),
        (dataset_report_sha256, "dataset hash"),
        (candidate_descriptor_sha256, "candidate descriptor hash"),
        (model_artifact_sha256, "model hash"),
        (model_manifest_sha256, "model manifest hash"),
    ):
        _hex(digest, label)
    assert_no_protected_input_paths(
        [
            bundle_root,
            dataset_root,
            starting_adapter.root,
            candidate_path,
            model_name_or_path,
            model_manifest_path,
            output_dir,
        ]
    )
    inputs = load_answer_free_selection_inputs(bundle_root, bundle_report_sha256)
    train_signals = verify_train_signals(dataset_root, dataset_report_sha256)
    _verify_signal_links(
        inputs.records, dataset_root=dataset_root.resolve(), train_signals=train_signals
    )
    pool_sha = str(_object(inputs.report.get("sources"), "selection sources")["rl_pool_report_sha256"])
    _require(
        sha256_file(candidate_path.resolve()) == candidate_descriptor_sha256,
        "Candidate descriptor hash mismatch",
    )
    candidate = _read_json(candidate_path.resolve(), "candidate descriptor")
    bound_candidate, runtime_weights = verify_selection_candidate_runtime(
        candidate,
        expected_pool_report_sha256=pool_sha,
        starting_adapter=starting_adapter,
    )
    output_dir = output_dir.resolve()
    work_dir = output_dir.parent / f".{output_dir.name}.work"
    _require(not output_dir.exists(), f"Selection inference output exists: {output_dir}")
    configuration = {
        "schema_version": SELECTION_INFERENCE_CONFIG_SCHEMA,
        "code_revision": code_revision,
        "candidate": bound_candidate,
        "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64},
        "sources": {
            "selection_bundle_report_sha256": bundle_report_sha256,
            "selection_prompt_manifest_sha256": inputs.prompt_manifest_sha256,
            "dataset_report_sha256": dataset_report_sha256,
            "candidate_descriptor_sha256": candidate_descriptor_sha256,
            "model_revision": model_revision,
            "model_artifact_sha256": model_artifact_sha256,
            "model_manifest_sha256": model_manifest_sha256,
        },
        "contracts": {
            "selection_answers_opened": False,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "visual_tower_unused": True,
        },
    }
    config_path = work_dir / "selection_inference_config.json"
    journal_path = work_dir / "selection_predictions.jsonl"
    if work_dir.exists():
        _require(resume, f"Incomplete selection inference exists: {work_dir}")
        _require(_read_json(config_path, "selection inference config") == configuration, "Selection inference resume drift")
    else:
        work_dir.mkdir(parents=True)
        _write_json(config_path, configuration)
    completed = (
        _read_jsonl(journal_path, "selection prediction journal")
        if journal_path.exists() and journal_path.stat().st_size > 0
        else []
    )
    _require(
        [row.get("instruction_id") for row in completed]
        == [row["instruction_id"] for row in inputs.records[: len(completed)]],
        "Selection prediction journal is not an exact prefix",
    )
    factory = runtime_factory or XicSamplingRuntime
    runtime = factory(
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        model_artifact_sha256=model_artifact_sha256,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=model_manifest_sha256,
        starting_adapter=starting_adapter,
        train_signals=train_signals,
        runtime_weights=runtime_weights,
        attention_implementation="sdpa",
        seed=17,
    )
    rows = list(completed)
    try:
        with journal_path.open("a", encoding="utf-8") as journal:
            for index, record in enumerate(inputs.records[len(rows) :], len(rows) + 1):
                completion = runtime.greedy_completion(record, max_new_tokens=64)
                row = {
                    "schema_version": SELECTION_PREDICTION_SCHEMA,
                    "instruction_id": record["instruction_id"],
                    "task": record["task"],
                    "language": record["language"],
                    "candidate_id": bound_candidate["candidate_id"],
                    "response": completion.response,
                    "answer_tokens": len(completion.token_ids),
                    "terminated_with_eos": completion.terminated_with_eos,
                }
                journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                rows.append(row)
                if index % 100 == 0 or index == len(inputs.records):
                    print(f"[selection {index}/{len(inputs.records)}] candidate={bound_candidate['candidate_id']}", flush=True)
        metadata = dict(runtime.metadata)
    finally:
        runtime.close()
    _require(len(rows) == len(inputs.records), "Selection inference coverage is incomplete")
    _require(
        metadata.get("visual_forward_calls") == 0,
        "Selection inference invoked the visual tower",
    )
    for sampling_key in (
        "temperature",
        "top_p",
        "max_new_tokens",
        "prefill_once_per_rollout_group",
    ):
        metadata.pop(sampling_key, None)
    metadata["selection_decode"] = configuration["decode"]
    report = {
        "schema_version": SELECTION_INFERENCE_REPORT_SCHEMA,
        "code_revision": code_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": bound_candidate,
        "decode": configuration["decode"],
        "sources": configuration["sources"],
        "counts": {
            "prediction_records": len(rows),
            "source_groups": len({row.get("group_id") for row in inputs.records}),
        },
        "runtime": metadata,
        "contracts": configuration["contracts"],
        "artifacts": {
            "predictions": {
                "path": journal_path.name,
                "sha256": sha256_file(journal_path),
                "records": len(rows),
            },
            "configuration": {
                "path": config_path.name,
                "sha256": sha256_file(config_path),
            },
        },
    }
    report_path = work_dir / "selection_inference_report.json"
    _write_json(report_path, report)
    manifest_path = work_dir / "artifact_manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (journal_path, config_path, report_path)
        ),
        encoding="utf-8",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.replace(output_dir)
    final_report = output_dir / report_path.name
    final_predictions = output_dir / journal_path.name
    return SelectionInferenceResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        predictions_path=final_predictions,
        prediction_records=len(rows),
    )
