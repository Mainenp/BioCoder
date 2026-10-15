"""Build the leakage-safe v2 RL training pool and source-group selection set."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.data.splits import classify_group


RL_POOL_SCHEMA = "chrompeak-qwen3vl-rl-pool-v1"
RL_RECORD_SCHEMA = "chrompeak-qwen3vl-rl-record-v1"
RL_SELECTION_GROUP_SCHEMA = "chrompeak-qwen3vl-rl-selection-group-v1"
SELECTION_HASH_NAMESPACE = "biocoder-v2-rl-selection"
EXPECTED_TRAIN_SOURCE_GROUPS = 87
SELECTION_SOURCE_GROUPS = 8
AUXILIARY_TASKS = (
    "peak_presence",
    "peak_presence_metadata",
    "scientific_qc",
)
GROUNDING_TASK = "peak_grounding"
FORBIDDEN_INPUT_TOKENS = ("internal_test", "validation_answers")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), context)


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            records.append(_object(json.loads(line), f"{context}:{line_number}"))
    _require(bool(records), f"Empty {context}: {path}")
    return records


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


def _safe_child(root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Invalid {context} path")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected relative {context} path")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} path escapes its root") from error
    return candidate


def assert_no_protected_input_paths(paths: Iterable[Path | str]) -> None:
    """Reject every v2 training-side reference to protected evaluation inputs."""

    for path in paths:
        normalized = str(path).replace("\\", "/").casefold()
        for token in FORBIDDEN_INPUT_TOKENS:
            _require(token not in normalized, f"Protected input path is forbidden: {path}")


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    source_path: str
    stratum: str
    provenance_sha256: str


@dataclass(frozen=True)
class RLPoolResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    training_pool_path: Path
    selection_set_path: Path
    training_records: int
    selection_records: int
    training_source_groups: int
    selection_source_groups: int


@dataclass(frozen=True)
class VerifiedRLTrainingPool:
    root: Path
    report: dict[str, Any]
    report_sha256: str
    records: tuple[dict[str, Any], ...]
    selection_group_ids: frozenset[str]


def load_rl_training_pool(
    root: Path, expected_report_sha256: str
) -> VerifiedRLTrainingPool:
    """Load only the training artifact; selection samples stay unopened."""

    root = root.resolve()
    assert_no_protected_input_paths([root])
    report_path = root / "rl_pool_report.json"
    _require(report_path.is_file(), f"Missing RL pool report: {report_path}")
    actual_report_sha256 = sha256_file(report_path)
    _require(
        actual_report_sha256 == expected_report_sha256,
        "RL pool report hash mismatch",
    )
    report = _read_json(report_path, "RL pool report")
    _require(report.get("schema_version") == RL_POOL_SCHEMA, "Wrong RL pool schema")
    contracts = _object(report.get("contracts"), "RL pool contracts")
    for name, expected in (
        ("train_artifacts_only", True),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
        ("selection_groups_trained", False),
        ("vision_inputs_present", False),
    ):
        _require(contracts.get(name) is expected, f"RL pool contract failed: {name}")
    selection = _object(report.get("selection"), "RL selection lock")
    group_ids = selection.get("group_ids")
    _require(
        isinstance(group_ids, list)
        and len(group_ids) == SELECTION_SOURCE_GROUPS
        and all(isinstance(group_id, str) and bool(group_id) for group_id in group_ids),
        "RL selection group lock is invalid",
    )
    _require(len(set(group_ids)) == len(group_ids), "Duplicate selection group lock")
    training_path, artifact = _artifact(
        root, report, "training_pool", "RL training pool"
    )
    records = _read_jsonl(training_path, "RL training pool")
    _require(artifact.get("records") == len(records), "RL training pool count mismatch")
    assert_training_pool_isolation(records, group_ids)
    return VerifiedRLTrainingPool(
        root=root,
        report=report,
        report_sha256=actual_report_sha256,
        records=tuple(records),
        selection_group_ids=frozenset(group_ids),
    )


def selection_group_key(source_path: str) -> str:
    payload = f"{SELECTION_HASH_NAMESPACE}\0{source_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_source_groups(
    groups: Sequence[SourceGroup], *, count: int = SELECTION_SOURCE_GROUPS
) -> tuple[SourceGroup, ...]:
    """Apply the frozen hash order and three-stratum repair deterministically."""

    _require(count > 0, "Selection group count must be positive")
    by_id = {group.group_id: group for group in groups}
    _require(len(by_id) == len(groups), "Duplicate source group IDs")
    by_path = {group.source_path: group for group in groups}
    _require(len(by_path) == len(groups), "Duplicate source paths")
    _require(len(groups) > count, "Selection must leave at least one training group")
    required_strata = {"blank", "qc", "sample"}
    _require(
        {group.stratum for group in groups} == required_strata,
        "Source inventory must contain exactly blank, qc, and sample strata",
    )

    ordered = sorted(groups, key=lambda group: selection_group_key(group.source_path))
    selected = list(ordered[:count])
    missing = sorted(required_strata - {group.stratum for group in selected})
    for stratum in missing:
        replacement = next(group for group in ordered if group.stratum == stratum)
        removable = [group for group in selected if group.stratum == "sample"]
        _require(bool(removable), "Selection coverage repair has no sample group to replace")
        latest_sample = max(
            removable, key=lambda group: selection_group_key(group.source_path)
        )
        selected.remove(latest_sample)
        _require(replacement not in selected, "Coverage replacement duplicates a group")
        selected.append(replacement)

    selected.sort(key=lambda group: selection_group_key(group.source_path))
    _require(len(selected) == count, "Selection group count changed during repair")
    _require(
        {group.stratum for group in selected} == required_strata,
        "Selection set lacks a required source stratum",
    )
    return tuple(selected)


def assert_training_pool_isolation(
    records: Sequence[dict[str, Any]], selection_group_ids: Iterable[str]
) -> None:
    selected = {str(group_id) for group_id in selection_group_ids}
    _require(bool(selected), "Selection source-group lock is empty")
    for row, record in enumerate(records):
        group_id = str(record.get("group_id") or "")
        _require(group_id != "", f"Training record lacks group_id: {row}")
        _require(
            group_id not in selected,
            f"Selection-set sample entered a training process: {group_id}",
        )


def _stable_record_key(seed: int, purpose: str, record: dict[str, Any]) -> str:
    payload = (
        f"{seed}\0{purpose}\0{record.get('task')}\0"
        f"{record.get('instruction_id')}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def construct_training_pool(
    records: Sequence[dict[str, Any]],
    selection_group_ids: Iterable[str],
    *,
    seed: int = 17,
) -> tuple[dict[str, Any], ...]:
    """Keep all grounding prompts and a task-stratified 20% auxiliary slice."""

    selected = {str(group_id) for group_id in selection_group_ids}
    eligible = [record for record in records if str(record.get("group_id")) not in selected]
    assert_training_pool_isolation(eligible, selected)
    grounding = [record for record in eligible if record.get("task") == GROUNDING_TASK]
    _require(bool(grounding), "RL pool has no grounding prompts")

    auxiliary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        task = str(record.get("task") or "")
        if task in AUXILIARY_TASKS:
            auxiliary[task].append(record)
        else:
            _require(task == GROUNDING_TASK, f"Unsupported RL pool task: {task}")
    _require(
        set(auxiliary) == set(AUXILIARY_TASKS),
        "RL pool lacks an auxiliary task stratum",
    )
    for task in AUXILIARY_TASKS:
        auxiliary[task].sort(
            key=lambda record: _stable_record_key(seed, "auxiliary", record)
        )

    target_auxiliary = int(math.floor(len(grounding) / 4.0 + 0.5))
    _require(target_auxiliary > 0, "RL pool is too small for an 80/20 task mix")
    _require(
        sum(len(rows) for rows in auxiliary.values()) >= target_auxiliary,
        "Insufficient auxiliary prompts for the preregistered task mix",
    )
    chosen_auxiliary = []
    offsets = {task: 0 for task in AUXILIARY_TASKS}
    while len(chosen_auxiliary) < target_auxiliary:
        made_progress = False
        for task in AUXILIARY_TASKS:
            offset = offsets[task]
            if offset < len(auxiliary[task]):
                chosen_auxiliary.append(auxiliary[task][offset])
                offsets[task] += 1
                made_progress = True
                if len(chosen_auxiliary) == target_auxiliary:
                    break
        _require(made_progress, "Auxiliary task strata were exhausted")

    pool = [*grounding, *chosen_auxiliary]
    pool.sort(key=lambda record: _stable_record_key(seed, "training-pool", record))
    assert_training_pool_isolation(pool, selected)
    return tuple(pool)


def _artifact(
    root: Path, report: dict[str, Any], name: str, context: str
) -> tuple[Path, dict[str, Any]]:
    artifact = _object(_object(report.get("artifacts"), "artifacts").get(name), name)
    path = _safe_child(root, artifact.get("path"), context)
    _require(path.is_file(), f"Missing {context}: {path}")
    _require(sha256_file(path) == artifact.get("sha256"), f"{context} hash mismatch")
    return path, artifact


def _conversation(record: dict[str, Any], row: int) -> tuple[str, str]:
    messages = record.get("conversations")
    _require(isinstance(messages, list) and len(messages) == 2, f"Bad conversation: {row}")
    human = _object(messages[0], f"human message {row}")
    assistant = _object(messages[1], f"assistant message {row}")
    _require(human.get("from") == "human", f"Bad human role: {row}")
    _require(assistant.get("from") == "gpt", f"Bad assistant role: {row}")
    prompt = human.get("value")
    response = assistant.get("value")
    _require(isinstance(prompt, str) and bool(prompt), f"Empty prompt: {row}")
    _require(isinstance(response, str) and bool(response), f"Empty response: {row}")
    return prompt, response


def _source_group(assets_root: Path, group_id: str) -> SourceGroup:
    provenance_path = (
        assets_root / "jobs" / "train" / group_id / "derivation_provenance.json"
    ).resolve()
    try:
        provenance_path.relative_to(assets_root)
    except ValueError as error:
        raise ValueError("Train provenance escapes the assets root") from error
    provenance = _read_json(provenance_path, f"train provenance {group_id}")
    _require(provenance.get("job_id") == group_id, "Train provenance job mismatch")
    _require(provenance.get("split") == "train", "Non-train provenance entered RL data")
    source_path = provenance.get("source_mzml")
    _require(isinstance(source_path, str) and bool(source_path), "Missing train source path")
    assert_no_protected_input_paths([source_path, provenance_path])
    stratum = classify_group([{"sample_id": Path(source_path).name}])
    return SourceGroup(
        group_id=group_id,
        source_path=source_path.replace("\\", "/"),
        stratum=stratum,
        provenance_sha256=sha256_file(provenance_path),
    )


def _merge_train_inputs(
    *,
    train_rows: Sequence[dict[str, Any]],
    selections: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
    assets_root: Path,
) -> tuple[list[dict[str, Any]], list[SourceGroup]]:
    _require(len(train_rows) == len(selections), "Train row/selection count mismatch")
    links_by_id = {str(link.get("instruction_id") or ""): link for link in links}
    _require(len(links_by_id) == len(links), "Duplicate train XIC link IDs")
    group_cache: dict[str, SourceGroup] = {}
    merged = []
    for row, (train_row, selection) in enumerate(zip(train_rows, selections)):
        instruction_id = str(selection.get("instruction_id") or "")
        _require(instruction_id in links_by_id, f"Missing train XIC link: {instruction_id}")
        link = links_by_id[instruction_id]
        group_id = str(selection.get("group_id") or "")
        _require(group_id != "" and link.get("group_id") == group_id, f"Group mismatch: {row}")
        _require(link.get("split") == "train", f"Non-train XIC link: {row}")
        signal = _object(link.get("signal"), f"signal link {row}")
        scalars = _object(link.get("scalar_features"), f"scalar link {row}")
        for relative in (signal.get("array"), scalars.get("array")):
            _require(
                isinstance(relative, str)
                and relative.replace("\\", "/").startswith("train/"),
                f"Non-train tensor link: {row}",
            )
        if group_id not in group_cache:
            group_cache[group_id] = _source_group(assets_root, group_id)
        source_group = group_cache[group_id]
        prompt, response = _conversation(train_row, row)
        _require(train_row.get("image") == selection.get("image"), f"Image drift: {row}")
        _require(selection.get("asset_id") == link.get("asset_id"), f"Asset drift: {row}")
        _require(selection.get("task") == link.get("task"), f"Task drift: {row}")
        _require(selection.get("language") == link.get("language"), f"Language drift: {row}")
        merged.append(
            {
                "schema_version": RL_RECORD_SCHEMA,
                "source_train_row": row,
                "instruction_id": instruction_id,
                "asset_id": selection.get("asset_id"),
                "group_id": group_id,
                "source_path": source_group.source_path,
                "source_stratum": source_group.stratum,
                "task": selection.get("task"),
                "language": selection.get("language"),
                "target_peak_present": selection.get("target_peak_present"),
                "prompt": prompt,
                "expected_response": response,
                "input_modality": "xic_only",
                "signal": dict(signal),
                "scalar_features": dict(scalars),
            }
        )
    _require(len(links_by_id) == len(merged), "Unused train XIC links")
    return merged, list(group_cache.values())


def build_rl_pool(
    *,
    code_revision: str,
    lora_bundle_root: Path,
    lora_bundle_report_sha256: str,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    assets_root: Path,
    output_dir: Path,
    seed: int = 17,
) -> RLPoolResult:
    """Build the formal 79/8 source-group split from train-only V1.1 inputs."""

    lora_bundle_root = lora_bundle_root.resolve()
    fusion_bundle_root = fusion_bundle_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    assert_no_protected_input_paths(
        [lora_bundle_root, fusion_bundle_root, assets_root, output_dir]
    )
    _require(len(code_revision) == 40, "Code revision must be a full Git SHA")
    _require(seed == 17, "The preregistered RL pool seed is 17")

    lora_report_path = lora_bundle_root / "lora_bundle_report.json"
    fusion_report_path = fusion_bundle_root / "fusion_bundle_report.json"
    _require(
        sha256_file(lora_report_path) == lora_bundle_report_sha256,
        "LoRA bundle report hash mismatch",
    )
    _require(
        sha256_file(fusion_report_path) == fusion_bundle_report_sha256,
        "Fusion bundle report hash mismatch",
    )
    lora = _read_json(lora_report_path, "LoRA bundle report")
    fusion = _read_json(fusion_report_path, "fusion bundle report")
    lora_contracts = _object(lora.get("contracts"), "LoRA contracts")
    fusion_contracts = _object(fusion.get("contracts"), "fusion contracts")
    _require(lora_contracts.get("train_split_only") is True, "LoRA bundle is not train-only")
    _require(
        lora_contracts.get("validation_answers_opened") is False,
        "LoRA bundle opened validation answers",
    )
    _require(lora.get("internal_test_accessed") is False, "LoRA bundle used internal test")
    _require(
        fusion_contracts.get("train_split_only_for_supervision") is True,
        "Fusion bundle supervision is not train-only",
    )
    _require(
        fusion_contracts.get("validation_answer_key_opened") is False,
        "Fusion bundle opened validation answers",
    )
    _require(fusion.get("internal_test_accessed") is False, "Fusion bundle used internal test")

    train_rows_path, train_rows_artifact = _artifact(
        lora_bundle_root, lora, "train_qwen", "LoRA train rows"
    )
    selections_path, selections_artifact = _artifact(
        lora_bundle_root, lora, "selection_manifest", "LoRA train manifest"
    )
    links_path, links_artifact = _artifact(
        fusion_bundle_root, fusion, "train_xic_links", "fusion train links"
    )
    train_rows = _read_jsonl(train_rows_path, "LoRA train rows")
    selections = _read_jsonl(selections_path, "LoRA train manifest")
    links = _read_jsonl(links_path, "fusion train links")
    for artifact, records, label in (
        (train_rows_artifact, train_rows, "LoRA train rows"),
        (selections_artifact, selections, "LoRA train manifest"),
        (links_artifact, links, "fusion train links"),
    ):
        _require(artifact.get("records") == len(records), f"{label} count mismatch")

    merged, groups = _merge_train_inputs(
        train_rows=train_rows,
        selections=selections,
        links=links,
        assets_root=assets_root,
    )
    _require(
        len(groups) == EXPECTED_TRAIN_SOURCE_GROUPS,
        f"Expected {EXPECTED_TRAIN_SOURCE_GROUPS} train source groups, found {len(groups)}",
    )
    selection_groups = select_source_groups(groups)
    selection_group_ids = {group.group_id for group in selection_groups}
    selection_records = [
        record for record in merged if str(record["group_id"]) in selection_group_ids
    ]
    training_pool = list(construct_training_pool(merged, selection_group_ids, seed=seed))
    training_group_ids = {str(record["group_id"]) for record in training_pool}
    _require(not training_group_ids & selection_group_ids, "RL pool/selection leakage")
    _require(
        len(training_group_ids) == EXPECTED_TRAIN_SOURCE_GROUPS - SELECTION_SOURCE_GROUPS,
        "RL training pool lost a source group",
    )
    _require(
        {str(record["group_id"]) for record in selection_records} == selection_group_ids,
        "Selection records do not cover all selected source groups",
    )

    _require(not output_dir.exists(), f"RL pool output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        training_path = staging / "rl_training_pool.jsonl"
        selection_set_path = staging / "rl_selection_set.jsonl"
        selection_groups_path = staging / "rl_selection_groups.jsonl"
        config_path = staging / "rl_pool_config.json"
        report_path = staging / "rl_pool_report.json"
        _write_jsonl(training_path, training_pool)
        _write_jsonl(selection_set_path, selection_records)
        selection_group_rows = [
            {
                "schema_version": RL_SELECTION_GROUP_SCHEMA,
                "selection_rank": rank,
                "selection_key": selection_group_key(group.source_path),
                "group_id": group.group_id,
                "source_path": group.source_path,
                "source_stratum": group.stratum,
                "provenance_sha256": group.provenance_sha256,
                "instruction_records": sum(
                    record["group_id"] == group.group_id for record in selection_records
                ),
            }
            for rank, group in enumerate(selection_groups)
        ]
        _write_jsonl(selection_groups_path, selection_group_rows)
        _write_json(
            config_path,
            {
                "schema_version": "chrompeak-qwen3vl-rl-pool-config-v1",
                "selection_namespace": SELECTION_HASH_NAMESPACE,
                "selection_source_groups": SELECTION_SOURCE_GROUPS,
                "expected_train_source_groups": EXPECTED_TRAIN_SOURCE_GROUPS,
                "grounding_fraction": 0.8,
                "auxiliary_fraction": 0.2,
                "auxiliary_sampling": "task_stratified_stable_hash_round_robin",
                "seed": seed,
            },
        )
        task_counts = Counter(str(record["task"]) for record in training_pool)
        language_counts = Counter(str(record["language"]) for record in training_pool)
        target_counts = Counter(
            str(bool(record["target_peak_present"])).lower() for record in training_pool
        )
        report = {
            "schema_version": RL_POOL_SCHEMA,
            "code_revision": code_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "dataset_report_sha256": _object(
                    fusion.get("sources"), "fusion sources"
                ).get("dataset_report_sha256"),
                "lora_bundle_report_sha256": lora_bundle_report_sha256,
                "lora_train_qwen_sha256": train_rows_artifact["sha256"],
                "lora_selection_manifest_sha256": selections_artifact["sha256"],
                "fusion_bundle_report_sha256": fusion_bundle_report_sha256,
                "fusion_train_xic_links_sha256": links_artifact["sha256"],
            },
            "selection": {
                "namespace": SELECTION_HASH_NAMESPACE,
                "source_groups": len(selection_group_ids),
                "group_ids": sorted(selection_group_ids),
                "source_strata": dict(
                    sorted(Counter(group.stratum for group in selection_groups).items())
                ),
                "absolute_scores_are_optimistically_biased": True,
                "allowed_use": "relative_rl_configuration_and_checkpoint_selection_only",
            },
            "counts": {
                "available_train_source_groups": len(groups),
                "training_source_groups": len(training_group_ids),
                "selection_source_groups": len(selection_group_ids),
                "available_train_instructions": len(merged),
                "training_instructions": len(training_pool),
                "selection_instructions": len(selection_records),
                "by_task": dict(sorted(task_counts.items())),
                "by_language": dict(sorted(language_counts.items())),
                "by_target_peak_present": dict(sorted(target_counts.items())),
                "actual_grounding_fraction": (
                    task_counts[GROUNDING_TASK] / len(training_pool)
                ),
            },
            "contracts": {
                "train_artifacts_only": True,
                "selection_hash_matches_preregistration": True,
                "selection_has_blank_qc_sample_coverage": True,
                "training_selection_source_group_overlap": 0,
                "validation_prompts_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
                "selection_groups_trained": False,
                "vision_inputs_present": False,
                "signals_external_to_bundle": True,
            },
            "development_training_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "artifacts": {
                "training_pool": {
                    "path": training_path.name,
                    "sha256": sha256_file(training_path),
                    "records": len(training_pool),
                },
                "selection_set": {
                    "path": selection_set_path.name,
                    "sha256": sha256_file(selection_set_path),
                    "records": len(selection_records),
                },
                "selection_groups": {
                    "path": selection_groups_path.name,
                    "sha256": sha256_file(selection_groups_path),
                    "records": len(selection_group_rows),
                },
                "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
            },
        }
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (
                    report_path,
                    config_path,
                    selection_groups_path,
                    selection_set_path,
                    training_path,
                )
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    final_report = output_dir / "rl_pool_report.json"
    return RLPoolResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        training_pool_path=output_dir / "rl_training_pool.jsonl",
        selection_set_path=output_dir / "rl_selection_set.jsonl",
        training_records=len(training_pool),
        selection_records=len(selection_records),
        training_source_groups=len(training_group_ids),
        selection_source_groups=len(selection_group_ids),
    )
