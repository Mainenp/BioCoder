"""Build a train-only, provenance-bound Qwen3-VL LoRA bundle."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.instruction_data import BILINGUAL_DATASET_SCHEMA


LORA_BUNDLE_SCHEMA = "chrompeak-qwen3vl-lora-bundle-v1"
LORA_SELECTION_SCHEMA = "chrompeak-qwen3vl-lora-selection-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LoraBundleResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    training_path: Path
    training_records: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _read_object(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON for {context}: {path}") from error


def _safe_child(root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Invalid path for {context}")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected relative path for {context}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes root for {context}") from error
    return candidate


def _portable_image(assets_root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Missing image for {context}")
    _require("\\" not in relative, f"Non-portable image path for {context}")
    pure_path = PurePosixPath(relative)
    _require(
        not pure_path.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure_path.parts),
        f"Unsafe image path for {context}",
    )
    candidate = (assets_root / Path(*pure_path.parts)).resolve()
    try:
        candidate.relative_to(assets_root)
    except ValueError as error:
        raise ValueError(f"Image path escapes assets root for {context}") from error
    _require(candidate.is_file(), f"Training image does not exist: {relative}")
    return candidate


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {context} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{context} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {context} JSON at line {line_number}") from error
    return records


def _read_manifest_prefix(
    path: Path,
    expected_records: int,
) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for output_row in range(expected_records):
            line = stream.readline()
            _require(bool(line), "Instruction manifest ended before train rows")
            try:
                record = _object(json.loads(line), f"train manifest row {output_row}")
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid train manifest JSON at row {output_row}") from error
            _require(record.get("split") == "train", "Validation record entered train prefix")
            _require(record.get("output_artifact") == "train_qwen.jsonl", "Wrong train artifact")
            _require(record.get("output_row") == output_row, "Train manifest row mismatch")
            records.append(record)
    return records


def _conversation(record: dict[str, Any], row: int) -> tuple[str, str]:
    conversations = record.get("conversations")
    _require(isinstance(conversations, list) and len(conversations) == 2, f"Bad turns: {row}")
    human = _object(conversations[0], f"human turn {row}")
    assistant = _object(conversations[1], f"assistant turn {row}")
    _require(human.get("from") == "human", f"First turn is not human: {row}")
    _require(assistant.get("from") == "gpt", f"Second turn is not gpt: {row}")
    prompt = human.get("value")
    response = assistant.get("value")
    _require(isinstance(prompt, str) and prompt.count("<image>") == 1, f"Bad prompt: {row}")
    _require(isinstance(response, str) and bool(response), f"Empty response: {row}")
    _require(
        "<image>" not in response and "<video>" not in response,
        f"Visual token in answer: {row}",
    )
    try:
        _object(json.loads(response), f"response {row}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Response is not JSON at row {row}") from error
    return prompt, response


def _stable_score(seed: int, manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{seed}\0{manifest['instruction_id']}\0{manifest['output_row']}".encode("utf-8")
    ).hexdigest()


def _select_rows(
    manifests: list[dict[str, Any]],
    max_records: int | None,
    seed: int,
) -> list[int]:
    if max_records is None or max_records >= len(manifests):
        return list(range(len(manifests)))
    _require(max_records >= 1, "max_records must be positive")
    strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in manifests:
        key = (
            str(record.get("task")),
            str(record.get("language", "unspecified")),
            bool(record.get("target_peak_present")),
        )
        strata[key].append(record)
    queues = {
        key: deque(sorted(value, key=lambda item: _stable_score(seed, item)))
        for key, value in strata.items()
    }
    selected = []
    keys = sorted(
        queues,
        key=lambda key: hashlib.sha256(
            f"{seed}\0{'|'.join(str(part) for part in key)}".encode("utf-8")
        ).hexdigest(),
    )
    while len(selected) < max_records:
        made_progress = False
        for key in keys:
            if queues[key] and len(selected) < max_records:
                selected.append(int(queues[key].popleft()["output_row"]))
                made_progress = True
        _require(made_progress, "Could not satisfy LoRA sampling request")
    return sorted(selected, key=lambda row: _stable_score(seed, manifests[row]))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def build_lora_training_bundle(
    *,
    instruction_root: Path,
    instruction_report_sha256: str,
    assets_root: Path,
    output_dir: Path,
    max_records: int | None = None,
    seed: int = 17,
) -> LoraBundleResult:
    """Publish an official-format train subset without opening validation answers."""

    instruction_root = instruction_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    _require(bool(_HEX_64.fullmatch(instruction_report_sha256)), "Bad instruction report hash")
    _require(isinstance(seed, int), "Seed must be an integer")
    if max_records is not None:
        _require(isinstance(max_records, int) and max_records >= 1, "Bad max_records")
    report_path = instruction_root / "instruction_dataset_report.json"
    report = _read_object(report_path, "instruction dataset report")
    _require(
        sha256_file(report_path) == instruction_report_sha256,
        "Instruction report hash mismatch",
    )
    _require(
        report.get("schema_version") == BILINGUAL_DATASET_SCHEMA,
        "Bilingual instructions required",
    )
    _require(
        report.get("internal_test_accessed") is False,
        "Instruction data accessed internal test",
    )
    _require(report.get("final_benchmark_eligible") is False, "Unexpected final benchmark state")
    contracts = _object(report.get("contracts"), "instruction contracts")
    for name in (
        "one_image_token_per_train_record",
        "visual_tokens_forbidden_in_answers",
        "validation_answers_separated_from_prompts",
        "image_paths_relative_to_external_assets_root",
        "train_has_one_language_per_semantic_instruction",
    ):
        _require(contracts.get(name) is True, f"Instruction contract failed: {name}")

    artifacts = _object(report.get("artifacts"), "instruction artifacts")
    train_artifact = _object(artifacts.get("train_qwen"), "train_qwen artifact")
    manifest_artifact = _object(artifacts.get("instruction_manifest"), "manifest artifact")
    train_path = _safe_child(instruction_root, train_artifact.get("path"), "train_qwen")
    manifest_path = _safe_child(
        instruction_root, manifest_artifact.get("path"), "instruction manifest"
    )
    _require(
        sha256_file(train_path) == train_artifact.get("sha256"),
        "Train artifact hash mismatch",
    )
    _require(
        sha256_file(manifest_path) == manifest_artifact.get("sha256"),
        "Instruction manifest hash mismatch",
    )
    train_records = _read_jsonl(train_path, "train instruction")
    train_count = train_artifact.get("records")
    _require(
        isinstance(train_count, int) and train_count == len(train_records),
        "Train count mismatch",
    )
    train_manifests = _read_manifest_prefix(manifest_path, train_count)
    _require(len(train_manifests) == train_count, "Train manifest count mismatch")

    seen_ids = set()
    declared_images: dict[str, str] = {}
    for row, (record, manifest) in enumerate(zip(train_records, train_manifests)):
        instruction_id = manifest.get("instruction_id")
        _require(
            isinstance(instruction_id, str) and bool(_HEX_24.fullmatch(instruction_id)),
            f"Bad instruction id: {row}",
        )
        _require(instruction_id not in seen_ids, f"Duplicate instruction id: {instruction_id}")
        seen_ids.add(instruction_id)
        _require(
            isinstance(manifest.get("task"), str) and bool(manifest["task"]),
            f"Bad task: {row}",
        )
        _require(
            manifest.get("language") in {"en", "zh-CN"},
            f"Bad language: {row}",
        )
        _require(
            isinstance(manifest.get("target_peak_present"), bool),
            f"Bad peak-presence target: {row}",
        )
        for field in ("asset_id", "group_id"):
            _require(
                isinstance(manifest.get(field), str) and bool(manifest[field]),
                f"Bad {field}: {row}",
            )
        prompt, response = _conversation(record, row)
        image = record.get("image")
        _require(image == manifest.get("image_path"), f"Image/manifest mismatch: {row}")
        _require(
            hashlib.sha256(response.encode("utf-8")).hexdigest()
            == manifest.get("response_sha256"),
            f"Response/manifest mismatch: {row}",
        )
        declared_image_hash = manifest.get("image_sha256")
        _require(
            isinstance(declared_image_hash, str)
            and bool(_HEX_64.fullmatch(declared_image_hash)),
            f"Bad image hash: {row}",
        )
        if image not in declared_images:
            declared_images[image] = declared_image_hash
        else:
            _require(
                declared_images[image] == declared_image_hash,
                f"Inconsistent image hash: {instruction_id}",
            )
        _require(bool(prompt.strip()), f"Empty prompt: {row}")

    selected_rows = _select_rows(train_manifests, max_records, seed)
    checked_images = set()
    for row in selected_rows:
        manifest = train_manifests[row]
        image = str(manifest["image_path"])
        if image in checked_images:
            continue
        image_path = _portable_image(assets_root, image, str(manifest["instruction_id"]))
        _require(
            sha256_file(image_path) == declared_images[image],
            f"Image hash mismatch: {manifest['instruction_id']}",
        )
        checked_images.add(image)
    selected_records = [train_records[row] for row in selected_rows]
    selection_records = []
    strata = Counter()
    for output_row, source_row in enumerate(selected_rows):
        manifest = train_manifests[source_row]
        stratum = {
            "task": manifest["task"],
            "language": manifest.get("language", "unspecified"),
            "target_peak_present": bool(manifest["target_peak_present"]),
        }
        stratum_key = (
            stratum["task"],
            stratum["language"],
            str(stratum["target_peak_present"]).lower(),
        )
        strata[stratum_key] += 1
        selection_records.append(
            {
                "schema_version": LORA_SELECTION_SCHEMA,
                "output_row": output_row,
                "source_train_row": source_row,
                "instruction_id": manifest["instruction_id"],
                "asset_id": manifest["asset_id"],
                "group_id": manifest["group_id"],
                "image": manifest["image_path"],
                "image_sha256": manifest["image_sha256"],
                **stratum,
            }
        )

    _require(not output_dir.exists(), f"LoRA bundle output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        training_path = staging / "train_qwen.jsonl"
        selection_path = staging / "selection_manifest.jsonl"
        config_path = staging / "bundle_config.json"
        result_report_path = staging / "lora_bundle_report.json"
        _write_jsonl(training_path, selected_records)
        _write_jsonl(selection_path, selection_records)
        _write_json(
            config_path,
            {
                "schema_version": "chrompeak-qwen3vl-lora-bundle-config-v1",
                "seed": seed,
                "max_records": max_records,
                "selection": (
                    "all_train_records"
                    if len(selected_rows) == train_count
                    else "stratified_round_robin_task_language_label"
                ),
            },
        )
        output_artifacts = {
            "train_qwen": training_path,
            "selection_manifest": selection_path,
            "bundle_config": config_path,
        }
        source_dataset = _object(report.get("source_dataset"), "source dataset")
        split_counts = _object(
            _object(report.get("counts"), "instruction counts").get("by_split"),
            "split counts",
        )
        train_split = _object(split_counts.get("train"), "train counts")
        _write_json(
            result_report_path,
            {
                "schema_version": LORA_BUNDLE_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "instruction_report_path": str(report_path),
                    "instruction_report_sha256": instruction_report_sha256,
                    "train_qwen_sha256": train_artifact["sha256"],
                    "instruction_manifest_sha256": manifest_artifact["sha256"],
                    "dataset_report_sha256": source_dataset.get("dataset_report_sha256"),
                    "asset_index_sha256": source_dataset.get("asset_index_sha256"),
                },
                "counts": {
                    "available_train_instructions": train_count,
                    "selected_train_instructions": len(selected_rows),
                    "selected_unique_assets": len(
                        {record["asset_id"] for record in selection_records}
                    ),
                    "verified_selected_images": len(checked_images),
                    "selected_source_groups": len(
                        {record["group_id"] for record in selection_records}
                    ),
                    "available_train_source_groups": train_split.get("source_groups"),
                    "by_stratum": {
                        "|".join(key): value for key, value in sorted(strata.items())
                    },
                },
                "selection": {
                    "seed": seed,
                    "max_records": max_records,
                    "sample_capped": len(selected_rows) < train_count,
                    "method": (
                        "all_train_records"
                        if len(selected_rows) == train_count
                        else "stratified_round_robin_task_language_label"
                    ),
                },
                "contracts": {
                    "train_split_only": True,
                    "validation_prompts_opened": False,
                    "validation_answers_opened": False,
                    "internal_test_accessed": False,
                    "official_image_conversations_format": True,
                    "assets_external_to_bundle": True,
                },
                "development_training_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
                "artifacts": {
                    name: {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        **(
                            {"records": len(selected_rows)}
                            if name in {"train_qwen", "selection_manifest"}
                            else {}
                        ),
                    }
                    for name, path in output_artifacts.items()
                },
            },
        )
        manifest_output = staging / "artifact_manifest.sha256"
        manifest_output.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (
                    config_path,
                    result_report_path,
                    selection_path,
                    training_path,
                )
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    final_report = output_dir / "lora_bundle_report.json"
    return LoraBundleResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        training_path=output_dir / "train_qwen.jsonl",
        training_records=len(selected_rows),
    )
