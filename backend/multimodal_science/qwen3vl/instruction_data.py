"""Build leakage-resistant Qwen3-VL SFT and validation-evaluation records."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from multimodal_science.baselines.dataset import load_sequence_split
from multimodal_science.data.manifest import sha256_file

DATASET_SCHEMA = "chrompeak-qwen3vl-instructions-v1"
MANIFEST_SCHEMA = "chrompeak-qwen3vl-instruction-manifest-v1"
VALIDATION_PROMPT_SCHEMA = "chrompeak-qwen3vl-validation-prompt-v1"
VALIDATION_ANSWER_SCHEMA = "chrompeak-qwen3vl-validation-answer-v1"
OFFICIAL_FORMAT_REFERENCE = (
    "https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md"
)
TASKS = (
    "peak_presence",
    "peak_presence_metadata",
    "peak_grounding",
    "scientific_qc",
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InstructionDatasetResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    source_dataset_report_sha256: str
    source_assets: int
    train_instructions: int
    validation_instructions: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, description: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{description} must be an object")
    return value


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), description)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} JSON: {path}") from exc


def _safe_artifact_path(root: Path, artifact: dict[str, Any], name: str) -> Path:
    relative = Path(str(artifact.get("path") or ""))
    _require(relative != Path("."), f"Dataset artifact path is empty: {name}")
    _require(not relative.is_absolute(), f"Dataset artifact path must be relative: {name}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Dataset artifact path escapes root: {name}") from exc
    _require(path.is_file(), f"Dataset artifact is missing: {name}")
    expected_hash = artifact.get("sha256")
    _require(
        isinstance(expected_hash, str) and bool(_HEX_64.fullmatch(expected_hash)),
        f"Dataset artifact SHA-256 is invalid: {name}",
    )
    _require(sha256_file(path) == expected_hash, f"Dataset artifact hash mismatch: {name}")
    return path


def _portable_image_path(value: Any, row: int) -> str:
    _require(isinstance(value, str) and value != "", f"Image path is missing at row {row}")
    _require("\\" not in value, f"Image path is not portable at row {row}")
    _require(
        re.match(r"^[A-Za-z]:/", value) is None,
        f"Image path must not contain a drive prefix at row {row}",
    )
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"Image path must be relative at row {row}")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"Image path is unsafe at row {row}",
    )
    return path.as_posix()


def _finite(value: Any, description: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{description} must be finite")
    return result


def _load_examples(
    dataset_root: Path,
) -> tuple[dict[str, Any], str, dict[str, list[dict[str, Any]]]]:
    report_path = dataset_root / "dataset_report.json"
    _require(report_path.is_file(), f"Dataset report not found: {report_path}")
    report = _read_json(report_path, "multimodal Dataset report")
    _require(
        report.get("schema_version") == "chrompeak-multimodal-dataset-v1",
        "Unsupported multimodal Dataset schema",
    )
    splits = report.get("splits")
    _require(
        isinstance(splits, list) and set(splits) == {"train", "validation"},
        "Instruction building requires a train/validation-only Dataset",
    )
    artifacts = _object(report.get("artifacts"), "Dataset artifacts")
    _require(
        not any(
            "internal" in name.lower() or name.lower().startswith("test")
            for name in artifacts
        ),
        "Dataset artifacts expose a protected evaluation split",
    )
    verified_paths = {
        name: _safe_artifact_path(dataset_root, _object(artifact, f"artifact {name}"), name)
        for name, artifact in artifacts.items()
    }
    verified_splits = {
        split: load_sequence_split(dataset_root, split, verify_hashes=False)
        for split in ("train", "validation")
    }
    counts = _object(
        _object(report.get("counts"), "Dataset counts").get("by_split"),
        "split counts",
    )
    examples_by_split: dict[str, list[dict[str, Any]]] = {}
    groups_by_split: dict[str, set[str]] = {}
    global_assets = set()
    for split in ("train", "validation"):
        key = f"{split}_examples"
        _require(key in verified_paths, f"Dataset examples artifact is missing: {split}")
        records = []
        groups = set()
        with verified_paths[key].open(encoding="utf-8") as stream:
            for expected_row, line in enumerate(stream):
                _require(line.strip() != "", f"Blank Dataset example at {split} row {expected_row}")
                try:
                    example = _object(json.loads(line), f"{split} example row {expected_row}")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid Dataset example at {split} row {expected_row}"
                    ) from exc
                _validate_example(example, split, expected_row, report)
                asset_id = example["asset_id"]
                _require(asset_id not in global_assets, f"Asset crosses Dataset splits: {asset_id}")
                global_assets.add(asset_id)
                groups.add(example["group_id"])
                records.append(example)
        declared = _object(counts.get(split), f"declared {split} counts")
        _require(declared.get("assets") == len(records), f"Dataset {split} count mismatch")
        artifact_records = _object(artifacts[key], f"artifact {key}").get("records")
        _require(artifact_records == len(records), f"Dataset {split} artifact count mismatch")
        _require(bool(records), f"Dataset split is empty: {split}")
        _require(
            tuple(record["asset_id"] for record in records)
            == verified_splits[split].asset_ids,
            f"Verified array/example asset alignment mismatch: {split}",
        )
        _require(
            tuple(record["group_id"] for record in records)
            == verified_splits[split].group_ids,
            f"Verified array/example group alignment mismatch: {split}",
        )
        examples_by_split[split] = records
        groups_by_split[split] = groups
    _require(
        not (groups_by_split["train"] & groups_by_split["validation"]),
        "Source-group leakage between train and validation",
    )
    return report, sha256_file(report_path), examples_by_split


def _validate_example(
    example: dict[str, Any], split: str, expected_row: int, report: dict[str, Any]
) -> None:
    _require(
        example.get("schema_version") == "chrompeak-multimodal-example-v1",
        f"Unsupported Dataset example schema at {split} row {expected_row}",
    )
    _require(example.get("row") == expected_row, f"Non-contiguous Dataset row: {split}")
    _require(example.get("split") == split, f"Dataset split mismatch at row {expected_row}")
    for identity in ("asset_id", "record_id", "group_id"):
        _require(
            isinstance(example.get(identity), str) and example[identity] != "",
            f"Missing {identity} at {split} row {expected_row}",
        )
    image = _object(example.get("image"), f"image at {split} row {expected_row}")
    _portable_image_path(image.get("path"), expected_row)
    _require(
        isinstance(image.get("sha256"), str) and bool(_HEX_64.fullmatch(image["sha256"])),
        f"Invalid image SHA-256 at {split} row {expected_row}",
    )
    _require(
        isinstance(image.get("width"), int)
        and image["width"] > 0
        and isinstance(image.get("height"), int)
        and image["height"] > 0,
        f"Invalid image dimensions at {split} row {expected_row}",
    )
    metadata = _object(example.get("metadata"), f"metadata at {split} row {expected_row}")
    _require(
        isinstance(metadata.get("component"), str) and metadata["component"] != "",
        f"Missing component at {split} row {expected_row}",
    )
    _require(
        isinstance(metadata.get("channel"), str) and metadata["channel"] != "",
        f"Missing channel at {split} row {expected_row}",
    )
    for name in ("q1", "q3", "expected_rt_minutes"):
        _finite(metadata.get(name), f"{name} at {split} row {expected_row}")
    window = metadata.get("roi_window_minutes")
    _require(isinstance(window, list) and len(window) == 2, "ROI window must have two values")
    _require(
        _finite(window[0], "ROI lower bound") < _finite(window[1], "ROI upper bound"),
        f"ROI window is invalid at {split} row {expected_row}",
    )
    target = _object(example.get("target"), f"target at {split} row {expected_row}")
    present = target.get("peak_present")
    _require(
        isinstance(present, bool),
        f"Peak target must be boolean at {split} row {expected_row}",
    )
    _require(target.get("supervision_source") == "human", "Primary target must be human-supervised")
    _require(
        target.get("coordinate_system") == "roi_fraction_0_1",
        "Unsupported target coordinate system",
    )
    if present:
        start = _finite(target.get("start_normalized"), "target start")
        end = _finite(target.get("end_normalized"), "target end")
        _require(0.0 <= start < end <= 1.0, "Positive target boundary is invalid")
    else:
        _require(
            target.get("start_normalized") is None and target.get("end_normalized") is None,
            "Negative target boundaries must be null",
        )
    provenance = _object(example.get("provenance"), f"provenance at {split} row {expected_row}")
    _require(
        provenance.get("asset_index_sha256") == report.get("asset_index_sha256"),
        f"Asset-index provenance mismatch at {split} row {expected_row}",
    )


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _metadata_context(example: dict[str, Any]) -> str:
    metadata = example["metadata"]
    return (
        f"Analyte: {metadata['component']}\n"
        f"Channel: {metadata['channel']}\n"
        f"Transition: Q1={float(metadata['q1']):.6g}, Q3={float(metadata['q3']):.6g}\n"
        f"Expected retention time: {float(metadata['expected_rt_minutes']):.6g} min"
    )


def _task_record(example: dict[str, Any], task: str) -> tuple[str, str, str, list[str]] | None:
    target = example["target"]
    present = bool(target["peak_present"])
    image = example["image"]
    if task == "peak_presence":
        prompt = (
            "<image>\nAnalyze this chromatographic ROI image. Return JSON only with exactly "
            "this schema: {\"peak_present\":true|false}."
        )
        response = _json_response({"peak_present": present})
        return prompt, response, "human", ["image"]
    if task == "peak_presence_metadata":
        prompt = (
            "<image>\nAnalyze this chromatographic ROI using the image and acquisition metadata.\n"
            f"{_metadata_context(example)}\n"
            "Return JSON only with exactly this schema: {\"peak_present\":true|false}."
        )
        response = _json_response({"peak_present": present})
        return prompt, response, "human", ["image", "metadata"]
    if task == "peak_grounding":
        if not present:
            return None
        width = int(image["width"])
        height = int(image["height"])
        x1 = max(0, min(width - 1, round(float(target["start_normalized"]) * width)))
        x2 = max(x1 + 1, min(width, round(float(target["end_normalized"]) * width)))
        prompt = (
            "<image>\nLocate the chromatographic peak interval in this ROI image. Return JSON only "
            "with exactly this schema: {\"bbox_2d\":[x1,y1,x2,y2]}. Coordinates must be "
            f"source-image pixels for a {width}x{height} image; x follows retention time and the "
            "box spans the full image height."
        )
        response = _json_response({"bbox_2d": [x1, 0, x2, height]})
        return prompt, response, "human", ["image"]
    if task == "scientific_qc":
        prompt = (
            "<image>\nPerform chromatographic peak QC using the image and acquisition metadata.\n"
            f"{_metadata_context(example)}\n"
            "Return JSON only with exactly this schema: "
            "{\"qc_state\":\"ok|no_peak\",\"reason\":\"peak_detected|no_visible_peak\"}."
        )
        response = _json_response(
            {
                "qc_state": "ok" if present else "no_peak",
                "reason": "peak_detected" if present else "no_visible_peak",
            }
        )
        return prompt, response, "deterministic_rule", ["image", "metadata"]
    raise ValueError(f"Unsupported instruction task: {task}")


def _instruction_id(
    dataset_report_sha256: str, split: str, asset_id: str, task: str
) -> str:
    payload = f"{dataset_report_sha256}\0{split}\0{asset_id}\0{task}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _records_for_split(
    examples: Iterable[dict[str, Any]],
    split: str,
    tasks: tuple[str, ...],
    dataset_report_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = []
    validation_prompts = []
    validation_answers = []
    manifest = []
    for example in examples:
        for task in tasks:
            built = _task_record(example, task)
            if built is None:
                continue
            prompt, response, supervision_source, modalities = built
            _require(prompt.count("<image>") == 1, "Prompt must contain exactly one image token")
            _require(
                "<image>" not in response and "<video>" not in response,
                "Answer contains a visual token",
            )
            _object(json.loads(response), "instruction response")
            instruction_id = _instruction_id(
                dataset_report_sha256, split, example["asset_id"], task
            )
            image_path = _portable_image_path(example["image"]["path"], int(example["row"]))
            if split == "train":
                output_row = len(train_records)
                train_records.append(
                    {
                        "image": image_path,
                        "conversations": [
                            {"from": "human", "value": prompt},
                            {"from": "gpt", "value": response},
                        ],
                    }
                )
                output_artifact = "train_qwen.jsonl"
            else:
                output_row = len(validation_prompts)
                validation_prompts.append(
                    {
                        "schema_version": VALIDATION_PROMPT_SCHEMA,
                        "instruction_id": instruction_id,
                        "task": task,
                        "image": image_path,
                        "conversations": [{"from": "human", "value": prompt}],
                    }
                )
                validation_answers.append(
                    {
                        "schema_version": VALIDATION_ANSWER_SCHEMA,
                        "instruction_id": instruction_id,
                        "task": task,
                        "expected_response": response,
                    }
                )
                output_artifact = "validation_prompts.jsonl"
            manifest.append(
                {
                    "schema_version": MANIFEST_SCHEMA,
                    "instruction_id": instruction_id,
                    "split": split,
                    "task": task,
                    "output_artifact": output_artifact,
                    "output_row": output_row,
                    "asset_id": example["asset_id"],
                    "record_id": example["record_id"],
                    "group_id": example["group_id"],
                    "image_path": image_path,
                    "image_sha256": example["image"]["sha256"],
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    "target_peak_present": bool(example["target"]["peak_present"]),
                    "input_modalities": modalities,
                    "supervision_source": supervision_source,
                    "source_dataset_report_sha256": dataset_report_sha256,
                    "asset_index_sha256": example["provenance"]["asset_index_sha256"],
                }
            )
    return train_records, validation_prompts, validation_answers, manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            stream.write("\n")


def build_instruction_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    tasks: Iterable[str] = TASKS,
) -> InstructionDatasetResult:
    """Publish official-format train records and answer-separated validation records."""

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    _require(dataset_root.is_dir(), f"Dataset root not found: {dataset_root}")
    _require(not output_dir.exists(), f"Instruction output already exists: {output_dir}")
    selected_tasks = tuple(dict.fromkeys(tasks))
    _require(bool(selected_tasks), "At least one instruction task is required")
    unsupported = sorted(set(selected_tasks) - set(TASKS))
    _require(not unsupported, f"Unsupported instruction tasks: {unsupported}")
    source_report, source_report_sha256, examples = _load_examples(dataset_root)
    train, _, _, train_manifest = _records_for_split(
        examples["train"], "train", selected_tasks, source_report_sha256
    )
    _, validation_prompts, validation_answers, validation_manifest = _records_for_split(
        examples["validation"], "validation", selected_tasks, source_report_sha256
    )
    _require(bool(train), "Selected tasks produced no train instructions")
    _require(bool(validation_prompts), "Selected tasks produced no validation instructions")
    _require(
        len(validation_prompts) == len(validation_answers),
        "Validation prompts and answer key are misaligned",
    )
    manifest = train_manifest + validation_manifest
    _require(
        len({record["instruction_id"] for record in manifest}) == len(manifest),
        "Duplicate instruction IDs",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        config_path = staging / "builder_config.json"
        train_path = staging / "train_qwen.jsonl"
        prompts_path = staging / "validation_prompts.jsonl"
        answers_path = staging / "validation_answers.jsonl"
        manifest_path = staging / "instruction_manifest.jsonl"
        _write_json(
            config_path,
            {
                "schema_version": "chrompeak-qwen3vl-builder-config-v1",
                "tasks": list(selected_tasks),
                "train_split": "train",
                "evaluation_split": "validation",
                "internal_test_accessed": False,
                "official_format_reference": OFFICIAL_FORMAT_REFERENCE,
            },
        )
        _write_jsonl(train_path, train)
        _write_jsonl(prompts_path, validation_prompts)
        _write_jsonl(answers_path, validation_answers)
        _write_jsonl(manifest_path, manifest)
        paths = {
            "builder_config": config_path,
            "train_qwen": train_path,
            "validation_prompts": prompts_path,
            "validation_answers": answers_path,
            "instruction_manifest": manifest_path,
        }
        record_counts = {
            "train_qwen": len(train),
            "validation_prompts": len(validation_prompts),
            "validation_answers": len(validation_answers),
            "instruction_manifest": len(manifest),
        }
        artifacts = {}
        for name, path in paths.items():
            artifact: dict[str, Any] = {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            if name in record_counts:
                artifact["records"] = record_counts[name]
            artifacts[name] = artifact
        task_counts: dict[str, Counter[str]] = defaultdict(Counter)
        label_counts: dict[str, Counter[str]] = defaultdict(Counter)
        instruction_assets: dict[str, set[str]] = defaultdict(set)
        instruction_groups: dict[str, set[str]] = defaultdict(set)
        for record in manifest:
            split = record["split"]
            task_counts[split][record["task"]] += 1
            label_counts[split][str(record["target_peak_present"]).lower()] += 1
            instruction_assets[split].add(record["asset_id"])
            instruction_groups[split].add(record["group_id"])
        report_path = staging / "instruction_dataset_report.json"
        _write_json(
            report_path,
            {
                "schema_version": DATASET_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_dataset": {
                    "dataset_report_sha256": source_report_sha256,
                    "asset_index_sha256": source_report.get("asset_index_sha256"),
                    "dataset_version": source_report.get("dataset_version"),
                    "source_assets": sum(len(value) for value in examples.values()),
                    "by_split": {split: len(examples[split]) for split in ("train", "validation")},
                },
                "counts": {
                    "instructions": len(manifest),
                    "by_split": {
                        split: {
                            "source_assets": len(examples[split]),
                            "instruction_backed_assets": len(instruction_assets[split]),
                            "source_groups": len(instruction_groups[split]),
                            "instructions": sum(task_counts[split].values()),
                            "by_task": dict(sorted(task_counts[split].items())),
                            "by_target_presence": dict(sorted(label_counts[split].items())),
                        }
                        for split in ("train", "validation")
                    },
                },
                "contracts": {
                    "tasks": list(selected_tasks),
                    "train_format": "qwen3vl-image-conversations-jsonl",
                    "one_image_token_per_train_record": True,
                    "visual_tokens_forbidden_in_answers": True,
                    "validation_answers_separated_from_prompts": True,
                    "image_paths_relative_to_external_assets_root": True,
                    "images_reopened": False,
                    "official_format_reference": OFFICIAL_FORMAT_REFERENCE,
                    "supervision_sources": ["human", "deterministic_rule"],
                },
                "internal_test_accessed": False,
                "final_benchmark_eligible": False,
                "artifacts": artifacts,
            },
        )
        staging.replace(output_dir)

    final_report = output_dir / "instruction_dataset_report.json"
    return InstructionDatasetResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        source_dataset_report_sha256=source_report_sha256,
        source_assets=sum(len(value) for value in examples.values()),
        train_instructions=len(train),
        validation_instructions=len(validation_prompts),
    )
