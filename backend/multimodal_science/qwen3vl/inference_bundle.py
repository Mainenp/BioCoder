"""Build a hash-bound Qwen3-VL inference bundle that contains no answer key."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.instruction_data import (
    BILINGUAL_DATASET_SCHEMA,
    BILINGUAL_VALIDATION_PROMPT_SCHEMA,
    DATASET_SCHEMA,
    LANGUAGES,
    TASKS,
    VALIDATION_PROMPT_SCHEMA,
)

BUNDLE_SCHEMA = "chrompeak-qwen3vl-inference-bundle-v1"
BUNDLE_PROMPT_SCHEMA = "chrompeak-qwen3vl-inference-prompt-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_INSTRUCTION_SCHEMAS = {DATASET_SCHEMA, BILINGUAL_DATASET_SCHEMA}


@dataclass(frozen=True)
class InferenceBundleResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    prompt_records: int
    source_instruction_report_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON for {context}: {path}") from error


def _safe_child(root: Path, relative: Any, context: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"Invalid path for {context}")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected a relative path for {context}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes instruction root for {context}") from error
    return candidate


def _portable_image_path(value: Any, instruction_id: str) -> str:
    _require(isinstance(value, str) and bool(value), f"Missing image: {instruction_id}")
    _require("\\" not in value, f"Non-portable image path: {instruction_id}")
    _require(re.match(r"^[A-Za-z]:/", value) is None, "Image path has a drive prefix")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"Image path must be relative: {instruction_id}")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"Unsafe image path: {instruction_id}",
    )
    return path.as_posix()


def _read_prompts(path: Path, expected_records: Any) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Validation prompts not found: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank validation prompt at line {line_number}")
            try:
                records.append(
                    _object(json.loads(line), f"validation prompt line {line_number}")
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid validation prompt JSON at line {line_number}"
                ) from error
    _require(
        isinstance(expected_records, int) and len(records) == expected_records,
        "Validation prompt record count mismatch",
    )
    _require(bool(records), "Validation prompts are empty")
    return records


def _bundle_prompt(
    record: dict[str, Any],
    *,
    bilingual: bool,
) -> dict[str, Any]:
    instruction_id = record.get("instruction_id")
    _require(
        isinstance(instruction_id, str) and bool(_HEX_24.fullmatch(instruction_id)),
        f"Invalid instruction_id: {instruction_id}",
    )
    expected_schema = (
        BILINGUAL_VALIDATION_PROMPT_SCHEMA if bilingual else VALIDATION_PROMPT_SCHEMA
    )
    _require(record.get("schema_version") == expected_schema, "Invalid prompt schema")
    task = record.get("task")
    _require(task in TASKS, f"Unsupported prompt task: {task}")
    conversations = record.get("conversations")
    _require(
        isinstance(conversations, list) and len(conversations) == 1,
        f"Prompt must contain exactly one message: {instruction_id}",
    )
    message = _object(conversations[0], f"prompt message {instruction_id}")
    prompt = message.get("value")
    _require(
        message.get("from") == "human" and isinstance(prompt, str),
        f"Prompt must be a human string: {instruction_id}",
    )
    _require(
        prompt.count("<image>") == 1 and "<video>" not in prompt,
        f"Prompt visual-token contract failed: {instruction_id}",
    )
    _require("expected_response" not in record, f"Prompt leaks an answer: {instruction_id}")
    result = {
        "schema_version": BUNDLE_PROMPT_SCHEMA,
        "instruction_id": instruction_id,
        "task": task,
        "image": _portable_image_path(record.get("image"), instruction_id),
        "prompt": prompt,
    }
    if bilingual:
        pair_id = record.get("pair_id")
        language = record.get("language")
        _require(
            isinstance(pair_id, str) and bool(_HEX_24.fullmatch(pair_id)),
            f"Invalid pair_id: {instruction_id}",
        )
        _require(language in LANGUAGES, f"Invalid prompt language: {instruction_id}")
        result.update({"pair_id": pair_id, "language": language})
    return result


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
                + "\n"
            )


def build_inference_bundle(
    instruction_root: Path,
    output_dir: Path,
    *,
    expected_instruction_report_sha256: str,
) -> InferenceBundleResult:
    """Publish validation prompts without copying or opening the answer artifact."""

    instruction_root = instruction_root.resolve()
    output_dir = output_dir.resolve()
    _require(instruction_root.is_dir(), f"Instruction root not found: {instruction_root}")
    _require(not output_dir.exists(), f"Inference bundle already exists: {output_dir}")
    _require(
        bool(_HEX_64.fullmatch(expected_instruction_report_sha256)),
        "Expected instruction report SHA-256 must be lowercase hexadecimal",
    )

    source_report_path = instruction_root / "instruction_dataset_report.json"
    source_report = _read_json(source_report_path, "instruction report")
    source_report_sha256 = sha256_file(source_report_path)
    _require(
        source_report_sha256 == expected_instruction_report_sha256,
        "Instruction report hash mismatch",
    )
    source_schema = source_report.get("schema_version")
    _require(
        source_schema in _SUPPORTED_INSTRUCTION_SCHEMAS,
        "Unsupported instruction report schema",
    )
    _require(
        source_report.get("internal_test_accessed") is False,
        "Instruction report must not access internal test data",
    )
    bilingual = source_schema == BILINGUAL_DATASET_SCHEMA
    artifacts = _object(source_report.get("artifacts"), "instruction artifacts")
    prompt_artifact = _object(
        artifacts.get("validation_prompts"), "validation prompt artifact"
    )
    prompt_path = _safe_child(
        instruction_root,
        prompt_artifact.get("path"),
        "validation prompts",
    )
    expected_prompt_hash = prompt_artifact.get("sha256")
    _require(
        isinstance(expected_prompt_hash, str)
        and bool(_HEX_64.fullmatch(expected_prompt_hash)),
        "Validation prompt SHA-256 is invalid",
    )
    _require(
        sha256_file(prompt_path) == expected_prompt_hash,
        "Validation prompt hash mismatch",
    )
    source_prompts = _read_prompts(prompt_path, prompt_artifact.get("records"))
    prompts = [_bundle_prompt(record, bilingual=bilingual) for record in source_prompts]
    instruction_ids = [record["instruction_id"] for record in prompts]
    _require(
        len(set(instruction_ids)) == len(instruction_ids),
        "Duplicate instruction IDs in inference prompts",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}-staging-",
    ) as staging_name:
        staging = Path(staging_name)
        prompts_path = staging / "inference_prompts.jsonl"
        _write_jsonl(prompts_path, prompts)
        counts: dict[str, Any] = {
            "prompts": len(prompts),
            "by_task": dict(sorted(Counter(row["task"] for row in prompts).items())),
        }
        if bilingual:
            counts["by_language"] = dict(
                sorted(Counter(row["language"] for row in prompts).items())
            )
            counts["semantic_language_pairs"] = len(
                {str(row["pair_id"]) for row in prompts}
            )
        report_path = staging / "inference_bundle_report.json"
        _write_json(
            report_path,
            {
                "schema_version": BUNDLE_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "instruction_report_sha256": source_report_sha256,
                    "instruction_schema_version": source_schema,
                    "validation_prompts_sha256": expected_prompt_hash,
                    "source_dataset_report_sha256": _object(
                        source_report.get("source_dataset"), "source dataset"
                    ).get("dataset_report_sha256"),
                },
                "counts": counts,
                "contracts": {
                    "prompt_only": True,
                    "answer_key_opened": False,
                    "answer_key_materialized": False,
                    "instruction_manifest_materialized": False,
                    "one_image_token_per_prompt": True,
                    "image_paths_relative_to_external_assets_root": True,
                    "internal_test_accessed": False,
                    "language_variants_are_not_independent_assets": bilingual,
                },
                "artifacts": {
                    "inference_prompts": {
                        "path": prompts_path.name,
                        "sha256": sha256_file(prompts_path),
                        "records": len(prompts),
                    }
                },
                "internal_test_accessed": False,
            },
        )
        staging.replace(output_dir)

    final_report = output_dir / "inference_bundle_report.json"
    return InferenceBundleResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        prompt_records=len(prompts),
        source_instruction_report_sha256=source_report_sha256,
    )
