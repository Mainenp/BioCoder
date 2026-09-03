"""Resumable, provenance-bound Qwen3-VL inference over prompt-only bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.inference_bundle import (
    BUNDLE_PROMPT_SCHEMA,
    BUNDLE_SCHEMA,
)

GENERATION_CONFIG_SCHEMA = "chrompeak-qwen3vl-generation-config-v1"
GENERATION_RECORD_SCHEMA = "chrompeak-qwen3vl-generation-record-v1"
GENERATION_REPORT_SCHEMA = "chrompeak-qwen3vl-generation-v1"
PREDICTION_SCHEMA = "chrompeak-qwen3vl-prediction-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_40_TO_64 = re.compile(r"^[0-9a-f]{40,64}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GenerationSettings:
    batch_size: int = 1
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 17
    dtype: str = "auto"
    device_map: str = "auto"
    attention_implementation: str | None = None


@dataclass(frozen=True)
class PromptRequest:
    instruction_id: str
    task: str
    image_path: Path
    prompt: str
    language: str | None = None
    pair_id: str | None = None


@dataclass(frozen=True)
class QwenInferenceResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    predictions_path: Path
    prediction_records: int
    complete_prompt_coverage: bool
    development_comparison_candidate: bool


class BatchGenerator(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate(
        self,
        requests: Sequence[PromptRequest],
        settings: GenerationSettings,
    ) -> list[str]: ...


GeneratorFactory = Callable[[str, str, GenerationSettings], BatchGenerator]


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
        raise ValueError(f"Path escapes root for {context}") from error
    return candidate


def _resolve_image(assets_root: Path, relative: Any, instruction_id: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Image path is missing")
    _require("\\" not in relative, f"Non-portable image path: {instruction_id}")
    path = PurePosixPath(relative)
    _require(
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"Unsafe image path: {instruction_id}",
    )
    candidate = (assets_root / Path(*path.parts)).resolve()
    try:
        candidate.relative_to(assets_root)
    except ValueError as error:
        raise ValueError(f"Image path escapes assets root: {instruction_id}") from error
    _require(candidate.is_file(), f"Image not found: {relative}")
    return candidate


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {context} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{context} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {context} JSON at line {line_number}") from error
    _require(bool(records), f"{context} is empty")
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
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


def _append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
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
        stream.flush()
        os.fsync(stream.fileno())


def _validate_settings(settings: GenerationSettings) -> None:
    _require(settings.batch_size >= 1, "Batch size must be positive")
    _require(settings.max_new_tokens >= 1, "max_new_tokens must be positive")
    _require(isinstance(settings.seed, int), "Seed must be an integer")
    _require(bool(settings.dtype), "dtype must not be empty")
    _require(bool(settings.device_map), "device_map must not be empty")
    if settings.do_sample:
        _require(
            settings.temperature is not None
            and math.isfinite(settings.temperature)
            and settings.temperature > 0.0,
            "Sampling requires a positive finite temperature",
        )
        _require(
            settings.top_p is not None
            and math.isfinite(settings.top_p)
            and 0.0 < settings.top_p <= 1.0,
            "Sampling requires top_p in (0, 1]",
        )
    else:
        _require(
            settings.temperature is None and settings.top_p is None,
            "Greedy generation must not declare temperature or top_p",
        )


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(f"Cannot parse package version: {value}")
    return tuple(int(part) for part in match.groups())


class _TransformersGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        model_revision: str,
        settings: GenerationSettings,
    ) -> None:
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "Qwen3-VL inference requires transformers>=4.57.0"
            ) from error
        if _version_tuple(transformers_version) < (4, 57, 0):
            raise RuntimeError(
                f"Qwen3-VL requires transformers>=4.57.0; found {transformers_version}"
            )
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL inference requires torch and Transformers image-text support"
            ) from error

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        model_kwargs: dict[str, Any] = {
            "revision": model_revision,
            "dtype": settings.dtype,
            "device_map": settings.device_map,
            "trust_remote_code": False,
        }
        if settings.attention_implementation is not None:
            model_kwargs["attn_implementation"] = settings.attention_implementation
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            trust_remote_code=False,
        )
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        resolved_revision = getattr(self._model.config, "_commit_hash", None)
        self._torch = torch
        self._metadata = {
            "backend": "transformers",
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "model_class": type(self._model).__name__,
            "processor_class": type(self._processor).__name__,
            "resolved_model_revision": resolved_revision,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def generate(
        self,
        requests: Sequence[PromptRequest],
        settings: GenerationSettings,
    ) -> list[str]:
        messages = []
        for request in requests:
            text = request.prompt.replace("<image>", "", 1).strip()
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": request.image_path.as_uri()},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self._model.device)
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
        }
        if settings.do_sample:
            generation_arguments.update(
                {"temperature": settings.temperature, "top_p": settings.top_p}
            )
        with self._torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_arguments)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return list(
            self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )


def _transformers_generator_factory(
    model_name_or_path: str,
    model_revision: str,
    settings: GenerationSettings,
) -> BatchGenerator:
    return _TransformersGenerator(model_name_or_path, model_revision, settings)


def _validate_prompt(record: dict[str, Any]) -> None:
    instruction_id = record.get("instruction_id")
    _require(record.get("schema_version") == BUNDLE_PROMPT_SCHEMA, "Invalid prompt schema")
    _require(
        isinstance(instruction_id, str) and bool(_HEX_24.fullmatch(instruction_id)),
        f"Invalid inference instruction_id: {instruction_id}",
    )
    _require(isinstance(record.get("task"), str) and bool(record["task"]), "Missing task")
    _require(
        isinstance(record.get("prompt"), str)
        and record["prompt"].count("<image>") == 1,
        f"Invalid prompt text: {instruction_id}",
    )


def _load_bundle(
    bundle_root: Path,
    expected_bundle_report_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    report_path = bundle_root / "inference_bundle_report.json"
    report = _read_json(report_path, "inference bundle report")
    report_hash = sha256_file(report_path)
    _require(report_hash == expected_bundle_report_sha256, "Inference bundle hash mismatch")
    _require(report.get("schema_version") == BUNDLE_SCHEMA, "Unsupported bundle schema")
    contracts = _object(report.get("contracts"), "inference bundle contracts")
    for name, expected in (
        ("prompt_only", True),
        ("answer_key_opened", False),
        ("answer_key_materialized", False),
        ("internal_test_accessed", False),
    ):
        _require(contracts.get(name) is expected, f"Bundle contract failed: {name}")
    _require(report.get("internal_test_accessed") is False, "Bundle accessed internal test")
    artifacts = _object(report.get("artifacts"), "inference bundle artifacts")
    artifact = _object(artifacts.get("inference_prompts"), "inference prompts artifact")
    prompt_path = _safe_child(bundle_root, artifact.get("path"), "inference prompts")
    expected_hash = artifact.get("sha256")
    _require(
        isinstance(expected_hash, str) and bool(_HEX_64.fullmatch(expected_hash)),
        "Inference prompt SHA-256 is invalid",
    )
    _require(sha256_file(prompt_path) == expected_hash, "Inference prompt hash mismatch")
    prompts = _read_jsonl(prompt_path, "inference prompts")
    _require(artifact.get("records") == len(prompts), "Inference prompt count mismatch")
    for record in prompts:
        _validate_prompt(record)
    ids = [record["instruction_id"] for record in prompts]
    _require(len(set(ids)) == len(ids), "Duplicate inference instruction IDs")
    return report, prompts, expected_hash


def _journal_records(
    path: Path,
    selected_prompts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path, "generation journal")
    _require(len(records) <= len(selected_prompts), "Generation journal is too long")
    for index, record in enumerate(records):
        prompt = selected_prompts[index]
        expected_id = prompt["instruction_id"]
        _require(
            record.get("schema_version") == GENERATION_RECORD_SCHEMA,
            "Invalid generation journal schema",
        )
        _require(record.get("instruction_id") == expected_id, "Journal is not a prompt prefix")
        _require(isinstance(record.get("response"), str), "Journal response is invalid")
        _require(record.get("task") == prompt.get("task"), "Journal task mismatch")
        _require(record.get("image") == prompt.get("image"), "Journal image mismatch")
        _require(
            record.get("prompt_sha256")
            == hashlib.sha256(str(prompt["prompt"]).encode("utf-8")).hexdigest(),
            "Journal prompt hash mismatch",
        )
        _require(
            isinstance(record.get("image_sha256"), str)
            and bool(_HEX_64.fullmatch(record["image_sha256"])),
            "Journal image hash is invalid",
        )
        _require(
            record.get("response_sha256")
            == hashlib.sha256(record["response"].encode("utf-8")).hexdigest(),
            "Journal response hash mismatch",
        )
        for name in ("language", "pair_id"):
            _require(record.get(name) == prompt.get(name), f"Journal {name} mismatch")
    return records


def _prompt_order_sha256(prompts: Sequence[dict[str, Any]]) -> str:
    payload = "".join(f"{record['instruction_id']}\n" for record in prompts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_qwen_inference(
    bundle_root: Path,
    assets_root: Path,
    output_dir: Path,
    *,
    expected_bundle_report_sha256: str,
    model_name_or_path: str,
    model_revision: str,
    settings: GenerationSettings = GenerationSettings(),
    model_artifact_sha256: str | None = None,
    max_records: int | None = None,
    resume: bool = False,
    generator_factory: GeneratorFactory | None = None,
) -> QwenInferenceResult:
    """Generate predictions without accepting an instruction-root or answer-key path."""

    bundle_root = bundle_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    _require(bundle_root.is_dir(), f"Inference bundle not found: {bundle_root}")
    _require(assets_root.is_dir(), f"Assets root not found: {assets_root}")
    _require(not output_dir.exists(), f"Inference output already exists: {output_dir}")
    _require(
        bool(_HEX_64.fullmatch(expected_bundle_report_sha256)),
        "Expected bundle SHA-256 must be lowercase hexadecimal",
    )
    _require(bool(model_name_or_path), "Model name or path must not be empty")
    _require(bool(model_revision), "Model revision must not be empty")
    if model_artifact_sha256 is not None:
        _require(
            bool(_HEX_64.fullmatch(model_artifact_sha256)),
            "Model artifact SHA-256 must be lowercase hexadecimal",
        )
    _validate_settings(settings)
    if max_records is not None:
        _require(max_records >= 1, "max_records must be positive")

    bundle_report, prompts, prompt_artifact_sha256 = _load_bundle(
        bundle_root,
        expected_bundle_report_sha256,
    )
    selected_prompts = prompts[:max_records] if max_records is not None else prompts
    complete_prompt_coverage = len(selected_prompts) == len(prompts)
    run_config = {
        "schema_version": GENERATION_CONFIG_SCHEMA,
        "bundle_report_sha256": expected_bundle_report_sha256,
        "prompt_artifact_sha256": prompt_artifact_sha256,
        "prompt_order_sha256": _prompt_order_sha256(selected_prompts),
        "model": {
            "name_or_path": model_name_or_path,
            "requested_revision": model_revision,
            "artifact_sha256": model_artifact_sha256,
        },
        "generation": asdict(settings),
        "scope": {
            "bundle_prompts": len(prompts),
            "selected_prompts": len(selected_prompts),
            "max_records": max_records,
            "complete_prompt_coverage": complete_prompt_coverage,
        },
        "contracts": {
            "input_is_prompt_only_bundle": True,
            "instruction_root_not_accepted": True,
            "answer_key_path_not_accepted": True,
            "internal_test_accessed": False,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.parent / f".{output_dir.name}.work"
    run_config_path = work_dir / "run_config.json"
    journal_path = work_dir / "generation_records.jsonl"
    runtime_path = work_dir / "runtime_metadata.json"
    if work_dir.exists():
        _require(resume, f"Inference work directory exists; use resume: {work_dir}")
        _require(
            _read_json(run_config_path, "existing run config") == run_config,
            "Resume configuration does not match the existing run",
        )
    else:
        work_dir.mkdir()
        _write_json(run_config_path, run_config)

    journal = _journal_records(journal_path, selected_prompts)
    generator: BatchGenerator | None = None
    if len(journal) == len(selected_prompts):
        runtime_metadata = _read_json(runtime_path, "existing runtime metadata")
    else:
        factory = generator_factory or _transformers_generator_factory
        generator = factory(model_name_or_path, model_revision, settings)
        runtime_metadata = _object(generator.metadata, "generator metadata")
        _require(
            isinstance(runtime_metadata.get("backend"), str)
            and bool(runtime_metadata["backend"]),
            "Generator backend metadata is missing",
        )
        if runtime_path.exists():
            _require(
                _read_json(runtime_path, "existing runtime metadata")
                == runtime_metadata,
                "Resume runtime does not match the existing run",
            )
        else:
            _write_json(runtime_path, runtime_metadata)
    _require(
        isinstance(runtime_metadata.get("backend"), str)
        and bool(runtime_metadata["backend"]),
        "Generator backend metadata is missing",
    )

    image_hash_cache: dict[str, str] = {}
    for batch_start in range(len(journal), len(selected_prompts), settings.batch_size):
        batch_prompts = selected_prompts[
            batch_start : batch_start + settings.batch_size
        ]
        requests = []
        for record in batch_prompts:
            image_path = _resolve_image(
                assets_root,
                record.get("image"),
                str(record["instruction_id"]),
            )
            requests.append(
                PromptRequest(
                    instruction_id=str(record["instruction_id"]),
                    task=str(record["task"]),
                    image_path=image_path,
                    prompt=str(record["prompt"]),
                    language=record.get("language"),
                    pair_id=record.get("pair_id"),
                )
            )
        _require(generator is not None, "Generator is unavailable for an incomplete run")
        responses = generator.generate(requests, settings)
        _require(len(responses) == len(requests), "Generator returned the wrong batch size")
        new_records = []
        for record, request, response in zip(batch_prompts, requests, responses):
            _require(isinstance(response, str), "Generator response must be a string")
            image_key = str(request.image_path)
            if image_key not in image_hash_cache:
                image_hash_cache[image_key] = sha256_file(request.image_path)
            generated_record: dict[str, Any] = {
                "schema_version": GENERATION_RECORD_SCHEMA,
                "instruction_id": request.instruction_id,
                "task": request.task,
                "image": str(record["image"]),
                "image_sha256": image_hash_cache[image_key],
                "prompt_sha256": hashlib.sha256(
                    request.prompt.encode("utf-8")
                ).hexdigest(),
                "response": response,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            }
            if request.language is not None:
                generated_record["language"] = request.language
            if request.pair_id is not None:
                generated_record["pair_id"] = request.pair_id
            new_records.append(generated_record)
        _append_jsonl(journal_path, new_records)
        journal.extend(new_records)

    _require(len(journal) == len(selected_prompts), "Generation did not cover selected prompts")
    predictions = [
        {
            "schema_version": PREDICTION_SCHEMA,
            "instruction_id": record["instruction_id"],
            "response": record["response"],
        }
        for record in journal
    ]
    predictions_path = work_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    resolved_revision = runtime_metadata.get("resolved_model_revision")
    model_identity_immutable = bool(
        model_artifact_sha256
        or (
            isinstance(resolved_revision, str)
            and bool(_HEX_40_TO_64.fullmatch(resolved_revision))
        )
    )
    development_candidate = bool(
        complete_prompt_coverage
        and runtime_metadata.get("backend") == "transformers"
        and model_identity_immutable
    )
    source = _object(bundle_report.get("source"), "bundle source")
    report_path = work_dir / "generation_report.json"
    _write_json(
        report_path,
        {
            "schema_version": GENERATION_REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "bundle_report_sha256": expected_bundle_report_sha256,
                "prompt_artifact_sha256": prompt_artifact_sha256,
                "instruction_report_sha256": source.get(
                    "instruction_report_sha256"
                ),
                "validation_prompts_sha256": source.get(
                    "validation_prompts_sha256"
                ),
                "source_dataset_report_sha256": source.get(
                    "source_dataset_report_sha256"
                ),
            },
            "model": {
                "name_or_path": model_name_or_path,
                "requested_revision": model_revision,
                "resolved_revision": resolved_revision,
                "artifact_sha256": model_artifact_sha256,
                "identity_immutable": model_identity_immutable,
            },
            "runtime": runtime_metadata,
            "generation": asdict(settings),
            "scope": run_config["scope"],
            "counts": {
                "predictions": len(predictions),
                "unique_images_opened": len({record["image"] for record in journal}),
            },
            "contracts": {
                "input_is_prompt_only_bundle": True,
                "answer_key_available_to_runner": False,
                "answer_key_opened": False,
                "instruction_manifest_available_to_runner": False,
                "internal_test_accessed": False,
                "predictions_preserve_prompt_order": True,
            },
            "artifacts": {
                "run_config": {
                    "path": run_config_path.name,
                    "sha256": sha256_file(run_config_path),
                },
                "runtime_metadata": {
                    "path": runtime_path.name,
                    "sha256": sha256_file(runtime_path),
                },
                "generation_records": {
                    "path": journal_path.name,
                    "sha256": sha256_file(journal_path),
                    "records": len(journal),
                },
                "predictions": {
                    "path": predictions_path.name,
                    "sha256": sha256_file(predictions_path),
                    "records": len(predictions),
                },
            },
            "prediction_generation_provenance_available": True,
            "development_comparison_candidate": development_candidate,
            "internal_test_accessed": False,
            "final_benchmark_eligible": False,
        },
    )
    work_dir.replace(output_dir)

    final_report = output_dir / "generation_report.json"
    final_predictions = output_dir / "predictions.jsonl"
    return QwenInferenceResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        predictions_path=final_predictions,
        prediction_records=len(predictions),
        complete_prompt_coverage=complete_prompt_coverage,
        development_comparison_candidate=development_candidate,
    )
