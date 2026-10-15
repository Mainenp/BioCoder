"""Single-GPU, resumable BF16 LoRA training for Qwen3-VL."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.lora_data import LORA_BUNDLE_SCHEMA


LORA_TRAINING_CONFIG_SCHEMA = "chrompeak-qwen3vl-lora-training-config-v1"
LORA_TRAINING_REPORT_SCHEMA = "chrompeak-qwen3vl-lora-training-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_IGNORE_INDEX = -100


@dataclass(frozen=True)
class LoraTrainingSettings:
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    max_length: int = 1024
    min_pixels: int = 16 * 28 * 28
    max_pixels: int = 160 * 28 * 28
    save_steps: int = 50
    log_steps: int = 1
    seed: int = 17
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    deterministic_warn_only: bool = True


@dataclass(frozen=True)
class LoraTrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    adapter_dir: Path
    global_steps: int
    development_training_complete: bool


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


def _image_path(assets_root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Missing training image")
    _require("\\" not in relative, "Non-portable training image path")
    pure_path = PurePosixPath(relative)
    _require(
        not pure_path.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure_path.parts),
        "Unsafe training image path",
    )
    candidate = (assets_root / Path(*pure_path.parts)).resolve()
    try:
        candidate.relative_to(assets_root)
    except ValueError as error:
        raise ValueError("Training image escapes assets root") from error
    _require(candidate.is_file(), f"Training image not found: {relative}")
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
    _require(bool(records), f"{context} is empty")
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _validate_settings(settings: LoraTrainingSettings) -> None:
    _require(settings.epochs >= 1, "epochs must be positive")
    if settings.max_steps is not None:
        _require(settings.max_steps >= 1, "max_steps must be positive")
    for name in ("batch_size", "gradient_accumulation_steps", "save_steps", "log_steps"):
        _require(getattr(settings, name) >= 1, f"{name} must be positive")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(settings, name)
        _require(math.isfinite(value) and value > 0.0, f"{name} must be positive and finite")
    _require(
        math.isfinite(settings.weight_decay) and settings.weight_decay >= 0.0,
        "weight_decay must be finite and nonnegative",
    )
    _require(0.0 <= settings.warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)")
    _require(settings.lora_rank >= 1 and settings.lora_alpha >= 1, "Bad LoRA dimensions")
    _require(0.0 <= settings.lora_dropout < 1.0, "lora_dropout must be in [0, 1)")
    _require(settings.max_length >= 64, "max_length is too small")
    _require(
        settings.min_pixels >= 28 * 28 and settings.max_pixels >= settings.min_pixels,
        "Invalid image pixel bounds",
    )
    _require(settings.attention_implementation in {"sdpa", "eager"}, "Unsupported attention")


def assistant_supervision_labels(
    full_input_ids: Sequence[int],
    prompt_input_ids: Sequence[int],
    *,
    ignore_index: int = _IGNORE_INDEX,
) -> list[int]:
    """Mask all user/image tokens and retain only assistant-generation targets."""

    full = list(full_input_ids)
    prompt = list(prompt_input_ids)
    _require(len(full) > len(prompt), "Assistant answer produced no supervision tokens")
    _require(full[: len(prompt)] == prompt, "Prompt is not an exact prefix of full conversation")
    return [ignore_index] * len(prompt) + full[len(prompt) :]


def _load_bundle(
    bundle_root: Path,
    expected_report_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    _require(bool(_HEX_64.fullmatch(expected_report_sha256)), "Bad LoRA bundle hash")
    report_path = bundle_root / "lora_bundle_report.json"
    report = _read_object(report_path, "LoRA bundle report")
    _require(sha256_file(report_path) == expected_report_sha256, "LoRA bundle hash mismatch")
    _require(report.get("schema_version") == LORA_BUNDLE_SCHEMA, "Unsupported LoRA bundle")
    _require(report.get("development_training_eligible") is True, "Bundle is not trainable")
    _require(report.get("final_benchmark_eligible") is False, "Unexpected benchmark scope")
    _require(report.get("internal_test_accessed") is False, "Bundle accessed internal test")
    contracts = _object(report.get("contracts"), "LoRA bundle contracts")
    for name, expected in (
        ("train_split_only", True),
        ("validation_prompts_opened", False),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
        ("official_image_conversations_format", True),
    ):
        _require(contracts.get(name) is expected, f"LoRA bundle contract failed: {name}")
    artifact = _object(
        _object(report.get("artifacts"), "LoRA bundle artifacts").get("train_qwen"),
        "LoRA train artifact",
    )
    train_path = _safe_child(bundle_root, artifact.get("path"), "LoRA train artifact")
    train_hash = sha256_file(train_path)
    _require(train_hash == artifact.get("sha256"), "LoRA training rows hash mismatch")
    records = _read_jsonl(train_path, "LoRA training row")
    _require(artifact.get("records") == len(records), "LoRA training row count mismatch")
    return report, records, train_hash


def _messages(
    record: dict[str, Any], image_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversations = record.get("conversations")
    _require(isinstance(conversations, list) and len(conversations) == 2, "Bad conversation")
    human = _object(conversations[0], "human turn")
    assistant = _object(conversations[1], "assistant turn")
    _require(human.get("from") == "human" and assistant.get("from") == "gpt", "Bad roles")
    prompt = human.get("value")
    response = assistant.get("value")
    _require(isinstance(prompt, str) and prompt.count("<image>") == 1, "Bad prompt")
    _require(isinstance(response, str) and bool(response), "Bad assistant response")
    prompt_text = prompt.replace("<image>", "", 1).strip()
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt_text},
        ],
    }
    answer = {"role": "assistant", "content": [{"type": "text", "text": response}]}
    return [user], [user, answer]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in (output_dir / "checkpoints").glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None))[1]


def reconcile_history(path: Path, checkpoint_step: int) -> int:
    """Drop only the uncheckpointed history suffix before a resumed run."""

    _require(checkpoint_step >= 0, "Checkpoint step cannot be negative")
    if not path.exists():
        _require(checkpoint_step == 0, "Checkpoint has no matching training history")
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"Blank training history line: {index}")
        try:
            record = _object(json.loads(line), f"training history line {index}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid training history JSON at line {index}") from error
        _require(record.get("global_step") == index, "Training history steps are not contiguous")
        records.append(record)
    _require(len(records) >= checkpoint_step, "Checkpoint is ahead of training history")
    discarded = len(records) - checkpoint_step
    if discarded:
        temporary = path.with_name(f".{path.name}.resume")
        temporary.write_text(
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for record in records[:checkpoint_step]
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    return discarded


def epoch_sample_indices(
    record_count: int,
    batch_size: int,
    seed: int,
    start_batch: int = 0,
) -> list[int]:
    """Return the deterministic epoch-order suffix needed by a resumed run."""

    _require(record_count >= 1, "record_count must be positive")
    _require(batch_size >= 1, "batch_size must be positive")
    _require(start_batch >= 0, "start_batch must be nonnegative")
    batch_count = math.ceil(record_count / batch_size)
    _require(start_batch <= batch_count, "start_batch exceeds the epoch")
    indices = list(range(record_count))
    random.Random(seed).shuffle(indices)
    return indices[min(start_batch * batch_size, record_count) :]


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    output_dir: Path,
    state: dict[str, Any],
) -> Path:
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"step-{state['global_step']:08d}"
    _require(not final.exists(), f"Checkpoint already exists: {final}")
    with tempfile.TemporaryDirectory(dir=checkpoints, prefix=f".{final.name}-") as staging_name:
        staging = Path(staging_name)
        model.save_pretrained(staging, safe_serialization=True)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        torch.save(
            {
                "python": random.getstate(),
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            staging / "rng_state.pt",
        )
        _write_json(staging / "trainer_state.json", state)
        Path(staging_name).replace(final)
    return final


def run_lora_training(
    *,
    bundle_root: Path,
    bundle_report_sha256: str,
    assets_root: Path,
    output_dir: Path,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    code_revision: str,
    settings: LoraTrainingSettings,
    resume: bool = False,
) -> LoraTrainingResult:
    """Train language-attention LoRA adapters while keeping image and split contracts sealed."""

    _validate_settings(settings)
    _require(bool(model_name_or_path), "Model path is required")
    _require(bool(model_revision), "Model revision is required")
    _require(bool(_HEX_64.fullmatch(model_artifact_sha256)), "Bad model artifact hash")
    _require(bool(_HEX_40.fullmatch(code_revision)), "Bad code revision")
    bundle_root = bundle_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    bundle_report, records, train_rows_sha256 = _load_bundle(
        bundle_root, bundle_report_sha256
    )
    bundle_counts = _object(bundle_report.get("counts"), "LoRA bundle counts")
    available_records = bundle_counts.get("available_train_instructions")
    _require(
        isinstance(available_records, int) and available_records >= len(records),
        "Invalid available train count",
    )
    sample_capped = len(records) < available_records
    smoke_test = sample_capped or settings.max_steps is not None
    for record in records:
        _image_path(assets_root, record.get("image"))

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "LoRA training requires torch, transformers>=4.57, peft and safetensors"
        ) from error

    _require(torch.cuda.is_available(), "Qwen3-VL LoRA training requires CUDA")
    _require(torch.cuda.device_count() == 1, "Expose exactly one CUDA device per LoRA process")
    _require(torch.cuda.is_bf16_supported(), "Visible GPU does not support BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=settings.deterministic_warn_only)

    configuration = {
        "schema_version": LORA_TRAINING_CONFIG_SCHEMA,
        "bundle_report_sha256": bundle_report_sha256,
        "train_rows_sha256": train_rows_sha256,
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "code_revision": code_revision,
        "settings": asdict(settings),
        "adapter_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "vision_tower_trainable": False,
        "vision_merger_trainable": False,
        "language_base_trainable": False,
    }
    config_path = output_dir / "training_config.json"
    report_path = output_dir / "lora_training_report.json"
    if output_dir.exists():
        _require(resume, f"Training output already exists: {output_dir}")
        _require(not report_path.exists(), "Completed LoRA run cannot be resumed")
        _require(
            _read_object(config_path, "training config") == configuration,
            "Resume config mismatch",
        )
    else:
        output_dir.mkdir(parents=True)
        _write_json(config_path, configuration)
    history_path = output_dir / "training_history.jsonl"

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        trust_remote_code=False,
    )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = settings.max_length
    image_processor = processor.image_processor
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = settings.min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = settings.max_pixels
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        image_processor.size["shortest_edge"] = settings.min_pixels
        image_processor.size["longest_edge"] = settings.max_pixels

    class TrainingRows(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return records[index]

    class Collator:
        def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
            encoded = []
            for record in instances:
                image = _image_path(assets_root, record.get("image"))
                prompt_messages, full_messages = _messages(record, image)
                full = processor.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt = processor.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_ids = full["input_ids"][0]
                prompt_ids = prompt["input_ids"][0]
                _require(len(full_ids) <= settings.max_length, "Encoded example exceeds max_length")
                labels = torch.tensor(
                    assistant_supervision_labels(full_ids.tolist(), prompt_ids.tolist()),
                    dtype=torch.long,
                )
                encoded.append(
                    {
                        "input_ids": full_ids,
                        "labels": labels,
                        "pixel_values": full["pixel_values"],
                        "image_grid_thw": full["image_grid_thw"],
                    }
                )
            input_ids = pad_sequence(
                [item["input_ids"] for item in encoded],
                batch_first=True,
                padding_value=processor.tokenizer.pad_token_id,
            )
            labels = pad_sequence(
                [item["labels"] for item in encoded],
                batch_first=True,
                padding_value=_IGNORE_INDEX,
            )
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": input_ids.ne(processor.tokenizer.pad_token_id),
                "pixel_values": torch.cat(
                    [item["pixel_values"] for item in encoded], dim=0
                ),
                "image_grid_thw": torch.cat(
                    [item["image_grid_thw"] for item in encoded], dim=0
                ),
            }

    device = torch.device("cuda:0")
    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation=settings.attention_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    _require(bool(trainable_parameter_names), "LoRA injected no trainable parameters")
    _require(
        all("lora_" in name for name in trainable_parameter_names),
        "A non-LoRA base parameter remained trainable",
    )
    _require(
        not any(
            marker in name.lower()
            for name in trainable_parameter_names
            for marker in ("visual", "vision", "merger")
        ),
        "LoRA unexpectedly targeted the visual encoder or merger",
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    batches_per_epoch = math.ceil(len(records) / settings.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / settings.gradient_accumulation_steps)
    planned_updates = settings.epochs * updates_per_epoch
    total_updates = (
        min(planned_updates, settings.max_steps)
        if settings.max_steps is not None
        else planned_updates
    )
    _require(total_updates >= 1, "Training configuration produces no optimizer updates")
    warmup_steps = int(total_updates * settings.warmup_ratio)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    state: dict[str, Any] = {
        "global_step": 0,
        "epoch": 0,
        "next_batch_in_epoch": 0,
        "total_updates": total_updates,
    }
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    if checkpoint is not None:
        adapter_state = load_file(str(checkpoint / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, adapter_state)
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=False))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", weights_only=False))
        _optimizer_to_device(optimizer, device)
        state = _read_object(checkpoint / "trainer_state.json", "trainer state")
        _require(state.get("total_updates") == total_updates, "Resume step plan mismatch")
        rng = torch.load(checkpoint / "rng_state.pt", weights_only=False)
        random.setstate(rng["python"])
        torch.random.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    discarded_history_records = (
        reconcile_history(history_path, int(state["global_step"])) if resume else 0
    )

    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(int(state["epoch"]), settings.epochs):
        start_batch = (
            int(state["next_batch_in_epoch"])
            if epoch == int(state["epoch"])
            else 0
        )
        sample_indices = epoch_sample_indices(
            len(records),
            settings.batch_size,
            settings.seed + epoch,
            start_batch,
        )
        loader = DataLoader(
            TrainingRows(),
            batch_size=settings.batch_size,
            sampler=sample_indices,
            num_workers=0,
            collate_fn=Collator(),
            pin_memory=False,
        )
        accumulated = 0
        loss_sum = 0.0
        for batch_index, batch in enumerate(loader, start=start_batch):
            batch = {key: value.to(device) for key, value in batch.items()}
            if accumulated == 0:
                accumulation_target = min(
                    settings.gradient_accumulation_steps,
                    batches_per_epoch - batch_index,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / accumulation_target
            loss.backward()
            accumulated += 1
            loss_sum += float(raw_loss.detach().cpu())
            is_last_batch = batch_index + 1 == batches_per_epoch
            if accumulated < accumulation_target and not is_last_batch:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                settings.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] = int(state["global_step"]) + 1
            state["epoch"] = epoch
            state["next_batch_in_epoch"] = batch_index + 1
            if is_last_batch:
                state["epoch"] = epoch + 1
                state["next_batch_in_epoch"] = 0
            history_record = {
                "global_step": state["global_step"],
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "micro_batches": accumulated,
                "loss": loss_sum / accumulated,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(grad_norm.detach().cpu()),
                "elapsed_seconds": time.monotonic() - wall_start,
            }
            _append_jsonl(history_path, history_record)
            if int(state["global_step"]) % settings.log_steps == 0:
                print(
                    f"[step {state['global_step']}/{total_updates}] "
                    f"loss={history_record['loss']:.6f} "
                    f"lr={history_record['learning_rate']:.3e}",
                    flush=True,
                )
            if int(state["global_step"]) % settings.save_steps == 0:
                _checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    torch=torch,
                    output_dir=output_dir,
                    state=state,
                )
            accumulated = 0
            loss_sum = 0.0
            if int(state["global_step"]) >= total_updates:
                stop = True
                break
        if stop:
            break

    _require(int(state["global_step"]) == total_updates, "LoRA training stopped early")
    latest = _latest_checkpoint(output_dir)
    if latest is None or latest.name != f"step-{total_updates:08d}":
        latest = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            output_dir=output_dir,
            state=state,
        )

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".final-") as staging_name:
        staging = Path(staging_name)
        staged_adapter = staging / "adapter"
        staged_processor = staging / "processor"
        model.save_pretrained(staged_adapter, safe_serialization=True)
        processor.save_pretrained(staged_processor)
        _require(not (output_dir / "adapter").exists(), "Final adapter already exists")
        _require(not (output_dir / "processor").exists(), "Final processor already exists")
        staged_adapter.replace(output_dir / "adapter")
        staged_processor.replace(output_dir / "processor")

    finished_at = datetime.now(timezone.utc)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    source = _object(bundle_report.get("source"), "bundle source")
    report = {
        "schema_version": LORA_TRAINING_REPORT_SCHEMA,
        "code_revision": code_revision,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_time_seconds": time.monotonic() - wall_start,
        "source": {
            "bundle_report_sha256": bundle_report_sha256,
            "train_rows_sha256": train_rows_sha256,
            "instruction_report_sha256": source.get("instruction_report_sha256"),
            "dataset_report_sha256": source.get("dataset_report_sha256"),
            "asset_index_sha256": source.get("asset_index_sha256"),
        },
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "artifact_sha256": model_artifact_sha256,
            "class": type(model.get_base_model()).__name__,
            "adapter_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "trainable_parameter_tensors": len(trainable_parameter_names),
            "trainable_parameter_names_sha256": hashlib.sha256(
                "".join(f"{name}\n" for name in trainable_parameter_names).encode("utf-8")
            ).hexdigest(),
        },
        "training": {
            **asdict(settings),
            "training_records": len(records),
            "available_training_records": available_records,
            "sample_capped": sample_capped,
            "smoke_test": smoke_test,
            "optimizer_updates": total_updates,
            "effective_batch_size": (
                settings.batch_size * settings.gradient_accumulation_steps
            ),
            "resumed_from": (
                checkpoint.relative_to(output_dir).as_posix()
                if checkpoint is not None
                else None
            ),
            "discarded_uncheckpointed_history_records": discarded_history_records,
            "final_checkpoint": latest.relative_to(output_dir).as_posix(),
        },
        "runtime": {
            "python_packages": {
                name: _package_version(name)
                for name in ("torch", "torchvision", "transformers", "peft", "safetensors")
            },
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_cuda_memory_bytes": peak_memory,
            "attention_implementation": settings.attention_implementation,
        },
        "contracts": {
            "base_weights_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "assistant_tokens_only_supervision": True,
            "train_split_only": True,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "seed_controlled": True,
            "bitwise_determinism_claimed": False,
        },
        "development_training_complete": not smoke_test,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(report_path, report)
    artifact_paths = [config_path, history_path, report_path]
    artifact_paths.extend(
        sorted(
            path
            for directory in (output_dir / "adapter", output_dir / "processor")
            for path in directory.rglob("*")
            if path.is_file()
        )
    )
    manifest_path = output_dir / "artifact_manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in artifact_paths
        ),
        encoding="utf-8",
    )
    model.config.use_cache = True
    return LoraTrainingResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        adapter_dir=output_dir / "adapter",
        global_steps=total_updates,
        development_training_complete=not smoke_test,
    )
