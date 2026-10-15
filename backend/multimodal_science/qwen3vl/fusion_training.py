"""Resumable single-GPU Qwen3-VL LoRA plus XIC-projector training."""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from multimodal_science.chrompeakformer.multimodal_dataset import DATASET_SCHEMA
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_data import SUPPORTED_FUSION_BUNDLE_SCHEMAS
from multimodal_science.qwen3vl.auxiliary_pretraining import (
    load_verified_pretrained_projector,
)
from multimodal_science.qwen3vl.fusion_smoke import (
    GPU_ALLOCATION_MODE,
    REQUIRED_RUNTIME_PACKAGES,
    _artifact,
    _bind_fusion_lora_artifacts,
    _object,
    _parameter_digest,
    _read_json,
    _read_jsonl,
    _require,
    _verify_and_bind_initial_adapter,
    _verify_model_manifest,
    _verify_mrope_insertion,
)
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.lora_data import LORA_BUNDLE_SCHEMA
from multimodal_science.qwen3vl.lora_training import (
    _append_jsonl,
    _image_path,
    _messages,
    _optimizer_to_device,
    assistant_supervision_labels,
    epoch_sample_indices,
    reconcile_history,
)
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)


FUSION_TRAINING_CONFIG_SCHEMA = "chrompeak-qwen3vl-xic-fusion-training-config-v1"
FUSION_TRAINING_REPORT_SCHEMA = "chrompeak-qwen3vl-xic-fusion-training-v1"
XIC_ONLY_TRAINING_CONFIG_SCHEMA = "chrompeak-qwen3vl-xic-only-training-config-v1"
XIC_ONLY_TRAINING_REPORT_SCHEMA = "chrompeak-qwen3vl-xic-only-training-v1"
FUSION_INPUT_MODALITIES = ("image_xic", "xic_only")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FusionTrainingSettings:
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_learning_rate: float = 2e-7
    projector_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_length: int = 1024
    min_pixels: int = 16 * 28 * 28
    max_pixels: int = 160 * 28 * 28
    save_steps: int = 250
    log_steps: int = 10
    seed: int = 17
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    deterministic_warn_only: bool = True
    sensor_tokens: int = 4
    input_modality: str = "image_xic"


@dataclass(frozen=True)
class FusionTrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    adapter_dir: Path
    projector_path: Path
    global_steps: int
    development_training_complete: bool


@dataclass(frozen=True)
class _FusionInputs:
    records: list[tuple[dict[str, Any], dict[str, Any]]]
    signals_path: Path
    lora_content_binding: dict[str, Any]
    adapter_dir: Path
    adapter_metadata: dict[str, Any]
    adapter_train_rows_sha256: str
    model_inventory: dict[str, int]
    availability_counts: dict[str, int]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _validate_settings(settings: FusionTrainingSettings) -> None:
    _require(settings.epochs >= 1, "epochs must be positive")
    if settings.max_steps is not None:
        _require(settings.max_steps >= 1, "max_steps must be positive")
    _require(
        settings.batch_size == 1,
        "Fusion training currently requires batch_size=1 for unpadded M-RoPE insertion",
    )
    for name in ("gradient_accumulation_steps", "save_steps", "log_steps"):
        _require(getattr(settings, name) >= 1, f"{name} must be positive")
    for name in (
        "lora_learning_rate",
        "projector_learning_rate",
        "max_grad_norm",
    ):
        value = getattr(settings, name)
        _require(math.isfinite(value) and value > 0.0, f"{name} must be positive and finite")
    _require(
        math.isfinite(settings.weight_decay) and settings.weight_decay >= 0.0,
        "weight_decay must be finite and nonnegative",
    )
    _require(0.0 <= settings.warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)")
    _require(settings.max_length >= 64, "max_length is too small")
    _require(
        settings.sensor_tokens in {1, 4, 8},
        "sensor_tokens must be one of 1, 4, or 8",
    )
    _require(
        settings.input_modality in FUSION_INPUT_MODALITIES,
        "input_modality must be image_xic or xic_only",
    )
    _require(
        settings.min_pixels >= 28 * 28 and settings.max_pixels >= settings.min_pixels,
        "Invalid image pixel bounds",
    )
    _require(
        settings.attention_implementation in {"sdpa", "eager"},
        "Unsupported attention implementation",
    )


def xic_only_prompt_text(prompt: str, language: str) -> str:
    """Return the canonical language-matched prompt for the XIC-only ablation."""

    _require(prompt.count("<image>") == 1, "Bad XIC-only prompt")
    _require(language in {"en", "zh-CN"}, "Unsupported XIC-only language")
    statement = (
        "Input modality: aligned XIC sensor signal only; no image pixels are available."
        if language == "en"
        else "输入模态：仅提供对齐的 XIC 传感器信号，不提供任何图像像素。"
    )
    return f"{statement}\n{prompt.replace('<image>', '', 1).strip()}"


def _xic_only_messages(
    record: dict[str, Any],
    *,
    language: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a text/XIC conversation without exposing image pixels to Qwen.

    The original task wording remains intact, including declared canvas dimensions used
    by the grounding task.  A language-matched modality statement makes the ablation
    explicit instead of silently replacing the chromatogram with a blank image.
    """

    conversations = record.get("conversations")
    _require(isinstance(conversations, list) and len(conversations) == 2, "Bad conversation")
    human = _object(conversations[0], "human turn")
    assistant = _object(conversations[1], "assistant turn")
    _require(human.get("from") == "human" and assistant.get("from") == "gpt", "Bad roles")
    prompt = human.get("value")
    response = assistant.get("value")
    _require(isinstance(prompt, str), "Bad prompt")
    _require(isinstance(response, str) and bool(response), "Bad assistant response")
    resolved_language = language if language is not None else record.get("language")
    prompt_text = xic_only_prompt_text(prompt, str(resolved_language))
    user = {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    answer = {"role": "assistant", "content": [{"type": "text", "text": response}]}
    return [user], [user, answer]


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in (output_dir / "checkpoints").glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None))[1]


def _checkpoint(
    *,
    model: Any,
    projector: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    save_file: Any,
    output_dir: Path,
    state: dict[str, Any],
) -> Path:
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"step-{state['global_step']:08d}"
    _require(not final.exists(), f"Checkpoint already exists: {final}")
    with tempfile.TemporaryDirectory(dir=checkpoints, prefix=f".{final.name}-") as staging_name:
        staging = Path(staging_name)
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        save_file(
            {key: value.detach().cpu() for key, value in projector.state_dict().items()},
            str(staging / "sensor_projector.safetensors"),
        )
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


def _load_training_inputs(
    *,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    lora_bundle_root: Path,
    lora_bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    assets_root: Path,
    initial_adapter: AdapterSpec,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    model_manifest_path: Path,
    model_manifest_sha256: str,
) -> _FusionInputs:
    fusion_root = fusion_bundle_root.resolve()
    lora_root = lora_bundle_root.resolve()
    data_root = dataset_root.resolve()
    image_root = assets_root.resolve()
    model_inventory = _verify_model_manifest(
        Path(model_name_or_path),
        model_manifest_path,
        model_manifest_sha256,
    )

    fusion_path = fusion_root / "fusion_bundle_report.json"
    lora_path = lora_root / "lora_bundle_report.json"
    dataset_path = data_root / "dataset_report.json"
    fusion_report = _read_json(fusion_path, "fusion report")
    lora_report = _read_json(lora_path, "LoRA bundle report")
    dataset_report = _read_json(dataset_path, "Dataset report")
    _require(sha256_file(fusion_path) == fusion_bundle_report_sha256, "Fusion hash mismatch")
    _require(sha256_file(lora_path) == lora_bundle_report_sha256, "LoRA hash mismatch")
    _require(sha256_file(dataset_path) == dataset_report_sha256, "Dataset hash mismatch")
    _require(
        fusion_report.get("schema_version") in SUPPORTED_FUSION_BUNDLE_SCHEMAS,
        "Bad fusion schema",
    )
    _require(lora_report.get("schema_version") == LORA_BUNDLE_SCHEMA, "Bad LoRA schema")
    _require(dataset_report.get("schema_version") == DATASET_SCHEMA, "Bad Dataset schema")
    _require(dataset_report.get("splits") == ["train", "validation"], "Bad Dataset splits")
    _require(dataset_report.get("target_points") == 160, "Unexpected XIC point count")

    fusion_contracts = _object(fusion_report.get("contracts"), "fusion contracts")
    for name, expected in (
        ("train_split_only_for_supervision", True),
        ("validation_answer_key_opened", False),
        ("signals_external_to_bundle", True),
        ("source_group_overlap", 0),
        ("internal_test_accessed", False),
    ):
        _require(fusion_contracts.get(name) == expected, f"Fusion contract failed: {name}")
    lora_contracts = _object(lora_report.get("contracts"), "LoRA bundle contracts")
    for name, expected in (
        ("train_split_only", True),
        ("validation_prompts_opened", False),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
    ):
        _require(lora_contracts.get(name) is expected, f"LoRA contract failed: {name}")
    _require(fusion_report.get("development_training_eligible") is True, "Bad fusion scope")
    _require(lora_report.get("development_training_eligible") is True, "Bad LoRA scope")
    _require(fusion_report.get("final_benchmark_eligible") is False, "Bad fusion benchmark scope")
    _require(lora_report.get("final_benchmark_eligible") is False, "Bad LoRA benchmark scope")
    fusion_sources = _object(fusion_report.get("sources"), "fusion sources")
    _require(
        fusion_sources.get("dataset_report_sha256") == dataset_report_sha256,
        "Dataset drift",
    )
    lora_source = _object(lora_report.get("source"), "LoRA bundle source")
    _require(
        lora_source.get("dataset_report_sha256") == dataset_report_sha256,
        "Runtime LoRA Dataset drift",
    )
    _require(fusion_report.get("internal_test_accessed") is False, "Internal test accessed")

    validation_artifact = _object(
        _object(fusion_report.get("artifacts"), "fusion artifacts").get(
            "validation_xic_links"
        ),
        "validation XIC links",
    )
    validation_path = fusion_root / Path(str(validation_artifact.get("path") or ""))
    _require(not validation_path.exists(), "Train-only fusion view contains validation links")
    _require(not (data_root / "validation").exists(), "Train-only Dataset view contains validation")
    _require(
        not (image_root / "jobs" / "validation").exists(),
        "Train-only asset view contains validation images",
    )

    train_rows_path = _artifact(lora_root, lora_report, "train_qwen")
    selection_path = _artifact(lora_root, lora_report, "selection_manifest")
    lora_content_binding = _bind_fusion_lora_artifacts(
        fusion_report,
        lora_bundle_report_sha256,
        train_rows_path,
        selection_path,
    )
    train_links = _read_jsonl(
        _artifact(fusion_root, fusion_report, "train_xic_links"),
        "fusion train links",
    )
    train_rows = _read_jsonl(train_rows_path, "LoRA train rows")
    selections = _read_jsonl(selection_path, "LoRA selection rows")
    _require(
        len(train_links) == len(train_rows) == len(selections),
        "Fusion/train row count mismatch",
    )
    train_examples = _read_jsonl(
        _artifact(data_root, dataset_report, "train_examples"),
        "Dataset train examples",
    )
    examples_by_asset = {str(item.get("asset_id")): item for item in train_examples}
    _require(len(examples_by_asset) == len(train_examples), "Duplicate train asset IDs")

    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row, selection, link in zip(train_rows, selections, train_links):
        for field in ("instruction_id", "asset_id", "group_id", "task", "language"):
            _require(selection.get(field) == link.get(field), f"Fusion mismatch: {field}")
        _require(row.get("image") == link.get("image"), "Image alignment mismatch")
        _require(
            selection.get("image_sha256") == link.get("image_sha256"),
            "Image digest alignment mismatch",
        )
        example = _object(examples_by_asset.get(str(link.get("asset_id"))), "train example")
        example_image = _object(example.get("image"), "train example image")
        example_signal = _object(example.get("sequence"), "train example sequence")
        signal_link = _object(link.get("signal"), "fusion signal link")
        for actual, expected, label in (
            (example.get("split"), "train", "split"),
            (example.get("group_id"), link.get("group_id"), "group"),
            (example_image.get("path"), link.get("image"), "image path"),
            (example_image.get("sha256"), link.get("image_sha256"), "image hash"),
            (example_signal.get("array"), signal_link.get("array"), "signal array"),
            (example_signal.get("row"), signal_link.get("row"), "signal row"),
            (example_signal.get("length"), signal_link.get("length"), "signal length"),
            (
                example_signal.get("signal_available"),
                signal_link.get("available"),
                "signal availability",
            ),
        ):
            _require(actual == expected, f"Dataset/fusion mismatch: {label}")
        records.append((row, link))

    availability_counts = {
        "available": sum(
            _object(link.get("signal"), "signal").get("available") is True
            for _, link in records
        ),
        "unavailable": sum(
            _object(link.get("signal"), "signal").get("available") is False
            for _, link in records
        ),
    }
    _require(availability_counts["available"] >= 1, "Fusion data has no measured XIC")
    adapter_dir, adapter_metadata, adapter_train_rows_sha256 = (
        _verify_and_bind_initial_adapter(
            initial_adapter,
            train_rows_path,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            model_artifact_sha256=model_artifact_sha256,
        )
    )
    return _FusionInputs(
        records=records,
        signals_path=_artifact(data_root, dataset_report, "train_signals"),
        lora_content_binding=lora_content_binding,
        adapter_dir=adapter_dir,
        adapter_metadata=adapter_metadata,
        adapter_train_rows_sha256=adapter_train_rows_sha256,
        model_inventory=model_inventory,
        availability_counts=availability_counts,
    )


def run_fusion_training(
    *,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    lora_bundle_root: Path,
    lora_bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    assets_root: Path,
    initial_adapter: AdapterSpec,
    output_dir: Path,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    model_manifest_path: Path,
    model_manifest_sha256: str,
    code_revision: str,
    settings: FusionTrainingSettings,
    resume: bool = False,
    pretrained_projector_root: Path | None = None,
    pretrained_projector_report_sha256: str | None = None,
    pretrained_projector_manifest_sha256: str | None = None,
) -> FusionTrainingResult:
    """Train a controlled Qwen/XIC candidate without validation supervision.

    ``image_xic`` preserves the original image-plus-XIC path.  ``xic_only`` uses the
    same records, initial adapter, targets, schedule, and sensor projector while
    withholding image pixels from both the processor and model forward pass.
    """

    _validate_settings(settings)
    for value, label, pattern in (
        (fusion_bundle_report_sha256, "fusion hash", _HEX_64),
        (lora_bundle_report_sha256, "LoRA bundle hash", _HEX_64),
        (dataset_report_sha256, "Dataset hash", _HEX_64),
        (model_artifact_sha256, "model hash", _HEX_64),
        (model_manifest_sha256, "model manifest hash", _HEX_64),
        (code_revision, "code revision", _HEX_40),
    ):
        _require(bool(pattern.fullmatch(value)), f"Invalid {label}")
    pretraining_values = (
        pretrained_projector_root,
        pretrained_projector_report_sha256,
        pretrained_projector_manifest_sha256,
    )
    _require(
        all(value is None for value in pretraining_values)
        or all(value is not None for value in pretraining_values),
        "Pretrained projector root, report hash, and manifest hash must be supplied together",
    )
    for value, label in (
        (pretrained_projector_report_sha256, "pretrained projector report hash"),
        (pretrained_projector_manifest_sha256, "pretrained projector manifest hash"),
    ):
        if value is not None:
            _require(bool(_HEX_64.fullmatch(value)), f"Invalid {label}")
    _require(
        os.environ.get("BIOCODER_VERIFIED_CODE_REVISION") == code_revision,
        "Code revision was not verified by the immutable launcher",
    )
    _require(
        os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE") == "staged_train_artifacts_only",
        "Fusion training requires staged train-only input roots",
    )
    _require(
        os.environ.get("BIOCODER_GPU_ALLOCATION_MODE") == GPU_ALLOCATION_MODE,
        "GPU allocation mode was not declared",
    )
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be set before Python starts",
    )
    output_dir = output_dir.resolve()
    image_root = assets_root.resolve()
    inputs = _load_training_inputs(
        fusion_bundle_root=fusion_bundle_root,
        fusion_bundle_report_sha256=fusion_bundle_report_sha256,
        lora_bundle_root=lora_bundle_root,
        lora_bundle_report_sha256=lora_bundle_report_sha256,
        dataset_root=dataset_root,
        dataset_report_sha256=dataset_report_sha256,
        assets_root=assets_root,
        initial_adapter=initial_adapter,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        model_artifact_sha256=model_artifact_sha256,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=model_manifest_sha256,
    )

    configuration = {
        "schema_version": (
            FUSION_TRAINING_CONFIG_SCHEMA
            if settings.input_modality == "image_xic"
            else XIC_ONLY_TRAINING_CONFIG_SCHEMA
        ),
        "code_revision": code_revision,
        "fusion_bundle_report_sha256": fusion_bundle_report_sha256,
        "lora_bundle_report_sha256": lora_bundle_report_sha256,
        "dataset_report_sha256": dataset_report_sha256,
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "model_artifact_sha256": model_artifact_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "initial_adapter_report_sha256": initial_adapter.training_report_sha256,
        "initial_adapter_manifest_sha256": initial_adapter.manifest_sha256,
        "pretrained_projector_report_sha256": pretrained_projector_report_sha256,
        "pretrained_projector_manifest_sha256": pretrained_projector_manifest_sha256,
        "settings": asdict(settings),
        "train_input_scope": "staged_train_artifacts_only",
        "vision_tower_trainable": False,
        "vision_merger_trainable": False,
        "language_base_trainable": False,
    }
    config_path = output_dir / "training_config.json"
    report_path = output_dir / "fusion_training_report.json"
    if output_dir.exists():
        _require(resume, f"Training output already exists: {output_dir}")
        _require(not report_path.exists(), "Completed fusion run cannot be resumed")
        _require(_read_json(config_path, "training config") == configuration, "Resume mismatch")
    else:
        output_dir.mkdir(parents=True)
        _write_json(config_path, configuration)
    history_path = output_dir / "training_history.jsonl"

    try:
        import numpy as np
        import torch
        from peft import PeftModel, set_peft_model_state_dict
        from safetensors.torch import load_file, save_file
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "Fusion training requires NumPy, torch, transformers, peft and safetensors"
        ) from error

    runtime_packages = {name: version(name) for name in REQUIRED_RUNTIME_PACKAGES}
    _require(
        runtime_packages == REQUIRED_RUNTIME_PACKAGES,
        f"Fusion runtime drift: expected={REQUIRED_RUNTIME_PACKAGES} actual={runtime_packages}",
    )
    _require(torch.cuda.is_available(), "Fusion training requires CUDA")
    _require(torch.cuda.device_count() == 1, "Expose exactly one CUDA device")
    _require(torch.cuda.is_bf16_supported(), "Visible GPU does not support BF16")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=settings.deterministic_warn_only)

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

    device = torch.device("cuda:0")
    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation=settings.attention_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, inputs.adapter_dir, is_trainable=True)
    model.config.use_cache = False
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.to(device).train()
    generation_model = model.get_base_model()
    hidden_size = int(generation_model.config.text_config.hidden_size)
    position_token_id = processor.tokenizer.pad_token_id
    _require(isinstance(position_token_id, int), "Tokenizer has no position shadow token")
    for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
        _require(
            position_token_id != getattr(generation_model.config, token_name),
            f"Position shadow token collides with {token_name}",
        )
    projector_spec = SensorProjectorSpec(
        hidden_size=hidden_size,
        sensor_tokens=settings.sensor_tokens,
    )
    projector = build_sensor_projector(projector_spec).to(device).train()
    auxiliary_pretraining_metadata: dict[str, Any] | None = None
    auxiliary_source_sensor_tokens: int | None = None
    if pretrained_projector_root is not None:
        projector_weights, auxiliary_pretraining_metadata = (
            load_verified_pretrained_projector(
                pretrained_projector_root,
                report_sha256=str(pretrained_projector_report_sha256),
                manifest_sha256=str(pretrained_projector_manifest_sha256),
                expected_spec=projector_spec,
            )
        )
        projector.load_state_dict(load_file(str(projector_weights)), strict=True)
        auxiliary_source_sensor_tokens = int(
            _object(
                _object(
                    auxiliary_pretraining_metadata.get("model"),
                    "auxiliary pretraining model",
                ).get("sensor_projector"),
                "auxiliary sensor-projector specification",
            )["sensor_tokens"]
        )
    lora_named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    projector_named_parameters = list(projector.named_parameters())
    lora_parameters = [parameter for _, parameter in lora_named_parameters]
    projector_parameters = [parameter for _, parameter in projector_named_parameters]
    _require(lora_parameters, "Initial adapter is not trainable")
    _require(
        all("lora_" in name for name, _ in lora_named_parameters),
        "Non-LoRA base parameter is trainable",
    )
    _require(
        not any(
            marker in name.lower()
            for name, _ in lora_named_parameters
            for marker in ("visual", "vision", "merger")
        ),
        "The loaded adapter unexpectedly trains the visual path",
    )
    initial_lora_sha256 = _parameter_digest(lora_named_parameters)
    initial_projector_sha256 = _parameter_digest(projector_named_parameters)
    initial_gate_logit = float(projector.gate_logit.detach().float().cpu())
    initial_gate_probability = float(
        projector.gate_logit.detach().float().sigmoid().cpu()
    )

    batches_per_epoch = len(inputs.records)
    updates_per_epoch = math.ceil(
        batches_per_epoch / settings.gradient_accumulation_steps
    )
    planned_updates = settings.epochs * updates_per_epoch
    total_updates = (
        min(planned_updates, settings.max_steps)
        if settings.max_steps is not None
        else planned_updates
    )
    _require(total_updates >= 1, "Training configuration produces no optimizer updates")
    warmup_steps = int(total_updates * settings.warmup_ratio)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": settings.lora_learning_rate},
            {"params": projector_parameters, "lr": settings.projector_learning_rate},
        ],
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
        "processed_micro_batches": 0,
        "vision_forward_calls": 0,
        "mrope_contract": None,
        "first_gradient_groups": None,
    }
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    if checkpoint is not None:
        adapter_state = load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, adapter_state)
        projector.load_state_dict(
            load_file(str(checkpoint / "sensor_projector.safetensors"))
        )
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=False))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", weights_only=False))
        _optimizer_to_device(optimizer, device)
        state = _read_json(checkpoint / "trainer_state.json", "trainer state")
        _require(state.get("total_updates") == total_updates, "Resume step plan mismatch")
        rng = torch.load(checkpoint / "rng_state.pt", weights_only=False)
        random.setstate(rng["python"])
        torch.random.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    discarded_history_records = (
        reconcile_history(history_path, int(state["global_step"])) if resume else 0
    )
    resumed_micro_batches = int(state.get("processed_micro_batches", 0))
    resumed_vision_forward_calls = int(state.get("vision_forward_calls", 0))
    _require(resumed_micro_batches >= 0, "Bad resumed micro-batch count")
    _require(resumed_vision_forward_calls >= 0, "Bad resumed visual-forward count")

    signals = np.load(inputs.signals_path, mmap_mode="r", allow_pickle=False)
    _require(
        signals.ndim == 2 and signals.shape[1] == projector_spec.input_points,
        "Unexpected train signal matrix shape",
    )
    _require(signals.dtype == np.float32, f"Unexpected train signal dtype: {signals.dtype}")
    _require(bool(np.isfinite(signals).all()), "Train signal matrix contains non-finite values")
    _require(
        float(signals.min()) >= 0.0 and float(signals.max()) <= 1.0,
        "Train signals violate the normalized [0, 1] contract",
    )
    vision_module = getattr(generation_model, "visual", None)
    _require(vision_module is not None, "Qwen visual tower was not found")
    vision_forward_calls = 0

    def count_vision_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal vision_forward_calls
        vision_forward_calls += 1

    vision_hook = vision_module.register_forward_hook(count_vision_forward)
    verified_images: dict[Path, str] = {}
    restored_mrope = state.get("mrope_contract")
    _require(restored_mrope is None or isinstance(restored_mrope, dict), "Bad M-RoPE state")
    mrope_contract: dict[str, int] | None = restored_mrope
    restored_gradients = state.get("first_gradient_groups")
    _require(
        restored_gradients is None or isinstance(restored_gradients, dict),
        "Bad first-gradient state",
    )
    first_gradient_groups: dict[str, float] | None = restored_gradients
    processed_micro_batches = 0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    try:
        for epoch in range(int(state["epoch"]), settings.epochs):
            start_batch = (
                int(state["next_batch_in_epoch"])
                if epoch == int(state["epoch"])
                else 0
            )
            sample_indices = epoch_sample_indices(
                len(inputs.records),
                settings.batch_size,
                settings.seed + epoch,
                start_batch,
            )
            accumulated = 0
            loss_sum = 0.0
            for batch_index, sample_index in enumerate(sample_indices, start=start_batch):
                if accumulated == 0:
                    accumulation_target = min(
                        settings.gradient_accumulation_steps,
                        batches_per_epoch - batch_index,
                    )
                record, link = inputs.records[sample_index]
                image = _image_path(image_root, record.get("image"))
                expected_image_sha256 = str(link.get("image_sha256"))
                previous = verified_images.get(image)
                if previous is None:
                    previous = sha256_file(image)
                    verified_images[image] = previous
                _require(previous == expected_image_sha256, "Training image hash mismatch")
                prompt_messages, full_messages = (
                    _messages(record, image)
                    if settings.input_modality == "image_xic"
                    else _xic_only_messages(record, language=str(link.get("language")))
                )
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
                input_ids = full["input_ids"].to(device)
                prompt_length = int(prompt["input_ids"].shape[1])
                _require(input_ids.shape[1] <= settings.max_length, "Example exceeds max_length")
                labels = torch.tensor(
                    assistant_supervision_labels(
                        full["input_ids"][0].tolist(),
                        prompt["input_ids"][0].tolist(),
                    ),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                signal_link = _object(link.get("signal"), "signal")
                signal_row = int(signal_link["row"])
                _require(0 <= signal_row < signals.shape[0], "XIC row is outside the array")
                signal_available = signal_link.get("available")
                _require(isinstance(signal_available, bool), "Missing XIC availability flag")
                signal = torch.from_numpy(np.array(signals[signal_row], copy=True)).unsqueeze(0)
                signal = signal.to(device=device, dtype=torch.float32)
                availability = torch.tensor(
                    [signal_available], dtype=torch.bool, device=device
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sensor_embeddings = projector(signal, availability)
                    inputs_embeds, fused_labels, attention_mask, shadow_input_ids = (
                        insert_sensor_embeddings(
                            model,
                            input_ids,
                            labels,
                            sensor_embeddings,
                            [prompt_length],
                            position_token_id=position_token_id,
                        )
                    )
                    rope_kwargs: dict[str, Any] = {
                        "input_ids": shadow_input_ids,
                        "attention_mask": attention_mask,
                    }
                    if settings.input_modality == "image_xic":
                        rope_kwargs["image_grid_thw"] = full["image_grid_thw"].to(device)
                    position_ids, fused_rope_delta = (
                        generation_model.model.get_rope_index(**rope_kwargs)
                    )
                    if mrope_contract is None:
                        original_rope_kwargs: dict[str, Any] = {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                        }
                        if settings.input_modality == "image_xic":
                            original_rope_kwargs["image_grid_thw"] = rope_kwargs[
                                "image_grid_thw"
                            ]
                        original_position_ids, original_rope_delta = (
                            generation_model.model.get_rope_index(**original_rope_kwargs)
                        )
                        mrope_contract = _verify_mrope_insertion(
                            original_position_ids,
                            position_ids,
                            boundary=prompt_length,
                            sensor_tokens=projector_spec.sensor_tokens,
                        )
                        _require(
                            torch.equal(original_rope_delta, fused_rope_delta),
                            "Sensor insertion changed Qwen's native RoPE delta",
                        )
                        state["mrope_contract"] = mrope_contract
                    forward_kwargs: dict[str, Any] = {
                        "input_ids": None,
                        "inputs_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "labels": fused_labels,
                    }
                    if settings.input_modality == "image_xic":
                        forward_kwargs.update(
                            {
                                "pixel_values": full["pixel_values"].to(device),
                                "image_grid_thw": rope_kwargs["image_grid_thw"],
                            }
                        )
                    vision_calls_before = vision_forward_calls
                    outputs = model(
                        **forward_kwargs,
                    )
                    if settings.input_modality == "xic_only":
                        _require(
                            vision_forward_calls == vision_calls_before,
                            "XIC-only training invoked the visual tower",
                        )
                    raw_loss = outputs.loss
                    loss = raw_loss / accumulation_target
                loss.backward()
                processed_micro_batches += 1
                accumulated += 1
                loss_sum += float(raw_loss.detach().cpu())
                is_last_batch = batch_index + 1 == batches_per_epoch
                if accumulated < accumulation_target and not is_last_batch:
                    continue
                if first_gradient_groups is None:
                    first_gradient_groups = {}
                    for group, parameters in (
                        ("lora", lora_parameters),
                        ("projector", projector_parameters),
                    ):
                        squared = sum(
                            float(parameter.grad.detach().float().square().sum())
                            for parameter in parameters
                            if parameter.grad is not None
                        )
                        norm = math.sqrt(squared)
                        _require(math.isfinite(norm) and norm > 0.0, f"No gradient: {group}")
                        first_gradient_groups[group] = norm
                    state["first_gradient_groups"] = first_gradient_groups
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    lora_parameters + projector_parameters,
                    settings.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] = int(state["global_step"]) + 1
                state["epoch"] = epoch
                state["next_batch_in_epoch"] = batch_index + 1
                state["processed_micro_batches"] = (
                    resumed_micro_batches + processed_micro_batches
                )
                state["vision_forward_calls"] = (
                    resumed_vision_forward_calls + vision_forward_calls
                )
                if is_last_batch:
                    state["epoch"] = epoch + 1
                    state["next_batch_in_epoch"] = 0
                learning_rates = scheduler.get_last_lr()
                history_record = {
                    "global_step": state["global_step"],
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "micro_batches": accumulated,
                    "loss": loss_sum / accumulated,
                    "lora_learning_rate": learning_rates[0],
                    "projector_learning_rate": learning_rates[1],
                    "gradient_norm": float(grad_norm.detach().cpu()),
                    "elapsed_seconds": time.monotonic() - wall_start,
                }
                _append_jsonl(history_path, history_record)
                if int(state["global_step"]) % settings.log_steps == 0:
                    print(
                        f"[fusion step {state['global_step']}/{total_updates}] "
                        f"loss={history_record['loss']:.6f} "
                        f"lora_lr={history_record['lora_learning_rate']:.3e} "
                        f"projector_lr={history_record['projector_learning_rate']:.3e}",
                        flush=True,
                    )
                if int(state["global_step"]) % settings.save_steps == 0:
                    _checkpoint(
                        model=model,
                        projector=projector,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        torch=torch,
                        save_file=save_file,
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
    finally:
        vision_hook.remove()

    _require(int(state["global_step"]) == total_updates, "Fusion training stopped early")
    total_micro_batches = int(state["processed_micro_batches"])
    total_vision_forward_calls = int(state["vision_forward_calls"])
    if settings.input_modality == "image_xic":
        _require(
            total_vision_forward_calls >= total_micro_batches,
            "Visual path did not run every batch",
        )
    else:
        _require(
            total_vision_forward_calls == 0,
            "XIC-only training invoked the visual path",
        )
    history_records = _read_jsonl(history_path, "fusion training history")
    _require(len(history_records) == total_updates, "Training history is incomplete")
    _require(
        sum(int(record.get("micro_batches", 0)) for record in history_records)
        == total_micro_batches,
        "Training history micro-batch count disagrees with the checkpoint state",
    )
    latest = _latest_checkpoint(output_dir)
    if latest is None or latest.name != f"step-{total_updates:08d}":
        latest = _checkpoint(
            model=model,
            projector=projector,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            save_file=save_file,
            output_dir=output_dir,
            state=state,
        )

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".final-") as staging_name:
        staging = Path(staging_name)
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        processor.save_pretrained(staging / "processor")
        save_file(
            {key: value.detach().cpu() for key, value in projector.state_dict().items()},
            str(staging / "sensor_projector.safetensors"),
        )
        for target in ("adapter", "processor", "sensor_projector.safetensors"):
            _require(not (output_dir / target).exists(), f"Final artifact exists: {target}")
            (staging / target).replace(output_dir / target)

    lora_after_sha256 = _parameter_digest(lora_named_parameters)
    projector_after_sha256 = _parameter_digest(projector_named_parameters)
    _require(lora_after_sha256 != initial_lora_sha256, "LoRA parameters did not update")
    _require(
        projector_after_sha256 != initial_projector_sha256,
        "Sensor-projector parameters did not update",
    )
    smoke_test = settings.max_steps is not None
    development_complete = not smoke_test
    finished_at = datetime.now(timezone.utc)
    adapter_dir = output_dir / "adapter"
    projector_path = output_dir / "sensor_projector.safetensors"
    final_gate_logit = float(projector.gate_logit.detach().float().cpu())
    final_gate_probability = float(
        projector.gate_logit.detach().float().sigmoid().cpu()
    )
    report = {
        "schema_version": (
            FUSION_TRAINING_REPORT_SCHEMA
            if settings.input_modality == "image_xic"
            else XIC_ONLY_TRAINING_REPORT_SCHEMA
        ),
        "code_revision": code_revision,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_time_seconds": time.monotonic() - wall_start,
        "sources": {
            "fusion_bundle_report_sha256": fusion_bundle_report_sha256,
            "lora_bundle_report_sha256": lora_bundle_report_sha256,
            "fusion_lora_content_binding": inputs.lora_content_binding,
            "dataset_report_sha256": dataset_report_sha256,
            "adapter_train_rows_sha256": inputs.adapter_train_rows_sha256,
            "initial_adapter": inputs.adapter_metadata,
            "auxiliary_pretraining": (
                {
                    "report_sha256": pretrained_projector_report_sha256,
                    "manifest_sha256": pretrained_projector_manifest_sha256,
                    "code_revision": auxiliary_pretraining_metadata.get("code_revision"),
                    "dataset_report_sha256": _object(
                        auxiliary_pretraining_metadata.get("sources"),
                        "auxiliary pretraining sources",
                    ).get("dataset_report_sha256"),
                    "projector_sha256": _object(
                        auxiliary_pretraining_metadata.get("model"),
                        "auxiliary pretraining model",
                    ).get("persisted_projector_sha256"),
                    "source_sensor_tokens": auxiliary_source_sensor_tokens,
                    "target_sensor_tokens": projector_spec.sensor_tokens,
                    "token_pooling_remapped": (
                        auxiliary_source_sensor_tokens != projector_spec.sensor_tokens
                    ),
                }
                if auxiliary_pretraining_metadata is not None
                else None
            ),
        },
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "base_artifact_sha256": model_artifact_sha256,
            "verification_manifest_sha256": model_manifest_sha256,
            "verified_files": inputs.model_inventory["files"],
            "verified_bytes": inputs.model_inventory["bytes"],
            "sensor_projector": projector_spec.as_dict(),
            "sensor_gate": {
                "initial_logit": initial_gate_logit,
                "initial_probability": initial_gate_probability,
                "final_logit": final_gate_logit,
                "final_probability": final_gate_probability,
                "absolute_probability_change": (
                    final_gate_probability - initial_gate_probability
                ),
            },
            "lora_trainable_parameters": sum(value.numel() for value in lora_parameters),
            "projector_trainable_parameters": sum(
                value.numel() for value in projector_parameters
            ),
        },
        "training": {
            **asdict(settings),
            "training_records": len(inputs.records),
            "optimizer_updates": total_updates,
            "calibration_run": smoke_test,
            "processed_micro_batches": total_micro_batches,
            "processed_micro_batches_this_invocation": processed_micro_batches,
            "effective_batch_size": (
                settings.batch_size * settings.gradient_accumulation_steps
            ),
            "availability_counts": inputs.availability_counts,
            "vision_forward_calls": total_vision_forward_calls,
            "vision_forward_calls_this_invocation": vision_forward_calls,
            "mrope_contract": mrope_contract,
            "first_gradient_groups": first_gradient_groups,
            "resumed_from": (
                checkpoint.relative_to(output_dir).as_posix()
                if checkpoint is not None
                else None
            ),
            "discarded_uncheckpointed_history_records": discarded_history_records,
            "final_checkpoint": latest.relative_to(output_dir).as_posix(),
            "parameter_state_sha256": {
                "lora_initial": initial_lora_sha256,
                "lora_after": lora_after_sha256,
                "projector_initial": initial_projector_sha256,
                "projector_after": projector_after_sha256,
            },
        },
        "runtime": {
            "python_packages": runtime_packages,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "attention_implementation": settings.attention_implementation,
            "gpu_allocation_mode": os.environ.get("BIOCODER_GPU_ALLOCATION_MODE"),
            "train_input_scope": os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "contracts": {
            "input_modality": settings.input_modality,
            "base_weights_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "assistant_tokens_only_supervision": True,
            "image_and_xic_forward": settings.input_modality == "image_xic",
            "xic_only_forward": settings.input_modality == "xic_only",
            "image_pixels_forwarded": settings.input_modality == "image_xic",
            "images_opened_for_provenance_only": settings.input_modality == "xic_only",
            "native_multimodal_rope_positions": settings.input_modality == "image_xic",
            "native_qwen_rope_positions": True,
            "lora_and_projector_backward": True,
            "parameter_updates_verified": True,
            "train_split_only": True,
            "staged_train_only_input_roots": True,
            "fusion_lora_training_artifacts_bound": True,
            "initial_adapter_training_rows_bound": True,
            "auxiliary_pretraining_bound": auxiliary_pretraining_metadata is not None,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "seed_controlled": True,
            "bitwise_determinism_claimed": False,
        },
        "development_training_complete": development_complete,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(report_path, report)
    artifact_paths = [config_path, history_path, report_path, projector_path]
    artifact_paths.extend(
        sorted(
            path
            for directory in (adapter_dir, output_dir / "processor")
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
    return FusionTrainingResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        adapter_dir=adapter_dir,
        projector_path=projector_path,
        global_steps=total_updates,
        development_training_complete=development_complete,
    )
