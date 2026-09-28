"""End-to-end Qwen3-VL LoRA plus XIC-projector training smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from multimodal_science.chrompeakformer.multimodal_dataset import DATASET_SCHEMA
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_data import FUSION_BUNDLE_SCHEMA
from multimodal_science.qwen3vl.inference import AdapterSpec, _verify_adapter
from multimodal_science.qwen3vl.lora_data import LORA_BUNDLE_SCHEMA
from multimodal_science.qwen3vl.lora_training import (
    _image_path,
    _messages,
    assistant_supervision_labels,
)
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)


FUSION_SMOKE_SCHEMA = "chrompeak-qwen3vl-xic-fusion-smoke-v1"
GPU_ALLOCATION_MODE = "manual_physical_index_guard_no_slurm_gres"
REQUIRED_RUNTIME_PACKAGES = {
    "numpy": "1.26.4",
    "torch": "2.11.0+cu128",
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "safetensors": "0.6.2",
}
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FusionSmokeResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    manifest_sha256: str
    optimizer_updates: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), label)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(line.strip() != "", f"Blank {label} line {line_number}")
        records.append(_object(json.loads(line), f"{label} line {line_number}"))
    return records


def _artifact(root: Path, report: dict[str, Any], key: str) -> Path:
    item = _object(_object(report.get("artifacts"), "artifacts").get(key), key)
    relative = Path(str(item.get("path") or ""))
    _require(not relative.is_absolute() and str(relative) != ".", f"Unsafe {key} path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), f"{key} escapes root")
    _require(path.is_file() and sha256_file(path) == item.get("sha256"), f"Bad {key}")
    return path


def _bind_adapter_training_rows(
    adapter_metadata: dict[str, Any],
    train_rows_path: Path,
) -> str:
    """Require the initial adapter to have trained on the exact fused train rows."""

    source = _object(adapter_metadata.get("source"), "LoRA training source")
    expected = source.get("train_rows_sha256")
    _require(
        isinstance(expected, str) and bool(_HEX_64.fullmatch(expected)),
        "Initial adapter omits a valid training-row hash",
    )
    actual = sha256_file(train_rows_path)
    _require(
        expected == actual,
        "Initial adapter training rows do not match the fusion LoRA bundle",
    )
    return actual


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_model_manifest(
    model_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, int]:
    model_root = model_root.resolve()
    manifest_path = manifest_path.resolve()
    _require(model_root.is_dir(), f"Missing model root: {model_root}")
    _require(manifest_path.is_file(), f"Missing model manifest: {manifest_path}")
    _require(
        sha256_file(manifest_path) == expected_manifest_sha256,
        "Model manifest hash mismatch",
    )
    inventory = {
        path.relative_to(model_root).as_posix(): path
        for path in model_root.rglob("*")
        if path.is_file()
    }
    _require(bool(inventory), "Model root contains no files")
    verified = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        _require(line.strip() != "", f"Blank model manifest line {line_number}")
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed model manifest line {line_number}")
        digest, declared_path = parts
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad model digest at line {line_number}")
        declared = declared_path.lstrip("*").replace("\\", "/")
        matches = [
            relative
            for relative in inventory
            if declared == relative or declared.endswith(f"/{relative}")
        ]
        _require(len(matches) == 1, f"Ambiguous model path at line {line_number}")
        relative = matches[0]
        _require(relative not in verified, f"Duplicate model path: {relative}")
        _require(
            sha256_file(inventory[relative]) == digest,
            f"Model file hash mismatch: {relative}",
        )
        verified.add(relative)
    _require(verified == set(inventory), "Model manifest does not cover the local model exactly")
    _require("config.json" in verified, "Model manifest omits config.json")
    _require(
        any(path.endswith(".safetensors") for path in verified),
        "Model manifest omits safetensors weights",
    )
    return {
        "files": len(verified),
        "bytes": sum(inventory[path].stat().st_size for path in verified),
    }


def _gradient_l2_norm(parameters: list[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def _parameter_digest(named_parameters: list[tuple[str, Any]]) -> str:
    """Hash exact trainable tensor state without depending on tensor dtype support in NumPy."""

    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters):
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _group_gradient_norms(
    named_parameters: list[tuple[str, Any]],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    norms = {}
    for group, markers in groups.items():
        parameters = [
            parameter
            for name, parameter in named_parameters
            if any(marker in name for marker in markers)
        ]
        _require(parameters, f"Gradient group contains no parameters: {group}")
        norms[group] = _gradient_l2_norm(parameters)
    return norms


def _verify_mrope_insertion(
    original_position_ids: Any,
    fused_position_ids: Any,
    *,
    boundary: int,
    sensor_tokens: int,
) -> dict[str, int]:
    """Verify that four text-like sensor slots preserve Qwen's image M-RoPE prefix."""

    import torch

    _require(original_position_ids.ndim == 3, "Original M-RoPE positions are not rank three")
    _require(fused_position_ids.ndim == 3, "Fused M-RoPE positions are not rank three")
    _require(original_position_ids.shape[0] == 3, "Qwen M-RoPE must have three axes")
    _require(fused_position_ids.shape[0] == 3, "Fused Qwen M-RoPE must have three axes")
    _require(original_position_ids.shape[1] == 1, "Smoke expects one sample per step")
    _require(fused_position_ids.shape[1] == 1, "Smoke expects one fused sample per step")
    original_length = int(original_position_ids.shape[2])
    fused_length = int(fused_position_ids.shape[2])
    _require(fused_length == original_length + sensor_tokens, "Bad fused M-RoPE length")
    _require(0 <= boundary <= original_length, "Bad M-RoPE insertion boundary")
    _require(
        torch.equal(
            fused_position_ids[:, :, :boundary],
            original_position_ids[:, :, :boundary],
        ),
        "Sensor insertion changed the original image/prompt M-RoPE prefix",
    )
    inserted = fused_position_ids[:, :, boundary : boundary + sensor_tokens]
    _require(
        torch.equal(inserted[0], inserted[1]) and torch.equal(inserted[1], inserted[2]),
        "Inserted sensor slots are not text-like across all M-RoPE axes",
    )
    if boundary < original_length:
        _require(
            torch.equal(inserted[:, :, 0], original_position_ids[:, :, boundary]),
            "First sensor slot did not inherit the assistant-boundary M-RoPE position",
        )
    if sensor_tokens > 1:
        _require(
            bool(torch.all(inserted[0, 0, 1:] - inserted[0, 0, :-1] == 1).item()),
            "Inserted sensor M-RoPE positions are not contiguous",
        )
    suffix_shift = (
        fused_position_ids[:, :, boundary + sensor_tokens :]
        - original_position_ids[:, :, boundary:]
    )
    _require(
        bool(torch.all(suffix_shift == sensor_tokens).item()),
        "Assistant M-RoPE suffix did not shift by the sensor-token count",
    )
    return {
        "axes": 3,
        "original_length": original_length,
        "fused_length": fused_length,
        "sensor_tokens": sensor_tokens,
    }


def _bounded_training_rows(rows: list[Any], max_steps: int) -> list[Any]:
    _require(
        len(rows) >= max_steps,
        f"Requested {max_steps} updates but only {len(rows)} aligned rows exist",
    )
    return rows[:max_steps]


def run_fusion_smoke(
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
    max_records: int = 8,
    max_steps: int = 2,
    seed: int = 17,
) -> FusionSmokeResult:
    """Prove image, XIC, LoRA, projector and backward integration on bounded train rows."""

    for value, label, pattern in (
        (fusion_bundle_report_sha256, "fusion hash", _HEX_64),
        (lora_bundle_report_sha256, "LoRA bundle hash", _HEX_64),
        (dataset_report_sha256, "Dataset hash", _HEX_64),
        (model_artifact_sha256, "model hash", _HEX_64),
        (model_manifest_sha256, "model manifest hash", _HEX_64),
        (code_revision, "code revision", _HEX_40),
    ):
        _require(bool(pattern.fullmatch(value)), f"Invalid {label}")
    _require(1 <= max_records <= 128, "max_records must be between 1 and 128")
    _require(1 <= max_steps <= max_records, "max_steps must not exceed max_records")
    _require(0 <= seed < 2**32, "seed must fit an unsigned 32-bit integer")
    _require(
        os.environ.get("BIOCODER_VERIFIED_CODE_REVISION") == code_revision,
        "Code revision was not verified by the immutable launcher",
    )
    _require(
        os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE") == "staged_train_artifacts_only",
        "Fusion smoke requires staged train-only input roots",
    )
    _require(
        os.environ.get("BIOCODER_GPU_ALLOCATION_MODE") == GPU_ALLOCATION_MODE,
        "GPU allocation mode was not declared",
    )
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be set before Python starts",
    )
    roots = [fusion_bundle_root, lora_bundle_root, dataset_root, assets_root]
    fusion_root, lora_root, data_root, image_root = [path.resolve() for path in roots]
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Output already exists: {output_dir}")
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
    _require(fusion_report.get("schema_version") == FUSION_BUNDLE_SCHEMA, "Bad fusion schema")
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
    sources = _object(fusion_report.get("sources"), "fusion sources")
    _require(sources.get("dataset_report_sha256") == dataset_report_sha256, "Dataset drift")
    _require(sources.get("lora_bundle_report_sha256") == lora_bundle_report_sha256, "LoRA drift")
    _require(fusion_report.get("internal_test_accessed") is False, "Internal test accessed")
    validation_link = _object(
        _object(fusion_report.get("artifacts"), "fusion artifacts").get(
            "validation_xic_links"
        ),
        "validation XIC links",
    )
    validation_link_path = fusion_root / Path(str(validation_link.get("path") or ""))
    _require(
        not validation_link_path.exists(),
        "Train-only fusion view unexpectedly contains validation links",
    )
    _require(
        not (data_root / "validation").exists(),
        "Train-only Dataset view unexpectedly contains validation data",
    )
    _require(
        not (image_root / "jobs" / "validation").exists(),
        "Train-only asset view unexpectedly contains validation images",
    )

    train_links = _read_jsonl(_artifact(fusion_root, fusion_report, "train_xic_links"), "links")
    train_rows_path = _artifact(lora_root, lora_report, "train_qwen")
    train_rows = _read_jsonl(train_rows_path, "train rows")
    selections = _read_jsonl(
        _artifact(lora_root, lora_report, "selection_manifest"), "selection rows"
    )
    _require(
        len(train_links) == len(train_rows) == len(selections),
        "Fusion/train row count mismatch",
    )
    train_examples = _read_jsonl(
        _artifact(data_root, dataset_report, "train_examples"), "train examples"
    )
    examples_by_asset = {str(item.get("asset_id")): item for item in train_examples}
    _require(len(examples_by_asset) == len(train_examples), "Duplicate train asset IDs")
    selected = []
    for row, selection, link in zip(
        train_rows[:max_records], selections[:max_records], train_links[:max_records]
    ):
        for field in ("instruction_id", "asset_id", "group_id", "task", "language"):
            _require(
                selection.get(field) == link.get(field),
                f"Fusion alignment mismatch: {field}",
            )
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
        selected.append((row, link))

    processed = _bounded_training_rows(selected, max_steps)
    availability_counts = {
        "available": sum(
            _object(link.get("signal"), "signal").get("available") is True
            for _, link in processed
        ),
        "unavailable": sum(
            _object(link.get("signal"), "signal").get("available") is False
            for _, link in processed
        ),
    }
    _require(
        availability_counts["available"] >= 1,
        "Fusion smoke must exercise at least one measured XIC",
    )

    signals_path = _artifact(data_root, dataset_report, "train_signals")
    adapter_dir, adapter_metadata = _verify_adapter(
        initial_adapter,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        model_artifact_sha256=model_artifact_sha256,
    )
    _require(
        adapter_metadata.get("development_training_complete") is True,
        "Fusion must initialize from the completed image-only LoRA adapter",
    )
    adapter_train_rows_sha256 = _bind_adapter_training_rows(
        adapter_metadata,
        train_rows_path,
    )

    try:
        import numpy as np
        import torch
        from peft import (
            PeftModel,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Fusion smoke requires NumPy, torch, transformers, peft and safetensors"
        ) from error

    runtime_packages = {name: version(name) for name in REQUIRED_RUNTIME_PACKAGES}
    _require(
        runtime_packages == REQUIRED_RUNTIME_PACKAGES,
        f"Fusion runtime drift: expected={REQUIRED_RUNTIME_PACKAGES} actual={runtime_packages}",
    )
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose one CUDA GPU")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(model_name_or_path, revision=model_revision)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    hidden_size = int(model.get_base_model().config.text_config.hidden_size)
    position_token_id = processor.tokenizer.pad_token_id
    _require(isinstance(position_token_id, int), "Tokenizer has no position shadow token")
    for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
        _require(
            position_token_id != getattr(model.get_base_model().config, token_name),
            f"Position shadow token collides with {token_name}",
        )
    projector_spec = SensorProjectorSpec(hidden_size=hidden_size)
    projector = build_sensor_projector(projector_spec).to(device).train()
    lora_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    projector_named_parameters = list(projector.named_parameters())
    lora_parameters = [parameter for _, parameter in lora_named_parameters]
    projector_parameters = [parameter for _, parameter in projector_named_parameters]
    _require(lora_parameters, "Initial adapter is not trainable")
    _require(
        all(
            "lora_" in name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "Non-LoRA base parameter is trainable",
    )
    _require(
        not any(
            marker in name.lower()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            for marker in ("visual", "vision", "merger")
        ),
        "The loaded adapter unexpectedly trains the visual path",
    )
    lora_before_sha256 = _parameter_digest(lora_named_parameters)
    projector_before_sha256 = _parameter_digest(projector_named_parameters)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": 1e-6},
            {"params": projector_parameters, "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    signals = np.load(signals_path, mmap_mode="r", allow_pickle=False)
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
    generation_model = model.get_base_model()
    vision_module = getattr(generation_model, "visual", None)
    _require(vision_module is not None, "Qwen visual tower was not found")
    vision_forward_calls = 0

    def count_vision_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal vision_forward_calls
        vision_forward_calls += 1

    vision_hook = vision_module.register_forward_hook(count_vision_forward)
    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    history = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for step, (record, link) in enumerate(processed, 1):
            image = _image_path(image_root, record.get("image"))
            _require(
                sha256_file(image) == link.get("image_sha256"),
                "Training image hash mismatch",
            )
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
            labels = torch.tensor(
                assistant_supervision_labels(
                    full["input_ids"][0].tolist(), prompt["input_ids"][0].tolist()
                ),
                dtype=torch.long,
            ).unsqueeze(0)
            input_ids = full["input_ids"].to(device)
            labels = labels.to(device)
            prompt_length = int(prompt["input_ids"].shape[1])
            signal_link = _object(link.get("signal"), "signal")
            signal_row = int(signal_link["row"])
            _require(
                0 <= signal_row < signals.shape[0],
                "XIC row is outside the signal array",
            )
            signal_available = signal_link.get("available")
            _require(isinstance(signal_available, bool), "Missing XIC availability flag")
            signal = (
                torch.from_numpy(np.array(signals[signal_row], copy=True))
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
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
                image_grid_thw = full["image_grid_thw"].to(device)
                original_position_ids, original_rope_delta = (
                    generation_model.model.get_rope_index(
                        input_ids=input_ids,
                        image_grid_thw=image_grid_thw,
                        attention_mask=torch.ones_like(input_ids),
                    )
                )
                position_ids, fused_rope_delta = generation_model.model.get_rope_index(
                    input_ids=shadow_input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )
                mrope_contract = _verify_mrope_insertion(
                    original_position_ids,
                    position_ids,
                    boundary=prompt_length,
                    sensor_tokens=projector_spec.sensor_tokens,
                )
                _require(
                    torch.equal(original_rope_delta, fused_rope_delta),
                    "Sensor insertion changed Qwen's multimodal RoPE delta",
                )
                vision_calls_before = vision_forward_calls
                outputs = model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=fused_labels,
                    pixel_values=full["pixel_values"].to(device),
                    image_grid_thw=image_grid_thw,
                )
                _require(
                    vision_forward_calls > vision_calls_before,
                    "Qwen visual tower did not execute for the fused forward pass",
                )
            outputs.loss.backward()
            lora_gradient_norm = _gradient_l2_norm(lora_parameters)
            projector_gradient_norm = _gradient_l2_norm(projector_parameters)
            lora_target_gradient_norms = _group_gradient_norms(
                lora_named_parameters,
                {
                    "q_proj": (".q_proj.",),
                    "k_proj": (".k_proj.",),
                    "v_proj": (".v_proj.",),
                    "o_proj": (".o_proj.",),
                },
            )
            projector_gradient_norms = _group_gradient_norms(
                projector_named_parameters,
                {
                    "encoder": ("encoder.",),
                    "projection": ("projector.",),
                    "availability": ("availability_embedding.",),
                    "gate": ("gate_logit",),
                },
            )
            for group, norm in {
                "lora": lora_gradient_norm,
                "projector": projector_gradient_norm,
                **{f"lora_{key}": value for key, value in lora_target_gradient_norms.items()},
                **{
                    f"projector_{key}": value
                    for key, value in projector_gradient_norms.items()
                },
            }.items():
                _require(
                    math.isfinite(norm) and norm > 0.0,
                    f"Gradient group received no finite gradient: {group}",
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_parameters + projector_parameters, 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "step": step,
                    "loss": float(outputs.loss.detach()),
                    "gradient_norm": float(grad_norm),
                    "lora_gradient_norm": lora_gradient_norm,
                    "projector_gradient_norm": projector_gradient_norm,
                    "lora_target_gradient_norms": lora_target_gradient_norms,
                    "projector_gradient_norms": projector_gradient_norms,
                    "mrope": mrope_contract,
                    "vision_forward_calls": vision_forward_calls,
                    "signal_available": signal_available,
                }
            )
            print(
                f"[fusion step {step}/{max_steps}] loss={history[-1]['loss']:.6f}",
                flush=True,
            )
    finally:
        vision_hook.remove()

    _require(len(history) == max_steps, "Fusion smoke stopped before all requested updates")
    _require(
        [record["step"] for record in history] == list(range(1, max_steps + 1)),
        "Fusion smoke history is not contiguous",
    )
    _require(vision_forward_calls >= max_steps, "Visual tower did not run for every update")
    _require(math.isfinite(history[-1]["loss"]), "Non-finite fusion loss")
    lora_after_sha256 = _parameter_digest(lora_named_parameters)
    projector_after_sha256 = _parameter_digest(projector_named_parameters)
    _require(lora_after_sha256 != lora_before_sha256, "LoRA parameters did not update")
    _require(
        projector_after_sha256 != projector_before_sha256,
        "Sensor-projector parameters did not update",
    )
    output_dir.mkdir(parents=True)
    model.save_pretrained(output_dir / "adapter", safe_serialization=True)
    processor.save_pretrained(output_dir / "processor")
    adapter_weights_path = output_dir / "adapter" / "adapter_model.safetensors"
    adapter_config_path = output_dir / "adapter" / "adapter_config.json"
    projector_path = output_dir / "sensor_projector.safetensors"
    history_path = output_dir / "history.json"
    adapter_state = load_file(str(adapter_weights_path))
    _require(bool(adapter_state), "Saved adapter contains no tensors")
    _require(
        all(bool(torch.isfinite(value).all().item()) for value in adapter_state.values()),
        "Saved adapter contains non-finite tensors",
    )
    with torch.no_grad():
        for _, parameter in lora_named_parameters:
            parameter.zero_()
    lora_cleared_sha256 = _parameter_digest(lora_named_parameters)
    _require(
        lora_cleared_sha256 != lora_after_sha256,
        "LoRA reload control did not change the trained adapter state",
    )
    set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    reloaded_adapter_state = get_peft_model_state_dict(model, adapter_name="default")
    _require(
        set(reloaded_adapter_state) == set(adapter_state),
        "Reloaded adapter keys disagree with the serialized adapter",
    )
    _require(
        all(
            torch.equal(
                adapter_state[key],
                reloaded_adapter_state[key].detach().cpu(),
            )
            for key in adapter_state
        ),
        "Reloaded adapter tensors disagree with the serialized adapter",
    )
    lora_reloaded_sha256 = _parameter_digest(lora_named_parameters)
    _require(
        lora_reloaded_sha256 == lora_after_sha256,
        "Reloaded LoRA parameters do not restore the trained in-memory state",
    )
    projector_state = {key: value.detach().cpu() for key, value in projector.state_dict().items()}
    save_file(projector_state, str(projector_path))
    reloaded = build_sensor_projector(projector_spec)
    reloaded.load_state_dict(load_file(str(projector_path)))
    reloaded_projector_state = reloaded.state_dict()
    _require(
        set(reloaded_projector_state) == set(projector_state),
        "Reloaded projector keys disagree with the serialized projector",
    )
    _require(
        all(
            torch.equal(projector_state[key], reloaded_projector_state[key])
            for key in projector_state
        ),
        "Reloaded projector tensors disagree with the serialized projector",
    )
    _write_json(history_path, {"updates": history})
    report_path = output_dir / "fusion_smoke_report.json"
    report = {
        "schema_version": FUSION_SMOKE_SCHEMA,
        "code_revision": code_revision,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": time.monotonic() - wall_start,
        "sources": {
            "fusion_bundle_report_sha256": fusion_bundle_report_sha256,
            "lora_bundle_report_sha256": lora_bundle_report_sha256,
            "dataset_report_sha256": dataset_report_sha256,
            "adapter_train_rows_sha256": adapter_train_rows_sha256,
            "initial_adapter": adapter_metadata,
        },
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "base_artifact_sha256": model_artifact_sha256,
            "verification_manifest_sha256": model_manifest_sha256,
            "verified_files": model_inventory["files"],
            "verified_bytes": model_inventory["bytes"],
            "sensor_tokens_inserted": projector_spec.sensor_tokens,
            "sensor_projector": projector_spec.as_dict(),
            "projector_parameters": sum(value.numel() for value in projector.parameters()),
            "lora_trainable_parameters": sum(value.numel() for value in lora_parameters),
        },
        "training": {
            "selected_records": len(selected),
            "processed_records": len(history),
            "optimizer_updates": len(history),
            "seed": seed,
            "availability_counts": availability_counts,
            "vision_forward_calls": vision_forward_calls,
            "parameter_state_sha256": {
                "lora_before": lora_before_sha256,
                "lora_after": lora_after_sha256,
                "lora_cleared_before_reload": lora_cleared_sha256,
                "lora_reloaded": lora_reloaded_sha256,
                "projector_before": projector_before_sha256,
                "projector_after": projector_after_sha256,
            },
            "history": history,
        },
        "runtime": {
            "python_packages": runtime_packages,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "attention_implementation": "sdpa",
            "gpu_allocation_mode": os.environ.get("BIOCODER_GPU_ALLOCATION_MODE"),
            "train_input_scope": os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "contracts": {
            "image_and_xic_forward": True,
            "vision_tower_executed": True,
            "native_multimodal_rope_positions": True,
            "lora_and_projector_backward": True,
            "named_gradient_groups_verified": True,
            "parameter_updates_verified": True,
            "adapter_state_reloaded": True,
            "projector_reloaded": True,
            "vision_tower_frozen": True,
            "reproducibility_seeded": True,
            "bitwise_determinism_claimed": False,
            "staged_train_only_input_roots": True,
            "initial_adapter_training_rows_bound": True,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
        "artifacts": {
            "adapter_weights": {
                "path": adapter_weights_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(adapter_weights_path),
            },
            "adapter_config": {
                "path": adapter_config_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(adapter_config_path),
            },
            "sensor_projector": {
                "path": projector_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(projector_path),
            },
            "history": {
                "path": history_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(history_path),
            },
        },
        "development_training_complete": False,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(report_path, report)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = output_dir / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    listed_files = {
        line.split("  ", maxsplit=1)[1]
        for line in manifest.read_text(encoding="utf-8").splitlines()
    }
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest
    }
    _require(listed_files == actual_files, "Artifact manifest does not exactly cover output files")
    return FusionSmokeResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        optimizer_updates=len(history),
    )
