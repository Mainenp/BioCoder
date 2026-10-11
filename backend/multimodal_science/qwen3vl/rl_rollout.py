"""Hash-bound XIC-only sampling runtime shared by v2 difficulty and GRPO."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from multimodal_science.chrompeakformer.multimodal_dataset import DATASET_SCHEMA
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_inference import (
    _FusionAdapterVerifier,
    _eos_token_ids,
)
from multimodal_science.qwen3vl.fusion_training import (
    _verify_model_manifest,
    xic_only_prompt_text,
)
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import build_sensor_projector


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), context)


def _safe_artifact(
    root: Path, report: dict[str, Any], name: str, context: str
) -> tuple[Path, dict[str, Any]]:
    artifact = _object(_object(report.get("artifacts"), "Dataset artifacts").get(name), name)
    relative = artifact.get("path")
    _require(isinstance(relative, str) and bool(relative), f"Missing {context} path")
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes Dataset root") from error
    _require(path.is_file(), f"Missing {context}: {path}")
    _require(sha256_file(path) == artifact.get("sha256"), f"{context} hash mismatch")
    return path, artifact


@dataclass(frozen=True)
class SampledCompletion:
    response: str
    token_ids: tuple[int, ...]
    terminated_with_eos: bool
    sample_seed: int


@dataclass(frozen=True)
class VerifiedTrainSignals:
    path: Path
    report_sha256: str
    shape: tuple[int, int]


def verify_train_signals(
    dataset_root: Path, expected_report_sha256: str
) -> VerifiedTrainSignals:
    """Open only a staged train tensor view and reject protected split material."""

    dataset_root = dataset_root.resolve()
    assert_no_protected_input_paths([dataset_root])
    _require(
        not (dataset_root / "validation").exists(),
        "RL Dataset view contains validation tensors",
    )
    _require(
        not (dataset_root / "internal_test").exists(),
        "RL Dataset view contains internal-test tensors",
    )
    report_path = dataset_root / "dataset_report.json"
    _require(report_path.is_file(), f"Missing Dataset report: {report_path}")
    actual_report_sha256 = sha256_file(report_path)
    _require(actual_report_sha256 == expected_report_sha256, "Dataset report hash mismatch")
    report = _read_json(report_path, "Dataset report")
    _require(report.get("splits") == ["train", "validation"], "Unexpected Dataset split contract")
    _require(report.get("target_points") == 160, "Unexpected XIC point count")
    internal_test_accessed = report.get("internal_test_accessed")
    if internal_test_accessed is None:
        _require(
            report.get("schema_version") == DATASET_SCHEMA,
            "Legacy Dataset report has an unsupported schema",
        )
        artifacts = _object(report.get("artifacts"), "Dataset artifacts")
        _require(
            all("internal_test" not in str(name).lower() for name in artifacts),
            "Legacy Dataset report references internal test",
        )
    else:
        _require(internal_test_accessed is False, "Dataset report used internal test")
    signals_path, artifact = _safe_artifact(
        dataset_root, report, "train_signals", "train signals"
    )
    shape = artifact.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) and value > 0 for value in shape),
        "Train signal artifact shape is invalid",
    )
    _require(shape[1] == 160, "Train signals have the wrong point count")
    return VerifiedTrainSignals(
        path=signals_path,
        report_sha256=actual_report_sha256,
        shape=(shape[0], shape[1]),
    )


def sample_seed(seed: int, instruction_id: str, sample_index: int) -> int:
    payload = f"{seed}\0{instruction_id}\0{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def increment_mrope_position(position_ids: Any) -> Any:
    """Increment all three Qwen mRoPE axes together for one decoded token."""

    _require(getattr(position_ids, "ndim", None) == 3, "mRoPE positions must be rank three")
    _require(position_ids.shape[0] == 3, "Qwen mRoPE must expose three axes")
    return position_ids + 1


class XicSamplingRuntime:
    """Single-GPU, no-padding XIC-only rollout runtime with one prefill per group."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        model_revision: str,
        model_artifact_sha256: str,
        model_manifest_path: Path,
        model_manifest_sha256: str,
        starting_adapter: AdapterSpec,
        train_signals: VerifiedTrainSignals,
        attention_implementation: str = "sdpa",
        seed: int = 17,
    ) -> None:
        assert_no_protected_input_paths(
            [model_name_or_path, model_manifest_path, starting_adapter.root, train_signals.path]
        )
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError as error:
            raise RuntimeError("RL rollout requires transformers>=4.57.0") from error
        try:
            import numpy as np
            import torch
            from peft import PeftModel
            from safetensors.torch import load_file
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("RL rollout dependencies are incomplete") from error

        _require(torch.cuda.is_available(), "RL rollout requires CUDA")
        _verify_model_manifest(
            Path(model_name_or_path), model_manifest_path, model_manifest_sha256
        )
        verifier = _FusionAdapterVerifier(input_modality="xic_only")
        verifier(
            starting_adapter,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            model_artifact_sha256=model_artifact_sha256,
        )
        verified = verifier.verified
        _require(verified is not None, "Starting XIC-only adapter verification failed")
        training = _object(verified.report.get("training"), "starting training settings")
        trained_attention = training.get("attention_implementation")
        _require(
            trained_attention == attention_implementation,
            "Rollout attention differs from the starting artifact",
        )

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self._model = PeftModel.from_pretrained(
            base_model,
            verified.adapter_dir,
            is_trainable=False,
        ).to(device).eval()
        self._model.config.use_cache = True
        self._generation_model = self._model.get_base_model()
        visual = getattr(self._generation_model, "visual", None)
        _require(visual is not None, "Qwen visual tower was not found")
        self._visual_forward_calls = 0

        def reject_visual_forward(_module: Any, _inputs: Any) -> None:
            self._visual_forward_calls += 1
            raise RuntimeError("XIC-only RL rollout invoked the visual tower")

        self._visual_guard = visual.register_forward_pre_hook(reject_visual_forward)
        self._processor = AutoProcessor.from_pretrained(
            verified.processor_dir, trust_remote_code=False
        )
        self._projector = build_sensor_projector(verified.projector_spec).to(device).eval()
        self._projector.load_state_dict(load_file(str(verified.projector_path)))
        self._signals = np.load(train_signals.path, mmap_mode="r", allow_pickle=False)
        _require(tuple(self._signals.shape) == train_signals.shape, "Train signal shape drift")
        _require(self._signals.dtype == np.float32, "Train signals must be float32")
        _require(bool(np.isfinite(self._signals).all()), "Train signals contain non-finite values")
        _require(
            float(self._signals.min()) >= 0.0 and float(self._signals.max()) <= 1.0,
            "Train signals violate the normalized [0, 1] contract",
        )
        tokenizer = self._processor.tokenizer
        self._position_token_id = tokenizer.pad_token_id
        _require(isinstance(self._position_token_id, int), "Tokenizer has no shadow token")
        for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            _require(
                self._position_token_id != getattr(self._generation_model.config, token_name),
                f"Position shadow token collides with {token_name}",
            )
        torch.cuda.reset_peak_memory_stats(device)
        self._torch = torch
        self._np = np
        self._device = device
        self._seed = seed
        self._metadata = {
            "backend": "transformers-qwen3vl-xic-only-rl-sampling",
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "peft_version": version("peft"),
            "model_revision": model_revision,
            "model_artifact_sha256": model_artifact_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "starting_training_report_sha256": starting_adapter.training_report_sha256,
            "starting_manifest_sha256": starting_adapter.manifest_sha256,
            "dataset_report_sha256": train_signals.report_sha256,
            "input_modality": "xic_only",
            "sensor_projector": verified.projector_spec.as_dict(),
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 64,
            "prefill_once_per_rollout_group": True,
            "padding_used": False,
            "explicit_mrope_increment_all_three_axes": True,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        metadata["visual_forward_calls"] = self._visual_forward_calls
        metadata["peak_cuda_bytes"] = int(self._torch.cuda.max_memory_allocated(self._device))
        return metadata

    def close(self) -> None:
        self._visual_guard.remove()

    def _encode_prompt(self, record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        torch = self._torch
        prompt = record.get("prompt")
        language = record.get("language")
        _require(isinstance(prompt, str) and bool(prompt), "RL record has no prompt")
        _require(language in {"en", "zh-CN"}, "RL record has an invalid language")
        text = xic_only_prompt_text(prompt, str(language))
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        encoded = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        signal_link = _object(record.get("signal"), "RL signal link")
        row = signal_link.get("row")
        available = signal_link.get("available")
        _require(
            isinstance(row, int) and 0 <= row < self._signals.shape[0],
            "RL signal row is out of range",
        )
        _require(isinstance(available, bool), "RL signal availability is invalid")
        signal = torch.from_numpy(self._np.array(self._signals[row], copy=True)).unsqueeze(0)
        signal = signal.to(device=self._device, dtype=torch.float32)
        availability = torch.tensor([available], dtype=torch.bool, device=self._device)
        labels = torch.full_like(input_ids, -100)
        sensor_embeddings = self._projector(signal, availability)
        inputs_embeds, _, attention_mask, shadow_input_ids = insert_sensor_embeddings(
            self._model,
            input_ids,
            labels,
            sensor_embeddings,
            [int(input_ids.shape[1])],
            position_token_id=self._position_token_id,
        )
        position_ids, _ = self._generation_model.model.get_rope_index(
            input_ids=shadow_input_ids,
            attention_mask=attention_mask,
        )
        return inputs_embeds, attention_mask, position_ids, shadow_input_ids

    def sample_group(
        self,
        record: dict[str, Any],
        *,
        group_size: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 64,
    ) -> list[SampledCompletion]:
        _require(group_size == 8, "The preregistered rollout group size is 8")
        _require(temperature == 1.0, "The preregistered temperature is 1.0")
        _require(top_p == 1.0, "The preregistered top-p is 1.0")
        _require(max_new_tokens == 64, "The preregistered answer cap is 64 tokens")
        instruction_id = str(record.get("instruction_id") or "")
        _require(instruction_id != "", "RL record has no instruction ID")
        torch = self._torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inputs_embeds, attention_mask, position_ids, _ = self._encode_prompt(record)
            outputs = self._model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            _require(self._visual_forward_calls == 0, "RL rollout used image pixels")
            prefill_logits = outputs.logits[:, -1, :]
            prefill_cache = outputs.past_key_values
            next_position = position_ids[:, :, -1:] + 1
            _require(
                torch.equal(next_position[0], next_position[1])
                and torch.equal(next_position[1], next_position[2]),
                "XIC-only mRoPE axes diverged before decoding",
            )
            eos_ids = _eos_token_ids(self._model.generation_config.eos_token_id)
            completions = []
            for sample_index in range(group_size):
                current_logits = prefill_logits
                current_cache = copy.deepcopy(prefill_cache)
                current_mask = attention_mask.clone()
                current_position = next_position.clone()
                generated: list[int] = []
                terminated = False
                deterministic_seed = sample_seed(self._seed, instruction_id, sample_index)
                generator = torch.Generator(device=self._device)
                generator.manual_seed(deterministic_seed)
                for _ in range(max_new_tokens):
                    probabilities = torch.softmax(current_logits.float() / temperature, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    )
                    token_id = int(next_token.item())
                    generated.append(token_id)
                    if token_id in eos_ids:
                        terminated = True
                        break
                    current_mask = torch.cat(
                        (
                            current_mask,
                            torch.ones_like(next_token, dtype=current_mask.dtype),
                        ),
                        dim=1,
                    )
                    outputs = self._model(
                        input_ids=next_token,
                        attention_mask=current_mask,
                        position_ids=current_position,
                        past_key_values=current_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    current_logits = outputs.logits[:, -1, :]
                    current_cache = outputs.past_key_values
                    current_position = increment_mrope_position(current_position)
                completions.append(
                    SampledCompletion(
                        response=self._processor.decode(
                            generated,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        token_ids=tuple(generated),
                        terminated_with_eos=terminated,
                        sample_seed=deterministic_seed,
                    )
                )
        return completions
