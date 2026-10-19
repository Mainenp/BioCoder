"""Hash-bound, split-isolated single-sample Qwen/XIC deployment runtime.

Research evaluation entry points intentionally require validation bundles.  This
module is the production boundary: it accepts only explicitly frozen model
artifacts and one user-provided sample, and has no dataset, answer, or split
loading surface.
"""

from __future__ import annotations

import gc
import json
import math
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_training import xic_only_prompt_text
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)
from multimodal_science.qwen3vl.task_scoring import strict_json_object, valid_bbox

DEPLOYMENT_SCHEMA = "chrompeak-qwen3vl-agent-deployment-v1"
DEPLOYMENT_TASKS = {
    "peak_presence",
    "peak_presence_metadata",
    "peak_grounding",
    "scientific_qc",
}
DEPLOYMENT_MODALITIES = {"image_only", "image_xic", "xic_only"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_PATH_MARKERS = (
    "answer",
    "internal-test",
    "selection-set",
    "training-pool",
    "validation",
)
_ISOLATION_CONTRACTS = (
    "internal_test_accessed",
    "validation_answers_opened",
    "validation_prompts_opened",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None, f"Bad {label}")
    return value


def _deployment_path(value: Any, label: str, *, directory: bool = False) -> Path:
    _require(isinstance(value, str) and bool(value), f"Missing {label}")
    path = Path(value).expanduser().resolve()
    overlap = _protected_markers(path.parts)
    _require(not overlap, f"Protected research artifact path is forbidden for {label}: {overlap}")
    _require(path.is_dir() if directory else path.is_file(), f"Missing {label}: {path}")
    return path


def _protected_markers(parts: Sequence[str]) -> list[str]:
    normalized = {part.lower().replace("_", "-") for part in parts}
    return sorted(
        marker for marker in _PROTECTED_PATH_MARKERS if any(marker in part for part in normalized)
    )


def _verify_file(path: Path, expected: Any, label: str) -> str:
    digest = sha256_file(path)
    _require(digest == _sha(expected, f"{label} SHA-256"), f"{label} hash mismatch")
    return digest


def _manifest_entries(path: Path, root: Path, label: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw.strip()), f"Blank {label} line {line_number}")
        fields = raw.split(maxsplit=1)
        _require(len(fields) == 2, f"Malformed {label} line {line_number}")
        digest, relative = fields[0], fields[1].lstrip("* ")
        _sha(digest, f"{label} entry")
        relative_path = Path(relative)
        _require(not relative_path.is_absolute(), f"Absolute path in {label}")
        _require(
            all(part not in {"", ".", ".."} for part in relative_path.parts),
            f"Unsafe path in {label}",
        )
        protected = _protected_markers(relative_path.parts)
        _require(not protected, f"Protected research artifact in {label}: {protected}")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes root in {label}") from exc
        _require(relative not in entries, f"Duplicate path in {label}: {relative}")
        _require(candidate.is_file(), f"Missing file bound by {label}: {relative}")
        _require(sha256_file(candidate) == digest, f"File hash mismatch in {label}: {relative}")
        entries[relative] = digest
    _require(bool(entries), f"{label} is empty")
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    }
    _require(
        set(entries) == actual,
        f"{label} does not bind the exact directory contents",
    )
    return entries


def _assert_safe_provenance(value: Any, path: str = "provenance") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _ISOLATION_CONTRACTS:
                _require(child is False, f"{path}.{key} violates deployment data isolation")
            _assert_safe_provenance(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_safe_provenance(child, f"{path}[{index}]")


def _contract_values(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for candidate, child in value.items():
            if str(candidate).lower() == key:
                values.append(child)
            values.extend(_contract_values(child, key))
    elif isinstance(value, list):
        for child in value:
            values.extend(_contract_values(child, key))
    return values


@dataclass(frozen=True)
class VerifiedDeployment:
    manifest_path: Path
    manifest_sha256: str
    deployment_id: str
    model_root: Path
    model_revision: str
    adapter_dir: Path
    processor_dir: Path
    projector_path: Path | None
    projector_spec: SensorProjectorSpec | None
    input_modality: str
    attention_implementation: str
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    provenance: dict[str, Any]


def verify_deployment_manifest(path: Path) -> VerifiedDeployment:
    """Verify every deployment byte and reject research-only data lineage."""

    manifest_path = _deployment_path(str(path.expanduser().resolve()), "deployment manifest")
    manifest = _read_json(manifest_path, "deployment manifest")
    _require(manifest.get("schema_version") == DEPLOYMENT_SCHEMA, "Unexpected deployment schema")
    contracts = _object(manifest.get("contracts"), "deployment contracts")
    for key in _ISOLATION_CONTRACTS:
        _require(contracts.get(key) is False, f"Deployment contract {key} must be false")
    _require(contracts.get("user_samples_only") is True, "Deployment must be user-sample-only")

    model = _object(manifest.get("model"), "deployment model")
    model_root = _deployment_path(model.get("root"), "model root", directory=True)
    model_manifest = _deployment_path(model.get("manifest_path"), "model manifest")
    _verify_file(model_manifest, model.get("manifest_sha256"), "model manifest")
    _manifest_entries(model_manifest, model_root, "model manifest")
    revision = model.get("revision")
    _require(isinstance(revision, str) and bool(revision), "Model revision is missing")

    artifacts = _object(manifest.get("artifacts"), "deployment artifacts")
    adapter_dir = _deployment_path(
        artifacts.get("adapter_dir"), "adapter directory", directory=True
    )
    _verify_file(
        adapter_dir / "adapter_config.json",
        artifacts.get("adapter_config_sha256"),
        "adapter config",
    )
    _verify_file(
        adapter_dir / "adapter_model.safetensors",
        artifacts.get("adapter_weights_sha256"),
        "adapter weights",
    )
    processor_dir = _deployment_path(
        artifacts.get("processor_dir"), "processor directory", directory=True
    )
    processor_manifest = _deployment_path(
        artifacts.get("processor_manifest_path"), "processor manifest"
    )
    _verify_file(
        processor_manifest,
        artifacts.get("processor_manifest_sha256"),
        "processor manifest",
    )
    _manifest_entries(processor_manifest, processor_dir, "processor manifest")

    provenance = _object(manifest.get("provenance"), "deployment provenance")
    _assert_safe_provenance(provenance)
    report_path = _deployment_path(provenance.get("report_path"), "training report")
    _verify_file(report_path, provenance.get("report_sha256"), "training report")
    training_report = _read_json(report_path, "training report")
    _assert_safe_provenance(training_report, "training_report")
    for key in _ISOLATION_CONTRACTS:
        values = _contract_values(training_report, key)
        _require(values, f"Training report does not prove {key}=false")
        _require(
            all(value is False for value in values),
            f"Training report does not prove {key}=false",
        )

    runtime = _object(manifest.get("runtime"), "deployment runtime")
    modality = runtime.get("input_modality")
    _require(modality in DEPLOYMENT_MODALITIES, "Unsupported deployment modality")
    projector_path: Path | None = None
    spec: SensorProjectorSpec | None = None
    if modality == "image_only":
        _require(
            artifacts.get("projector_path") is None
            and artifacts.get("projector_sha256") is None
            and runtime.get("sensor_projector") is None,
            "Image-only deployment must not bind a sensor projector",
        )
    else:
        projector_path = _deployment_path(artifacts.get("projector_path"), "sensor projector")
        _verify_file(projector_path, artifacts.get("projector_sha256"), "sensor projector")
        spec = SensorProjectorSpec(
            **_object(runtime.get("sensor_projector"), "sensor projector spec")
        )
        spec.validate()
    attention = runtime.get("attention_implementation")
    _require(attention in {"sdpa", "eager"}, "Unsupported attention implementation")
    min_pixels = runtime.get("min_pixels")
    max_pixels = runtime.get("max_pixels")
    max_new_tokens = runtime.get("max_new_tokens")
    _require(isinstance(min_pixels, int) and min_pixels >= 28 * 28, "Invalid min_pixels")
    _require(isinstance(max_pixels, int) and max_pixels >= min_pixels, "Invalid max_pixels")
    _require(
        isinstance(max_new_tokens, int) and 1 <= max_new_tokens <= 256, "Invalid max_new_tokens"
    )
    deployment_id = manifest.get("deployment_id")
    _require(isinstance(deployment_id, str) and bool(deployment_id), "Deployment ID is missing")
    return VerifiedDeployment(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        deployment_id=deployment_id,
        model_root=model_root,
        model_revision=revision,
        adapter_dir=adapter_dir,
        processor_dir=processor_dir,
        projector_path=projector_path,
        projector_spec=spec,
        input_modality=modality,
        attention_implementation=attention,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        provenance=provenance,
    )


def build_task_prompt(
    task: str,
    language: str,
    *,
    width: int,
    height: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Build the same fixed JSON-only task surface used during training."""

    _require(task in DEPLOYMENT_TASKS, "Unsupported chromatography task")
    _require(language in {"en", "zh-CN"}, "Unsupported language")
    _require(width > 0 and height > 0, "Image dimensions must be positive")
    metadata = metadata or {}
    if task in {"peak_presence_metadata", "scientific_qc"}:
        required = {"component", "channel", "q1", "q3", "expected_rt_minutes"}
        _require(
            required <= set(metadata),
            f"Missing acquisition metadata: {sorted(required - set(metadata))}",
        )
        if language == "zh-CN":
            context = (
                f"分析物：{metadata['component']}\n通道：{metadata['channel']}\n"
                f"离子对：Q1={float(metadata['q1']):.6g}，Q3={float(metadata['q3']):.6g}\n"
                f"预期保留时间：{float(metadata['expected_rt_minutes']):.6g} 分钟"
            )
        else:
            context = (
                f"Analyte: {metadata['component']}\nChannel: {metadata['channel']}\n"
                f"Transition: Q1={float(metadata['q1']):.6g}, Q3={float(metadata['q3']):.6g}\n"
                f"Expected retention time: {float(metadata['expected_rt_minutes']):.6g} min"
            )
    if task == "peak_presence":
        return (
            "<image>\n请分析这张色谱 ROI 图像，判断目标色谱峰是否存在。仅返回 JSON，并严格使用此结构："
            '{"peak_present":true|false}。'
            if language == "zh-CN"
            else '<image>\nAnalyze this chromatographic ROI. Return JSON only with exactly this schema: {"peak_present":true|false}.'
        )
    if task == "peak_presence_metadata":
        return (
            f'<image>\n请结合色谱 ROI 图像和采集元数据，判断目标色谱峰是否存在。\n{context}\n仅返回 JSON：{{"peak_present":true|false}}。'
            if language == "zh-CN"
            else f'<image>\nAnalyze this chromatographic ROI using the image and acquisition metadata.\n{context}\nReturn JSON only with exactly this schema: {{"peak_present":true|false}}.'
        )
    if task == "peak_grounding":
        return (
            f'<image>\n请在这张色谱 ROI 图像中定位目标峰区间。仅返回 JSON：{{"bbox_2d":[x1,y1,x2,y2]}}。坐标使用原图像素；图像尺寸为 {width}×{height}，x 轴对应保留时间，边界框覆盖图像完整高度。'
            if language == "zh-CN"
            else f'<image>\nLocate the chromatographic peak interval in this ROI image. Return JSON only with exactly this schema: {{"bbox_2d":[x1,y1,x2,y2]}}. Coordinates must be source-image pixels for a {width}x{height} image; x follows retention time and the box spans the full image height.'
        )
    return (
        f'<image>\n请结合图像和采集元数据完成色谱峰质量检查。\n{context}\n仅返回 JSON：{{"qc_state":"ok|no_peak","reason":"peak_detected|no_visible_peak"}}。'
        if language == "zh-CN"
        else f'<image>\nPerform chromatographic peak QC using the image and acquisition metadata.\n{context}\nReturn JSON only with exactly this schema: {{"qc_state":"ok|no_peak","reason":"peak_detected|no_visible_peak"}}.'
    )


def validate_task_output(task: str, raw: str, width: int, height: int) -> dict[str, Any]:
    parsed, valid_json = strict_json_object(raw)
    _require(valid_json and parsed is not None, "Model did not return one JSON object")
    if task in {"peak_presence", "peak_presence_metadata"}:
        _require(
            set(parsed) == {"peak_present"} and isinstance(parsed["peak_present"], bool),
            "Invalid presence result schema",
        )
    elif task == "peak_grounding":
        _require(
            set(parsed) == {"bbox_2d"} and valid_bbox(parsed["bbox_2d"], width, height) is not None,
            "Invalid grounding result schema",
        )
    else:
        _require(set(parsed) == {"qc_state", "reason"}, "Invalid QC result schema")
        _require(parsed["qc_state"] in {"ok", "no_peak"}, "Invalid QC state")
        _require(parsed["reason"] in {"peak_detected", "no_visible_peak"}, "Invalid QC reason")
    return parsed


def _eos_token_ids(value: Any) -> set[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, (list, tuple)):
        result = {item for item in value if isinstance(item, int) and not isinstance(item, bool)}
        _require(bool(result), "Model declares no usable EOS token")
        return result
    raise ValueError("Model declares no usable EOS token")


class _TransformersDeploymentBackend:
    def __init__(self, deployment: VerifiedDeployment) -> None:
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError as exc:
            raise RuntimeError("Chromatography deployment requires transformers") from exc
        try:
            import numpy as np
            import torch
            from peft import PeftModel
            from safetensors.torch import load_file
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Chromatography deployment dependencies are incomplete") from exc
        _require(torch.cuda.is_available(), "Chromatography deployment requires CUDA")
        device = torch.device("cuda:0")
        base = AutoModelForImageTextToText.from_pretrained(
            str(deployment.model_root),
            revision=deployment.model_revision,
            dtype=torch.bfloat16,
            attn_implementation=deployment.attention_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.model = (
            PeftModel.from_pretrained(base, deployment.adapter_dir, is_trainable=False)
            .to(device)
            .eval()
        )
        self.model.config.use_cache = True
        self.generation_model = self.model.get_base_model()
        self.visual_forward_calls = 0
        self.visual_guard = None
        if deployment.input_modality == "xic_only":
            visual = getattr(self.generation_model, "visual", None)
            _require(visual is not None, "Qwen visual tower was not found")

            def reject_visual_forward(_module: Any, _inputs: Any) -> None:
                self.visual_forward_calls += 1
                raise RuntimeError("XIC-only deployment invoked the visual tower")

            self.visual_guard = visual.register_forward_pre_hook(reject_visual_forward)
        self.processor = AutoProcessor.from_pretrained(
            deployment.processor_dir, trust_remote_code=False
        )
        image_processor = self.processor.image_processor
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = deployment.min_pixels
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = deployment.max_pixels
        if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
            image_processor.size["shortest_edge"] = deployment.min_pixels
            image_processor.size["longest_edge"] = deployment.max_pixels
        self.projector = None
        if deployment.input_modality != "image_only":
            _require(
                deployment.projector_spec is not None and deployment.projector_path is not None,
                "Sensor deployment is missing its projector",
            )
            self.projector = build_sensor_projector(deployment.projector_spec).to(device).eval()
            self.projector.load_state_dict(load_file(str(deployment.projector_path)))
        self.torch = torch
        self.np = np
        self.device = device
        self.deployment = deployment
        self.position_token_id = self.processor.tokenizer.pad_token_id
        _require(isinstance(self.position_token_id, int), "Tokenizer has no shadow token")
        for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            _require(
                self.position_token_id != getattr(self.generation_model.config, token_name),
                f"Position shadow token collides with {token_name}",
            )
        self.metadata = {
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "visual_tower_guard_installed": deployment.input_modality == "xic_only",
        }
        if self.projector is not None:
            self.metadata["sensor_gate_probability"] = float(
                self.projector.gate_logit.detach().float().sigmoid().cpu()
            )

    def generate(
        self, prompt: str, signal: Sequence[float], available: bool, image_path: Path | None
    ) -> str:
        torch = self.torch
        modality = self.deployment.input_modality
        if modality in {"image_only", "image_xic"}:
            _require(
                image_path is not None and image_path.is_file(),
                "Image/XIC deployment requires an image",
            )
            text = prompt.replace("<image>", "", 1).strip()
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            ]
        else:
            language = "zh-CN" if "请" in prompt else "en"
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": xic_only_prompt_text(prompt, language)}
                        ],
                    }
                ]
            ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        if modality == "image_only":
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.deployment.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated = output_ids[0, input_ids.shape[1] :].tolist()
            return self.processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        _require(
            self.deployment.projector_spec is not None and self.projector is not None,
            "Sensor deployment is missing its projector",
        )
        array = self.np.asarray(signal, dtype=self.np.float32)
        _require(
            array.shape == (self.deployment.projector_spec.input_points,), "Bad XIC signal shape"
        )
        _require(bool(self.np.isfinite(array).all()), "XIC signal contains non-finite values")
        _require(
            float(array.min()) >= 0.0 and float(array.max()) <= 1.0,
            "XIC signal must be normalized to [0, 1]",
        )
        signal_tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(self.device)
        availability = torch.tensor([available], dtype=torch.bool, device=self.device)
        labels = torch.full_like(input_ids, -100)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sensor = self.projector(signal_tensor, availability)
            embeds, _, attention_mask, shadow = insert_sensor_embeddings(
                self.model,
                input_ids,
                labels,
                sensor,
                [int(input_ids.shape[1])],
                position_token_id=self.position_token_id,
            )
            rope: dict[str, Any] = {"input_ids": shadow, "attention_mask": attention_mask}
            if modality == "image_xic":
                rope["image_grid_thw"] = encoded["image_grid_thw"].to(self.device)
            position_ids, _ = self.generation_model.model.get_rope_index(**rope)
            kwargs: dict[str, Any] = {
                "input_ids": None,
                "inputs_embeds": embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
                "return_dict": True,
            }
            if modality == "image_xic":
                kwargs.update(
                    pixel_values=encoded["pixel_values"].to(self.device),
                    image_grid_thw=rope["image_grid_thw"],
                )
            outputs = self.model(**kwargs)
            _require(
                modality == "image_xic" or self.visual_forward_calls == 0,
                "XIC-only deployment used image pixels",
            )
            eos_ids = _eos_token_ids(self.model.generation_config.eos_token_id)
            generated: list[int] = []
            next_position = position_ids[:, :, -1:] + 1
            for _ in range(self.deployment.max_new_tokens):
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                if token_id in eos_ids:
                    break
                generated.append(token_id)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1
                )
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_position = next_position + 1
        return self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def close(self) -> None:
        """Release hooks and CUDA-owned model objects before a modality switch."""

        if self.visual_guard is not None:
            self.visual_guard.remove()
            self.visual_guard = None
        self.projector = None
        self.generation_model = None
        self.model = None


BackendFactory = Callable[[VerifiedDeployment], Any]


class ChromatogramDeploymentRuntime:
    """Lazy, serialized inference facade safe to share between Agent calls."""

    def __init__(self, manifest_path: Path, backend_factory: BackendFactory | None = None) -> None:
        self.deployment = verify_deployment_manifest(manifest_path)
        self._backend_factory = backend_factory or _TransformersDeploymentBackend
        self._backend: Any | None = None
        self._lock = threading.Lock()

    def _get_backend(self) -> Any:
        if self._backend is None:
            verified = verify_deployment_manifest(self.deployment.manifest_path)
            _require(
                verified.manifest_sha256 == self.deployment.manifest_sha256,
                "Deployment manifest drift before model load",
            )
            self.deployment = verified
            self._backend = self._backend_factory(verified)
        return self._backend

    def unload(self) -> None:
        """Unload the lazy backend so another deployment can use the same GPU."""

        with self._lock:
            backend = self._backend
            self._backend = None
            if backend is not None and callable(getattr(backend, "close", None)):
                backend.close()
            del backend
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def analyze(
        self,
        *,
        task: str,
        language: str,
        xic_signal: Sequence[float] | None,
        xic_available: bool,
        image_path: Path | None,
        image_width: int,
        image_height: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modality = self.deployment.input_modality
        if modality == "image_only":
            _require(xic_signal is None, "Image-only deployment must not receive XIC values")
            _require(image_path is not None, "Image-only deployment requires an image")
            signal: Sequence[float] = ()
        else:
            _require(xic_signal is not None, "This deployment requires XIC values")
            _require(self.deployment.projector_spec is not None, "Missing projector spec")
            _require(
                len(xic_signal) == self.deployment.projector_spec.input_points,
                "XIC signal length does not match deployment",
            )
            _require(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in xic_signal
                ),
                "XIC signal must contain finite numbers",
            )
            _require(
                all(0.0 <= float(value) <= 1.0 for value in xic_signal),
                "XIC signal must be normalized to [0, 1]",
            )
            signal = xic_signal
            if modality == "image_xic":
                _require(image_path is not None, "Image/XIC deployment requires an image")
            else:
                _require(image_path is None, "XIC-only deployment must not receive image pixels")
        prompt = build_task_prompt(
            task, language, width=image_width, height=image_height, metadata=metadata
        )
        with self._lock:
            backend = self._get_backend()
            raw = backend.generate(prompt, signal, xic_available, image_path)
        result = validate_task_output(task, raw, image_width, image_height)
        return {
            "deployment_id": self.deployment.deployment_id,
            "deployment_manifest_sha256": self.deployment.manifest_sha256,
            "input_modality": self.deployment.input_modality,
            "task": task,
            "language": language,
            "result": result,
            "raw_response": raw,
            "runtime": dict(getattr(backend, "metadata", {})),
        }


__all__ = [
    "DEPLOYMENT_MODALITIES",
    "DEPLOYMENT_SCHEMA",
    "DEPLOYMENT_TASKS",
    "ChromatogramDeploymentRuntime",
    "VerifiedDeployment",
    "build_task_prompt",
    "validate_task_output",
    "verify_deployment_manifest",
]
