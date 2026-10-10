"""Answer-isolated Qwen3-VL image/XIC inference with explicit M-RoPE decoding."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from multimodal_science.chrompeakformer.multimodal_dataset import DATASET_SCHEMA
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_data import (
    FUSION_BUNDLE_SCHEMA,
    FUSION_LINK_SCHEMA,
)
from multimodal_science.qwen3vl.fusion_training import (
    FUSION_INPUT_MODALITIES,
    FUSION_TRAINING_REPORT_SCHEMA,
    XIC_ONLY_TRAINING_REPORT_SCHEMA,
    xic_only_prompt_text,
)
from multimodal_science.qwen3vl.inference import (
    AdapterSpec,
    BatchGenerator,
    FinalBenchmarkAccessSpec,
    GenerationSettings,
    PromptRequest,
    QwenInferenceResult,
    _load_bundle,
    _require,
    _version_tuple,
    run_qwen_inference,
)
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)


FUSION_GENERATOR_BACKEND = "transformers-qwen3vl-image-xic"
XIC_ONLY_GENERATOR_BACKEND = "transformers-qwen3vl-xic-only"
XIC_INTERVENTIONS = ("aligned", "shuffled", "zero", "availability-off")
_HEX_40_TO_64 = re.compile(r"^[0-9a-f]{40,64}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _VerifiedFusionAdapter:
    root: Path
    adapter_dir: Path
    projector_path: Path
    processor_dir: Path
    report: dict[str, Any]
    metadata: dict[str, Any]
    projector_spec: SensorProjectorSpec


@dataclass(frozen=True)
class _ValidationInputs:
    links_by_instruction_id: dict[str, dict[str, Any]]
    signals_path: Path
    signal_shape: tuple[int, int]
    fusion_report_sha256: str
    dataset_report_sha256: str
    validation_links_sha256: str


@dataclass(frozen=True)
class _XicInterventionPlan:
    mode: str
    seed: int
    row_mapping: dict[int, int]
    availability_by_row: dict[int, bool]
    metadata: dict[str, Any]


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {label}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {label} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{label} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {label} JSON at line {line_number}") from error
    _require(records, f"{label} is empty")
    return records


def _artifact(
    root: Path,
    report: dict[str, Any],
    key: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    item = _object(_object(report.get("artifacts"), f"{label} artifacts").get(key), key)
    relative = item.get("path")
    _require(isinstance(relative, str) and bool(relative), f"Missing {label} path")
    _require("\\" not in relative, f"Non-portable {label} path")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute() and all(part not in {"", ".", ".."} for part in posix.parts),
        f"Unsafe {label} path",
    )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} artifact escapes its root") from error
    digest = item.get("sha256")
    _require(isinstance(digest, str) and bool(_HEX_64.fullmatch(digest)), f"Bad {label} hash")
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(sha256_file(path) == digest, f"{label} hash mismatch")
    return path, item


def _verified_manifest(root: Path, expected_sha256: str) -> dict[str, str]:
    _require(bool(_HEX_64.fullmatch(expected_sha256)), "Fusion manifest SHA-256 is invalid")
    manifest_path = root / "artifact_manifest.sha256"
    _require(manifest_path.is_file(), f"Missing fusion manifest: {manifest_path}")
    _require(sha256_file(manifest_path) == expected_sha256, "Fusion manifest hash mismatch")
    manifest: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split("  ", maxsplit=1)
        _require(len(parts) == 2, f"Malformed fusion manifest line: {line_number}")
        digest, relative = parts
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad fusion digest: {line_number}")
        _require("\\" not in relative, f"Non-portable fusion path: {relative}")
        posix = PurePosixPath(relative)
        _require(
            not posix.is_absolute()
            and all(part not in {"", ".", ".."} for part in posix.parts),
            f"Unsafe fusion path: {relative}",
        )
        _require(relative not in manifest, f"Duplicate fusion artifact: {relative}")
        path = (root / Path(*posix.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Fusion artifact escapes root: {relative}") from error
        _require(path.is_file(), f"Missing fusion artifact: {relative}")
        _require(sha256_file(path) == digest, f"Fusion artifact hash mismatch: {relative}")
        manifest[relative] = digest
    return manifest


def _projector_spec(value: Any) -> SensorProjectorSpec:
    payload = _object(value, "sensor-projector specification")
    required = {"input_points", "hidden_size", "sensor_tokens", "base_channels", "dropout"}
    _require(set(payload) == required, "Unexpected sensor-projector fields")
    spec = SensorProjectorSpec(
        input_points=payload["input_points"],
        hidden_size=payload["hidden_size"],
        sensor_tokens=payload["sensor_tokens"],
        base_channels=payload["base_channels"],
        dropout=payload["dropout"],
    )
    spec.validate()
    return spec


class _FusionAdapterVerifier:
    def __init__(
        self,
        input_modality: str = "image_xic",
        xic_intervention: dict[str, Any] | None = None,
    ) -> None:
        _require(input_modality in FUSION_INPUT_MODALITIES, "Unsupported input modality")
        self.verified: _VerifiedFusionAdapter | None = None
        self._input_modality = input_modality
        self._xic_intervention = (
            dict(xic_intervention) if xic_intervention is not None else None
        )

    def __call__(
        self,
        specification: AdapterSpec,
        *,
        model_name_or_path: str,
        model_revision: str,
        model_artifact_sha256: str | None,
    ) -> tuple[Path, dict[str, Any]]:
        root = specification.root.resolve()
        _require(root.is_dir(), f"Fusion adapter root not found: {root}")
        _require(
            bool(_HEX_64.fullmatch(specification.training_report_sha256)),
            "Fusion training report SHA-256 is invalid",
        )
        _require(model_artifact_sha256 is not None, "Fusion inference requires a model hash")
        manifest = _verified_manifest(root, specification.manifest_sha256)
        required = {
            "fusion_training_report.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "sensor_projector.safetensors",
            "processor/preprocessor_config.json",
            "processor/tokenizer_config.json",
        }
        _require(
            required <= set(manifest),
            f"Fusion manifest is missing: {sorted(required - set(manifest))}",
        )
        for directory in ("adapter", "processor"):
            unlisted = sorted(
                path.relative_to(root).as_posix()
                for path in (root / directory).rglob("*")
                if path.is_file() and path.relative_to(root).as_posix() not in manifest
            )
            _require(not unlisted, f"Fusion {directory} contains unlisted artifacts: {unlisted}")

        report_path = root / "fusion_training_report.json"
        _require(
            sha256_file(report_path) == specification.training_report_sha256,
            "Fusion training report hash mismatch",
        )
        report = _read_json(report_path, "fusion training report")
        expected_schema = (
            FUSION_TRAINING_REPORT_SCHEMA
            if self._input_modality == "image_xic"
            else XIC_ONLY_TRAINING_REPORT_SCHEMA
        )
        _require(report.get("schema_version") == expected_schema, "Wrong training report schema")
        _require(
            isinstance(report.get("code_revision"), str)
            and bool(_HEX_40_TO_64.fullmatch(report["code_revision"])),
            "Fusion code revision is invalid",
        )
        contracts = _object(report.get("contracts"), "fusion training contracts")
        shared_contracts = (
            ("base_weights_frozen", True),
            ("vision_tower_frozen", True),
            ("vision_merger_frozen", True),
            ("assistant_tokens_only_supervision", True),
            ("lora_and_projector_backward", True),
            ("parameter_updates_verified", True),
            ("train_split_only", True),
            ("validation_prompts_opened", False),
            ("validation_answers_opened", False),
            ("internal_test_accessed", False),
        )
        for name, expected in shared_contracts:
            _require(contracts.get(name) is expected, f"Fusion contract failed: {name}")
        if self._input_modality == "image_xic":
            for name, expected in (
                ("image_and_xic_forward", True),
                ("native_multimodal_rope_positions", True),
            ):
                _require(contracts.get(name) is expected, f"Fusion contract failed: {name}")
        else:
            for name, expected in (
                ("input_modality", "xic_only"),
                ("image_and_xic_forward", False),
                ("xic_only_forward", True),
                ("image_pixels_forwarded", False),
                ("images_opened_for_provenance_only", True),
                ("native_multimodal_rope_positions", False),
                ("native_qwen_rope_positions", True),
            ):
                _require(contracts.get(name) == expected, f"XIC-only contract failed: {name}")
        _require(
            report.get("development_training_complete") is True,
            "Fusion training is incomplete",
        )
        _require(report.get("development_comparison_eligible") is False, "Training self-qualified")
        _require(report.get("final_benchmark_eligible") is False, "Training is not a benchmark")
        _require(
            report.get("internal_test_accessed") is False,
            "Fusion training accessed test data",
        )

        model = _object(report.get("model"), "fusion model")
        _require(model.get("revision") == model_revision, "Fusion base revision mismatch")
        _require(
            model.get("base_artifact_sha256") == model_artifact_sha256,
            "Fusion base-model artifact hash mismatch",
        )
        spec = _projector_spec(model.get("sensor_projector"))
        training = _object(report.get("training"), "fusion training")
        optimizer_updates = training.get("optimizer_updates")
        training_records = training.get("training_records")
        _require(
            isinstance(optimizer_updates, int)
            and not isinstance(optimizer_updates, bool)
            and optimizer_updates >= 1,
            "Fusion optimizer-update count is invalid",
        )
        _require(
            isinstance(training_records, int)
            and not isinstance(training_records, bool)
            and training_records >= 1,
            "Fusion training-record count is invalid",
        )
        if self._input_modality == "xic_only":
            _require(training.get("vision_forward_calls") == 0, "XIC-only visual calls are nonzero")
        metadata = {
            "kind": (
                "image_xic_fusion"
                if self._input_modality == "image_xic"
                else "xic_only_qwen"
            ),
            "input_modality": self._input_modality,
            "training_report_sha256": specification.training_report_sha256,
            "manifest_sha256": specification.manifest_sha256,
            "code_revision": report["code_revision"],
            "trained_base_name_or_path": model.get("name_or_path"),
            "development_training_complete": True,
            "training_records": training_records,
            "optimizer_updates": optimizer_updates,
            "sensor_projector": spec.as_dict(),
            "fusion_bundle_report_sha256": _object(
                report.get("sources"), "fusion sources"
            ).get("fusion_bundle_report_sha256"),
            "xic_intervention": self._xic_intervention,
        }
        self.verified = _VerifiedFusionAdapter(
            root=root,
            adapter_dir=root / "adapter",
            projector_path=root / "sensor_projector.safetensors",
            processor_dir=root / "processor",
            report=report,
            metadata=metadata,
            projector_spec=spec,
        )
        return self.verified.adapter_dir, dict(metadata)


def _load_validation_inputs(
    *,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    inference_bundle_root: Path,
    inference_bundle_report_sha256: str,
    final_access: dict[str, str] | None = None,
) -> _ValidationInputs:
    for digest, label in (
        (fusion_bundle_report_sha256, "fusion bundle"),
        (dataset_report_sha256, "Dataset"),
        (inference_bundle_report_sha256, "inference bundle"),
    ):
        _require(bool(_HEX_64.fullmatch(digest)), f"Invalid {label} SHA-256")
    fusion_root = fusion_bundle_root.resolve()
    data_root = dataset_root.resolve()
    inference_root = inference_bundle_root.resolve()
    fusion_path = fusion_root / "fusion_bundle_report.json"
    dataset_path = data_root / "dataset_report.json"
    fusion = _read_json(fusion_path, "fusion bundle report")
    dataset = _read_json(dataset_path, "Dataset report")
    _require(
        sha256_file(fusion_path) == fusion_bundle_report_sha256,
        "Fusion bundle hash mismatch",
    )
    _require(sha256_file(dataset_path) == dataset_report_sha256, "Dataset hash mismatch")
    final_mode = final_access is not None
    expected_fusion_schema = (
        "chrompeak-qwen3vl-final-xic-bundle-v1"
        if final_mode
        else FUSION_BUNDLE_SCHEMA
    )
    _require(
        fusion.get("schema_version") == expected_fusion_schema,
        "Unsupported fusion inference bundle",
    )
    _require(dataset.get("schema_version") == DATASET_SCHEMA, "Unsupported Dataset schema")
    _require(dataset.get("target_points") == 160, "Unexpected XIC point count")
    expected_split = "internal_test" if final_mode else "validation"
    expected_splits = ["internal_test"] if final_mode else ["train", "validation"]
    _require(dataset.get("splits") == expected_splits, "Unexpected Dataset splits")
    contracts = _object(fusion.get("contracts"), "fusion contracts")
    _require(contracts.get("signals_external_to_bundle") is True, "Signals are embedded")
    sources = _object(
        fusion.get("source" if final_mode else "sources"), "fusion sources"
    )
    _require(sources.get("dataset_report_sha256") == dataset_report_sha256, "Dataset drift")
    _require(
        sources.get("inference_bundle_report_sha256") == inference_bundle_report_sha256,
        "Inference-bundle drift",
    )
    if final_mode:
        _require(contracts.get("answer_key_opened_by_generation") is False, "Answer leak")
        _require(contracts.get("split") == "internal_test", "Bad final fusion split")
        _require(fusion.get("internal_test_accessed") is True, "Fusion is not test-bound")
        _require(
            sources.get("protocol_sha256") == final_access["protocol_sha256"],
            "Fusion protocol drift",
        )
        _require(sources.get("access_id") == final_access["access_id"], "Fusion access drift")
    else:
        for name, expected in (
            ("validation_answer_key_opened", False),
            ("source_group_overlap", 0),
            ("internal_test_accessed", False),
        ):
            _require(contracts.get(name) == expected, f"Fusion contract failed: {name}")
        _require(fusion.get("internal_test_accessed") is False, "Fusion accessed test data")
    _, prompts, _ = _load_bundle(
        inference_root,
        inference_bundle_report_sha256,
        final_access=final_access,
    )
    link_key = "internal_test_xic_links" if final_mode else "validation_xic_links"
    links_path, links_artifact = _artifact(
        fusion_root,
        fusion,
        link_key,
        label=f"{expected_split} XIC links",
    )
    examples_path, examples_artifact = _artifact(
        data_root,
        dataset,
        f"{expected_split}_examples",
        label=f"{expected_split} examples",
    )
    signals_path, signals_artifact = _artifact(
        data_root,
        dataset,
        f"{expected_split}_signals",
        label=f"{expected_split} signals",
    )
    links = _read_jsonl(links_path, "validation XIC links")
    examples = _read_jsonl(examples_path, "validation examples")
    _require(links_artifact.get("records") == len(links), "Validation link count mismatch")
    _require(
        examples_artifact.get("records") == len(examples),
        "Validation example count mismatch",
    )
    _require(len(links) == len(prompts), "Validation prompt/link count mismatch")
    signal_shape = signals_artifact.get("shape")
    _require(signal_shape == [len(examples), 160], "Validation signal shape mismatch")
    examples_by_asset = {str(item.get("asset_id")): item for item in examples}
    _require(len(examples_by_asset) == len(examples), "Duplicate validation asset IDs")
    links_by_id: dict[str, dict[str, Any]] = {}
    for prompt, link in zip(prompts, links):
        instruction_id = str(prompt.get("instruction_id") or "")
        _require(link.get("schema_version") == FUSION_LINK_SCHEMA, "Bad validation link schema")
        _require(link.get("split") == expected_split, "Bad fusion link split")
        for field in ("instruction_id", "task", "language", "pair_id", "image"):
            _require(prompt.get(field) == link.get(field), f"Prompt/link mismatch: {field}")
        _require(instruction_id not in links_by_id, "Duplicate validation instruction ID")
        example = _object(examples_by_asset.get(str(link.get("asset_id"))), "validation example")
        image = _object(example.get("image"), "validation image")
        sequence = _object(example.get("sequence"), "validation sequence")
        signal = _object(link.get("signal"), "validation signal link")
        for actual, expected, label in (
            (example.get("split"), expected_split, "split"),
            (example.get("group_id"), link.get("group_id"), "group"),
            (image.get("path"), link.get("image"), "image path"),
            (image.get("sha256"), link.get("image_sha256"), "image hash"),
            (sequence.get("row"), signal.get("row"), "signal row"),
            (sequence.get("length"), signal.get("length"), "signal length"),
            (sequence.get("signal_available"), signal.get("available"), "availability"),
        ):
            _require(actual == expected, f"Dataset/link mismatch: {label}")
        row = signal.get("row")
        _require(isinstance(row, int) and 0 <= row < len(examples), "Bad validation signal row")
        _require(signal.get("length") == 160, "Bad validation signal length")
        _require(isinstance(signal.get("available"), bool), "Missing signal availability")
        links_by_id[instruction_id] = link
    return _ValidationInputs(
        links_by_instruction_id=links_by_id,
        signals_path=signals_path,
        signal_shape=(len(examples), 160),
        fusion_report_sha256=fusion_bundle_report_sha256,
        dataset_report_sha256=dataset_report_sha256,
        validation_links_sha256=sha256_file(links_path),
    )


def _build_xic_intervention_plan(
    links_by_instruction_id: dict[str, dict[str, Any]],
    *,
    mode: str,
    seed: int,
) -> _XicInterventionPlan:
    """Build an answer-independent, asset-stable XIC intervention plan."""

    _require(mode in XIC_INTERVENTIONS, f"Unsupported XIC intervention: {mode}")
    _require(isinstance(seed, int) and seed >= 0, "XIC intervention seed is invalid")
    availability_by_row: dict[int, bool] = {}
    for link in links_by_instruction_id.values():
        signal = _object(link.get("signal"), "validation signal link")
        row = signal.get("row")
        available = signal.get("available")
        _require(isinstance(row, int) and row >= 0, "Bad intervention signal row")
        _require(isinstance(available, bool), "Bad intervention availability")
        previous = availability_by_row.setdefault(row, available)
        _require(previous is available, "Availability differs across one validation asset")

    rows = sorted(availability_by_row)
    _require(rows, "XIC intervention has no validation rows")
    row_mapping = dict(zip(rows, rows))
    algorithm = "identity-v1"
    if mode == "shuffled":
        _require(len(rows) >= 2, "Shuffled XIC requires at least two validation assets")
        donors = list(rows)
        generator = random.Random(seed)
        for index in range(len(donors) - 1, 0, -1):
            other = generator.randrange(index)
            donors[index], donors[other] = donors[other], donors[index]
        _require(
            all(source != donor for source, donor in zip(rows, donors)),
            "Shuffled XIC permutation has a fixed point",
        )
        row_mapping = dict(zip(rows, donors))
        algorithm = "seeded-sattolo-single-cycle-v1"

    mapping_payload = [
        {"source_row": source, "donor_row": row_mapping[source]}
        for source in rows
    ]
    mapping_sha256 = hashlib.sha256(
        json.dumps(
            mapping_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "mode": mode,
        "seed": seed,
        "algorithm": algorithm,
        "mapping_sha256": mapping_sha256,
        "unique_signal_rows": len(rows),
        "changed_signal_rows": sum(row_mapping[row] != row for row in rows),
        "signal_values_zeroed": mode in {"zero", "availability-off"},
        "availability_forced_off": mode == "availability-off",
        "language_variants_share_one_asset_intervention": True,
        "answer_key_used": False,
    }
    return _XicInterventionPlan(
        mode=mode,
        seed=seed,
        row_mapping=row_mapping,
        availability_by_row=availability_by_row,
        metadata=metadata,
    )


def _eos_token_ids(value: Any) -> set[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, (list, tuple)):
        result = {item for item in value if isinstance(item, int) and not isinstance(item, bool)}
        _require(result, "Model declares no usable EOS token")
        return result
    raise ValueError("Model declares no usable EOS token")


class _FusionTransformersGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        model_revision: str,
        settings: GenerationSettings,
        verified: _VerifiedFusionAdapter,
        validation: _ValidationInputs,
        intervention: _XicInterventionPlan,
        input_modality: str = "image_xic",
    ) -> None:
        _require(input_modality in FUSION_INPUT_MODALITIES, "Unsupported input modality")
        _require(settings.batch_size == 1, "Fusion inference requires batch_size=1")
        _require(not settings.do_sample, "Fusion development evaluation requires greedy decoding")
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError as error:
            raise RuntimeError("Fusion inference requires transformers>=4.57.0") from error
        _require(
            _version_tuple(transformers_version) >= (4, 57, 0),
            f"Qwen3-VL requires transformers>=4.57.0; found {transformers_version}",
        )
        try:
            import numpy as np
            import torch
            from peft import PeftModel
            from safetensors.torch import load_file
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Fusion inference dependencies are incomplete") from error

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        _require(torch.cuda.is_available(), "Fusion inference requires CUDA")
        device = torch.device("cuda:0")
        training = _object(verified.report.get("training"), "fusion training settings")
        attention = training.get("attention_implementation")
        _require(attention in {"sdpa", "eager"}, "Unsupported trained attention implementation")
        _require(
            settings.attention_implementation == attention,
            "Requested attention implementation differs from the trained fusion artifact",
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            dtype=torch.bfloat16,
            attn_implementation=attention,
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
        self._input_modality = input_modality
        self._visual_forward_calls = 0
        self._visual_guard = None
        if input_modality == "xic_only":
            visual = getattr(self._generation_model, "visual", None)
            _require(visual is not None, "Qwen visual tower was not found")

            def reject_visual_forward(_module: Any, _inputs: Any) -> None:
                self._visual_forward_calls += 1
                raise RuntimeError("XIC-only inference invoked the visual tower")

            self._visual_guard = visual.register_forward_pre_hook(reject_visual_forward)
        self._processor = AutoProcessor.from_pretrained(
            verified.processor_dir,
            trust_remote_code=False,
        )
        image_processor = self._processor.image_processor
        min_pixels = training.get("min_pixels")
        max_pixels = training.get("max_pixels")
        _require(
            isinstance(min_pixels, int) and isinstance(max_pixels, int),
            "Missing pixel bounds",
        )
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = min_pixels
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels
        if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
            image_processor.size["shortest_edge"] = min_pixels
            image_processor.size["longest_edge"] = max_pixels
        self._projector = build_sensor_projector(verified.projector_spec).to(device).eval()
        self._projector.load_state_dict(load_file(str(verified.projector_path)))
        self._signals = np.load(validation.signals_path, mmap_mode="r", allow_pickle=False)
        _require(tuple(self._signals.shape) == validation.signal_shape, "Signal array shape drift")
        _require(self._signals.dtype == np.float32, "Validation signals must be float32")
        _require(
            bool(np.isfinite(self._signals).all()),
            "Validation signals contain non-finite values",
        )
        _require(
            float(self._signals.min()) >= 0.0 and float(self._signals.max()) <= 1.0,
            "Validation signals violate the normalized [0, 1] contract",
        )
        self._links = validation.links_by_instruction_id
        self._intervention = intervention
        self._verified_images: dict[str, str] = {}
        self._torch = torch
        self._np = np
        self._device = device
        tokenizer = self._processor.tokenizer
        self._position_token_id = tokenizer.pad_token_id
        _require(isinstance(self._position_token_id, int), "Tokenizer has no shadow token")
        for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            _require(
                self._position_token_id != getattr(self._generation_model.config, token_name),
                f"Position shadow token collides with {token_name}",
            )
        resolved_revision = getattr(self._generation_model.config, "_commit_hash", None)
        self._metadata = {
            "backend": "transformers",
            "fusion_execution_backend": (
                FUSION_GENERATOR_BACKEND
                if input_modality == "image_xic"
                else XIC_ONLY_GENERATOR_BACKEND
            ),
            "input_modality": input_modality,
            "image_pixels_forwarded": input_modality == "image_xic",
            "visual_tower_guard_installed": input_modality == "xic_only",
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "peft_version": version("peft"),
            "model_class": type(self._model).__name__,
            "processor_class": type(self._processor).__name__,
            "projector_class": type(self._projector).__name__,
            "resolved_model_revision": resolved_revision,
            "cuda_available": True,
            "cuda_device_count": int(torch.cuda.device_count()),
            "adapter_loaded": True,
            "manual_cached_greedy_decode": True,
            "explicit_fused_mrope_positions": True,
            "fusion_bundle_report_sha256": validation.fusion_report_sha256,
            "dataset_report_sha256": validation.dataset_report_sha256,
            "validation_links_sha256": validation.validation_links_sha256,
            "sensor_projector": verified.projector_spec.as_dict(),
            "sensor_gate": {
                "logit": float(self._projector.gate_logit.detach().float().cpu()),
                "probability": float(
                    self._projector.gate_logit.detach().float().sigmoid().cpu()
                ),
            },
            "xic_intervention": dict(intervention.metadata),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _decode_one(self, request: PromptRequest, settings: GenerationSettings) -> str:
        torch = self._torch
        link = _object(self._links.get(request.instruction_id), "validation fusion link")
        for actual, expected, label in (
            (request.task, link.get("task"), "task"),
            (request.language, link.get("language"), "language"),
            (request.pair_id, link.get("pair_id"), "pair ID"),
        ):
            _require(actual == expected, f"Request/link mismatch: {label}")
        image_key = str(request.image_path)
        image_digest = self._verified_images.get(image_key)
        if image_digest is None:
            image_digest = sha256_file(request.image_path)
            self._verified_images[image_key] = image_digest
        _require(image_digest == link.get("image_sha256"), "Validation image hash mismatch")
        if self._input_modality == "image_xic":
            text = request.prompt.replace("<image>", "", 1).strip()
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": str(request.image_path)},
                {"type": "text", "text": text},
            ]}]]
        else:
            text = xic_only_prompt_text(request.prompt, request.language)
            messages = [[{"role": "user", "content": [
                {"type": "text", "text": text},
            ]}]]
        encoded = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        signal_link = _object(link.get("signal"), "validation signal")
        source_signal_row = int(signal_link["row"])
        signal_row = self._intervention.row_mapping[source_signal_row]
        signal = torch.from_numpy(
            self._np.array(self._signals[signal_row], copy=True)
        ).unsqueeze(0).to(device=self._device, dtype=torch.float32)
        source_available = self._intervention.availability_by_row[source_signal_row]
        signal_available = source_available
        if self._intervention.mode == "shuffled":
            signal_available = self._intervention.availability_by_row[signal_row]
        elif self._intervention.mode == "zero":
            signal.zero_()
        elif self._intervention.mode == "availability-off":
            signal.zero_()
            signal_available = False
        availability = torch.tensor(
            [signal_available], dtype=torch.bool, device=self._device
        )
        labels = torch.full_like(input_ids, -100)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sensor_embeddings = self._projector(signal, availability)
            inputs_embeds, _, attention_mask, shadow_input_ids = insert_sensor_embeddings(
                self._model,
                input_ids,
                labels,
                sensor_embeddings,
                [int(input_ids.shape[1])],
                position_token_id=self._position_token_id,
            )
            rope_kwargs: dict[str, Any] = {
                "input_ids": shadow_input_ids,
                "attention_mask": attention_mask,
            }
            if self._input_modality == "image_xic":
                rope_kwargs["image_grid_thw"] = encoded["image_grid_thw"].to(
                    self._device
                )
            position_ids, _ = self._generation_model.model.get_rope_index(**rope_kwargs)
            forward_kwargs: dict[str, Any] = {
                "input_ids": None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
                "return_dict": True,
            }
            if self._input_modality == "image_xic":
                forward_kwargs.update(
                    {
                        "pixel_values": encoded["pixel_values"].to(self._device),
                        "image_grid_thw": rope_kwargs["image_grid_thw"],
                    }
                )
            outputs = self._model(**forward_kwargs)
            _require(
                self._input_modality == "image_xic" or self._visual_forward_calls == 0,
                "XIC-only inference used image pixels",
            )
            eos_ids = _eos_token_ids(self._model.generation_config.eos_token_id)
            generated: list[int] = []
            next_position = position_ids[:, :, -1:] + 1
            for _ in range(settings.max_new_tokens):
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                if token_id in eos_ids:
                    break
                generated.append(token_id)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)),
                    dim=1,
                )
                outputs = self._model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_position = next_position + 1
        return self._processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate(
        self,
        requests: Sequence[PromptRequest],
        settings: GenerationSettings,
    ) -> list[str]:
        _require(len(requests) == 1, "Fusion generator accepts one request at a time")
        return [self._decode_one(requests[0], settings)]


def run_fusion_inference(
    *,
    inference_bundle_root: Path,
    inference_bundle_report_sha256: str,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    assets_root: Path,
    fusion_adapter: AdapterSpec,
    output_dir: Path,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    settings: GenerationSettings = GenerationSettings(),
    max_records: int | None = None,
    resume: bool = False,
    xic_intervention: str = "aligned",
    xic_intervention_seed: int = 17,
    input_modality: str = "image_xic",
    final_benchmark_access: FinalBenchmarkAccessSpec | None = None,
) -> QwenInferenceResult:
    """Generate answer-isolated validation predictions for image/XIC ablations."""

    _require(input_modality in FUSION_INPUT_MODALITIES, "Unsupported input modality")
    _require(settings.batch_size == 1, "Fusion inference requires batch_size=1")
    _require(not settings.do_sample, "Fusion development evaluation requires greedy decoding")
    final_access = None
    if final_benchmark_access is not None:
        _require(
            final_benchmark_access.candidate_name == "qwen3vl_image_xic_fusion",
            "Final fusion inference requires the frozen fusion candidate",
        )
        _require(max_records is None, "Final fusion inference requires complete prompt coverage")
        _require(xic_intervention == "aligned", "Final benchmark forbids XIC interventions")
        _require(input_modality == "image_xic", "Final benchmark forbids new ablations")
        from multimodal_science.qwen3vl.final_benchmark_protocol import (
            verify_final_benchmark_access,
        )

        access_state = verify_final_benchmark_access(
            protocol_root=final_benchmark_access.protocol_root,
            expected_protocol_sha256=final_benchmark_access.protocol_sha256,
            ledger_dir=final_benchmark_access.ledger_dir,
        )
        _require(not access_state.completed, "Final benchmark access is already completed")
        final_access = {
            "access_id": access_state.access_id,
            "protocol_sha256": final_benchmark_access.protocol_sha256,
            "candidate_name": final_benchmark_access.candidate_name,
        }
    validation = _load_validation_inputs(
        fusion_bundle_root=fusion_bundle_root,
        fusion_bundle_report_sha256=fusion_bundle_report_sha256,
        dataset_root=dataset_root,
        dataset_report_sha256=dataset_report_sha256,
        inference_bundle_root=inference_bundle_root,
        inference_bundle_report_sha256=inference_bundle_report_sha256,
        final_access=final_access,
    )
    intervention = _build_xic_intervention_plan(
        validation.links_by_instruction_id,
        mode=xic_intervention,
        seed=xic_intervention_seed,
    )
    verifier = _FusionAdapterVerifier(input_modality, intervention.metadata)

    def generator_factory(
        runtime_model_name_or_path: str,
        runtime_model_revision: str,
        runtime_settings: GenerationSettings,
        adapter_dir: Path | None,
    ) -> BatchGenerator:
        _require(verifier.verified is not None, "Fusion adapter was not verified")
        _require(adapter_dir == verifier.verified.adapter_dir, "Fusion adapter path drift")
        return _FusionTransformersGenerator(
            runtime_model_name_or_path,
            runtime_model_revision,
            runtime_settings,
            verifier.verified,
            validation,
            intervention,
            input_modality,
        )

    return run_qwen_inference(
        inference_bundle_root,
        assets_root,
        output_dir,
        expected_bundle_report_sha256=inference_bundle_report_sha256,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        settings=settings,
        model_artifact_sha256=model_artifact_sha256,
        adapter=fusion_adapter,
        max_records=max_records,
        resume=resume,
        generator_factory=generator_factory,
        adapter_verifier=verifier,
        final_benchmark_access=final_benchmark_access,
    )
