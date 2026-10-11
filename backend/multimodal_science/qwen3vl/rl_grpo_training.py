"""Preregistered, train-only GRPO for the Qwen3-VL XIC-only policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_inference import (
    _FusionAdapterVerifier,
    _eos_token_ids,
)
from multimodal_science.qwen3vl.fusion_smoke import (
    GPU_ALLOCATION_MODE,
    REQUIRED_RUNTIME_PACKAGES,
    _parameter_digest,
)
from multimodal_science.qwen3vl.fusion_training import (
    _verify_model_manifest,
    xic_only_prompt_text,
)
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.lora_training import _optimizer_to_device
from multimodal_science.qwen3vl.rl_data import (
    GROUNDING_TASK,
    assert_no_protected_input_paths,
    assert_training_pool_isolation,
    load_rl_training_pool,
)
from multimodal_science.qwen3vl.rl_difficulty import DIFFICULTY_REPORT_SCHEMA
from multimodal_science.qwen3vl.rl_rewards import RLRewardResult, score_rl_response
from multimodal_science.qwen3vl.rl_rollout import (
    SampledCompletion,
    VerifiedTrainSignals,
    increment_mrope_position,
    sample_seed,
)
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.sensor_projector import build_sensor_projector


GRPO_CONFIG_SCHEMA = "chrompeak-qwen3vl-xic-grpo-config-v1"
GRPO_REPORT_SCHEMA = "chrompeak-qwen3vl-xic-grpo-training-v1"
GRPO_CHECKPOINT_SCHEMA = "chrompeak-qwen3vl-xic-grpo-checkpoint-v1"
DIFFICULTY_POOL_ARTIFACT = "difficulty_training_pool"
EXPERIMENTS = ("R1", "R2", "R3")
RUN_MODES = ("smoke", "calibration", "formal", "reduced-prompts", "reduced-steps")
GROUP_SIZE = 8
FORMAL_PROMPTS_PER_STEP = 16
REDUCED_PROMPTS_PER_STEP = 8
FORMAL_STEPS = 300
REDUCED_STEPS = 200
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 1.0
KL_BETA = 0.02
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 10
CHECKPOINT_INTERVAL = 100
ZERO_VARIANCE_STOP_THRESHOLD = 0.8
KL_STOP_THRESHOLD = 0.5
STOP_PATIENCE = 20
GROUNDING_FRACTION = 0.80
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"Expected an object for {context}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {context}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {context}: {path}") from error


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {context}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {context} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{context} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {context} JSON: {line_number}") from error
    _require(bool(records), f"{context} is empty")
    return records


def _safe_artifact(
    root: Path, report: dict[str, Any], key: str, context: str
) -> tuple[Path, dict[str, Any]]:
    artifact = _object(_object(report.get("artifacts"), "artifacts").get(key), key)
    relative = artifact.get("path")
    _require(isinstance(relative, str) and bool(relative), f"Missing {context} path")
    _require("\\" not in relative, f"Non-portable {context} path")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute() and all(part not in {"", ".", ".."} for part in posix.parts),
        f"Unsafe {context} path",
    )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes its root") from error
    _require(path.is_file(), f"Missing {context}: {path}")
    _require(sha256_file(path) == artifact.get("sha256"), f"{context} hash mismatch")
    return path, artifact


def starting_processor_pixel_bounds(report: dict[str, Any]) -> tuple[int, int]:
    """Return the immutable processor bounds inherited from the starting run."""
    training = _object(report.get("training"), "starting training")
    minimum = training.get("min_pixels")
    maximum = training.get("max_pixels")
    _require(
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and minimum > 0
        and maximum >= minimum,
        "Starting processor pixel bounds are invalid",
    )
    return minimum, maximum


@dataclass(frozen=True)
class GRPOSettings:
    experiment: str = "R1"
    run_mode: str = "formal"
    seed: int = 17
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    deterministic_warn_only: bool = True

    @property
    def prompts_per_step(self) -> int:
        if self.run_mode == "reduced-steps":
            return REDUCED_PROMPTS_PER_STEP
        if self.run_mode == "reduced-prompts":
            return REDUCED_PROMPTS_PER_STEP
        return FORMAL_PROMPTS_PER_STEP

    @property
    def total_steps(self) -> int:
        if self.run_mode in {"smoke", "calibration"}:
            return 2
        if self.run_mode == "reduced-steps":
            return REDUCED_STEPS
        return FORMAL_STEPS

    @property
    def lora_learning_rate(self) -> float:
        return 5e-6 if self.experiment == "R2" else 1e-6

    @property
    def projector_learning_rate(self) -> float | None:
        return 1e-5 if self.experiment == "R3" else None

    @property
    def train_projector(self) -> bool:
        return self.experiment == "R3"

    @property
    def formal_budget_complete(self) -> bool:
        return self.run_mode in {"formal", "reduced-prompts", "reduced-steps"}


def validate_settings(settings: GRPOSettings) -> None:
    _require(settings.experiment in EXPERIMENTS, "GRPO experiment must be R1, R2, or R3")
    _require(settings.run_mode in RUN_MODES, "Unsupported GRPO run mode")
    _require(settings.seed in {17, 29}, "The preregistered GRPO seeds are 17 and 29")
    _require(
        settings.attention_implementation in {"sdpa", "eager"},
        "Unsupported attention implementation",
    )
    _require(GROUP_SIZE == 8 and MAX_NEW_TOKENS == 64, "Frozen rollout settings drifted")
    _require(KL_BETA == 0.02 and WARMUP_STEPS == 10, "Frozen optimizer settings drifted")


@dataclass(frozen=True)
class VerifiedDifficultyPool:
    root: Path
    report: dict[str, Any]
    report_sha256: str
    records: tuple[dict[str, Any], ...]
    rl_pool_report_sha256: str
    selection_group_ids: frozenset[str]


@dataclass(frozen=True)
class GRPOTrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    adapter_dir: Path
    projector_path: Path
    global_step: int
    optimizer_updates: int
    development_training_complete: bool
    stopped_early: bool


@dataclass(frozen=True)
class RolloutGroup:
    record: dict[str, Any]
    completions: tuple[SampledCompletion, ...]
    scores: tuple[RLRewardResult, ...]
    advantages: tuple[float, ...] | None


def normalize_group_advantages(
    rewards: Sequence[float], *, epsilon: float = 1e-6
) -> tuple[float, ...] | None:
    """Return population-standardized group rewards; skip exact zero variance."""

    _require(len(rewards) == GROUP_SIZE, "GRPO requires exactly eight rewards per group")
    values = [float(value) for value in rewards]
    _require(all(math.isfinite(value) for value in values), "Rewards must be finite")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0.0:
        return None
    advantages = tuple((value - mean) / (std + epsilon) for value in values)
    _require(abs(statistics.fmean(advantages)) < 1e-9, "Advantages are not centered")
    return advantages


def kl_k3_values(
    policy_log_probs: Sequence[float], ref_log_probs: Sequence[float]
) -> tuple[float, ...]:
    """Compute the frozen nonnegative k3 KL estimator token by token."""

    _require(len(policy_log_probs) == len(ref_log_probs), "KL token counts disagree")
    values = []
    for policy, reference in zip(policy_log_probs, ref_log_probs):
        difference = float(reference) - float(policy)
        value = math.exp(difference) - difference - 1.0
        _require(math.isfinite(value) and value >= -1e-12, "KL k3 is invalid")
        values.append(max(0.0, value))
    return tuple(values)


def numpy_causal_answer_log_probs(
    logits: Any, labels: Any, *, ignore_index: int = -100
) -> tuple[Any, Any]:
    """NumPy reference for the exact shifted gather used by the CUDA loop."""

    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _require(logits.ndim == 3 and labels.ndim == 2, "Bad logits or labels rank")
    _require(logits.shape[:2] == labels.shape, "Logits and labels shape mismatch")
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    maxima = logits[:, :-1, :].max(axis=-1, keepdims=True)
    stabilized = logits[:, :-1, :] - maxima
    log_normalizer = np.log(np.exp(stabilized).sum(axis=-1, keepdims=True))
    log_probs = stabilized - log_normalizer
    safe_labels = np.where(mask, shifted_labels, 0)
    gathered = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    return gathered[mask], mask


def _torch_tail_answer_log_probs(
    logits: Any, target_labels: Any, *, torch: Any, ignore_index: int = -100
) -> tuple[Any, Any, Any]:
    """Gather answer targets from tail-only Qwen logits.

    ``logits`` contains one predictor position immediately before the answer plus
    one position for every answer token.  The final position predicts beyond the
    sampled answer and is intentionally discarded.
    """

    _require(logits.ndim == 3 and target_labels.ndim == 2, "Bad tail tensor rank")
    _require(
        logits.shape[0] == target_labels.shape[0]
        and logits.shape[1] == target_labels.shape[1] + 1,
        "Tail logits and answer labels do not align",
    )
    predictors = logits[:, :-1, :].float()
    mask = target_labels.ne(ignore_index)
    log_probs = torch.log_softmax(predictors, dim=-1)
    safe_labels = target_labels.masked_fill(~mask, 0)
    selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return selected[mask], entropy[mask], mask


def _is_adapter_parameter(name: str, adapter_name: str) -> bool:
    return "lora_" in name and f".{adapter_name}." in name


def configure_trainable_parameters(
    model_named_parameters: Sequence[tuple[str, Any]],
    projector_named_parameters: Sequence[tuple[str, Any]],
    *,
    experiment: str,
) -> dict[str, tuple[str, ...]]:
    """Freeze base/ref exactly and expose only the preregistered parameter groups."""

    _require(experiment in EXPERIMENTS, "Unknown GRPO experiment")
    groups: dict[str, list[str]] = {"policy": [], "ref": [], "base": [], "projector": []}
    for name, parameter in model_named_parameters:
        if _is_adapter_parameter(name, "policy"):
            parameter.requires_grad_(True)
            groups["policy"].append(name)
        elif _is_adapter_parameter(name, "ref"):
            parameter.requires_grad_(False)
            groups["ref"].append(name)
        else:
            parameter.requires_grad_(False)
            groups["base"].append(name)
    for name, parameter in projector_named_parameters:
        parameter.requires_grad_(experiment == "R3")
        groups["projector"].append(name)
    _require(bool(groups["policy"]), "Policy adapter has no parameters")
    _require(bool(groups["ref"]), "Reference adapter has no parameters")
    _require(bool(groups["base"]), "Base model parameter inventory is empty")
    _require(bool(groups["projector"]), "Projector parameter inventory is empty")
    return {name: tuple(values) for name, values in groups.items()}


def _stable_index_key(seed: int, step: int, lane: str, record: dict[str, Any]) -> str:
    instruction_id = str(record.get("instruction_id") or "")
    return hashlib.sha256(
        f"{seed}\0{step}\0{lane}\0{instruction_id}".encode("utf-8")
    ).hexdigest()


def _take_cycled(
    records: Sequence[dict[str, Any]], count: int, *, seed: int, step: int, lane: str
) -> list[dict[str, Any]]:
    _require(bool(records), f"Empty GRPO sampling lane: {lane}")
    ordered = sorted(records, key=lambda row: _stable_index_key(seed, step, lane, row))
    return [ordered[index % len(ordered)] for index in range(count)]


def prompt_batch(
    records: Sequence[dict[str, Any]],
    *,
    step: int,
    prompts_per_step: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Build a deterministic per-step 80/20 prompt batch with auxiliary strata."""

    _require(prompts_per_step in {8, 16}, "Prompt count violates the reduction plan")
    _require(step >= 1, "GRPO steps are one-based")
    grounding = [record for record in records if record.get("task") == GROUNDING_TASK]
    auxiliary_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = str(record.get("task") or "")
        if task != GROUNDING_TASK:
            auxiliary_by_task[task].append(record)
    _require(bool(grounding), "Difficulty pool has no grounding prompts")
    _require(len(auxiliary_by_task) == 3, "Difficulty pool lacks auxiliary strata")
    grounding_count = int(math.floor(prompts_per_step * GROUNDING_FRACTION + 0.5))
    auxiliary_count = prompts_per_step - grounding_count
    batch = _take_cycled(
        grounding,
        grounding_count,
        seed=seed,
        step=step,
        lane="grounding",
    )
    tasks = sorted(auxiliary_by_task)
    offset = (step - 1) % len(tasks)
    for position in range(auxiliary_count):
        task = tasks[(offset + position) % len(tasks)]
        selected = _take_cycled(
            auxiliary_by_task[task],
            1,
            seed=seed,
            step=step,
            lane=f"auxiliary:{task}:{position}",
        )[0]
        batch.append(selected)
    batch.sort(key=lambda row: _stable_index_key(seed, step, "batch", row))
    return tuple(batch)


def load_difficulty_training_pool(
    *,
    difficulty_root: Path,
    difficulty_report_sha256: str,
    rl_pool_root: Path,
    rl_pool_report_sha256: str,
) -> VerifiedDifficultyPool:
    """Load only train artifacts and prove every selected row came from the RL pool."""

    difficulty_root = difficulty_root.resolve()
    assert_no_protected_input_paths([difficulty_root, rl_pool_root])
    report_path = difficulty_root / "rl_difficulty_report.json"
    _require(sha256_file(report_path) == difficulty_report_sha256, "Difficulty hash mismatch")
    report = _read_json(report_path, "difficulty report")
    _require(report.get("schema_version") == DIFFICULTY_REPORT_SCHEMA, "Bad difficulty schema")
    contracts = _object(report.get("contracts"), "difficulty contracts")
    for name, expected in (
        ("train_pool_only", True),
        ("selection_groups_trained", False),
        ("validation_answers_opened", False),
        ("internal_test_accessed", False),
        ("visual_tower_unused", True),
    ):
        _require(contracts.get(name) is expected, f"Difficulty contract failed: {name}")
    _require(report.get("development_screening_complete") is True, "Difficulty is incomplete")
    source = _object(report.get("source"), "difficulty source")
    _require(source.get("rl_pool_report_sha256") == rl_pool_report_sha256, "RL pool drift")
    selected_path, selected_artifact = _safe_artifact(
        difficulty_root, report, DIFFICULTY_POOL_ARTIFACT, "difficulty training pool"
    )
    records = _read_jsonl(selected_path, "difficulty training pool")
    _require(selected_artifact.get("records") == len(records), "Difficulty count mismatch")
    pool = load_rl_training_pool(rl_pool_root, rl_pool_report_sha256)
    source_by_id = {str(row["instruction_id"]): row for row in pool.records}
    _require(len(source_by_id) == len(pool.records), "Duplicate RL source rows")
    for record in records:
        instruction_id = str(record.get("instruction_id") or "")
        _require(instruction_id in source_by_id, f"Difficulty row is outside RL pool: {instruction_id}")
        original = dict(record)
        difficulty = original.pop("difficulty", None)
        _require(isinstance(difficulty, dict), "Difficulty decision is missing")
        _require(original == source_by_id[instruction_id], f"Difficulty row drift: {instruction_id}")
    assert_training_pool_isolation(records, pool.selection_group_ids)
    return VerifiedDifficultyPool(
        root=difficulty_root,
        report=report,
        report_sha256=difficulty_report_sha256,
        records=tuple(records),
        rl_pool_report_sha256=rl_pool_report_sha256,
        selection_group_ids=pool.selection_group_ids,
    )


def _repeat_cache(cache: Any, repeats: int) -> Any:
    _require(hasattr(cache, "batch_repeat_interleave"), "KV cache cannot repeat batches")
    result = cache.batch_repeat_interleave(repeats)
    return cache if result is None else result


def _select_cache(cache: Any, indices: Any) -> Any:
    _require(hasattr(cache, "batch_select_indices"), "KV cache cannot select batches")
    result = cache.batch_select_indices(indices)
    return cache if result is None else result


def _save_policy_adapter(model: Any, destination: Path) -> None:
    """Save one named policy adapter as a normal standalone PEFT artifact."""

    _require(not destination.exists(), f"Adapter destination already exists: {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".policy-save-") as name:
        raw = Path(name) / "raw"
        model.save_pretrained(
            raw,
            safe_serialization=True,
            selected_adapters=["policy"],
        )
        source = raw / "policy" if (raw / "policy").is_dir() else raw
        _require((source / "adapter_config.json").is_file(), "Saved policy config is missing")
        _require(
            (source / "adapter_model.safetensors").is_file(),
            "Saved policy weights are missing",
        )
        source.replace(destination)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (output_dir / "checkpoints").glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def _manifest_entries(root: Path) -> list[tuple[str, str]]:
    entries = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "artifact_manifest.sha256":
            continue
        entries.append((sha256_file(path), relative))
    return entries


def _write_manifest(root: Path) -> Path:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in _manifest_entries(root)),
        encoding="utf-8",
    )
    return manifest


def _reconcile_history(path: Path, checkpoint_step: int) -> int:
    if not path.exists():
        return 0
    records = _read_jsonl(path, "GRPO history")
    retained = [row for row in records if int(row.get("global_step", -1)) <= checkpoint_step]
    discarded = len(records) - len(retained)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    return discarded


class _GRPORuntime:
    """One-GPU policy/ref runtime with parallel same-prompt sampling."""

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
        settings: GRPOSettings,
    ) -> None:
        assert_no_protected_input_paths(
            [model_name_or_path, model_manifest_path, starting_adapter.root, train_signals.path]
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
            raise RuntimeError("GRPO requires NumPy, torch, PEFT, safetensors and transformers") from error

        _require(torch.cuda.is_available(), "GRPO requires CUDA")
        _require(torch.cuda.device_count() == 1, "Expose exactly one CUDA device")
        _require(torch.cuda.is_bf16_supported(), "Visible GPU does not support BF16")
        runtime_packages = {name: version(name) for name in REQUIRED_RUNTIME_PACKAGES}
        _require(
            runtime_packages == REQUIRED_RUNTIME_PACKAGES,
            f"GRPO runtime drift: expected={REQUIRED_RUNTIME_PACKAGES} actual={runtime_packages}",
        )
        model_inventory = _verify_model_manifest(
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
        _require(verified is not None, "Starting adapter verification failed")
        adapter_config = _read_json(
            verified.adapter_dir / "adapter_config.json", "starting adapter config"
        )
        _require(
            float(adapter_config.get("lora_dropout", -1.0)) == 0.0,
            "On-policy GRPO requires the frozen zero-dropout LoRA",
        )
        trained_attention = _object(
            verified.report.get("training"), "starting training"
        ).get("attention_implementation")
        _require(
            trained_attention == settings.attention_implementation,
            "GRPO attention differs from the starting model",
        )

        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        torch.cuda.manual_seed_all(settings.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(
            True, warn_only=settings.deterministic_warn_only
        )
        device = torch.device("cuda:0")
        processor = AutoProcessor.from_pretrained(
            verified.processor_dir, trust_remote_code=False
        )
        processor.tokenizer.padding_side = "right"
        base = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            dtype=torch.bfloat16,
            attn_implementation=settings.attention_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(
            base,
            verified.adapter_dir,
            adapter_name="policy",
            is_trainable=True,
        )
        model.load_adapter(
            verified.adapter_dir,
            adapter_name="ref",
            is_trainable=False,
        )
        model.to(device)
        if settings.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        generation_model = model.get_base_model()
        _require(
            float(generation_model.config.text_config.attention_dropout) == 0.0,
            "On-policy GRPO requires zero text-attention dropout",
        )
        projector = build_sensor_projector(verified.projector_spec).to(device)
        projector.load_state_dict(load_file(str(verified.projector_path)), strict=True)
        ref_projector = copy.deepcopy(projector).to(device).eval()
        for parameter in ref_projector.parameters():
            parameter.requires_grad_(False)
        signals = np.load(train_signals.path, mmap_mode="r", allow_pickle=False)
        _require(tuple(signals.shape) == train_signals.shape, "Train signal shape drift")
        _require(signals.dtype == np.float32, "Train signals must be float32")
        _require(bool(np.isfinite(signals).all()), "Train signals contain non-finite values")
        _require(
            float(signals.min()) >= 0.0 and float(signals.max()) <= 1.0,
            "Train signals violate the normalized [0, 1] contract",
        )
        position_token_id = processor.tokenizer.pad_token_id
        _require(isinstance(position_token_id, int), "Tokenizer has no position shadow token")
        for token_name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            _require(
                position_token_id != getattr(generation_model.config, token_name),
                f"Position shadow token collides with {token_name}",
            )

        parameter_groups = configure_trainable_parameters(
            list(model.named_parameters()),
            list(projector.named_parameters()),
            experiment=settings.experiment,
        )
        policy_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name in set(parameter_groups["policy"])
        ]
        ref_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name in set(parameter_groups["ref"])
        ]
        projector_named = list(projector.named_parameters())
        policy_parameters = [parameter for _, parameter in policy_named]
        projector_parameters = [parameter for _, parameter in projector_named]
        optimizer_groups: list[dict[str, Any]] = [
            {"params": policy_parameters, "lr": settings.lora_learning_rate}
        ]
        if settings.train_projector:
            optimizer_groups.append(
                {
                    "params": projector_parameters,
                    "lr": settings.projector_learning_rate,
                }
            )
        optimizer_ids = {
            id(parameter)
            for group in optimizer_groups
            for parameter in group["params"]
        }
        expected_ids = {id(parameter) for parameter in policy_parameters}
        if settings.train_projector:
            expected_ids |= {id(parameter) for parameter in projector_parameters}
        _require(optimizer_ids == expected_ids, "Optimizer parameter scope drift")
        _require(
            all(not parameter.requires_grad for _, parameter in ref_named),
            "Reference adapter is trainable",
        )
        visual = getattr(generation_model, "visual", None)
        _require(visual is not None, "Qwen visual tower was not found")
        self.visual_forward_calls = 0

        def reject_visual_forward(_module: Any, _inputs: Any) -> None:
            self.visual_forward_calls += 1
            raise RuntimeError("XIC-only GRPO invoked the visual tower")

        self._visual_guard = visual.register_forward_pre_hook(reject_visual_forward)
        torch.cuda.reset_peak_memory_stats(device)
        self.torch = torch
        self.np = np
        self.load_file = load_file
        self.save_file = save_file
        self.set_peft_model_state_dict = set_peft_model_state_dict
        self.get_peft_model_state_dict = get_peft_model_state_dict
        self.device = device
        self.model = model
        self.generation_model = generation_model
        self.processor = processor
        self.projector = projector
        self.ref_projector = ref_projector
        self.signals = signals
        self.position_token_id = position_token_id
        self.settings = settings
        self.verified_start = verified
        self.model_inventory = model_inventory
        self.train_signals = train_signals
        self.parameter_groups = parameter_groups
        self.model_named = list(model.named_parameters())
        self.policy_named = policy_named
        self.ref_named = ref_named
        self.projector_named = projector_named
        self.policy_parameters = policy_parameters
        self.projector_parameters = projector_parameters
        self.initial_policy_sha256 = _parameter_digest(policy_named)
        self.initial_ref_sha256 = _parameter_digest(ref_named)
        self.initial_projector_sha256 = _parameter_digest(projector_named)
        self.adapter_switches = Counter()

    @property
    def peak_cuda_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_allocated(self.device))

    def close(self) -> None:
        self._visual_guard.remove()

    def activate_adapter(self, name: str, *, trainable: bool) -> None:
        _require(name in {"policy", "ref"}, "Unknown adapter")
        # PEFT 0.17 makes the selected adapter trainable.  Reassert the complete
        # scope after every switch so the reference and base never accumulate grads.
        self.model.set_adapter(name)
        for parameter_name, parameter in self.model_named:
            parameter.requires_grad_(
                name == "policy"
                and trainable
                and _is_adapter_parameter(parameter_name, "policy")
            )
        for parameter in self.projector.parameters():
            parameter.requires_grad_(self.settings.train_projector)
        _require(
            all(not parameter.requires_grad for _, parameter in self.ref_named),
            "Reference adapter became trainable",
        )
        self.adapter_switches[name] += 1

    def verify_saved_policy(self, adapter_dir: Path) -> bool:
        expected = self.get_peft_model_state_dict(
            self.model, adapter_name="policy"
        )
        actual = self.load_file(str(adapter_dir / "adapter_model.safetensors"))
        _require(set(actual) == set(expected), "Reloaded policy tensor keys drifted")
        for key, expected_tensor in expected.items():
            _require(
                self.torch.equal(actual[key].cpu(), expected_tensor.detach().cpu()),
                f"Reloaded policy tensor drifted: {key}",
            )
        return True

    def _prompt_and_signal(self, record: dict[str, Any]) -> tuple[Any, Any, Any]:
        prompt = record.get("prompt")
        language = record.get("language")
        _require(isinstance(prompt, str) and bool(prompt), "RL record has no prompt")
        _require(language in {"en", "zh-CN"}, "RL record has an invalid language")
        text = xic_only_prompt_text(prompt, str(language))
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        signal_link = _object(record.get("signal"), "RL signal link")
        row = signal_link.get("row")
        available = signal_link.get("available")
        _require(
            isinstance(row, int) and 0 <= row < self.signals.shape[0],
            "RL signal row is out of range",
        )
        _require(isinstance(available, bool), "RL signal availability is invalid")
        signal = self.torch.from_numpy(
            self.np.array(self.signals[row], copy=True)
        ).unsqueeze(0)
        signal = signal.to(device=self.device, dtype=self.torch.float32)
        availability = self.torch.tensor(
            [available], dtype=self.torch.bool, device=self.device
        )
        return input_ids, signal, availability

    def _fused_inputs(
        self,
        *,
        input_ids: Any,
        labels: Any,
        prompt_lengths: Sequence[int],
        signals: Any,
        availability: Any,
        projector: Any,
        real_lengths: Sequence[int] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        sensor_embeddings = projector(signals, availability)
        inputs_embeds, fused_labels, attention_mask, shadow_ids = insert_sensor_embeddings(
            self.model,
            input_ids,
            labels,
            sensor_embeddings,
            prompt_lengths,
            position_token_id=self.position_token_id,
        )
        if real_lengths is not None:
            sensor_tokens = int(sensor_embeddings.shape[1])
            fused_lengths = self.torch.tensor(
                [length + sensor_tokens for length in real_lengths],
                dtype=self.torch.long,
                device=self.device,
            )
            positions = self.torch.arange(
                fused_labels.shape[1], device=self.device
            ).unsqueeze(0)
            attention_mask = positions.lt(fused_lengths.unsqueeze(1)).long()
        position_ids, _ = self.generation_model.model.get_rope_index(
            input_ids=shadow_ids,
            attention_mask=attention_mask,
        )
        return inputs_embeds, fused_labels, attention_mask, position_ids

    def sample_group(
        self, record: dict[str, Any], *, global_step: int
    ) -> tuple[SampledCompletion, ...]:
        """Prefill once, repeat the KV cache, and sample eight active rows in parallel."""

        torch = self.torch
        instruction_id = str(record.get("instruction_id") or "")
        _require(instruction_id != "", "RL record has no instruction ID")
        self.activate_adapter("policy", trainable=False)
        self.model.eval()
        self.projector.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            input_ids, signal, availability = self._prompt_and_signal(record)
            labels = torch.full_like(input_ids, -100)
            inputs_embeds, _, attention_mask, position_ids = self._fused_inputs(
                input_ids=input_ids,
                labels=labels,
                prompt_lengths=[int(input_ids.shape[1])],
                signals=signal,
                availability=availability,
                projector=self.projector,
            )
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            _require(self.visual_forward_calls == 0, "GRPO sampling used image pixels")
            current_logits = outputs.logits[:, -1, :].repeat(GROUP_SIZE, 1)
            cache = _repeat_cache(copy.deepcopy(outputs.past_key_values), GROUP_SIZE)
            current_mask = attention_mask.repeat(GROUP_SIZE, 1)
            current_position = position_ids[:, :, -1:].repeat(1, GROUP_SIZE, 1) + 1
            _require(
                torch.equal(current_position[0], current_position[1])
                and torch.equal(current_position[1], current_position[2]),
                "XIC-only mRoPE axes diverged before decoding",
            )
            eos_ids = _eos_token_ids(self.model.generation_config.eos_token_id)
            generated: list[list[int]] = [[] for _ in range(GROUP_SIZE)]
            terminated = [False] * GROUP_SIZE
            seeds = [
                sample_seed(
                    self.settings.seed,
                    f"{instruction_id}\0step-{global_step}",
                    sample_index,
                )
                for sample_index in range(GROUP_SIZE)
            ]
            generators = []
            for seed in seeds:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                generators.append(generator)
            active = list(range(GROUP_SIZE))
            for _ in range(MAX_NEW_TOKENS):
                probabilities = torch.softmax(current_logits.float() / TEMPERATURE, dim=-1)
                sampled = torch.stack(
                    [
                        torch.multinomial(
                            probabilities[row_index],
                            num_samples=1,
                            generator=generators[original_index],
                        )
                        for row_index, original_index in enumerate(active)
                    ]
                )
                keep_rows = []
                next_active = []
                for row_index, original_index in enumerate(active):
                    token_id = int(sampled[row_index].item())
                    generated[original_index].append(token_id)
                    if token_id in eos_ids:
                        terminated[original_index] = True
                    else:
                        keep_rows.append(row_index)
                        next_active.append(original_index)
                if not keep_rows:
                    break
                keep = torch.tensor(keep_rows, dtype=torch.long, device=self.device)
                cache = _select_cache(cache, keep)
                next_tokens = sampled.index_select(0, keep)
                current_mask = current_mask.index_select(0, keep)
                current_position = current_position.index_select(1, keep)
                current_mask = torch.cat(
                    (
                        current_mask,
                        torch.ones(
                            (len(keep_rows), 1),
                            dtype=current_mask.dtype,
                            device=self.device,
                        ),
                    ),
                    dim=1,
                )
                outputs = self.model(
                    input_ids=next_tokens,
                    attention_mask=current_mask,
                    position_ids=current_position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                current_logits = outputs.logits[:, -1, :]
                cache = outputs.past_key_values
                current_position = increment_mrope_position(current_position)
                active = next_active
            completions = tuple(
                SampledCompletion(
                    response=self.processor.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    token_ids=tuple(token_ids),
                    terminated_with_eos=terminated[index],
                    sample_seed=seeds[index],
                )
                for index, token_ids in enumerate(generated)
            )
        self.activate_adapter("policy", trainable=True)
        return completions

    def _teacher_batch(
        self,
        record: dict[str, Any],
        completions: Sequence[SampledCompletion],
        *,
        projector: Any,
    ) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        prompt_ids, signal, availability = self._prompt_and_signal(record)
        prompt_length = int(prompt_ids.shape[1])
        answer_lengths = [len(completion.token_ids) for completion in completions]
        _require(all(length >= 1 for length in answer_lengths), "Empty sampled answer")
        maximum_answer = max(answer_lengths)
        pad_token_id = self.processor.tokenizer.pad_token_id
        _require(isinstance(pad_token_id, int), "Tokenizer has no pad token")
        input_rows = []
        label_rows = []
        prompt_tokens = prompt_ids[0].tolist()
        for completion in completions:
            answer = list(completion.token_ids)
            padding = maximum_answer - len(answer)
            input_rows.append(prompt_tokens + answer + [pad_token_id] * padding)
            label_rows.append([-100] * prompt_length + answer + [-100] * padding)
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        labels = torch.tensor(label_rows, dtype=torch.long, device=self.device)
        signals = signal.repeat(len(completions), 1)
        availabilities = availability.repeat(len(completions))
        real_lengths = [prompt_length + length for length in answer_lengths]
        return self._fused_inputs(
            input_ids=input_ids,
            labels=labels,
            prompt_lengths=[prompt_length] * len(completions),
            signals=signals,
            availability=availabilities,
            projector=projector,
            real_lengths=real_lengths,
        )

    def token_log_probs(
        self,
        record: dict[str, Any],
        completions: Sequence[SampledCompletion],
        *,
        adapter_name: str,
    ) -> tuple[list[Any], list[Any]]:
        """Teacher-force complete sampled answers and return answer-only token values."""

        torch = self.torch
        is_policy = adapter_name == "policy"
        _require(is_policy or adapter_name == "ref", "Unknown log-prob adapter")
        self.activate_adapter(adapter_name, trainable=is_policy)
        if is_policy:
            self.model.train()
        else:
            self.model.eval()
        projector = self.projector if is_policy else self.ref_projector
        projector.eval()
        context = torch.enable_grad() if is_policy else torch.no_grad()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inputs_embeds, labels, attention_mask, position_ids = self._teacher_batch(
                record, completions, projector=projector
            )
            maximum_answer = max(len(completion.token_ids) for completion in completions)
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=maximum_answer + 1,
            )
            target_labels = labels[:, -maximum_answer:]
            selected, entropy, _ = _torch_tail_answer_log_probs(
                outputs.logits, target_labels, torch=torch
            )
            lengths = [len(completion.token_ids) for completion in completions]
            _require(int(selected.numel()) == sum(lengths), "Answer log-prob count mismatch")
            log_prob_rows = []
            entropy_rows = []
            offset = 0
            for length in lengths:
                log_prob_rows.append(selected[offset : offset + length])
                entropy_rows.append(entropy[offset : offset + length])
                offset += length
        return log_prob_rows, entropy_rows

    def checkpoint(
        self,
        *,
        output_dir: Path,
        optimizer: Any,
        scheduler: Any,
        state: dict[str, Any],
    ) -> Path:
        torch = self.torch
        checkpoints = output_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        final = checkpoints / f"step-{int(state['global_step']):08d}"
        _require(not final.exists(), f"Checkpoint already exists: {final}")
        with tempfile.TemporaryDirectory(
            dir=checkpoints, prefix=f".{final.name}-"
        ) as staging_name:
            staging = Path(staging_name)
            _save_policy_adapter(self.model, staging / "adapter")
            self.save_file(
                {
                    key: value.detach().cpu()
                    for key, value in self.projector.state_dict().items()
                },
                str(staging / "sensor_projector.safetensors"),
            )
            torch.save(optimizer.state_dict(), staging / "optimizer.pt")
            torch.save(scheduler.state_dict(), staging / "scheduler.pt")
            torch.save(
                {
                    "python": random.getstate(),
                    "numpy": self.np.random.get_state(),
                    "torch_cpu": torch.random.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                },
                staging / "rng_state.pt",
            )
            _write_json(
                staging / "checkpoint.json",
                {"schema_version": GRPO_CHECKPOINT_SCHEMA, **state},
            )
            Path(staging_name).replace(final)
        return final

    def restore_checkpoint(
        self,
        checkpoint: Path,
        *,
        optimizer: Any,
        scheduler: Any,
    ) -> dict[str, Any]:
        torch = self.torch
        state = _read_json(checkpoint / "checkpoint.json", "GRPO checkpoint")
        _require(state.pop("schema_version") == GRPO_CHECKPOINT_SCHEMA, "Bad checkpoint schema")
        adapter_state = self.load_file(
            str(checkpoint / "adapter" / "adapter_model.safetensors")
        )
        self.set_peft_model_state_dict(
            self.model, adapter_state, adapter_name="policy"
        )
        self.projector.load_state_dict(
            self.load_file(str(checkpoint / "sensor_projector.safetensors")),
            strict=True,
        )
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(checkpoint / "scheduler.pt", weights_only=False)
        )
        _optimizer_to_device(optimizer, self.device)
        rng = torch.load(checkpoint / "rng_state.pt", weights_only=False)
        random.setstate(rng["python"])
        self.np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
        return state


def _score_group(
    record: dict[str, Any], completions: Sequence[SampledCompletion]
) -> tuple[RLRewardResult, ...]:
    return tuple(
        score_rl_response(
            task=str(record["task"]),
            prompt={"conversations": [{"from": "human", "value": record["prompt"]}]},
            manifest={},
            expected_response=str(record["expected_response"]),
            predicted_response=completion.response,
            terminated_with_eos=completion.terminated_with_eos,
        )
        for completion in completions
    )


def _rollout_groups(
    runtime: _GRPORuntime,
    records: Sequence[dict[str, Any]],
    *,
    global_step: int,
) -> tuple[RolloutGroup, ...]:
    groups = []
    for record in records:
        completions = runtime.sample_group(record, global_step=global_step)
        _require(len(completions) == GROUP_SIZE, "Sampling returned the wrong group size")
        scores = _score_group(record, completions)
        advantages = normalize_group_advantages([score.reward for score in scores])
        groups.append(
            RolloutGroup(
                record=record,
                completions=completions,
                scores=scores,
                advantages=advantages,
            )
        )
    return tuple(groups)


def _backward_group(
    runtime: _GRPORuntime,
    group: RolloutGroup,
    *,
    total_answer_tokens: int,
) -> dict[str, Any]:
    _require(group.advantages is not None, "Cannot update a zero-variance group")
    ref_rows, _ = runtime.token_log_probs(
        group.record, group.completions, adapter_name="ref"
    )
    policy_rows, entropy_rows = runtime.token_log_probs(
        group.record, group.completions, adapter_name="policy"
    )
    torch = runtime.torch
    policy_objective = torch.zeros((), dtype=torch.float32, device=runtime.device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=runtime.device)
    token_kl: list[float] = []
    token_entropy: list[float] = []
    for advantage, policy, reference, entropy in zip(
        group.advantages, policy_rows, ref_rows, entropy_rows
    ):
        _require(policy.shape == reference.shape, "Policy/ref token shapes disagree")
        reference = reference.detach()
        policy_objective = policy_objective - float(advantage) * policy.sum()
        difference = reference - policy
        kl = torch.exp(difference) - difference - 1.0
        _require(bool(torch.isfinite(kl).all()), "Non-finite token KL")
        kl_sum = kl_sum + kl.sum()
        token_kl.extend(float(value) for value in kl.detach().cpu().tolist())
        token_entropy.extend(float(value) for value in entropy.detach().cpu().tolist())
    loss = (policy_objective + KL_BETA * kl_sum) / total_answer_tokens
    _require(bool(torch.isfinite(loss)), "Non-finite GRPO loss")
    loss.backward()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss_numerator": float(policy_objective.detach().cpu()),
        "kl_sum": float(kl_sum.detach().cpu()),
        "token_kl": token_kl,
        "token_entropy": token_entropy,
    }


def _means_by(
    groups: Sequence[RolloutGroup], field: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        key = str(group.record.get(field))
        values[key].extend(score.reward for score in group.scores)
    return {key: statistics.fmean(rows) for key, rows in sorted(values.items())}


def _step_diagnostics(
    groups: Sequence[RolloutGroup],
    *,
    token_kl: Sequence[float],
    token_entropy: Sequence[float],
) -> dict[str, Any]:
    completions = [completion for group in groups for completion in group.completions]
    scores = [score for group in groups for score in group.scores]
    grounding = [score for score in scores if score.task == GROUNDING_TASK]
    valid_grounding = [score for score in grounding if score.interval_iou is not None]
    width_ratios = [
        float(score.interval_width_ratio)
        for score in valid_grounding
        if score.interval_width_ratio is not None
    ]
    zero_variance_groups = sum(group.advantages is None for group in groups)
    return {
        "mean_reward_by_task": _means_by(groups, "task"),
        "mean_reward_by_language": _means_by(groups, "language"),
        "grounding_mean_iou": (
            statistics.fmean(float(score.interval_iou) for score in valid_grounding)
            if valid_grounding
            else None
        ),
        "grounding_mean_giou": (
            statistics.fmean(float(score.interval_giou) for score in valid_grounding)
            if valid_grounding
            else None
        ),
        "grounding_mean_interval_width_ratio": (
            statistics.fmean(width_ratios) if width_ratios else None
        ),
        "zero_variance_groups": zero_variance_groups,
        "zero_variance_group_fraction": zero_variance_groups / len(groups),
        "kl_mean": statistics.fmean(token_kl) if token_kl else 0.0,
        "mean_answer_tokens": statistics.fmean(
            len(completion.token_ids) for completion in completions
        ),
        "truncation_fraction": sum(
            not completion.terminated_with_eos for completion in completions
        )
        / len(completions),
        "format_error_fraction": sum(not score.valid for score in scores) / len(scores),
        "mean_sampling_entropy": (
            statistics.fmean(token_entropy) if token_entropy else None
        ),
        "rollouts": len(completions),
        "answer_tokens": sum(len(completion.token_ids) for completion in completions),
    }


def _checkpoint_state(
    *,
    global_step: int,
    total_steps: int,
    optimizer_updates: int,
    zero_variance_patience: int,
    kl_patience: int,
    first_gradient_norms: dict[str, float] | None,
    stopped_early: bool,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "total_steps": total_steps,
        "optimizer_updates": optimizer_updates,
        "zero_variance_patience": zero_variance_patience,
        "kl_patience": kl_patience,
        "first_gradient_norms": first_gradient_norms,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def run_grpo_training(
    *,
    code_revision: str,
    rl_pool_root: Path,
    rl_pool_report_sha256: str,
    difficulty_root: Path,
    difficulty_report_sha256: str,
    train_signals: VerifiedTrainSignals,
    starting_adapter: AdapterSpec,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    model_manifest_path: Path,
    model_manifest_sha256: str,
    output_dir: Path,
    settings: GRPOSettings,
    resume: bool = False,
) -> GRPOTrainingResult:
    """Run one-update-per-rollout on-policy GRPO without protected data access."""

    validate_settings(settings)
    for value, label, pattern in (
        (code_revision, "code revision", _HEX_40),
        (rl_pool_report_sha256, "RL pool hash", _HEX_64),
        (difficulty_report_sha256, "difficulty hash", _HEX_64),
        (train_signals.report_sha256, "Dataset hash", _HEX_64),
        (model_artifact_sha256, "model hash", _HEX_64),
        (model_manifest_sha256, "model manifest hash", _HEX_64),
    ):
        _require(bool(pattern.fullmatch(value)), f"Invalid {label}")
    _require(
        os.environ.get("BIOCODER_VERIFIED_CODE_REVISION") == code_revision,
        "Code revision was not verified by the immutable launcher",
    )
    _require(
        os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE") == "staged_train_artifacts_only",
        "GRPO requires staged train-only roots",
    )
    _require(
        os.environ.get("BIOCODER_GPU_ALLOCATION_MODE") == GPU_ALLOCATION_MODE,
        "GPU allocation mode was not declared",
    )
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be set before Python starts",
    )
    assert_no_protected_input_paths(
        [
            rl_pool_root,
            difficulty_root,
            train_signals.path,
            starting_adapter.root,
            output_dir,
        ]
    )
    pool = load_difficulty_training_pool(
        difficulty_root=difficulty_root,
        difficulty_report_sha256=difficulty_report_sha256,
        rl_pool_root=rl_pool_root,
        rl_pool_report_sha256=rl_pool_report_sha256,
    )
    assert_training_pool_isolation(pool.records, pool.selection_group_ids)
    base_pool = load_rl_training_pool(rl_pool_root, rl_pool_report_sha256)
    base_sources = _object(base_pool.report.get("sources"), "RL pool sources")
    _require(
        base_sources.get("dataset_report_sha256") == train_signals.report_sha256,
        "RL pool and staged train tensors disagree",
    )
    output_dir = output_dir.resolve()
    configuration = {
        "schema_version": GRPO_CONFIG_SCHEMA,
        "code_revision": code_revision,
        "algorithm": "on_policy_group_baseline_reinforce_with_k3_kl",
        "experiment": settings.experiment,
        "run_mode": settings.run_mode,
        "settings": asdict(settings),
        "frozen_hyperparameters": {
            "group_size": GROUP_SIZE,
            "prompts_per_step": settings.prompts_per_step,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "kl_beta": KL_BETA,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_steps": WARMUP_STEPS,
            "total_steps": settings.total_steps,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "lora_learning_rate": settings.lora_learning_rate,
            "projector_learning_rate": settings.projector_learning_rate,
        },
        "sources": {
            "rl_pool_report_sha256": rl_pool_report_sha256,
            "difficulty_report_sha256": difficulty_report_sha256,
            "dataset_report_sha256": train_signals.report_sha256,
            "starting_adapter_report_sha256": starting_adapter.training_report_sha256,
            "starting_adapter_manifest_sha256": starting_adapter.manifest_sha256,
            "model_revision": model_revision,
            "model_artifact_sha256": model_artifact_sha256,
            "model_manifest_sha256": model_manifest_sha256,
        },
        "data_contract": {
            "train_pool_only": True,
            "selection_groups_trained": False,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
    }
    config_path = output_dir / "training_config.json"
    report_path = output_dir / "fusion_training_report.json"
    if output_dir.exists():
        _require(resume, f"Training output already exists: {output_dir}")
        _require(not report_path.exists(), "Completed GRPO run cannot be resumed")
        _require(_read_json(config_path, "GRPO config") == configuration, "Resume mismatch")
    else:
        output_dir.mkdir(parents=True)
        _write_json(config_path, configuration)
    history_path = output_dir / "training_history.jsonl"

    runtime = _GRPORuntime(
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        model_artifact_sha256=model_artifact_sha256,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=model_manifest_sha256,
        starting_adapter=starting_adapter,
        train_signals=train_signals,
        settings=settings,
    )
    torch = runtime.torch
    try:
        from transformers import get_constant_schedule_with_warmup
    except ImportError as error:
        runtime.close()
        raise RuntimeError("GRPO requires the Transformers scheduler") from error
    optimizer_groups: list[dict[str, Any]] = [
        {"params": runtime.policy_parameters, "lr": settings.lora_learning_rate}
    ]
    if settings.train_projector:
        optimizer_groups.append(
            {
                "params": runtime.projector_parameters,
                "lr": settings.projector_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS
    )
    state = _checkpoint_state(
        global_step=0,
        total_steps=settings.total_steps,
        optimizer_updates=0,
        zero_variance_patience=0,
        kl_patience=0,
        first_gradient_norms=None,
        stopped_early=False,
        stop_reason=None,
    )
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    if checkpoint is not None:
        state = runtime.restore_checkpoint(
            checkpoint, optimizer=optimizer, scheduler=scheduler
        )
        _require(state.get("total_steps") == settings.total_steps, "Resume budget drift")
    discarded_history = _reconcile_history(history_path, int(state["global_step"]))
    wall_started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    runtime.activate_adapter("policy", trainable=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        for global_step in range(int(state["global_step"]) + 1, settings.total_steps + 1):
            step_started = time.monotonic()
            batch = prompt_batch(
                pool.records,
                step=global_step,
                prompts_per_step=settings.prompts_per_step,
                seed=settings.seed,
            )
            assert_training_pool_isolation(batch, pool.selection_group_ids)
            groups = _rollout_groups(runtime, batch, global_step=global_step)
            active = [group for group in groups if group.advantages is not None]
            total_answer_tokens = sum(
                len(completion.token_ids)
                for group in active
                for completion in group.completions
            )
            token_kl: list[float] = []
            token_entropy: list[float] = []
            loss_value = 0.0
            if active:
                _require(total_answer_tokens > 0, "Active rollout batch has no answer tokens")
                for group in active:
                    contribution = _backward_group(
                        runtime,
                        group,
                        total_answer_tokens=total_answer_tokens,
                    )
                    loss_value += contribution["loss"]
                    token_kl.extend(contribution["token_kl"])
                    token_entropy.extend(contribution["token_entropy"])
                if state.get("first_gradient_norms") is None:
                    first_gradient_norms = {}
                    parameter_groups = [("policy", runtime.policy_parameters)]
                    if settings.train_projector:
                        parameter_groups.append(("projector", runtime.projector_parameters))
                    for name, parameters in parameter_groups:
                        squared = sum(
                            float(parameter.grad.detach().float().square().sum())
                            for parameter in parameters
                            if parameter.grad is not None
                        )
                        norm = math.sqrt(squared)
                        _require(math.isfinite(norm) and norm > 0.0, f"No gradient: {name}")
                        first_gradient_norms[name] = norm
                    state["first_gradient_norms"] = first_gradient_norms
                trainable = list(runtime.policy_parameters)
                if settings.train_projector:
                    trainable.extend(runtime.projector_parameters)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                _require(bool(torch.isfinite(grad_norm)), "Non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                state["optimizer_updates"] = int(state["optimizer_updates"]) + 1
                optimizer.zero_grad(set_to_none=True)
            diagnostics = _step_diagnostics(
                groups,
                token_kl=token_kl,
                token_entropy=token_entropy,
            )
            zero_patience = (
                int(state["zero_variance_patience"]) + 1
                if diagnostics["zero_variance_group_fraction"] > ZERO_VARIANCE_STOP_THRESHOLD
                else 0
            )
            kl_patience = (
                int(state["kl_patience"]) + 1
                if diagnostics["kl_mean"] > KL_STOP_THRESHOLD
                else 0
            )
            state.update(
                {
                    "global_step": global_step,
                    "zero_variance_patience": zero_patience,
                    "kl_patience": kl_patience,
                }
            )
            stop_reason = None
            if zero_patience >= STOP_PATIENCE:
                stop_reason = "zero_variance_fraction_above_0.8_for_20_steps"
            elif kl_patience >= STOP_PATIENCE:
                stop_reason = "mean_kl_above_0.5_for_20_steps"
            if stop_reason is not None:
                state["stopped_early"] = True
                state["stop_reason"] = stop_reason
            learning_rates = scheduler.get_last_lr()
            history_record = {
                "schema_version": "chrompeak-qwen3vl-xic-grpo-step-v1",
                "global_step": global_step,
                "optimizer_updates": state["optimizer_updates"],
                "prompt_records": len(groups),
                "loss": loss_value,
                "gradient_norm": (
                    float(grad_norm.detach().cpu()) if active else None
                ),
                "lora_learning_rate": learning_rates[0],
                "projector_learning_rate": (
                    learning_rates[1] if settings.train_projector else None
                ),
                "step_wall_time_seconds": time.monotonic() - step_started,
                **diagnostics,
            }
            _append_jsonl(history_path, history_record)
            print(
                f"[grpo step {global_step}/{settings.total_steps}] "
                f"loss={loss_value:.6f} kl={diagnostics['kl_mean']:.6f} "
                f"zero_variance={diagnostics['zero_variance_group_fraction']:.4f}",
                flush=True,
            )
            if (
                global_step % CHECKPOINT_INTERVAL == 0
                or global_step == settings.total_steps
                or state["stopped_early"]
            ):
                runtime.checkpoint(
                    output_dir=output_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                )
            if state["stopped_early"]:
                break
    finally:
        runtime.close()

    history = _read_jsonl(history_path, "GRPO history")
    _require(len(history) == int(state["global_step"]), "GRPO history is incomplete")
    final_checkpoint = _latest_checkpoint(output_dir)
    _require(final_checkpoint is not None, "GRPO produced no checkpoint")
    if int(final_checkpoint.name.removeprefix("step-")) != int(state["global_step"]):
        final_checkpoint = runtime.checkpoint(
            output_dir=output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
        )
    adapter_dir = output_dir / "adapter"
    processor_dir = output_dir / "processor"
    projector_path = output_dir / "sensor_projector.safetensors"
    for final_artifact in (adapter_dir, processor_dir, projector_path):
        _require(not final_artifact.exists(), f"Final artifact already exists: {final_artifact}")
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".final-") as staging_name:
        staging = Path(staging_name)
        staged_adapter = staging / "adapter"
        staged_processor = staging / "processor"
        staged_projector = staging / "sensor_projector.safetensors"
        _save_policy_adapter(runtime.model, staged_adapter)
        runtime.processor.save_pretrained(staged_processor)
        runtime.save_file(
            {
                key: value.detach().cpu()
                for key, value in runtime.projector.state_dict().items()
            },
            str(staged_projector),
        )
        adapter_reloaded = runtime.verify_saved_policy(staged_adapter)
        staged_adapter.replace(adapter_dir)
        staged_processor.replace(processor_dir)
        staged_projector.replace(projector_path)
    final_policy_sha256 = _parameter_digest(runtime.policy_named)
    final_ref_sha256 = _parameter_digest(runtime.ref_named)
    final_projector_sha256 = _parameter_digest(runtime.projector_named)
    _require(int(state["optimizer_updates"]) >= 1, "GRPO made no optimizer update")
    _require(state["first_gradient_norms"] is not None, "Answer gradient was never observed")
    _require(final_policy_sha256 != runtime.initial_policy_sha256, "Policy adapter did not update")
    _require(final_ref_sha256 == runtime.initial_ref_sha256, "Reference adapter changed")
    if settings.train_projector and int(state["optimizer_updates"]) > 0:
        _require(
            final_projector_sha256 != runtime.initial_projector_sha256,
            "R3 projector did not update",
        )
    else:
        _require(
            final_projector_sha256 == runtime.initial_projector_sha256,
            "Frozen projector changed",
        )
    completed_budget = int(state["global_step"]) == settings.total_steps
    development_complete = (
        settings.formal_budget_complete
        and completed_budget
        and not bool(state["stopped_early"])
    )
    start_sources = _object(runtime.verified_start.report.get("sources"), "start sources")
    min_pixels, max_pixels = starting_processor_pixel_bounds(runtime.verified_start.report)
    report = {
        "schema_version": GRPO_REPORT_SCHEMA,
        "code_revision": code_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "sources": {
            "rl_pool_report_sha256": rl_pool_report_sha256,
            "difficulty_report_sha256": difficulty_report_sha256,
            "dataset_report_sha256": train_signals.report_sha256,
            "starting_adapter_report_sha256": starting_adapter.training_report_sha256,
            "starting_adapter_manifest_sha256": starting_adapter.manifest_sha256,
            "fusion_bundle_report_sha256": start_sources.get(
                "fusion_bundle_report_sha256"
            ),
        },
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "base_artifact_sha256": model_artifact_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "model_inventory": runtime.model_inventory,
            "sensor_projector": runtime.verified_start.projector_spec.as_dict(),
        },
        "training": {
            "algorithm": "grpo_on_policy_single_update",
            "experiment": settings.experiment,
            "run_mode": settings.run_mode,
            "global_steps": state["global_step"],
            "optimizer_updates": state["optimizer_updates"],
            "training_records": int(state["global_step"]) * settings.prompts_per_step,
            "group_size": GROUP_SIZE,
            "rollouts_per_step": settings.prompts_per_step * GROUP_SIZE,
            "lora_learning_rate": settings.lora_learning_rate,
            "projector_learning_rate": settings.projector_learning_rate,
            "kl_beta": KL_BETA,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "attention_implementation": settings.attention_implementation,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "vision_forward_calls": runtime.visual_forward_calls,
            "wall_time_seconds": time.monotonic() - wall_started,
            "peak_cuda_bytes": runtime.peak_cuda_bytes,
            "discarded_uncheckpointed_history_records": discarded_history,
            "final_checkpoint": final_checkpoint.relative_to(output_dir).as_posix(),
            "stopped_early": state["stopped_early"],
            "stop_reason": state["stop_reason"],
            "first_gradient_norms": state["first_gradient_norms"],
        },
        "diagnostics": {
            "final_step": history[-1],
            "adapter_switches": dict(sorted(runtime.adapter_switches.items())),
        },
        "parameter_verification": {
            "initial_policy_sha256": runtime.initial_policy_sha256,
            "final_policy_sha256": final_policy_sha256,
            "initial_ref_sha256": runtime.initial_ref_sha256,
            "final_ref_sha256": final_ref_sha256,
            "initial_projector_sha256": runtime.initial_projector_sha256,
            "final_projector_sha256": final_projector_sha256,
            "adapter_saved_and_reloaded": adapter_reloaded,
        },
        "contracts": {
            "input_modality": "xic_only",
            "base_weights_frozen": True,
            "reference_adapter_frozen": True,
            "policy_adapter_trainable": True,
            "projector_trainable": settings.train_projector,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "image_pixels_forwarded": False,
            "visual_tower_unused": runtime.visual_forward_calls == 0,
            "answer_tokens_only_log_prob": True,
            "token_level_loss_normalization": True,
            "single_update_per_on_policy_rollout": True,
            "train_split_only": True,
            "selection_groups_trained": False,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "parameter_updates_verified": True,
        },
        "development_training_complete": development_complete,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(report_path, report)
    manifest_path = _write_manifest(output_dir)
    _require(manifest_path.is_file(), "GRPO manifest was not written")
    return GRPOTrainingResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        adapter_dir=adapter_dir,
        projector_path=projector_path,
        global_step=int(state["global_step"]),
        optimizer_updates=int(state["optimizer_updates"]),
        development_training_complete=development_complete,
        stopped_early=bool(state["stopped_early"]),
    )
