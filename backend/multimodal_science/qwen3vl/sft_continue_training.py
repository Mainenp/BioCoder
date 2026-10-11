"""Preregistered C1 continue-SFT control for the Qwen3-VL XIC-only model.

This module deliberately reuses the v2 GRPO runtime for the fragile pieces of
the XIC-only path (adapter verification, sensor insertion, mRoPE, checkpoint
state, and the visual-tower guard).  C1 differs only in the objective: it
teacher-forces the standard answer rather than sampling a rollout reward.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_smoke import GPU_ALLOCATION_MODE, _parameter_digest
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.lora_training import assistant_supervision_labels
from multimodal_science.qwen3vl.rl_data import (
    assert_no_protected_input_paths,
    assert_training_pool_isolation,
    load_rl_training_pool,
)
from multimodal_science.qwen3vl.rl_grpo_training import (
    CHECKPOINT_INTERVAL,
    FORMAL_PROMPTS_PER_STEP,
    FORMAL_STEPS,
    MAX_GRAD_NORM,
    REDUCED_PROMPTS_PER_STEP,
    REDUCED_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    GRPOSettings,
    _GRPORuntime,
    _latest_checkpoint,
    _read_json,
    _reconcile_history,
    _require,
    _write_manifest,
    _write_json,
    configure_trainable_parameters,
    load_difficulty_training_pool,
    prompt_batch,
    starting_processor_pixel_bounds,
)
from multimodal_science.qwen3vl.rl_rollout import VerifiedTrainSignals


C1_CONFIG_SCHEMA = "chrompeak-qwen3vl-xic-c1-continue-sft-config-v1"
C1_REPORT_SCHEMA = "chrompeak-qwen3vl-xic-c1-continue-sft-training-v1"
C1_RUN_MODES = ("smoke", "calibration", "formal", "reduced-prompts", "reduced-steps")
C1_LEARNING_RATE_CHOICES = ("original", "1e-6")
_HEX_40 = "0123456789abcdef"


@dataclass(frozen=True)
class C1Settings:
    """Frozen C1 schedule, aligned one-for-one with the GRPO reduction plan."""

    run_mode: str = "formal"
    learning_rate_choice: str = "original"
    seed: int = 17
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    deterministic_warn_only: bool = True

    @property
    def prompts_per_step(self) -> int:
        return (
            REDUCED_PROMPTS_PER_STEP
            if self.run_mode in {"reduced-prompts", "reduced-steps"}
            else FORMAL_PROMPTS_PER_STEP
        )

    @property
    def total_steps(self) -> int:
        if self.run_mode in {"smoke", "calibration"}:
            return 2
        return REDUCED_STEPS if self.run_mode == "reduced-steps" else FORMAL_STEPS

    @property
    def formal_budget_complete(self) -> bool:
        return self.run_mode in {"formal", "reduced-prompts", "reduced-steps"}


@dataclass(frozen=True)
class C1TrainingResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    adapter_dir: Path
    projector_path: Path
    global_step: int
    optimizer_updates: int
    development_training_complete: bool


def validate_settings(settings: C1Settings) -> None:
    _require(settings.run_mode in C1_RUN_MODES, "Unsupported C1 run mode")
    _require(
        settings.learning_rate_choice in C1_LEARNING_RATE_CHOICES,
        "C1 learning rate must be original or 1e-6",
    )
    _require(settings.seed in {17, 29}, "The preregistered C1 seeds are 17 and 29")
    _require(
        settings.attention_implementation in {"sdpa", "eager"},
        "Unsupported attention implementation",
    )
    _require(
        FORMAL_PROMPTS_PER_STEP == 16 and FORMAL_STEPS == 300,
        "Formal C1 budget drifted",
    )
    _require(
        REDUCED_PROMPTS_PER_STEP == 8 and REDUCED_STEPS == 200,
        "C1 reduction plan drifted",
    )
    _require(WARMUP_STEPS == 10 and CHECKPOINT_INTERVAL == 100, "C1 schedule drifted")


def resolve_lora_learning_rate(choice: str, original_learning_rate: Any) -> float:
    """Resolve one of the two preregistered C1 rates without an arbitrary knob."""

    _require(choice in C1_LEARNING_RATE_CHOICES, "Unknown C1 learning-rate choice")
    if choice == "1e-6":
        return 1e-6
    _require(
        isinstance(original_learning_rate, (int, float))
        and not isinstance(original_learning_rate, bool)
        and math.isfinite(float(original_learning_rate))
        and float(original_learning_rate) > 0.0,
        "Starting C0 report has no valid LoRA learning rate",
    )
    return float(original_learning_rate)


def _starting_lora_learning_rate(starting_adapter: AdapterSpec) -> float:
    """Read only the hash-bound C0 report field required by the preregistration."""

    root = starting_adapter.root.resolve()
    assert_no_protected_input_paths([root])
    report_path = root / "fusion_training_report.json"
    _require(report_path.is_file(), f"Missing starting C0 report: {report_path}")
    _require(
        sha256_file(report_path) == starting_adapter.training_report_sha256,
        "Starting C0 report hash mismatch",
    )
    report = _read_json(report_path, "starting C0 report")
    training = report.get("training")
    _require(isinstance(training, dict), "Starting C0 report lacks training settings")
    return resolve_lora_learning_rate("original", training.get("lora_learning_rate"))


def configure_c1_trainable_parameters(
    model_named_parameters: Sequence[tuple[str, Any]],
    projector_named_parameters: Sequence[tuple[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Expose only the policy LoRA adapter; base, ref, and projector stay frozen."""

    groups = configure_trainable_parameters(
        model_named_parameters, projector_named_parameters, experiment="R1"
    )
    _require(not any(parameter.requires_grad for _, parameter in projector_named_parameters), "C1 projector is trainable")
    _require(bool(groups["policy"]), "C1 has no policy LoRA parameters")
    return groups


def _standard_answer_loss(runtime: _GRPORuntime, record: dict[str, Any]) -> Any:
    """Teacher-force exactly the difficulty-pool standard response on one prompt."""

    response = record.get("expected_response")
    _require(isinstance(response, str) and bool(response), "C1 record lacks standard response")
    runtime.activate_adapter("policy", trainable=True)
    runtime.model.train()
    runtime.projector.eval()
    prompt_ids, signal, availability = runtime._prompt_and_signal(record)
    prompt_length = int(prompt_ids.shape[1])
    prompt = record.get("prompt")
    language = record.get("language")
    _require(isinstance(prompt, str) and bool(prompt), "C1 record lacks prompt")
    _require(language in {"en", "zh-CN"}, "C1 record has invalid language")
    # Full tokenization uses the same canonical XIC-only prompt helper as the
    # rollout path; no image content or alternative chat template is introduced.
    from multimodal_science.qwen3vl.fusion_training import xic_only_prompt_text

    full_messages = [[
        {"role": "user", "content": [{"type": "text", "text": xic_only_prompt_text(prompt, str(language))}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]]
    full = runtime.processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = full["input_ids"].to(runtime.device)
    _require(
        input_ids.shape[1] > prompt_length,
        "C1 standard response produced no supervised tokens",
    )
    labels = runtime.torch.tensor(
        assistant_supervision_labels(
            input_ids[0].tolist(), prompt_ids[0].tolist()
        ),
        dtype=runtime.torch.long,
        device=runtime.device,
    ).unsqueeze(0)
    with runtime.torch.autocast(device_type="cuda", dtype=runtime.torch.bfloat16):
        inputs_embeds, fused_labels, attention_mask, position_ids = runtime._fused_inputs(
            input_ids=input_ids,
            labels=labels,
            prompt_lengths=[prompt_length],
            signals=signal,
            availability=availability,
            projector=runtime.projector,
        )
        outputs = runtime.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=fused_labels,
            use_cache=False,
            return_dict=True,
        )
    _require(runtime.visual_forward_calls == 0, "C1 XIC-only forward invoked visual tower")
    _require(outputs.loss is not None and runtime.torch.isfinite(outputs.loss), "C1 loss is invalid")
    return outputs.loss


def _state(total_steps: int) -> dict[str, Any]:
    return {
        "global_step": 0,
        "total_steps": total_steps,
        "optimizer_updates": 0,
        "zero_variance_patience": 0,
        "kl_patience": 0,
        "first_gradient_norms": None,
        "stopped_early": False,
        "stop_reason": None,
    }


def run_c1_continue_sft(
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
    settings: C1Settings,
    resume: bool = False,
) -> C1TrainingResult:
    """Train C1 on exactly the difficulty-selected train-only pool and budget."""

    validate_settings(settings)
    _require(len(code_revision) == 40 and all(char in _HEX_40 for char in code_revision), "Invalid code revision")
    for digest, label in (
        (rl_pool_report_sha256, "RL pool hash"),
        (difficulty_report_sha256, "difficulty hash"),
        (train_signals.report_sha256, "Dataset hash"),
        (model_artifact_sha256, "model hash"),
        (model_manifest_sha256, "model manifest hash"),
    ):
        _require(isinstance(digest, str) and len(digest) == 64 and all(char in _HEX_40 for char in digest), f"Invalid {label}")
    _require(os.environ.get("BIOCODER_VERIFIED_CODE_REVISION") == code_revision, "Code revision was not verified by the immutable launcher")
    _require(os.environ.get("BIOCODER_TRAIN_INPUT_SCOPE") == "staged_train_artifacts_only", "C1 requires staged train-only roots")
    _require(os.environ.get("BIOCODER_GPU_ALLOCATION_MODE") == GPU_ALLOCATION_MODE, "GPU allocation mode was not declared")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS_WORKSPACE_CONFIG must be set before Python starts")
    assert_no_protected_input_paths([rl_pool_root, difficulty_root, train_signals.path, starting_adapter.root, output_dir])

    pool = load_difficulty_training_pool(
        difficulty_root=difficulty_root,
        difficulty_report_sha256=difficulty_report_sha256,
        rl_pool_root=rl_pool_root,
        rl_pool_report_sha256=rl_pool_report_sha256,
    )
    assert_training_pool_isolation(pool.records, pool.selection_group_ids)
    base_pool = load_rl_training_pool(rl_pool_root, rl_pool_report_sha256)
    sources = base_pool.report.get("sources")
    _require(isinstance(sources, dict) and sources.get("dataset_report_sha256") == train_signals.report_sha256, "C1 pool and staged train tensors disagree")
    original_lr = _starting_lora_learning_rate(starting_adapter)
    lora_learning_rate = resolve_lora_learning_rate(settings.learning_rate_choice, original_lr)

    output_dir = output_dir.resolve()
    configuration = {
        "schema_version": C1_CONFIG_SCHEMA,
        "code_revision": code_revision,
        "algorithm": "teacher_forced_continue_sft",
        "settings": asdict(settings),
        "frozen_hyperparameters": {
            "prompts_per_step": settings.prompts_per_step,
            "supervised_examples_per_update": settings.prompts_per_step,
            "total_steps": settings.total_steps,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "lora_learning_rate": lora_learning_rate,
            "original_c0_lora_learning_rate": original_lr,
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
            "difficulty_training_pool_only": True,
            "selection_groups_trained": False,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
    }
    config_path = output_dir / "training_config.json"
    report_path = output_dir / "sft_continue_training_report.json"
    if output_dir.exists():
        _require(resume, f"Training output already exists: {output_dir}")
        _require(not report_path.exists(), "Completed C1 run cannot be resumed")
        _require(_read_json(config_path, "C1 config") == configuration, "Resume mismatch")
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
        settings=GRPOSettings(
            experiment="R1",
            run_mode=settings.run_mode,
            seed=settings.seed,
            attention_implementation=settings.attention_implementation,
            gradient_checkpointing=settings.gradient_checkpointing,
            deterministic_warn_only=settings.deterministic_warn_only,
        ),
    )
    torch = runtime.torch
    started_at = datetime.now(timezone.utc)
    wall_started = time.monotonic()
    try:
        from transformers import get_constant_schedule_with_warmup
    except ImportError as error:
        runtime.close()
        raise RuntimeError("C1 requires the Transformers scheduler") from error
    optimizer = torch.optim.AdamW(
        [{"params": runtime.policy_parameters, "lr": lora_learning_rate}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)
    state = _state(settings.total_steps)
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    if checkpoint is not None:
        state = runtime.restore_checkpoint(checkpoint, optimizer=optimizer, scheduler=scheduler)
        _require(state.get("total_steps") == settings.total_steps, "Resume step plan mismatch")
    discarded_history = _reconcile_history(history_path, int(state["global_step"])) if resume else 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for global_step in range(int(state["global_step"]) + 1, settings.total_steps + 1):
            started_step = time.monotonic()
            batch = prompt_batch(
                pool.records,
                step=global_step,
                prompts_per_step=settings.prompts_per_step,
                seed=settings.seed,
            )
            assert_training_pool_isolation(batch, pool.selection_group_ids)
            loss_sum = 0.0
            for record in batch:
                loss = _standard_answer_loss(runtime, record)
                (loss / len(batch)).backward()
                loss_sum += float(loss.detach().cpu())
            if state["first_gradient_norms"] is None:
                squared = sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in runtime.policy_parameters
                    if parameter.grad is not None
                )
                gradient = math.sqrt(squared)
                _require(math.isfinite(gradient) and gradient > 0.0, "C1 produced no LoRA gradient")
                state["first_gradient_norms"] = {"policy": gradient}
            grad_norm = torch.nn.utils.clip_grad_norm_(runtime.policy_parameters, MAX_GRAD_NORM)
            _require(bool(torch.isfinite(grad_norm)), "C1 gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] = global_step
            state["optimizer_updates"] = int(state["optimizer_updates"]) + 1
            _require(runtime.visual_forward_calls == 0, "C1 visual tower call detected")
            history = {
                "schema_version": "chrompeak-qwen3vl-xic-c1-step-v1",
                "global_step": global_step,
                "optimizer_updates": state["optimizer_updates"],
                "supervised_examples": len(batch),
                "loss": loss_sum / len(batch),
                "lora_learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(grad_norm.detach().cpu()),
                "step_wall_time_seconds": time.monotonic() - started_step,
            }
            with history_path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(history, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[c1 step {global_step}/{settings.total_steps}] loss={history['loss']:.6f}", flush=True)
            if global_step % CHECKPOINT_INTERVAL == 0 or global_step == settings.total_steps:
                runtime.checkpoint(output_dir=output_dir, optimizer=optimizer, scheduler=scheduler, state=state)
    finally:
        runtime.close()

    _require(int(state["global_step"]) == settings.total_steps, "C1 did not complete its fixed budget")
    history_records = []
    if history_path.is_file():
        history_records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
    _require(len(history_records) == settings.total_steps, "C1 history is incomplete")
    final_checkpoint = _latest_checkpoint(output_dir)
    _require(final_checkpoint is not None and int(final_checkpoint.name.removeprefix("step-")) == settings.total_steps, "C1 final checkpoint is missing")

    adapter_dir = output_dir / "adapter"
    projector_path = output_dir / "sensor_projector.safetensors"
    processor_dir = output_dir / "processor"
    _require(not any(path.exists() for path in (adapter_dir, projector_path, processor_dir)), "C1 final artifact already exists")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=output_dir, prefix=".final-") as staging_name:
        staging = Path(staging_name)
        staged_adapter = staging / "adapter"
        runtime_adapter_saved = runtime.verify_saved_policy  # make save/reload proof explicit below
        from multimodal_science.qwen3vl.rl_grpo_training import _save_policy_adapter

        _save_policy_adapter(runtime.model, staged_adapter)
        runtime_adapter_saved(staged_adapter)
        runtime.processor.save_pretrained(staging / "processor")
        runtime.save_file({key: value.detach().cpu() for key, value in runtime.projector.state_dict().items()}, str(staging / "sensor_projector.safetensors"))
        staged_adapter.replace(adapter_dir)
        (staging / "processor").replace(processor_dir)
        (staging / "sensor_projector.safetensors").replace(projector_path)
    final_policy_sha256 = _parameter_digest(runtime.policy_named)
    final_ref_sha256 = _parameter_digest(runtime.ref_named)
    final_projector_sha256 = _parameter_digest(runtime.projector_named)
    _require(final_policy_sha256 != runtime.initial_policy_sha256, "C1 policy adapter did not update")
    _require(final_ref_sha256 == runtime.initial_ref_sha256, "C1 reference adapter changed")
    _require(final_projector_sha256 == runtime.initial_projector_sha256, "C1 frozen projector changed")
    _require(runtime.visual_forward_calls == 0, "C1 visual tower was called")
    min_pixels, max_pixels = starting_processor_pixel_bounds(
        runtime.verified_start.report
    )
    report = {
        "schema_version": C1_REPORT_SCHEMA,
        "code_revision": code_revision,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "sources": configuration["sources"],
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "base_artifact_sha256": model_artifact_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "model_inventory": runtime.model_inventory,
            "sensor_projector": runtime.verified_start.projector_spec.as_dict(),
        },
        "training": {
            "algorithm": "teacher_forced_continue_sft",
            "run_mode": settings.run_mode,
            "seed": settings.seed,
            "learning_rate_choice": settings.learning_rate_choice,
            "lora_learning_rate": lora_learning_rate,
            "original_c0_lora_learning_rate": original_lr,
            "global_steps": state["global_step"],
            "optimizer_updates": state["optimizer_updates"],
            "training_records": int(state["global_step"]) * settings.prompts_per_step,
            "supervised_examples_per_update": settings.prompts_per_step,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_steps": WARMUP_STEPS,
            "attention_implementation": settings.attention_implementation,
            "gradient_checkpointing": settings.gradient_checkpointing,
            "deterministic_warn_only": settings.deterministic_warn_only,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "vision_forward_calls": runtime.visual_forward_calls,
            "wall_time_seconds": time.monotonic() - wall_started,
            "peak_cuda_bytes": runtime.peak_cuda_bytes,
            "discarded_uncheckpointed_history_records": discarded_history,
            "final_checkpoint": final_checkpoint.relative_to(output_dir).as_posix(),
            "first_gradient_norms": state["first_gradient_norms"],
        },
        "parameter_verification": {
            "initial_policy_sha256": runtime.initial_policy_sha256,
            "final_policy_sha256": final_policy_sha256,
            "initial_reference_sha256": runtime.initial_ref_sha256,
            "final_reference_sha256": final_ref_sha256,
            "initial_projector_sha256": runtime.initial_projector_sha256,
            "final_projector_sha256": final_projector_sha256,
            "adapter_saved_and_reloaded": True,
        },
        "contracts": {
            "input_modality": "xic_only",
            "base_weights_frozen": True,
            "policy_lora_only_trainable": True,
            "reference_adapter_frozen": True,
            "projector_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "image_pixels_forwarded": False,
            "visual_tower_unused": runtime.visual_forward_calls == 0,
            "xic_only_shadow_mrope_without_image_grid": True,
            "standard_expected_responses_only": True,
            "difficulty_training_pool_only": True,
            "selection_groups_trained": False,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
            "parameter_updates_verified": True,
        },
        "development_training_complete": settings.formal_budget_complete,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    _write_json(report_path, report)
    _write_manifest(output_dir)
    return C1TrainingResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        adapter_dir=adapter_dir,
        projector_path=projector_path,
        global_step=int(state["global_step"]),
        optimizer_updates=int(state["optimizer_updates"]),
        development_training_complete=settings.formal_budget_complete,
    )
