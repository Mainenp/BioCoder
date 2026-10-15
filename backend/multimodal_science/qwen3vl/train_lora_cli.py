"""Train a resumable Qwen3-VL BF16 LoRA adapter on a sealed train bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.lora_training import (
    LoraTrainingSettings,
    run_lora_training,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--bundle-root", type=Path, required=True)
    command.add_argument("--bundle-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--max-steps", type=int)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--gradient-accumulation-steps", type=int, default=16)
    command.add_argument("--learning-rate", type=float, default=1e-6)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--warmup-ratio", type=float, default=0.03)
    command.add_argument("--max-grad-norm", type=float, default=1.0)
    command.add_argument("--lora-rank", type=int, default=8)
    command.add_argument("--lora-alpha", type=int, default=16)
    command.add_argument("--lora-dropout", type=float, default=0.0)
    command.add_argument("--max-length", type=int, default=1024)
    command.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    command.add_argument("--max-pixels", type=int, default=160 * 28 * 28)
    command.add_argument("--save-steps", type=int, default=50)
    command.add_argument("--log-steps", type=int, default=1)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    command.add_argument("--no-gradient-checkpointing", action="store_true")
    command.add_argument("--strict-deterministic", action="store_true")
    command.add_argument("--resume", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    settings = LoraTrainingSettings(
        epochs=arguments.epochs,
        max_steps=arguments.max_steps,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        warmup_ratio=arguments.warmup_ratio,
        max_grad_norm=arguments.max_grad_norm,
        lora_rank=arguments.lora_rank,
        lora_alpha=arguments.lora_alpha,
        lora_dropout=arguments.lora_dropout,
        max_length=arguments.max_length,
        min_pixels=arguments.min_pixels,
        max_pixels=arguments.max_pixels,
        save_steps=arguments.save_steps,
        log_steps=arguments.log_steps,
        seed=arguments.seed,
        attention_implementation=arguments.attention_implementation,
        gradient_checkpointing=not arguments.no_gradient_checkpointing,
        deterministic_warn_only=not arguments.strict_deterministic,
    )
    result = run_lora_training(
        bundle_root=arguments.bundle_root,
        bundle_report_sha256=arguments.bundle_report_sha256,
        assets_root=arguments.assets_root,
        output_dir=arguments.output_dir,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        code_revision=arguments.code_revision,
        settings=settings,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "adapter_dir": str(result.adapter_dir),
                "global_steps": result.global_steps,
                "development_training_complete": result.development_training_complete,
                "development_comparison_eligible": False,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
