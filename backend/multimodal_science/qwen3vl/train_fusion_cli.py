"""Train resumable Qwen3-VL LoRA plus an XIC sensor projector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_training import (
    FusionTrainingSettings,
    run_fusion_training,
)
from multimodal_science.qwen3vl.inference import AdapterSpec


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--fusion-bundle-root", type=Path, required=True)
    command.add_argument("--fusion-bundle-report-sha256", required=True)
    command.add_argument("--lora-bundle-root", type=Path, required=True)
    command.add_argument("--lora-bundle-report-sha256", required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--initial-adapter-root", type=Path, required=True)
    command.add_argument("--initial-adapter-report-sha256", required=True)
    command.add_argument("--initial-adapter-manifest-sha256", required=True)
    command.add_argument("--pretrained-projector-root", type=Path)
    command.add_argument("--pretrained-projector-report-sha256")
    command.add_argument("--pretrained-projector-manifest-sha256")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--model-manifest-path", type=Path, required=True)
    command.add_argument("--model-manifest-sha256", required=True)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--max-steps", type=int)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--gradient-accumulation-steps", type=int, default=16)
    command.add_argument("--lora-learning-rate", type=float, default=2e-7)
    command.add_argument("--projector-learning-rate", type=float, default=1e-4)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--warmup-ratio", type=float, default=0.03)
    command.add_argument("--max-grad-norm", type=float, default=1.0)
    command.add_argument("--max-length", type=int, default=1024)
    command.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    command.add_argument("--max-pixels", type=int, default=160 * 28 * 28)
    command.add_argument("--save-steps", type=int, default=250)
    command.add_argument("--log-steps", type=int, default=10)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--sensor-tokens", type=int, choices=(1, 4, 8), default=4)
    command.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    command.add_argument("--no-gradient-checkpointing", action="store_true")
    command.add_argument("--strict-deterministic", action="store_true")
    command.add_argument("--resume", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    settings = FusionTrainingSettings(
        epochs=arguments.epochs,
        max_steps=arguments.max_steps,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        lora_learning_rate=arguments.lora_learning_rate,
        projector_learning_rate=arguments.projector_learning_rate,
        weight_decay=arguments.weight_decay,
        warmup_ratio=arguments.warmup_ratio,
        max_grad_norm=arguments.max_grad_norm,
        max_length=arguments.max_length,
        min_pixels=arguments.min_pixels,
        max_pixels=arguments.max_pixels,
        save_steps=arguments.save_steps,
        log_steps=arguments.log_steps,
        seed=arguments.seed,
        attention_implementation=arguments.attention_implementation,
        gradient_checkpointing=not arguments.no_gradient_checkpointing,
        deterministic_warn_only=not arguments.strict_deterministic,
        sensor_tokens=arguments.sensor_tokens,
    )
    result = run_fusion_training(
        fusion_bundle_root=arguments.fusion_bundle_root,
        fusion_bundle_report_sha256=arguments.fusion_bundle_report_sha256,
        lora_bundle_root=arguments.lora_bundle_root,
        lora_bundle_report_sha256=arguments.lora_bundle_report_sha256,
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        assets_root=arguments.assets_root,
        initial_adapter=AdapterSpec(
            root=arguments.initial_adapter_root,
            training_report_sha256=arguments.initial_adapter_report_sha256,
            manifest_sha256=arguments.initial_adapter_manifest_sha256,
        ),
        output_dir=arguments.output_dir,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        model_manifest_path=arguments.model_manifest_path,
        model_manifest_sha256=arguments.model_manifest_sha256,
        code_revision=arguments.code_revision,
        settings=settings,
        resume=arguments.resume,
        pretrained_projector_root=arguments.pretrained_projector_root,
        pretrained_projector_report_sha256=(
            arguments.pretrained_projector_report_sha256
        ),
        pretrained_projector_manifest_sha256=(
            arguments.pretrained_projector_manifest_sha256
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "adapter_dir": str(result.adapter_dir),
                "projector_path": str(result.projector_path),
                "global_steps": result.global_steps,
                "development_training_complete": (
                    result.development_training_complete
                ),
                "development_comparison_eligible": False,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
