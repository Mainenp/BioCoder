"""Train a preregistered Qwen3-VL XIC-only GRPO candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.rl_grpo_training import (
    EXPERIMENTS,
    RUN_MODES,
    GRPOSettings,
    run_grpo_training,
)
from multimodal_science.qwen3vl.rl_rollout import verify_train_signals


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--rl-pool-root", type=Path, required=True)
    command.add_argument("--rl-pool-report-sha256", required=True)
    command.add_argument("--difficulty-root", type=Path, required=True)
    command.add_argument("--difficulty-report-sha256", required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--starting-adapter-root", type=Path, required=True)
    command.add_argument("--starting-adapter-report-sha256", required=True)
    command.add_argument("--starting-adapter-manifest-sha256", required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--model-manifest-path", type=Path, required=True)
    command.add_argument("--model-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    command.add_argument("--run-mode", choices=RUN_MODES, required=True)
    command.add_argument("--seed", type=int, choices=(17, 29), default=17)
    command.add_argument(
        "--attention-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    command.add_argument("--strict-deterministic", action="store_true")
    command.add_argument("--no-gradient-checkpointing", action="store_true")
    command.add_argument("--resume", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    train_signals = verify_train_signals(
        arguments.dataset_root, arguments.dataset_report_sha256
    )
    result = run_grpo_training(
        code_revision=arguments.code_revision,
        rl_pool_root=arguments.rl_pool_root,
        rl_pool_report_sha256=arguments.rl_pool_report_sha256,
        difficulty_root=arguments.difficulty_root,
        difficulty_report_sha256=arguments.difficulty_report_sha256,
        train_signals=train_signals,
        starting_adapter=AdapterSpec(
            root=arguments.starting_adapter_root,
            training_report_sha256=arguments.starting_adapter_report_sha256,
            manifest_sha256=arguments.starting_adapter_manifest_sha256,
        ),
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        model_manifest_path=arguments.model_manifest_path,
        model_manifest_sha256=arguments.model_manifest_sha256,
        output_dir=arguments.output_dir,
        settings=GRPOSettings(
            experiment=arguments.experiment,
            run_mode=arguments.run_mode,
            seed=arguments.seed,
            attention_implementation=arguments.attention_implementation,
            gradient_checkpointing=not arguments.no_gradient_checkpointing,
            deterministic_warn_only=not arguments.strict_deterministic,
        ),
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "adapter_dir": str(result.adapter_dir),
                "projector_path": str(result.projector_path),
                "global_step": result.global_step,
                "optimizer_updates": result.optimizer_updates,
                "development_training_complete": result.development_training_complete,
                "stopped_early": result.stopped_early,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
