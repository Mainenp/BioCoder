"""Pretrain the XIC morphology encoder on verified auxiliary unlabeled traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.auxiliary_pretraining import (
    AuxiliaryPretrainingSettings,
    run_auxiliary_pretraining,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--dataset-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--max-steps", type=int)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--temperature", type=float, default=0.1)
    command.add_argument("--projection-size", type=int, default=256)
    command.add_argument("--noise-std", type=float, default=0.02)
    command.add_argument("--mask-fraction", type=float, default=0.10)
    command.add_argument("--max-shift-points", type=int, default=4)
    command.add_argument("--log-steps", type=int, default=25)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--hidden-size", type=int, default=2560)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = run_auxiliary_pretraining(
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        dataset_manifest_sha256=arguments.dataset_manifest_sha256,
        output_dir=arguments.output_dir,
        code_revision=arguments.code_revision,
        settings=AuxiliaryPretrainingSettings(
            epochs=arguments.epochs,
            max_steps=arguments.max_steps,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            temperature=arguments.temperature,
            projection_size=arguments.projection_size,
            noise_std=arguments.noise_std,
            mask_fraction=arguments.mask_fraction,
            max_shift_points=arguments.max_shift_points,
            log_steps=arguments.log_steps,
            seed=arguments.seed,
            hidden_size=arguments.hidden_size,
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "projector_path": str(result.projector_path),
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
