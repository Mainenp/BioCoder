"""CLI for the ChromPeakFormer sequence baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_science.baselines.sequence_training import (
    SequenceTrainConfig,
    run_sequence_training,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--modality",
        choices=("sequence", "sequence_metadata"),
        default="sequence",
    )
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--epochs", type=int, default=40)
    command.add_argument("--batch-size", type=int, default=256)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--boundary-weight", type=float, default=2.0)
    command.add_argument("--dropout", type=float, default=0.15)
    command.add_argument("--patience", type=int, default=8)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast on CUDA; ignored on CPU.",
    )
    command.add_argument("--num-workers", type=int, default=0)
    command.add_argument("--bootstrap-iterations", type=int, default=1000)
    command.add_argument("--smoke-test", action="store_true")
    command.add_argument("--max-train-samples", type=int)
    command.add_argument("--max-validation-samples", type=int)
    return command


def main() -> None:
    arguments = parser().parse_args()
    config = SequenceTrainConfig(
        modality=arguments.modality,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        boundary_weight=arguments.boundary_weight,
        dropout=arguments.dropout,
        patience=arguments.patience,
        device=arguments.device,
        amp=arguments.amp,
        num_workers=arguments.num_workers,
        bootstrap_iterations=arguments.bootstrap_iterations,
        smoke_test=arguments.smoke_test,
        max_train_samples=arguments.max_train_samples,
        max_validation_samples=arguments.max_validation_samples,
    )

    def progress(record: dict[str, object]) -> None:
        train = record["train"]
        validation = record["validation"]
        print(
            "[epoch {epoch}] train={train_loss:.6f} validation={validation_loss:.6f} "
            "macro_f1@0.5={macro_f1:.6f}".format(
                epoch=record["epoch"],
                train_loss=train["loss"],
                validation_loss=validation["loss"],
                macro_f1=record["validation_macro_f1_at_0_5"],
            ),
            file=sys.stderr,
            flush=True,
        )

    result = run_sequence_training(
        arguments.dataset_root,
        arguments.output_dir,
        config,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "checkpoint_path": str(result.checkpoint_path),
                "best_epoch": result.best_epoch,
                "selected_threshold": result.selected_threshold,
                "development_comparison_eligible": (
                    result.development_comparison_eligible
                ),
                "final_benchmark_eligible": result.final_benchmark_eligible,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
