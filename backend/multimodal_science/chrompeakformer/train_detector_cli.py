"""Train ChromPeakFormer through the audited external-source adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.detector_training import run_detector_training


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--source-config", type=Path, required=True)
    command.add_argument("--detector-dataset-root", type=Path, required=True)
    command.add_argument("--detector-dataset-report", type=Path, required=True)
    command.add_argument("--detector-dataset-report-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    command.add_argument("--num-workers", type=int, default=2)
    command.add_argument("--epochs", type=int)
    command.add_argument("--batch-size", type=int)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument(
        "--smoke-test",
        action="store_true",
        help="Mark this run as contract-only and ineligible for development comparisons.",
    )
    command.add_argument("--resume", type=Path)
    command.add_argument("--resume-sha256")
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = run_detector_training(
        source_root=arguments.source_root,
        source_config_path=arguments.source_config,
        detector_dataset_root=arguments.detector_dataset_root,
        detector_dataset_report_path=arguments.detector_dataset_report,
        expected_dataset_report_sha256=arguments.detector_dataset_report_sha256,
        output_dir=arguments.output_dir,
        device=arguments.device,
        num_workers=arguments.num_workers,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        smoke_test=arguments.smoke_test,
        resume_path=arguments.resume,
        expected_resume_sha256=arguments.resume_sha256,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "checkpoint_path": str(result.checkpoint_path),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "final_epoch": result.final_epoch,
                "best_observed_epoch": result.best_observed_epoch,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
