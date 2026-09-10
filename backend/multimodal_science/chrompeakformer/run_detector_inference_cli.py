"""Run ChromPeakFormer validation inference and emit standard COCO predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.detector_inference import run_detector_inference


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--checkpoint-sha256", required=True)
    command.add_argument("--training-report", type=Path, required=True)
    command.add_argument("--training-report-sha256", required=True)
    command.add_argument("--detector-dataset-root", type=Path, required=True)
    command.add_argument("--detector-dataset-report", type=Path, required=True)
    command.add_argument("--detector-dataset-report-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--num-workers", type=int, default=2)
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = run_detector_inference(
        source_root=arguments.source_root,
        checkpoint_path=arguments.checkpoint,
        expected_checkpoint_sha256=arguments.checkpoint_sha256,
        training_report_path=arguments.training_report,
        expected_training_report_sha256=arguments.training_report_sha256,
        detector_dataset_root=arguments.detector_dataset_root,
        detector_dataset_report_path=arguments.detector_dataset_report,
        expected_dataset_report_sha256=arguments.detector_dataset_report_sha256,
        output_dir=arguments.output_dir,
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        amp=arguments.amp,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "predictions_path": str(result.predictions_path),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "prediction_count": result.prediction_count,
                "image_count": result.image_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
