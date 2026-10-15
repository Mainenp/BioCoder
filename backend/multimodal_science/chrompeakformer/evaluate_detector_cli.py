"""Evaluate ChromPeakFormer COCO predictions on the audited validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.detector_evaluation import (
    evaluate_detector_predictions,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--validation-coco", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--inference-report", type=Path, required=True)
    command.add_argument("--inference-report-sha256", required=True)
    command.add_argument("--detector-dataset-report", type=Path, required=True)
    command.add_argument("--detector-dataset-report-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--score-threshold", type=float, default=0.5)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = evaluate_detector_predictions(
        validation_coco_path=arguments.validation_coco,
        predictions_path=arguments.predictions,
        inference_report_path=arguments.inference_report,
        expected_inference_report_sha256=arguments.inference_report_sha256,
        detector_dataset_report_path=arguments.detector_dataset_report,
        expected_dataset_report_sha256=arguments.detector_dataset_report_sha256,
        output_dir=arguments.output_dir,
        fixed_score_threshold=arguments.score_threshold,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "validation_images": result.validation_images,
                "predictions": result.predictions,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
