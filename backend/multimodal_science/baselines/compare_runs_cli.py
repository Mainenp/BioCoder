"""Create a validated ChromPeakFormer development-ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.baselines.comparison import build_development_comparison


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--detector-evaluation", type=Path, required=True)
    command.add_argument("--detector-dataset-report", type=Path, required=True)
    command.add_argument("--sequence-report", type=Path, required=True)
    command.add_argument("--sequence-verification", type=Path, required=True)
    command.add_argument("--sequence-metadata-report", type=Path, required=True)
    command.add_argument("--sequence-metadata-verification", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_development_comparison(
        detector_evaluation_path=arguments.detector_evaluation,
        detector_dataset_report_path=arguments.detector_dataset_report,
        sequence_report_path=arguments.sequence_report,
        sequence_verification_path=arguments.sequence_verification,
        sequence_metadata_report_path=arguments.sequence_metadata_report,
        sequence_metadata_verification_path=arguments.sequence_metadata_verification,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "markdown_path": str(result.markdown_path),
                "manifest_path": str(result.manifest_path),
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
