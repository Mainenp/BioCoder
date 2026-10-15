"""Verify a ChromPeakFormer sequence run from its immutable saved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.baselines.run_validation import validate_sequence_run


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument(
        "--output",
        type=Path,
        help="Defaults to <run-dir>/verification_report.json; refuses overwrite.",
    )
    return command


def main() -> None:
    arguments = parser().parse_args()
    output = arguments.output or arguments.run_dir / "verification_report.json"
    result = validate_sequence_run(
        arguments.run_dir,
        arguments.dataset_root,
        output_path=output,
    )
    print(
        json.dumps(
            {
                "quality_gate_passed": True,
                "source_report_path": str(result.source_report_path),
                "source_report_sha256": result.source_report_sha256,
                "dataset_report_sha256": result.dataset_report_sha256,
                "verification_path": str(result.verification_path),
                "verification_sha256": result.verification_sha256,
                "prediction_records": result.prediction_records,
                "validation_source_groups": result.validation_source_groups,
                "selected_threshold": result.selected_threshold,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
