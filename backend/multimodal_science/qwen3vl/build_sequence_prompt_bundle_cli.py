"""Build a target-free frozen SequencePeakNet prompt bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.sequence_prompt_data import (
    build_sequence_prompt_bundle,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--sequence-run-root", type=Path, required=True)
    command.add_argument("--sequence-report-sha256", required=True)
    command.add_argument("--sequence-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--code-revision", required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_sequence_prompt_bundle(
        sequence_run_root=arguments.sequence_run_root,
        sequence_report_sha256=arguments.sequence_report_sha256,
        sequence_manifest_sha256=arguments.sequence_manifest_sha256,
        output_dir=arguments.output_dir,
        code_revision=arguments.code_revision,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "predictions_path": str(result.predictions_path),
                "prediction_records": result.prediction_records,
                "target_fields_excluded": True,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
