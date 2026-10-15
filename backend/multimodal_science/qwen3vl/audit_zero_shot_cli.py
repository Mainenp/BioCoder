"""Audit zero-shot shortcuts and grounding coordinate interpretations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.zero_shot_audit import audit_zero_shot_failures


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--generation-report", type=Path, required=True)
    command.add_argument("--generation-report-sha256", required=True)
    command.add_argument("--evaluation-report", type=Path, required=True)
    command.add_argument("--evaluation-report-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--bootstrap-iterations", type=int, default=1000)
    command.add_argument("--seed", type=int, default=17)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = audit_zero_shot_failures(
        arguments.generation_report,
        arguments.evaluation_report,
        arguments.output_dir,
        expected_generation_report_sha256=arguments.generation_report_sha256,
        expected_evaluation_report_sha256=arguments.evaluation_report_sha256,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "coordinate_records": result.coordinate_records,
                "source_pixel_invalid_records": result.source_pixel_invalid_records,
                "internal_test_accessed": result.internal_test_accessed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
