"""Evaluate complete Qwen3-VL validation predictions against the separated answer key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.evaluation import evaluate_qwen_predictions


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--instruction-root", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--instruction-report-sha256", required=True)
    command.add_argument("--bootstrap-iterations", type=int, default=1000)
    command.add_argument("--seed", type=int, default=17)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = evaluate_qwen_predictions(
        arguments.instruction_root,
        arguments.predictions,
        arguments.output_dir,
        expected_instruction_report_sha256=arguments.instruction_report_sha256,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "prediction_records": result.prediction_records,
                "valid_json_records": result.valid_json_records,
                "schema_valid_records": result.schema_valid_records,
                "validation_source_groups": result.validation_source_groups,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
