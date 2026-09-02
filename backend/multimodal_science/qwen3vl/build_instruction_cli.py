"""Build Qwen3-VL train instructions and answer-separated validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.instruction_data import TASKS, build_instruction_dataset


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Repeat to select tasks; defaults to all declared tasks.",
    )
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_instruction_dataset(
        arguments.dataset_root,
        arguments.output_dir,
        tasks=arguments.task or TASKS,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "source_dataset_report_sha256": result.source_dataset_report_sha256,
                "source_assets": result.source_assets,
                "train_instructions": result.train_instructions,
                "validation_instructions": result.validation_instructions,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
