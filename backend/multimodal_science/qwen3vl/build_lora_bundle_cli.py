"""Build a train-only Qwen3-VL LoRA bundle from verified instructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.lora_data import build_lora_training_bundle


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--instruction-root", type=Path, required=True)
    command.add_argument("--instruction-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--max-records", type=int)
    command.add_argument("--seed", type=int, default=17)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_lora_training_bundle(
        instruction_root=arguments.instruction_root,
        instruction_report_sha256=arguments.instruction_report_sha256,
        assets_root=arguments.assets_root,
        output_dir=arguments.output_dir,
        max_records=arguments.max_records,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "training_path": str(result.training_path),
                "training_records": result.training_records,
                "train_split_only": True,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
