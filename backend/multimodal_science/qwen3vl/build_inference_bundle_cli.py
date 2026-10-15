"""Build a hash-bound prompt-only Qwen3-VL inference bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference_bundle import build_inference_bundle


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--instruction-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--instruction-report-sha256", required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_inference_bundle(
        arguments.instruction_root,
        arguments.output_dir,
        expected_instruction_report_sha256=arguments.instruction_report_sha256,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "prompt_records": result.prompt_records,
                "source_instruction_report_sha256": (
                    result.source_instruction_report_sha256
                ),
                "answer_key_materialized": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
