"""Compare provenance-verified Qwen and specialist development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.development_comparison import (
    build_cross_family_development_comparison,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--specialist-comparison", type=Path, required=True)
    command.add_argument("--zero-shot-generation", type=Path, required=True)
    command.add_argument("--zero-shot-evaluation", type=Path, required=True)
    command.add_argument("--lora-generation", type=Path, required=True)
    command.add_argument("--lora-evaluation", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_cross_family_development_comparison(
        specialist_comparison_path=arguments.specialist_comparison,
        zero_shot_generation_path=arguments.zero_shot_generation,
        zero_shot_evaluation_path=arguments.zero_shot_evaluation,
        lora_generation_path=arguments.lora_generation,
        lora_evaluation_path=arguments.lora_evaluation,
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
