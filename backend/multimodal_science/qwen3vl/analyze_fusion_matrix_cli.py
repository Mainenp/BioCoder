"""CLI for the formal five-cell Qwen3-VL image+XIC validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_matrix_analysis import (
    MATRIX_CONFIGURATIONS,
    FusionMatrixRun,
    analyze_fusion_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in MATRIX_CONFIGURATIONS:
        option = label.replace("-", "_")
        for artifact in ("training", "generation", "evaluation"):
            parser.add_argument(
                f"--{label}-{artifact}-root",
                dest=f"{option}_{artifact}_root",
                type=Path,
                required=True,
            )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    runs = {}
    for label in MATRIX_CONFIGURATIONS:
        option = label.replace("-", "_")
        runs[label] = FusionMatrixRun(
            training_root=getattr(arguments, f"{option}_training_root"),
            generation_root=getattr(arguments, f"{option}_generation_root"),
            evaluation_root=getattr(arguments, f"{option}_evaluation_root"),
        )
    result = analyze_fusion_matrix(runs=runs, output_dir=arguments.output_dir)
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
