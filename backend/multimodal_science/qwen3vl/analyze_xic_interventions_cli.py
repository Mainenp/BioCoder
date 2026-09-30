"""CLI for paired XIC intervention analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_inference import XIC_INTERVENTIONS
from multimodal_science.qwen3vl.xic_intervention_analysis import (
    XicInterventionRun,
    analyze_xic_interventions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in XIC_INTERVENTIONS:
        option = label.replace("-", "_")
        parser.add_argument(
            f"--{label}-generation-root",
            dest=f"{option}_generation_root",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{label}-evaluation-root",
            dest=f"{option}_evaluation_root",
            type=Path,
            required=True,
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    runs = {}
    for label in XIC_INTERVENTIONS:
        option = label.replace("-", "_")
        runs[label] = XicInterventionRun(
            generation_root=getattr(arguments, f"{option}_generation_root"),
            evaluation_root=getattr(arguments, f"{option}_evaluation_root"),
        )
    result = analyze_xic_interventions(
        runs=runs,
        output_dir=arguments.output_dir,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.seed,
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
