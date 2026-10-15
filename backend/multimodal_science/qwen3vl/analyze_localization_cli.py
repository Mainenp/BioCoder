"""Build the development-only v2 localization analysis artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.localization_analysis import analyze_localization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-evaluation-root", type=Path, required=True)
    parser.add_argument("--c1-evaluation-root", type=Path, required=True)
    parser.add_argument("--rl-evaluation-root", type=Path, required=True)
    parser.add_argument("--validation-metadata-path", type=Path, required=True)
    parser.add_argument("--expected-validation-metadata-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--failure-seed", type=int, default=17)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = analyze_localization(
        c0_evaluation_root=arguments.c0_evaluation_root,
        c1_evaluation_root=arguments.c1_evaluation_root,
        rl_evaluation_root=arguments.rl_evaluation_root,
        validation_metadata_path=arguments.validation_metadata_path,
        expected_validation_metadata_sha256=arguments.expected_validation_metadata_sha256,
        output_dir=arguments.output_dir,
        bootstrap_iterations=arguments.bootstrap_iterations,
        bootstrap_seed=arguments.bootstrap_seed,
        failure_seed=arguments.failure_seed,
    )
    print(json.dumps({"output_dir": str(result.output_dir), "report_path": str(result.report_path), "report_sha256": result.report_sha256, "markdown_path": str(result.markdown_path), "failure_cases_path": str(result.failure_cases_path), "manifest_path": str(result.manifest_path), "development_comparison_eligible": True, "final_benchmark_eligible": False, "internal_test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
