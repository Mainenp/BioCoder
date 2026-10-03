"""Freeze the final benchmark candidate and metric protocol before test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_protocol import (
    freeze_final_benchmark_protocol,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dossier-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--derivation-plan", type=Path, required=True)
    parser.add_argument("--derivation-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--instruction-root", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-internal-test-assets", type=int, default=1815)
    parser.add_argument("--expected-internal-test-source-groups", type=int, default=11)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = freeze_final_benchmark_protocol(
        development_dossier_root=arguments.development_dossier_root,
        split_manifest_path=arguments.split_manifest,
        split_report_path=arguments.split_report,
        derivation_plan_path=arguments.derivation_plan,
        derivation_report_path=arguments.derivation_report,
        dataset_root=arguments.dataset_root,
        instruction_root=arguments.instruction_root,
        base_model_manifest_path=arguments.base_model_manifest,
        output_dir=arguments.output_dir,
        expected_internal_test_assets=arguments.expected_internal_test_assets,
        expected_internal_test_source_groups=(
            arguments.expected_internal_test_source_groups
        ),
        bootstrap_iterations=arguments.bootstrap_iterations,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "candidate_lock_path": str(result.candidate_lock_path),
                "metrics_lock_path": str(result.metrics_lock_path),
                "pre_internal_test_ready": True,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
