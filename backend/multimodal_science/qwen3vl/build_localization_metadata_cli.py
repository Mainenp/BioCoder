"""Build hash-bound, answer-free validation metadata for v2 localization analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.localization_analysis import (
    build_localization_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-report-sha256", required=True)
    parser.add_argument("--fusion-bundle-root", type=Path, required=True)
    parser.add_argument("--fusion-bundle-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = build_localization_metadata(
        dataset_root=arguments.dataset_root,
        expected_dataset_report_sha256=arguments.dataset_report_sha256,
        fusion_bundle_root=arguments.fusion_bundle_root,
        expected_fusion_bundle_report_sha256=arguments.fusion_bundle_report_sha256,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "metadata_path": str(result.metadata_path),
                "metadata_sha256": result.metadata_sha256,
                "manifest_path": str(result.manifest_path),
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
