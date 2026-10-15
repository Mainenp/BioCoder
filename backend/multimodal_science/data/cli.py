from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.data.manifest import build_manifest


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Build an eligibility-gated chromatogram manifest from raw data."
    )
    argument_parser.add_argument("--data-root", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    argument_parser.add_argument("--source-archive-sha256", required=True)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = build_manifest(
        arguments.data_root,
        arguments.output_dir,
        source_archive_sha256=arguments.source_archive_sha256,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
                "record_count": result.record_count,
                "train_eligible_count": result.train_eligible_count,
                "benchmark_eligible_count": result.benchmark_eligible_count,
                "audit_count": result.audit_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
