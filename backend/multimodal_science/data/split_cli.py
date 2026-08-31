from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.data.splits import DEFAULT_SEED, build_splits


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Create deterministic, leakage-audited chromatogram dataset splits."
    )
    argument_parser.add_argument("--manifest", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    argument_parser.add_argument("--audit-report", type=Path)
    argument_parser.add_argument("--seed", default=DEFAULT_SEED)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = build_splits(
        arguments.manifest,
        arguments.output_dir,
        audit_report_path=arguments.audit_report,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "split_manifest_path": str(result.split_manifest_path),
                "report_path": str(result.report_path),
                "split_manifest_sha256": result.split_manifest_sha256,
                "record_count": result.record_count,
                "group_count": result.group_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
