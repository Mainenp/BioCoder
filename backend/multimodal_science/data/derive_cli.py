from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.data.derivation import build_derivation_plan


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Build hash-verified ChromPeakFormer ROI/XIC derivation jobs."
    )
    argument_parser.add_argument("--split-manifest", type=Path, required=True)
    argument_parser.add_argument("--data-root", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    argument_parser.add_argument("--audit-report", type=Path)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = build_derivation_plan(
        arguments.split_manifest,
        arguments.data_root,
        arguments.output_dir,
        audit_report_path=arguments.audit_report,
    )
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "plan_path": str(result.plan_path),
                "report_path": str(result.report_path),
                "plan_sha256": result.plan_sha256,
                "job_count": result.job_count,
                "source_count": result.source_count,
                "execution_ready": result.execution_ready,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
