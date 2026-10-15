from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_science.chrompeakformer.auxiliary_asset_index import (
    build_auxiliary_asset_index,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Build a complete, unlabeled-only auxiliary ROI/XIC asset index."
    )
    argument_parser.add_argument("--plan", type=Path, required=True)
    argument_parser.add_argument("--assets-root", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()

    def report_progress(completed: int, total: int, job_id: str) -> None:
        print(
            f"[auxiliary-index] verified job {completed}/{total}: {job_id}",
            file=sys.stderr,
            flush=True,
        )

    result = build_auxiliary_asset_index(
        arguments.plan,
        arguments.assets_root,
        arguments.output_dir,
        progress_callback=report_progress,
    )
    print(
        json.dumps(
            {
                "plan_sha256": result.plan_sha256,
                "index_path": str(result.index_path),
                "report_path": str(result.report_path),
                "manifest_path": str(result.manifest_path),
                "index_sha256": result.index_sha256,
                "report_sha256": result.report_sha256,
                "manifest_sha256": result.manifest_sha256,
                "selected_jobs": result.selected_jobs,
                "indexed_jobs": result.indexed_jobs,
                "missing_jobs": result.missing_jobs,
                "asset_count": result.asset_count,
                "source_group_count": result.source_group_count,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
