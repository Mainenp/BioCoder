"""Materialize the zero-label auxiliary XIC pretraining dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_science.qwen3vl.auxiliary_pretraining_data import (
    build_auxiliary_pretraining_dataset,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--auxiliary-index-root", type=Path, required=True)
    command.add_argument("--auxiliary-index-report-sha256", required=True)
    command.add_argument("--auxiliary-index-manifest-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--target-points", type=int, default=160)
    return command


def main() -> None:
    arguments = parser().parse_args()

    def progress(completed: int, total: int, asset_id: str) -> None:
        if completed == total or completed % 100 == 0:
            print(
                f"[auxiliary-materialize] {completed}/{total}: {asset_id}",
                file=sys.stderr,
                flush=True,
            )

    result = build_auxiliary_pretraining_dataset(
        arguments.auxiliary_index_root,
        arguments.auxiliary_index_report_sha256,
        arguments.auxiliary_index_manifest_sha256,
        arguments.assets_root,
        arguments.output_dir,
        target_points=arguments.target_points,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "asset_count": result.asset_count,
                "source_group_count": result.source_group_count,
                "target_points": result.target_points,
                "labels": 0,
                "metrics_allowed": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
