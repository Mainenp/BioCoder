from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_science.chrompeakformer.multimodal_dataset import (
    build_multimodal_dataset,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Materialize a provenance-bound multimodal chromatogram dataset."
    )
    argument_parser.add_argument("--index", type=Path, required=True)
    argument_parser.add_argument("--readiness-report", type=Path, required=True)
    argument_parser.add_argument("--sequence-preflight", type=Path, required=True)
    argument_parser.add_argument("--assets-root", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    argument_parser.add_argument("--split", action="append", dest="splits")
    argument_parser.add_argument("--target-points", type=int, default=160)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()

    def report_progress(completed: int, total: int, path: str) -> None:
        print(
            f"[materialize] processed matrix {completed}/{total}: {path}",
            file=sys.stderr,
            flush=True,
        )

    result = build_multimodal_dataset(
        arguments.index,
        arguments.readiness_report,
        arguments.sequence_preflight,
        arguments.assets_root,
        arguments.output_dir,
        include_splits=(
            frozenset(arguments.splits)
            if arguments.splits
            else frozenset({"train", "validation"})
        ),
        target_points=arguments.target_points,
        progress_callback=report_progress,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "asset_index_sha256": result.asset_index_sha256,
                "asset_count": result.asset_count,
                "target_points": result.target_points,
                "cached": result.cached,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
