from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_science.chrompeakformer.sequence_preflight import (
    build_sequence_preflight_report,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Profile ROI-cropped XIC signals before baseline training."
    )
    argument_parser.add_argument("--index", type=Path, required=True)
    argument_parser.add_argument("--readiness-report", type=Path, required=True)
    argument_parser.add_argument("--assets-root", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()

    def report_progress(completed: int, total: int, path: str) -> None:
        print(
            f"[sequence-preflight] verified matrix {completed}/{total}: {path}",
            file=sys.stderr,
            flush=True,
        )

    result = build_sequence_preflight_report(
        arguments.index,
        arguments.readiness_report,
        arguments.assets_root,
        arguments.output,
        progress_callback=report_progress,
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "asset_index_sha256": result.asset_index_sha256,
                "asset_count": result.asset_count,
                "matrix_count": result.matrix_count,
                "quality_gate_passed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
