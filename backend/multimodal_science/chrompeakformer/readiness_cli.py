from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.training_readiness import (
    build_training_readiness_report,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Validate a complete asset index before multimodal training."
    )
    argument_parser.add_argument("--index", type=Path, required=True)
    argument_parser.add_argument("--index-report", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--required-split", action="append", dest="required_splits")
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = build_training_readiness_report(
        arguments.index,
        arguments.index_report,
        arguments.output,
        required_splits=(
            frozenset(arguments.required_splits)
            if arguments.required_splits
            else frozenset({"train", "validation"})
        ),
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "asset_index_sha256": result.asset_index_sha256,
                "asset_count": result.asset_count,
                "splits": list(result.splits),
                "quality_gate_passed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
