from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.executor import (
    execute_plan,
    execution_payload,
    lazy_extractor,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Execute hash-verified ChromPeakFormer derivation jobs."
    )
    argument_parser.add_argument("--plan", type=Path, required=True)
    argument_parser.add_argument("--data-root", type=Path, required=True)
    argument_parser.add_argument("--output-root", type=Path, required=True)
    argument_parser.add_argument(
        "--extractor",
        required=True,
        help="Python callable in package.module:callable form",
    )
    argument_parser.add_argument("--split", action="append", dest="splits")
    argument_parser.add_argument("--max-jobs", type=int)
    argument_parser.add_argument("--continue-on-error", action="store_true")
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = execute_plan(
        arguments.plan,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        extractor=lazy_extractor(arguments.extractor),
        include_splits=frozenset(arguments.splits) if arguments.splits else None,
        max_jobs=arguments.max_jobs,
        continue_on_error=arguments.continue_on_error,
    )
    print(json.dumps(execution_payload(result), ensure_ascii=False))
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
