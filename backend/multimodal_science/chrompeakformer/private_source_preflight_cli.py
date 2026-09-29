from __future__ import annotations

import argparse
import json

from multimodal_science.chrompeakformer.private_adapter import (
    preflight_private_source,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Import and fingerprint the complete authorized private extractor source."
    )
    argument_parser.add_argument("--expected-sha256", required=True)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = preflight_private_source(arguments.expected_sha256)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
