"""Create, verify, or complete the one-time sealed benchmark access ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_protocol import (
    complete_final_benchmark_access,
    open_final_benchmark_access,
    verify_final_benchmark_access,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("open", "verify", "complete"))
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--final-evidence-manifest", type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    common = {
        "protocol_root": arguments.protocol_root,
        "expected_protocol_sha256": arguments.protocol_sha256,
        "ledger_dir": arguments.ledger_dir,
    }
    if arguments.mode == "open":
        if arguments.final_evidence_manifest is not None:
            raise ValueError("open does not accept --final-evidence-manifest")
        result = open_final_benchmark_access(**common)
    elif arguments.mode == "verify":
        if arguments.final_evidence_manifest is not None:
            raise ValueError("verify does not accept --final-evidence-manifest")
        result = verify_final_benchmark_access(**common)
    else:
        if arguments.final_evidence_manifest is None:
            raise ValueError("complete requires --final-evidence-manifest")
        result = complete_final_benchmark_access(
            **common,
            final_evidence_manifest_path=arguments.final_evidence_manifest,
        )
    print(
        json.dumps(
            {
                "ledger_dir": str(result.ledger_dir),
                "access_path": str(result.access_path),
                "access_sha256": result.access_sha256,
                "access_id": result.access_id,
                "protocol_sha256": result.protocol_sha256,
                "completed": result.completed,
                "internal_test_accessed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
