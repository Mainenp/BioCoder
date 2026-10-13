"""Build the immutable 16-cell v2 RL selection checkpoint matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.rl_candidate_matrix import build_candidate_matrix


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--code-revision", required=True)
    result.add_argument("--rl-pool-report-sha256", required=True)
    result.add_argument("--c0-root", required=True, type=Path)
    result.add_argument("--c0-report-sha256", required=True)
    result.add_argument("--c0-manifest-sha256", required=True)
    result.add_argument("--run-spec", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    run_spec = json.loads(args.run_spec.read_text(encoding="utf-8"))
    result = build_candidate_matrix(
        code_revision=args.code_revision,
        rl_pool_report_sha256=args.rl_pool_report_sha256,
        c0_root=args.c0_root,
        c0_report_sha256=args.c0_report_sha256,
        c0_manifest_sha256=args.c0_manifest_sha256,
        run_spec=run_spec,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.__dict__.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
