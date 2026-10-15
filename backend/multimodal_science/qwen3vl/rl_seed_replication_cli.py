"""Build or analyze the selection-only v2 RL seed-29 replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.rl_seed_replication import (
    analyze_seed_replication,
    build_seed_replication_candidates,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build-candidates")
    build.add_argument("--code-revision", required=True)
    build.add_argument("--rl-pool-report-sha256", required=True)
    build.add_argument("--c0-report-sha256", required=True)
    build.add_argument("--c0-manifest-sha256", required=True)
    build.add_argument("--r2-root", required=True, type=Path)
    build.add_argument("--c1-root", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--seed17-r2-evaluation", required=True, type=Path)
    analyze.add_argument("--seed17-c1-evaluation", required=True, type=Path)
    analyze.add_argument("--seed29-r2-evaluation", required=True, type=Path)
    analyze.add_argument("--seed29-c1-evaluation", required=True, type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.action == "build-candidates":
        value = build_seed_replication_candidates(
            code_revision=args.code_revision,
            rl_pool_report_sha256=args.rl_pool_report_sha256,
            c0_report_sha256=args.c0_report_sha256,
            c0_manifest_sha256=args.c0_manifest_sha256,
            r2_root=args.r2_root,
            c1_root=args.c1_root,
            output_dir=args.output_dir,
        )
    else:
        value = analyze_seed_replication(
            seed17_r2_evaluation=args.seed17_r2_evaluation,
            seed17_c1_evaluation=args.seed17_c1_evaluation,
            seed29_r2_evaluation=args.seed29_r2_evaluation,
            seed29_c1_evaluation=args.seed29_c1_evaluation,
            output_dir=args.output_dir,
        )
    print(
        json.dumps(
            {
                key: str(item) if isinstance(item, Path) else item
                for key, item in value.__dict__.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
