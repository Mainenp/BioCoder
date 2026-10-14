"""Verify and print the completed v2 RL development evidence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.v2_rl_evidence_summary import (
    summarize_v2_rl_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--slurm-root", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = summarize_v2_rl_evidence(
        analysis_root=arguments.analysis_root,
        run_root=arguments.run_root,
        slurm_root=arguments.slurm_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
