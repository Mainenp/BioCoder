"""CLI for CPU-only v2 selection-bundle construction, scoring, and sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.rl_selection import (
    build_selection_bundle,
    evaluate_selection_predictions,
    seal_selection_result,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("build-bundle")
    bundle.add_argument("--pool-root", required=True, type=Path); bundle.add_argument("--pool-report-sha256", required=True); bundle.add_argument("--output-dir", required=True, type=Path); bundle.add_argument("--code-revision", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--bundle-root", required=True, type=Path); evaluate.add_argument("--bundle-report-sha256", required=True); evaluate.add_argument("--predictions", required=True, type=Path); evaluate.add_argument("--candidate-json", required=True, type=Path); evaluate.add_argument("--output-dir", required=True, type=Path)
    seal = commands.add_parser("seal")
    seal.add_argument("--bundle-root", required=True, type=Path); seal.add_argument("--bundle-report-sha256", required=True); seal.add_argument("--c0-evaluation", required=True, type=Path); seal.add_argument("--rl-evaluation", required=True, type=Path, action="append"); seal.add_argument("--c1-evaluation", required=True, type=Path, action="append"); seal.add_argument("--output-dir", required=True, type=Path); seal.add_argument("--code-revision", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-bundle": result = build_selection_bundle(pool_root=args.pool_root, expected_pool_report_sha256=args.pool_report_sha256, output_dir=args.output_dir, code_revision=args.code_revision)
    elif args.command == "evaluate": result = evaluate_selection_predictions(bundle_root=args.bundle_root, expected_bundle_report_sha256=args.bundle_report_sha256, predictions_path=args.predictions, candidate=json.loads(args.candidate_json.read_text(encoding="utf-8")), output_dir=args.output_dir)
    else: result = seal_selection_result(bundle_root=args.bundle_root, expected_bundle_report_sha256=args.bundle_report_sha256, c0_evaluation=args.c0_evaluation, rl_evaluations=args.rl_evaluation, c1_evaluations=args.c1_evaluation, output_dir=args.output_dir, code_revision=args.code_revision)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
