"""CLI for CPU-only v2 selection-bundle construction, scoring, and sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.rl_selection import (
    begin_one_time_validation_access,
    build_selection_bundle,
    complete_one_time_validation_access,
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
    begin = commands.add_parser("begin-validation")
    begin.add_argument("--selection-root", required=True, type=Path); begin.add_argument("--selection-result-sha256", required=True); begin.add_argument("--ledger-root", required=True, type=Path)
    complete = commands.add_parser("complete-validation")
    complete.add_argument("--selection-root", required=True, type=Path); complete.add_argument("--selection-result-sha256", required=True); complete.add_argument("--ledger-root", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-bundle": result = build_selection_bundle(pool_root=args.pool_root, expected_pool_report_sha256=args.pool_report_sha256, output_dir=args.output_dir, code_revision=args.code_revision)
    elif args.command == "evaluate": result = evaluate_selection_predictions(bundle_root=args.bundle_root, expected_bundle_report_sha256=args.bundle_report_sha256, predictions_path=args.predictions, candidate=json.loads(args.candidate_json.read_text(encoding="utf-8")), output_dir=args.output_dir)
    elif args.command == "seal": result = seal_selection_result(bundle_root=args.bundle_root, expected_bundle_report_sha256=args.bundle_report_sha256, c0_evaluation=args.c0_evaluation, rl_evaluations=args.rl_evaluation, c1_evaluations=args.c1_evaluation, output_dir=args.output_dir, code_revision=args.code_revision)
    else:
        access = begin_one_time_validation_access(selection_root=args.selection_root, expected_selection_result_sha256=args.selection_result_sha256, ledger_root=args.ledger_root)
        if args.command == "complete-validation":
            complete_one_time_validation_access(access=access, selection_result_sha256=args.selection_result_sha256)
        result = access
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
