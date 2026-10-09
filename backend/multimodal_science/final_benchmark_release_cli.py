"""CLI for verification and public-safe packaging of the sealed benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.final_benchmark_release import (
    build_final_benchmark_release,
    verify_final_benchmark_evidence,
    verify_public_release,
)


def _add_source_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--protocol-root", type=Path, required=True)
    command.add_argument("--protocol-sha256", required=True)
    command.add_argument("--ledger-dir", type=Path, required=True)
    command.add_argument("--report-root", type=Path, required=True)


def add_release_subcommands(subparsers: argparse._SubParsersAction) -> None:
    verify = subparsers.add_parser(
        "verify-final",
        help="Verify the completed sealed benchmark without reopening test data.",
    )
    _add_source_arguments(verify)
    archive = subparsers.add_parser(
        "archive-final", help="Build a deterministic public-safe benchmark capsule."
    )
    _add_source_arguments(archive)
    archive.add_argument("--output-dir", type=Path, required=True)
    verify_release = subparsers.add_parser(
        "verify-release", help="Verify a standalone public benchmark capsule."
    )
    verify_release.add_argument("--release-root", type=Path, required=True)
    verify_release.add_argument("--archive", type=Path)
    results = subparsers.add_parser(
        "show-results", help="Verify a public capsule and print its frozen result table."
    )
    results.add_argument("--release-root", type=Path, required=True)
    results.add_argument("--archive", type=Path)


def dispatch_release_command(args: argparse.Namespace) -> None:
    if args.multimodal_command == "verify-final":
        result, _ = verify_final_benchmark_evidence(
            protocol_root=args.protocol_root,
            expected_protocol_sha256=args.protocol_sha256,
            ledger_dir=args.ledger_dir,
            report_root=args.report_root,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.multimodal_command == "archive-final":
        result = build_final_benchmark_release(
            protocol_root=args.protocol_root,
            expected_protocol_sha256=args.protocol_sha256,
            ledger_dir=args.ledger_dir,
            report_root=args.report_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.multimodal_command in {"verify-release", "show-results"}:
        result = verify_public_release(
            release_root=args.release_root,
            archive_path=args.archive,
        )
        if args.multimodal_command == "show-results":
            print((args.release_root / "benchmark_table.md").read_text(encoding="utf-8"))
        else:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse enforces the command set
        raise SystemExit(f"Unknown multimodal command: {args.multimodal_command}")


parser = argparse.ArgumentParser(
    prog="python -m multimodal_science.final_benchmark_release_cli"
)
subparsers = parser.add_subparsers(dest="multimodal_command", required=True)
add_release_subcommands(subparsers)


def main() -> None:
    dispatch_release_command(parser.parse_args())


if __name__ == "__main__":
    main()
