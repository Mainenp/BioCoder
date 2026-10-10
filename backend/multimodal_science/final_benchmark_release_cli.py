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
from multimodal_science.qwen3vl.public_development_evidence import (
    build_public_development_evidence_archive,
    verify_public_development_evidence,
)


def _add_source_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--protocol-root", type=Path, required=True)
    command.add_argument("--protocol-sha256", required=True)
    command.add_argument("--ledger-dir", type=Path, required=True)
    command.add_argument("--report-root", type=Path, required=True)


def _add_development_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--evidence-root", type=Path, required=True)
    command.add_argument("--report-sha256", required=True)


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
    verify_development = subparsers.add_parser(
        "verify-development",
        help="Verify the standalone validation-only v1.1 evidence.",
    )
    _add_development_arguments(verify_development)
    verify_development.add_argument("--archive", type=Path)
    archive_development = subparsers.add_parser(
        "archive-development",
        help="Build a deterministic ZIP from verified v1.1 evidence.",
    )
    _add_development_arguments(archive_development)
    archive_development.add_argument("--archive", type=Path, required=True)
    show_development = subparsers.add_parser(
        "show-development",
        help="Verify and print the validation-only v1.1 evidence table.",
    )
    _add_development_arguments(show_development)
    show_development.add_argument("--archive", type=Path)


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
    elif args.multimodal_command == "archive-development":
        result = build_public_development_evidence_archive(
            evidence_root=args.evidence_root,
            expected_report_sha256=args.report_sha256,
            archive_path=args.archive,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.multimodal_command in {"verify-development", "show-development"}:
        result = verify_public_development_evidence(
            evidence_root=args.evidence_root,
            expected_report_sha256=args.report_sha256,
            archive_path=args.archive,
        )
        if args.multimodal_command == "show-development":
            print(
                (args.evidence_root / "public_development_evidence.md").read_text(
                    encoding="utf-8"
                )
            )
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
