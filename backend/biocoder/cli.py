from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def run_agent(query: str, thread_id: str | None = None) -> dict:
    from app.main import chat, conversation_store
    from app.schemas import ChatRequest

    conversation_store.initialize()
    response = await chat(ChatRequest(message=query, thread_id=thread_id))
    return response.model_dump(mode="json")


def _add_multimodal_source_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--protocol-root", type=Path, required=True)
    command.add_argument("--protocol-sha256", required=True)
    command.add_argument("--ledger-dir", type=Path, required=True)
    command.add_argument("--report-root", type=Path, required=True)


def _add_multimodal_commands(command: argparse.ArgumentParser) -> None:
    subparsers = command.add_subparsers(dest="multimodal_command", required=True)
    verify = subparsers.add_parser(
        "verify-final",
        help="Verify the completed sealed benchmark without reopening test data.",
    )
    _add_multimodal_source_arguments(verify)
    archive = subparsers.add_parser(
        "archive-final", help="Build a deterministic public-safe benchmark capsule."
    )
    _add_multimodal_source_arguments(archive)
    archive.add_argument("--output-dir", type=Path, required=True)
    for name, help_text in (
        ("verify-release", "Verify a standalone public benchmark capsule."),
        ("show-results", "Verify a public capsule and print its frozen result table."),
    ):
        release = subparsers.add_parser(name, help=help_text)
        release.add_argument("--release-root", type=Path, required=True)
        release.add_argument("--archive", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biocoder", description="BioCoder 2.0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one Agent request and save its trajectory.")
    run.add_argument("--query", "-q")
    run.add_argument("--thread-id")
    multimodal = subparsers.add_parser(
        "multimodal",
        help="Verify, archive, or inspect the sealed LC-MS multimodal benchmark.",
    )
    _add_multimodal_commands(multimodal)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        query = (args.query or input("BioCoder query: ")).strip()
        if not query:
            raise SystemExit("Query must not be empty")
        result = asyncio.run(run_agent(query, args.thread_id))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "multimodal":
        from multimodal_science.final_benchmark_release_cli import (
            dispatch_release_command,
        )

        dispatch_release_command(args)


if __name__ == "__main__":
    main()
