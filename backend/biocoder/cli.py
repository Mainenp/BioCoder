from __future__ import annotations

import argparse
import asyncio
import json


async def run_agent(query: str, thread_id: str | None = None) -> dict:
    from app.main import chat, conversation_store
    from app.schemas import ChatRequest

    conversation_store.initialize()
    response = await chat(ChatRequest(message=query, thread_id=thread_id))
    return response.model_dump(mode="json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biocoder", description="BioCoder 2.0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one Agent request and save its trajectory.")
    run.add_argument("--query", "-q")
    run.add_argument("--thread-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        query = (args.query or input("BioCoder query: ")).strip()
        if not query:
            raise SystemExit("Query must not be empty")
        result = asyncio.run(run_agent(query, args.thread_id))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
