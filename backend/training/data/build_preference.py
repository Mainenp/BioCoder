from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.data.build_sft import load_records


def _query_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().casefold())


def _pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    source: str,
    chosen_record: dict[str, Any],
    rejected_record: dict[str, Any],
    dataset_version: str,
) -> dict[str, Any] | None:
    if not chosen.strip() or not rejected.strip() or chosen.strip() == rejected.strip():
        return None
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "preference_source": source,
        "provenance": {
            "chosen_task_id": chosen_record.get("task_id"),
            "rejected_task_id": rejected_record.get("task_id"),
            "chosen_record_id": chosen_record.get("record_id"),
            "rejected_record_id": rejected_record.get("record_id"),
            "chosen_quality_score": chosen_record.get("quality_score", 0),
            "rejected_quality_score": rejected_record.get("quality_score", 0),
            "dataset_version": dataset_version,
            "generation_method": source,
            "created_at": datetime.now(UTC).isoformat(),
        },
    }


def build_pairs(
    records: list[dict[str, Any]],
    *,
    dataset_version: str,
    min_quality_gap: float = 0.2,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("query", "")).strip():
            by_query[_query_key(str(record["query"]))].append(record)

        for feedback in record.get("feedback", []):
            correction = str(feedback.get("corrected_answer") or "").strip()
            if correction:
                candidate = _pair(
                    prompt=str(record.get("query", "")),
                    chosen=correction,
                    rejected=str(record.get("answer", "")),
                    source="human_corrected_vs_original",
                    chosen_record=record,
                    rejected_record=record,
                    dataset_version=dataset_version,
                )
                if candidate:
                    candidate["provenance"]["feedback_id"] = feedback.get("feedback_id")
                    pairs.append(candidate)

    for group in by_query.values():
        ranked = sorted(group, key=lambda row: float(row.get("quality_score", 0)), reverse=True)
        if len(ranked) < 2:
            continue
        chosen, rejected = ranked[0], ranked[-1]
        gap = float(chosen.get("quality_score", 0)) - float(rejected.get("quality_score", 0))
        if gap < min_quality_gap and bool(chosen.get("success")) == bool(rejected.get("success")):
            continue
        if chosen.get("success") and not rejected.get("success"):
            source = "successful_vs_failed"
        else:
            source = "high_score_vs_low_score"
        candidate = _pair(
            prompt=str(chosen["query"]),
            chosen=str(chosen.get("answer", "")),
            rejected=str(rejected.get("answer", "")),
            source=source,
            chosen_record=chosen,
            rejected_record=rejected,
            dataset_version=dataset_version,
        )
        if candidate:
            pairs.append(candidate)

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pair in pairs:
        unique[(pair["prompt"], pair["chosen"], pair["rejected"])] = pair
    return list(unique.values())


def build_preference_dataset(
    input_path: Path,
    output_path: Path,
    *,
    dataset_version: str,
    min_quality_gap: float = 0.2,
) -> dict[str, Any]:
    records = load_records(input_path)
    pairs = build_pairs(
        records,
        dataset_version=dataset_version,
        min_quality_gap=min_quality_gap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in pairs)
        + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    manifest = {
        "dataset_type": "preference",
        "dataset_version": dataset_version,
        "source": str(input_path),
        "created_at": datetime.now(UTC).isoformat(),
        "input_records": len(records),
        "preference_pairs": len(pairs),
        "min_quality_gap": min_quality_gap,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provenance-rich BioCoder preference data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--min-quality-gap", type=float, default=0.2)
    args = parser.parse_args()
    manifest = build_preference_dataset(
        args.input,
        args.output,
        dataset_version=args.dataset_version,
        min_quality_gap=args.min_quality_gap,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
