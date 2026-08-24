from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def dataset_digest(records: list[dict[str, Any]]) -> str:
    stable = [
        {
            "content_hash": row.get("content_hash"),
            "task_id": row.get("task_id"),
            "quality_score": row.get("quality_score"),
        }
        for row in records
    ]
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_version(
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    source_counts: dict[str, int],
) -> tuple[str, Path]:
    digest = dataset_digest(records)
    timestamp = datetime.now(UTC)
    version = f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:8]}"
    version_dir = output_root / version
    version_dir.mkdir(parents=True, exist_ok=False)
    records_path = version_dir / "flywheel_records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": version,
        "created_at": timestamp.isoformat(),
        "digest": digest,
        "record_count": len(records),
        "source_counts": source_counts,
        "records_file": records_path.name,
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return version, version_dir
