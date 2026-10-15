"""Build a target-free SequencePeakNet prediction bundle for Qwen prompting."""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from multimodal_science.baselines.sequence_training import REPORT_SCHEMA
from multimodal_science.data.manifest import sha256_file


SEQUENCE_PROMPT_BUNDLE_SCHEMA = "chrompeak-sequence-prompt-bundle-v1"
SEQUENCE_PROMPT_PREDICTION_SCHEMA = "chrompeak-sequence-prompt-prediction-v1"
SOURCE_PREDICTION_SCHEMA = "chrompeak-sequence-prediction-v1"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_PREDICTION_FIELDS = (
    "presence_probability",
    "start_normalized",
    "end_normalized",
)


@dataclass(frozen=True)
class SequencePromptBundleResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    predictions_path: Path
    prediction_records: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _safe_artifact(root: Path, relative: str, label: str) -> Path:
    _require(bool(relative) and "\\" not in relative, f"Bad {label} path")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute() and all(part not in {"", ".", ".."} for part in posix.parts),
        f"Unsafe {label} path",
    )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    _require(path.is_file(), f"Missing {label}: {path}")
    return path


def _verify_manifest(root: Path, expected_sha256: str) -> dict[str, str]:
    _require(bool(_HEX_64.fullmatch(expected_sha256)), "Bad sequence manifest hash")
    path = root / "artifact_manifest.sha256"
    _require(path.is_file(), "Missing sequence manifest")
    _require(sha256_file(path) == expected_sha256, "Sequence manifest hash mismatch")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", maxsplit=1)
        _require(len(parts) == 2, f"Malformed sequence manifest line: {line_number}")
        digest, relative = parts
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad manifest digest: {line_number}")
        artifact = _safe_artifact(root, relative, "sequence artifact")
        _require(sha256_file(artifact) == digest, f"Sequence artifact hash mismatch: {relative}")
        canonical = artifact.relative_to(root).as_posix()
        _require(
            canonical not in entries,
            f"Duplicate normalized sequence artifact: {canonical}",
        )
        entries[canonical] = digest
    return entries


def _finite_unit(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{label} outside [0, 1]")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_sequence_prompt_bundle(
    *,
    sequence_run_root: Path,
    sequence_report_sha256: str,
    sequence_manifest_sha256: str,
    output_dir: Path,
    code_revision: str,
) -> SequencePromptBundleResult:
    """Strip target columns from a frozen sequence run into an immutable bundle."""

    _require(bool(_HEX_64.fullmatch(sequence_report_sha256)), "Bad sequence report hash")
    _require(bool(_HEX_40.fullmatch(code_revision)), "Bad code revision")
    root = sequence_run_root.resolve()
    _require(root.is_dir(), f"Sequence run not found: {root}")
    manifest = _verify_manifest(root, sequence_manifest_sha256)
    report_path = root / "scientific_report.json"
    _require(report_path.is_file(), "Missing sequence report")
    _require(sha256_file(report_path) == sequence_report_sha256, "Sequence report hash mismatch")
    report = _read_json(report_path, "sequence report")
    _require(report.get("schema_version") == REPORT_SCHEMA, "Unsupported sequence report")
    _require(report.get("development_comparison_eligible") is True, "Sequence run ineligible")
    _require(
        report.get("final_benchmark_eligible") is False,
        "Sequence run is not development-only",
    )
    _require(report.get("internal_test_accessed") is False, "Sequence run accessed test data")
    _require(
        _object(report.get("config"), "sequence config").get("modality")
        == "sequence",
        "Sequence-only expert required",
    )
    source_artifact = _object(
        _object(report.get("artifacts"), "sequence artifacts").get("validation_predictions"),
        "validation predictions",
    )
    relative = source_artifact.get("path")
    source_digest = source_artifact.get("sha256")
    _require(isinstance(relative, str), "Missing source prediction path")
    _require(
        isinstance(source_digest, str) and bool(_HEX_64.fullmatch(source_digest)),
        "Bad source prediction hash",
    )
    source_path = _safe_artifact(root, relative, "sequence predictions")
    _require(sha256_file(source_path) == source_digest, "Source prediction hash mismatch")
    source_manifest_path = source_path.relative_to(root).as_posix()
    _require(
        manifest.get(source_manifest_path) == source_digest,
        "Source predictions are not manifest-bound",
    )

    clean_rows: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    discarded_fields: set[str] = set()
    with source_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            _require(bool(line.strip()), f"Blank sequence prediction line: {line_number}")
            try:
                row = _object(json.loads(line), f"prediction line {line_number}")
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid prediction JSON at line {line_number}") from error
            _require(row.get("schema_version") == SOURCE_PREDICTION_SCHEMA, "Bad source schema")
            asset_id = str(row.get("asset_id") or "")
            group_id = str(row.get("group_id") or "")
            _require(asset_id and group_id and asset_id not in seen_assets, "Bad asset identity")
            seen_assets.add(asset_id)
            probability = _finite_unit(row.get("presence_probability"), "presence probability")
            start = _finite_unit(row.get("start_normalized"), "interval start")
            end = _finite_unit(row.get("end_normalized"), "interval end")
            _require(start <= end, "Sequence interval is reversed")
            clean_rows.append(
                {
                    "schema_version": SEQUENCE_PROMPT_PREDICTION_SCHEMA,
                    "asset_id": asset_id,
                    "group_id": group_id,
                    "presence_probability": probability,
                    "start_normalized": start,
                    "end_normalized": end,
                }
            )
            discarded_fields.update(
                set(row)
                - {"schema_version", "asset_id", "group_id", *_PREDICTION_FIELDS}
            )
    _require(clean_rows, "No sequence predictions found")
    _require(
        source_artifact.get("records") == len(clean_rows),
        "Source sequence prediction count mismatch",
    )

    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Sequence prompt bundle exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        predictions_path = staging / "sequence_prompt_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in clean_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                stream.write("\n")
        clean_digest = sha256_file(predictions_path)
        bundle_report_path = staging / "sequence_prompt_bundle_report.json"
        _write_json(
            bundle_report_path,
            {
                "schema_version": SEQUENCE_PROMPT_BUNDLE_SCHEMA,
                "code_revision": code_revision,
                "sources": {
                    "sequence_report_sha256": sequence_report_sha256,
                    "sequence_manifest_sha256": sequence_manifest_sha256,
                    "source_predictions_sha256": source_digest,
                },
                "counts": {
                    "prediction_records": len(clean_rows),
                    "source_groups": len({row["group_id"] for row in clean_rows}),
                },
                "artifacts": {
                    "predictions": {
                        "path": predictions_path.name,
                        "sha256": clean_digest,
                        "records": len(clean_rows),
                    }
                },
                "sanitization": {
                    "allowlisted_fields": ["asset_id", "group_id", *_PREDICTION_FIELDS],
                    "discarded_source_fields": sorted(discarded_fields),
                },
                "contracts": {
                    "frozen_sequence_predictions_only": True,
                    "sequence_report_sha256_pinned": True,
                    "source_predictions_manifest_bound": True,
                    "target_fields_excluded": True,
                    "instruction_answers_opened": False,
                    "development_only": True,
                    "internal_test_accessed": False,
                },
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
        )
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (bundle_report_path, predictions_path)
            ),
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_dir)

    final_report = output_dir / "sequence_prompt_bundle_report.json"
    return SequencePromptBundleResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        predictions_path=output_dir / "sequence_prompt_predictions.jsonl",
        prediction_records=len(clean_rows),
    )
