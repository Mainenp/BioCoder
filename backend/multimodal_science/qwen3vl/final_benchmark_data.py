"""Materialize one access-bound internal-test view with answer isolation.

The generation-facing prompt bundle, evaluator-only answer key, XIC links, and
detector COCO view are separate hash trees.  Callers must present the already
opened one-time access ledger; this module never creates a second access event.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FINAL_BENCHMARK_PROTOCOL_SCHEMA,
    verify_final_benchmark_access,
)
from multimodal_science.qwen3vl.fusion_data import FUSION_LINK_SCHEMA
from multimodal_science.qwen3vl.inference_bundle import BUNDLE_PROMPT_SCHEMA
from multimodal_science.qwen3vl.instruction_data import (
    LANGUAGES,
    TASKS,
    _instruction_id,
    _localized_instruction_id,
    _portable_image_path,
    _task_record,
)


FINAL_BENCHMARK_DATA_SCHEMA = "chrompeak-final-benchmark-data-v1"
FINAL_INFERENCE_BUNDLE_SCHEMA = "chrompeak-qwen3vl-final-inference-bundle-v1"
FINAL_ANSWER_REPORT_SCHEMA = "chrompeak-qwen3vl-final-answer-key-v1"
FINAL_ANSWER_RECORD_SCHEMA = "chrompeak-qwen3vl-final-answer-v1"
FINAL_INSTRUCTION_MANIFEST_SCHEMA = "chrompeak-qwen3vl-final-manifest-v1"
FINAL_FUSION_BUNDLE_SCHEMA = "chrompeak-qwen3vl-final-xic-bundle-v1"
FINAL_DETECTOR_DATASET_SCHEMA = "chrompeak-final-detector-dataset-v1"
_DATASET_SCHEMA = "chrompeak-multimodal-dataset-v1"
_EXAMPLE_SCHEMA = "chrompeak-multimodal-example-v1"
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FinalBenchmarkDataResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path
    inference_root: Path
    answer_root: Path
    fusion_root: Path
    detector_root: Path
    assets: int
    prompts: int
    source_groups: int
    access_id: str
    cached: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {label}: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(line.strip() != "", f"Blank {label} line: {line_number}")
            try:
                records.append(_object(json.loads(line), f"{label} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {label} line: {line_number}") from error
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value != "", f"Invalid {label} path")
    relative = Path(value)
    _require(not relative.is_absolute() and str(relative) != ".", f"Unsafe {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its root") from error
    return path


def _manifest_entries(root: Path) -> dict[str, str]:
    manifest = root / "artifact_manifest.sha256"
    _require(manifest.is_file(), f"Artifact manifest not found: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Bad manifest line {line_number}: {manifest}")
        digest, relative = parts[0], parts[1].strip()
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad manifest digest: {relative}")
        path = _safe_relative(root, relative, f"manifest artifact {relative}")
        _require(path.is_file(), f"Manifest artifact missing: {relative}")
        _require(sha256_file(path) == digest, f"Manifest artifact drift: {relative}")
        _require(relative not in entries, f"Duplicate manifest artifact: {relative}")
        entries[relative] = digest
    _require(bool(entries), f"Artifact manifest is empty: {manifest}")
    return entries


def _write_manifest(root: Path, paths: Iterable[Path]) -> Path:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return manifest


def _dataset_artifact(
    dataset_root: Path,
    report: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    artifacts = _object(report.get("artifacts"), "Dataset artifacts")
    descriptor = _object(artifacts.get(key), f"Dataset artifact {key}")
    path = _safe_relative(dataset_root, descriptor.get("path"), f"Dataset artifact {key}")
    expected = descriptor.get("sha256")
    _require(
        isinstance(expected, str) and bool(_HEX_64.fullmatch(expected)),
        f"Invalid Dataset artifact digest: {key}",
    )
    _require(path.is_file() and sha256_file(path) == expected, f"Dataset artifact drift: {key}")
    return path, descriptor


def _asset_path(assets_root: Path, relative: str) -> Path:
    path = _safe_relative(assets_root, relative, "image asset")
    _require(path.is_file(), f"Image asset not found: {relative}")
    return path


def _access_context(
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    access = verify_final_benchmark_access(
        protocol_root=protocol_root,
        expected_protocol_sha256=expected_protocol_sha256,
        ledger_dir=ledger_dir,
    )
    _require(not access.completed, "Final benchmark access is already completed")
    protocol = _read_object(
        protocol_root / "final_benchmark_protocol.json", "final benchmark protocol"
    )
    _require(
        protocol.get("schema_version") == FINAL_BENCHMARK_PROTOCOL_SCHEMA,
        "Unsupported final benchmark protocol",
    )
    locks = _object(protocol.get("locks"), "protocol locks")
    candidate_descriptor = _object(locks.get("candidate"), "candidate lock")
    candidate_path = _safe_relative(
        protocol_root, candidate_descriptor.get("path"), "candidate lock"
    )
    _require(
        sha256_file(candidate_path) == candidate_descriptor.get("sha256"),
        "Candidate lock drift",
    )
    return protocol, _read_object(candidate_path, "candidate lock"), access.access_id


def _validate_example(
    example: dict[str, Any], row: int, dataset_report: dict[str, Any]
) -> None:
    _require(example.get("schema_version") == _EXAMPLE_SCHEMA, "Bad example schema")
    _require(example.get("row") == row, "Internal-test rows are not contiguous")
    _require(example.get("split") == "internal_test", "Example split is not internal_test")
    for key in ("asset_id", "record_id", "group_id"):
        _require(isinstance(example.get(key), str) and bool(example[key]), f"Missing {key}")
    image = _object(example.get("image"), "example image")
    _portable_image_path(image.get("path"), row)
    _require(
        isinstance(image.get("sha256"), str) and bool(_HEX_64.fullmatch(image["sha256"])),
        "Invalid image digest",
    )
    _require(
        isinstance(image.get("width"), int)
        and image["width"] > 0
        and isinstance(image.get("height"), int)
        and image["height"] > 0,
        "Invalid image dimensions",
    )
    sequence = _object(example.get("sequence"), "example sequence")
    scalars = _object(example.get("scalar_features"), "example scalar features")
    target = _object(example.get("target"), "example target")
    _require(
        sequence.get("row") == row and scalars.get("row") == row and target.get("row") == row,
        "Example array row mismatch",
    )
    _require(isinstance(sequence.get("signal_available"), bool), "Missing signal availability")
    _require(isinstance(target.get("peak_present"), bool), "Missing peak target")
    _require(target.get("supervision_source") == "human", "Test target is not human-supervised")
    provenance = _object(example.get("provenance"), "example provenance")
    _require(
        provenance.get("asset_index_sha256") == dataset_report.get("asset_index_sha256"),
        "Example asset-index provenance drift",
    )


def _load_internal_dataset(
    *,
    dataset_root: Path,
    expected_dataset_report_sha256: str,
    candidate_lock: dict[str, Any],
    protocol: dict[str, Any],
    assets_root: Path,
    verify_image_hashes: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    report_path = dataset_root / "dataset_report.json"
    report = _read_object(report_path, "internal-test Dataset report")
    _require(
        isinstance(expected_dataset_report_sha256, str)
        and bool(_HEX_64.fullmatch(expected_dataset_report_sha256)),
        "Invalid internal-test Dataset report SHA-256",
    )
    _require(sha256_file(report_path) == expected_dataset_report_sha256, "Dataset report drift")
    _require(report.get("schema_version") == _DATASET_SCHEMA, "Unsupported Dataset schema")
    _require(report.get("splits") == ["internal_test"], "Dataset is not internal-test-only")

    shared = _object(candidate_lock.get("shared"), "candidate shared inputs")
    normalization = _object(shared.get("scalar_normalization"), "frozen normalization")
    expected_normalization_sha = normalization.get("sha256")
    _require(
        report.get("frozen_scalar_normalization_sha256") == expected_normalization_sha,
        "Internal-test scalar normalization is not the frozen train fit",
    )
    normalization_path, _ = _dataset_artifact(dataset_root, report, "scalar_normalization")
    _require(
        sha256_file(normalization_path) == expected_normalization_sha,
        "Materialized scalar normalization drift",
    )

    artifacts: dict[str, Path] = {}
    for key in (
        "internal_test_signals",
        "internal_test_scalar_features",
        "internal_test_targets",
        "internal_test_examples",
    ):
        artifacts[key], _ = _dataset_artifact(dataset_root, report, key)
    examples = _read_jsonl(artifacts["internal_test_examples"], "internal-test examples")
    descriptor = _object(
        _object(report.get("artifacts"), "Dataset artifacts").get("internal_test_examples"),
        "internal-test examples descriptor",
    )
    _require(descriptor.get("records") == len(examples), "Internal-test example count drift")
    counts = _object(
        _object(report.get("counts"), "Dataset counts").get("by_split"),
        "Dataset split counts",
    )
    split_counts = _object(counts.get("internal_test"), "internal-test counts")
    _require(split_counts.get("assets") == len(examples), "Internal-test count mismatch")
    sealed = _object(protocol.get("sealed_split"), "sealed split")
    _require(len(examples) == sealed.get("expected_assets"), "Frozen test asset count drift")

    seen_assets: set[str] = set()
    groups: set[str] = set()
    for row, example in enumerate(examples):
        _validate_example(example, row, report)
        asset_id = str(example["asset_id"])
        _require(asset_id not in seen_assets, f"Duplicate internal-test asset: {asset_id}")
        seen_assets.add(asset_id)
        groups.add(str(example["group_id"]))
        image = _object(example.get("image"), "example image")
        image_path = _asset_path(assets_root, str(image["path"]))
        if verify_image_hashes:
            _require(sha256_file(image_path) == image["sha256"], f"Image drift: {asset_id}")
    _require(
        len(groups) == sealed.get("expected_source_groups"),
        "Frozen test source-group count drift",
    )
    return report, examples, artifacts


def _signal_link(example: dict[str, Any]) -> dict[str, Any]:
    image = _object(example.get("image"), "example image")
    sequence = _object(example.get("sequence"), "example sequence")
    scalars = _object(example.get("scalar_features"), "example scalar features")
    row = int(example["row"])
    return {
        "asset_id": example["asset_id"],
        "group_id": example["group_id"],
        "image": image["path"],
        "image_sha256": image["sha256"],
        "signal": {
            "array": sequence["array"],
            "row": row,
            "length": sequence["length"],
            "available": sequence["signal_available"],
        },
        "scalar_features": {"array": scalars["array"], "row": row},
    }


def _instruction_records(
    examples: list[dict[str, Any]], dataset_report_sha256: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prompts = []
    answers = []
    manifest = []
    fusion_links = []
    for example in examples:
        for task in TASKS:
            pair_id = _instruction_id(
                dataset_report_sha256, "internal_test", example["asset_id"], task
            )
            for language in LANGUAGES:
                built = _task_record(example, task, language)
                if built is None:
                    continue
                prompt, response, supervision_source, modalities = built
                instruction_id = _localized_instruction_id(pair_id, language)
                _require(bool(_HEX_24.fullmatch(instruction_id)), "Invalid instruction ID")
                image_path = _portable_image_path(example["image"]["path"], int(example["row"]))
                prompts.append(
                    {
                        "schema_version": BUNDLE_PROMPT_SCHEMA,
                        "instruction_id": instruction_id,
                        "pair_id": pair_id,
                        "language": language,
                        "task": task,
                        "image": image_path,
                        "prompt": prompt,
                    }
                )
                answers.append(
                    {
                        "schema_version": FINAL_ANSWER_RECORD_SCHEMA,
                        "instruction_id": instruction_id,
                        "pair_id": pair_id,
                        "language": language,
                        "task": task,
                        "expected_response": response,
                    }
                )
                manifest.append(
                    {
                        "schema_version": FINAL_INSTRUCTION_MANIFEST_SCHEMA,
                        "instruction_id": instruction_id,
                        "pair_id": pair_id,
                        "language": language,
                        "split": "internal_test",
                        "task": task,
                        "asset_id": example["asset_id"],
                        "record_id": example["record_id"],
                        "group_id": example["group_id"],
                        "image_path": image_path,
                        "image_sha256": example["image"]["sha256"],
                        "image_width": int(example["image"]["width"]),
                        "image_height": int(example["image"]["height"]),
                        "target_peak_present": bool(example["target"]["peak_present"]),
                        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        "input_modalities": modalities,
                        "supervision_source": supervision_source,
                        "source_dataset_report_sha256": dataset_report_sha256,
                    }
                )
                fusion_links.append(
                    {
                        "schema_version": FUSION_LINK_SCHEMA,
                        "split": "internal_test",
                        "instruction_id": instruction_id,
                        "pair_id": pair_id,
                        "task": task,
                        "language": language,
                        **_signal_link(example),
                    }
                )
    ids = [str(row["instruction_id"]) for row in prompts]
    _require(len(ids) == len(set(ids)), "Duplicate final benchmark instruction IDs")
    _require(
        ids == [str(row["instruction_id"]) for row in answers]
        == [str(row["instruction_id"]) for row in manifest]
        == [str(row["instruction_id"]) for row in fusion_links],
        "Final benchmark instruction artifacts are misaligned",
    )
    return prompts, answers, manifest, fusion_links


def _detector_coco(
    examples: list[dict[str, Any]],
    assets_root: Path,
    *,
    access_id: str,
    dataset_report_sha256: str,
) -> dict[str, Any]:
    images = []
    annotations = []
    next_annotation = 1
    for example in examples:
        image = _object(example.get("image"), "example image")
        target = _object(example.get("target"), "example target")
        image_id = int(example["row"]) + 1
        source = _asset_path(assets_root, str(image["path"]))
        width = int(image["width"])
        height = int(image["height"])
        images.append(
            {
                "id": image_id,
                "file_name": str(source),
                "width": width,
                "height": height,
                "asset_id": example["asset_id"],
                "group_id": example["group_id"],
                "image_sha256": image["sha256"],
            }
        )
        if bool(target["peak_present"]):
            x1 = max(0, min(width - 1, round(float(target["start_normalized"]) * width)))
            x2 = max(x1 + 1, min(width, round(float(target["end_normalized"]) * width)))
            annotations.append(
                {
                    "id": next_annotation,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, 0, x2 - x1, height],
                    "area": (x2 - x1) * height,
                    "iscrowd": 0,
                }
            )
            next_annotation += 1
    return {
        "info": {
            "description": "sealed internal-test view",
            "split": "internal_test",
            "access_id": access_id,
            "dataset_report_sha256": dataset_report_sha256,
        },
        "licenses": [],
        "categories": [{"id": 0, "name": "peak", "supercategory": "peak"}],
        "images": images,
        "annotations": annotations,
    }


def _result_from_existing(
    *,
    output_dir: Path,
    expected_protocol_sha256: str,
    dataset_report_sha256: str,
    access_id: str,
) -> FinalBenchmarkDataResult | None:
    if not output_dir.exists():
        return None
    entries = _manifest_entries(output_dir)
    report_path = output_dir / "final_benchmark_data_report.json"
    _require(
        entries.get(report_path.name) == sha256_file(report_path),
        "Final benchmark data report is not manifest-bound",
    )
    report = _read_object(report_path, "final benchmark data report")
    _require(report.get("schema_version") == FINAL_BENCHMARK_DATA_SCHEMA, "Bad data schema")
    source = _object(report.get("source"), "final benchmark data source")
    _require(source.get("protocol_sha256") == expected_protocol_sha256, "Protocol drift")
    _require(source.get("dataset_report_sha256") == dataset_report_sha256, "Dataset drift")
    _require(source.get("access_id") == access_id, "Access event drift")
    roots = _object(report.get("roots"), "final benchmark roots")
    resolved = {
        name: _safe_relative(output_dir, roots.get(name), f"{name} root")
        for name in ("inference", "answers", "fusion", "detector")
    }
    for name, root in resolved.items():
        _require(root.is_dir(), f"Missing final benchmark {name} root")
        _manifest_entries(root)
        relative_manifest = (root / "artifact_manifest.sha256").relative_to(output_dir).as_posix()
        _require(
            entries.get(relative_manifest) == sha256_file(root / "artifact_manifest.sha256"),
            f"Final benchmark {name} manifest drift",
        )
    counts = _object(report.get("counts"), "final benchmark counts")
    return FinalBenchmarkDataResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        manifest_path=output_dir / "artifact_manifest.sha256",
        inference_root=resolved["inference"],
        answer_root=resolved["answers"],
        fusion_root=resolved["fusion"],
        detector_root=resolved["detector"],
        assets=int(counts["assets"]),
        prompts=int(counts["prompts"]),
        source_groups=int(counts["source_groups"]),
        access_id=access_id,
        cached=True,
    )


def build_final_benchmark_data(
    *,
    protocol_root: Path,
    expected_protocol_sha256: str,
    ledger_dir: Path,
    dataset_root: Path,
    expected_dataset_report_sha256: str,
    assets_root: Path,
    output_dir: Path,
    verify_image_hashes: bool = True,
) -> FinalBenchmarkDataResult:
    """Build isolated final-evaluation artifacts under the sole access event."""

    protocol_root = protocol_root.resolve()
    ledger_dir = ledger_dir.resolve()
    dataset_root = dataset_root.resolve()
    assets_root = assets_root.resolve()
    output_dir = output_dir.resolve()
    _require(assets_root.is_dir(), f"Assets root not found: {assets_root}")
    protocol, candidate_lock, access_id = _access_context(
        protocol_root, expected_protocol_sha256, ledger_dir
    )
    cached = _result_from_existing(
        output_dir=output_dir,
        expected_protocol_sha256=expected_protocol_sha256,
        dataset_report_sha256=expected_dataset_report_sha256,
        access_id=access_id,
    )
    if cached is not None:
        return cached

    dataset_report, examples, _ = _load_internal_dataset(
        dataset_root=dataset_root,
        expected_dataset_report_sha256=expected_dataset_report_sha256,
        candidate_lock=candidate_lock,
        protocol=protocol,
        assets_root=assets_root,
        verify_image_hashes=verify_image_hashes,
    )
    prompts, answers, instruction_manifest, fusion_links = _instruction_records(
        examples, expected_dataset_report_sha256
    )
    groups = {str(row["group_id"]) for row in examples}
    task_counts = dict(sorted(Counter(str(row["task"]) for row in prompts).items()))
    language_counts = dict(
        sorted(Counter(str(row["language"]) for row in prompts).items())
    )
    positive_assets = sum(bool(row["target"]["peak_present"]) for row in examples)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        inference_root = staging / "inference"
        answer_root = staging / "answers"
        fusion_root = staging / "fusion"
        detector_root = staging / "detector"
        for root in (inference_root, answer_root, fusion_root, detector_root):
            root.mkdir()

        prompt_path = inference_root / "inference_prompts.jsonl"
        _write_jsonl(prompt_path, prompts)
        inference_report_path = inference_root / "inference_bundle_report.json"
        _write_json(
            inference_report_path,
            {
                "schema_version": FINAL_INFERENCE_BUNDLE_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                    "asset_index_sha256": dataset_report.get("asset_index_sha256"),
                },
                "counts": {
                    "prompts": len(prompts),
                    "independent_assets": len(examples),
                    "source_groups": len(groups),
                    "by_task": task_counts,
                    "by_language": language_counts,
                },
                "contracts": {
                    "split": "internal_test",
                    "prompt_only": True,
                    "answer_key_materialized_in_this_root": False,
                    "one_image_token_per_prompt": True,
                    "language_variants_are_not_independent_assets": True,
                },
                "artifacts": {
                    "inference_prompts": {
                        "path": prompt_path.name,
                        "sha256": sha256_file(prompt_path),
                        "records": len(prompts),
                    }
                },
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
        )
        inference_manifest = _write_manifest(
            inference_root, (prompt_path, inference_report_path)
        )

        answers_path = answer_root / "answers.jsonl"
        instruction_manifest_path = answer_root / "instruction_manifest.jsonl"
        _write_jsonl(answers_path, answers)
        _write_jsonl(instruction_manifest_path, instruction_manifest)
        answer_report_path = answer_root / "answer_key_report.json"
        _write_json(
            answer_report_path,
            {
                "schema_version": FINAL_ANSWER_REPORT_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                },
                "counts": {
                    "answers": len(answers),
                    "manifest_records": len(instruction_manifest),
                    "independent_assets": len(examples),
                    "source_groups": len(groups),
                    "by_task": task_counts,
                    "by_language": language_counts,
                },
                "contracts": {
                    "split": "internal_test",
                    "separate_from_generation_root": True,
                    "instruction_order_matches_prompt_bundle": True,
                    "threshold_selection_forbidden": True,
                },
                "artifacts": {
                    "answers": {
                        "path": answers_path.name,
                        "sha256": sha256_file(answers_path),
                        "records": len(answers),
                    },
                    "instruction_manifest": {
                        "path": instruction_manifest_path.name,
                        "sha256": sha256_file(instruction_manifest_path),
                        "records": len(instruction_manifest),
                    },
                },
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
        )
        answer_manifest = _write_manifest(
            answer_root, (answers_path, instruction_manifest_path, answer_report_path)
        )

        fusion_links_path = fusion_root / "internal_test_xic_links.jsonl"
        _write_jsonl(fusion_links_path, fusion_links)
        fusion_report_path = fusion_root / "fusion_bundle_report.json"
        _write_json(
            fusion_report_path,
            {
                "schema_version": FINAL_FUSION_BUNDLE_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                    "inference_bundle_report_sha256": sha256_file(inference_report_path),
                },
                "counts": {
                    "prompt_links": len(fusion_links),
                    "independent_assets": len(examples),
                    "source_groups": len(groups),
                },
                "contracts": {
                    "split": "internal_test",
                    "join_key": "instruction_id_and_image_path",
                    "answer_key_opened_by_generation": False,
                    "signals_external_to_bundle": True,
                },
                "artifacts": {
                    "internal_test_xic_links": {
                        "path": fusion_links_path.name,
                        "sha256": sha256_file(fusion_links_path),
                        "records": len(fusion_links),
                    }
                },
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
        )
        fusion_manifest = _write_manifest(
            fusion_root, (fusion_links_path, fusion_report_path)
        )

        coco_dir = detector_root / "coco" / "val"
        coco_dir.mkdir(parents=True)
        coco_path = coco_dir / "val_coco.json"
        coco = _detector_coco(
            examples,
            assets_root,
            access_id=access_id,
            dataset_report_sha256=expected_dataset_report_sha256,
        )
        _write_json(coco_path, coco)
        detector_report_path = detector_root / "detector_dataset_report.json"
        _write_json(
            detector_report_path,
            {
                "schema_version": FINAL_DETECTOR_DATASET_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                    "asset_index_sha256": dataset_report.get("asset_index_sha256"),
                },
                "split": {
                    "name": "internal_test",
                    "assets": len(examples),
                    "positive_assets": positive_assets,
                    "negative_assets": len(examples) - positive_assets,
                    "source_groups": len(groups),
                    "annotations": len(coco["annotations"]),
                },
                "contracts": {
                    "upstream_layout_alias": "coco/val/val_coco.json",
                    "semantic_split": "internal_test",
                    "absolute_verified_image_paths": True,
                    "threshold_selection_forbidden": True,
                },
                "artifacts": {
                    "internal_test_coco": {
                        "path": coco_path.relative_to(detector_root).as_posix(),
                        "sha256": sha256_file(coco_path),
                        "images": len(coco["images"]),
                        "annotations": len(coco["annotations"]),
                    }
                },
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
        )
        detector_manifest = _write_manifest(
            detector_root, (coco_path, detector_report_path)
        )

        report_path = staging / "final_benchmark_data_report.json"
        _write_json(
            report_path,
            {
                "schema_version": FINAL_BENCHMARK_DATA_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "protocol_sha256": expected_protocol_sha256,
                    "access_id": access_id,
                    "dataset_report_sha256": expected_dataset_report_sha256,
                    "asset_index_sha256": dataset_report.get("asset_index_sha256"),
                },
                "counts": {
                    "assets": len(examples),
                    "source_groups": len(groups),
                    "prompts": len(prompts),
                    "answers": len(answers),
                    "positive_assets": positive_assets,
                    "negative_assets": len(examples) - positive_assets,
                    "by_task": task_counts,
                    "by_language": language_counts,
                },
                "roots": {
                    "inference": inference_root.relative_to(staging).as_posix(),
                    "answers": answer_root.relative_to(staging).as_posix(),
                    "fusion": fusion_root.relative_to(staging).as_posix(),
                    "detector": detector_root.relative_to(staging).as_posix(),
                },
                "contracts": {
                    "one_time_access_ledger_verified": True,
                    "prompt_answer_roots_disjoint": True,
                    "frozen_train_scalar_normalization_reused": True,
                    "all_image_hashes_verified": verify_image_hashes,
                    "threshold_selection_forbidden": True,
                },
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
        )
        top_manifest = _write_manifest(
            staging,
            (
                report_path,
                inference_manifest,
                answer_manifest,
                fusion_manifest,
                detector_manifest,
            ),
        )
        _require(top_manifest.is_file(), "Top-level data manifest was not written")
        staging.replace(output_dir)

    return FinalBenchmarkDataResult(
        output_dir=output_dir,
        report_path=output_dir / "final_benchmark_data_report.json",
        report_sha256=sha256_file(output_dir / "final_benchmark_data_report.json"),
        manifest_path=output_dir / "artifact_manifest.sha256",
        inference_root=output_dir / "inference",
        answer_root=output_dir / "answers",
        fusion_root=output_dir / "fusion",
        detector_root=output_dir / "detector",
        assets=len(examples),
        prompts=len(prompts),
        source_groups=len(groups),
        access_id=access_id,
        cached=False,
    )
