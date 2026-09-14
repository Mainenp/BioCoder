"""Build answer-isolated image/XIC links for Qwen3-VL sensor fusion."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.inference_bundle import BUNDLE_SCHEMA
from multimodal_science.qwen3vl.lora_data import LORA_BUNDLE_SCHEMA


FUSION_BUNDLE_SCHEMA = "chrompeak-qwen3vl-xic-fusion-bundle-v1"
FUSION_LINK_SCHEMA = "chrompeak-qwen3vl-xic-link-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FusionBundleResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    train_links_path: Path
    validation_links_path: Path
    train_links: int
    validation_links: int


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


def _artifact(
    root: Path,
    report: dict[str, Any],
    key: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    artifacts = _object(report.get("artifacts"), f"{label} artifacts")
    record = _object(artifacts.get(key), f"{label} artifact {key}")
    relative = Path(str(record.get("path") or ""))
    _require(not relative.is_absolute() and str(relative) != ".", f"Bad {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} artifact escapes its root") from error
    _require(path.is_file(), f"Missing {label} artifact: {path}")
    _require(sha256_file(path) == record.get("sha256"), f"{label} hash mismatch")
    return path, record


def _dataset_examples(
    dataset_root: Path,
    dataset_report: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    signals_path, signals_artifact = _artifact(
        dataset_root,
        dataset_report,
        f"{split}_signals",
        label=f"{split} signals",
    )
    scalars_path, scalars_artifact = _artifact(
        dataset_root,
        dataset_report,
        f"{split}_scalar_features",
        label=f"{split} scalar features",
    )
    path, artifact = _artifact(
        dataset_root,
        dataset_report,
        f"{split}_examples",
        label=f"{split} examples",
    )
    records = _read_jsonl(path, f"{split} examples")
    _require(artifact.get("records") == len(records), f"{split} example count mismatch")
    target_points = dataset_report.get("target_points")
    _require(isinstance(target_points, int) and target_points > 0, "Invalid target points")
    _require(
        signals_artifact.get("shape") == [len(records), target_points],
        f"{split} signal shape mismatch",
    )
    scalar_shape = scalars_artifact.get("shape")
    _require(
        isinstance(scalar_shape, list)
        and len(scalar_shape) == 2
        and scalar_shape[0] == len(records),
        f"{split} scalar shape mismatch",
    )
    for row, record in enumerate(records):
        _require(record.get("row") == row, f"Non-contiguous {split} row")
        _require(record.get("split") == split, f"{split} example split mismatch")
        _require(
            record.get("schema_version") == "chrompeak-multimodal-example-v1",
            f"Unsupported {split} example schema",
        )
        sequence = _object(record.get("sequence"), f"{split} sequence row {row}")
        scalars = _object(record.get("scalar_features"), f"{split} scalars row {row}")
        image = _object(record.get("image"), f"{split} image row {row}")
        _require(str(record.get("asset_id") or "") != "", f"Missing {split} asset ID")
        _require(str(record.get("group_id") or "") != "", f"Missing {split} group ID")
        _require(str(image.get("path") or "") != "", f"Missing {split} image path")
        _require(
            bool(_HEX_64.fullmatch(str(image.get("sha256") or ""))),
            f"Invalid {split} image hash",
        )
        _require(
            sequence.get("array") == signals_path.relative_to(dataset_root).as_posix(),
            f"{split} sequence artifact mismatch",
        )
        _require(sequence.get("length") == target_points, f"{split} sequence length mismatch")
        _require(
            scalars.get("array") == scalars_path.relative_to(dataset_root).as_posix(),
            f"{split} scalar artifact mismatch",
        )
    return records


def _signal_link(example: dict[str, Any]) -> dict[str, Any]:
    sequence = _object(example.get("sequence"), "example sequence")
    scalars = _object(example.get("scalar_features"), "example scalar features")
    image = _object(example.get("image"), "example image")
    row = example.get("row")
    _require(
        isinstance(row, int)
        and sequence.get("row") == row
        and scalars.get("row") == row,
        "Example array rows disagree",
    )
    signal_available = sequence.get("signal_available")
    _require(isinstance(signal_available, bool), "Missing XIC availability flag")
    return {
        "asset_id": example.get("asset_id"),
        "group_id": example.get("group_id"),
        "image": image.get("path"),
        "image_sha256": image.get("sha256"),
        "signal": {
            "array": sequence.get("array"),
            "row": row,
            "length": sequence.get("length"),
            "available": signal_available,
        },
        "scalar_features": {"array": scalars.get("array"), "row": row},
    }


def build_fusion_bundle(
    *,
    dataset_root: Path,
    dataset_report_sha256: str,
    lora_bundle_root: Path,
    lora_bundle_report_sha256: str,
    inference_bundle_root: Path,
    inference_bundle_report_sha256: str,
    output_dir: Path,
) -> FusionBundleResult:
    """Join train instructions and validation prompts to XIC rows without answers."""

    for digest, label in (
        (dataset_report_sha256, "Dataset"),
        (lora_bundle_report_sha256, "LoRA bundle"),
        (inference_bundle_report_sha256, "inference bundle"),
    ):
        _require(bool(_HEX_64.fullmatch(digest)), f"Invalid {label} SHA-256")
    dataset_root = dataset_root.resolve()
    lora_bundle_root = lora_bundle_root.resolve()
    inference_bundle_root = inference_bundle_root.resolve()
    output_dir = output_dir.resolve()

    dataset_path = dataset_root / "dataset_report.json"
    lora_path = lora_bundle_root / "lora_bundle_report.json"
    inference_path = inference_bundle_root / "inference_bundle_report.json"
    dataset = _read_object(dataset_path, "Dataset report")
    lora = _read_object(lora_path, "LoRA bundle report")
    inference = _read_object(inference_path, "inference bundle report")
    _require(sha256_file(dataset_path) == dataset_report_sha256, "Dataset hash mismatch")
    _require(sha256_file(lora_path) == lora_bundle_report_sha256, "LoRA bundle hash mismatch")
    _require(
        sha256_file(inference_path) == inference_bundle_report_sha256,
        "Inference bundle hash mismatch",
    )
    _require(
        dataset.get("schema_version") == "chrompeak-multimodal-dataset-v1",
        "Unsupported Dataset schema",
    )
    _require(lora.get("schema_version") == LORA_BUNDLE_SCHEMA, "Unsupported LoRA bundle")
    _require(
        inference.get("schema_version") == BUNDLE_SCHEMA,
        "Unsupported inference bundle",
    )
    _require(
        lora.get("internal_test_accessed") is False
        and inference.get("internal_test_accessed") is False,
        "Fusion inputs accessed the internal test",
    )
    lora_source = _object(lora.get("source"), "LoRA source")
    inference_source = _object(inference.get("source"), "inference source")
    _require(
        lora_source.get("dataset_report_sha256") == dataset_report_sha256
        and inference_source.get("source_dataset_report_sha256")
        == dataset_report_sha256,
        "Fusion inputs do not share one Dataset",
    )

    train_examples = _dataset_examples(dataset_root, dataset, "train")
    validation_examples = _dataset_examples(dataset_root, dataset, "validation")
    train_by_asset = {str(item["asset_id"]): item for item in train_examples}
    validation_by_image = {
        str(_object(item.get("image"), "validation image")["path"]): item
        for item in validation_examples
    }
    _require(len(train_by_asset) == len(train_examples), "Duplicate train asset IDs")
    _require(
        len(validation_by_image) == len(validation_examples),
        "Duplicate validation image paths",
    )
    _require(
        not {str(item["group_id"]) for item in train_examples}
        & {str(item["group_id"]) for item in validation_examples},
        "Train/validation source-group leakage",
    )

    selection_path, selection_artifact = _artifact(
        lora_bundle_root, lora, "selection_manifest", label="LoRA selection"
    )
    prompt_path, prompt_artifact = _artifact(
        inference_bundle_root, inference, "inference_prompts", label="inference prompts"
    )
    selections = _read_jsonl(selection_path, "LoRA selection")
    prompts = _read_jsonl(prompt_path, "inference prompts")
    _require(selection_artifact.get("records") == len(selections), "Selection count mismatch")
    _require(prompt_artifact.get("records") == len(prompts), "Prompt count mismatch")
    _require(
        len({str(item.get("instruction_id") or "") for item in selections})
        == len(selections),
        "Duplicate train instruction IDs",
    )
    _require(
        len({str(item.get("instruction_id") or "") for item in prompts}) == len(prompts),
        "Duplicate validation instruction IDs",
    )

    train_links = []
    for selection in selections:
        asset_id = str(selection.get("asset_id") or "")
        _require(asset_id in train_by_asset, f"Unknown train asset: {asset_id}")
        link = _signal_link(train_by_asset[asset_id])
        _require(selection.get("image") == link["image"], "Train image/XIC identity mismatch")
        _require(
            selection.get("image_sha256") == link["image_sha256"],
            "Train image hash mismatch",
        )
        train_links.append(
            {
                "schema_version": FUSION_LINK_SCHEMA,
                "split": "train",
                "instruction_id": selection.get("instruction_id"),
                "task": selection.get("task"),
                "language": selection.get("language"),
                **link,
            }
        )

    validation_links = []
    for prompt in prompts:
        image_path = str(prompt.get("image") or "")
        _require(image_path in validation_by_image, f"Unknown validation image: {image_path}")
        validation_links.append(
            {
                "schema_version": FUSION_LINK_SCHEMA,
                "split": "validation",
                "instruction_id": prompt.get("instruction_id"),
                "pair_id": prompt.get("pair_id"),
                "task": prompt.get("task"),
                "language": prompt.get("language"),
                **_signal_link(validation_by_image[image_path]),
            }
        )

    _require(not output_dir.exists(), f"Fusion bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-staging-"
    ) as staging_name:
        staging = Path(staging_name)
        train_links_path = staging / "train_xic_links.jsonl"
        validation_links_path = staging / "validation_xic_links.jsonl"
        _write_jsonl(train_links_path, train_links)
        _write_jsonl(validation_links_path, validation_links)
        report_path = staging / "fusion_bundle_report.json"
        report = {
            "schema_version": FUSION_BUNDLE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "dataset_report_sha256": dataset_report_sha256,
                "lora_bundle_report_sha256": lora_bundle_report_sha256,
                "inference_bundle_report_sha256": inference_bundle_report_sha256,
                "asset_index_sha256": dataset.get("asset_index_sha256"),
            },
            "counts": {
                "train_instruction_links": len(train_links),
                "train_independent_assets": len({item["asset_id"] for item in train_links}),
                "validation_prompt_links": len(validation_links),
                "validation_independent_assets": len(
                    {item["asset_id"] for item in validation_links}
                ),
            },
            "contracts": {
                "join_key": "asset_id_train_and_image_path_validation",
                "train_split_only_for_supervision": True,
                "validation_answer_key_opened": False,
                "signals_external_to_bundle": True,
                "source_group_overlap": 0,
                "language_variants_are_not_independent_assets": True,
                "internal_test_accessed": False,
            },
            "artifacts": {
                "train_xic_links": {
                    "path": train_links_path.name,
                    "sha256": sha256_file(train_links_path),
                    "records": len(train_links),
                },
                "validation_xic_links": {
                    "path": validation_links_path.name,
                    "sha256": sha256_file(validation_links_path),
                    "records": len(validation_links),
                },
            },
            "development_training_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
        }
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (train_links_path, validation_links_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)

    final_report = output_dir / "fusion_bundle_report.json"
    return FusionBundleResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        train_links_path=output_dir / "train_xic_links.jsonl",
        validation_links_path=output_dir / "validation_xic_links.jsonl",
        train_links=len(train_links),
        validation_links=len(validation_links),
    )
