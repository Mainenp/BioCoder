"""Freeze external model artifacts into a production deployment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.deployment_runtime import (
    DEPLOYMENT_SCHEMA,
    verify_deployment_manifest,
)


def _directory_manifest(root: Path, destination: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Cannot freeze an empty directory: {root}")
    destination.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return sha256_file(destination)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Freeze one hash-bound Qwen/XIC deployment for BioCoder Agent."
    )
    command.add_argument("--deployment-id", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-root", type=Path, required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--adapter-dir", type=Path, required=True)
    command.add_argument("--projector-path", type=Path, required=True)
    command.add_argument("--processor-dir", type=Path, required=True)
    command.add_argument("--training-report-path", type=Path, required=True)
    command.add_argument("--input-modality", choices=("image_xic", "xic_only"), required=True)
    command.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    command.add_argument("--input-points", type=int, default=160)
    command.add_argument("--hidden-size", type=int, default=2560)
    command.add_argument("--sensor-tokens", type=int, default=4)
    command.add_argument("--base-channels", type=int, default=32)
    command.add_argument("--dropout", type=float, default=0.1)
    command.add_argument("--min-pixels", type=int, default=28 * 28)
    command.add_argument("--max-pixels", type=int, default=1_003_520)
    command.add_argument("--max-new-tokens", type=int, default=64)
    return command


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Deployment output already exists: {output}")
    output.mkdir(parents=True)
    model_root = args.model_root.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    projector = args.projector_path.expanduser().resolve()
    processor_dir = args.processor_dir.expanduser().resolve()
    report = args.training_report_path.expanduser().resolve()
    for path, label, directory in (
        (model_root, "model root", True),
        (adapter_dir, "adapter directory", True),
        (projector, "projector", False),
        (processor_dir, "processor directory", True),
        (report, "training report", False),
    ):
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"Missing {label}: {path}")

    model_manifest = output / "model.files.sha256"
    processor_manifest = output / "processor.files.sha256"
    model_manifest_sha = _directory_manifest(model_root, model_manifest)
    processor_manifest_sha = _directory_manifest(processor_dir, processor_manifest)
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    manifest = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "deployment_id": args.deployment_id,
        "model": {
            "root": str(model_root),
            "revision": args.model_revision,
            "manifest_path": str(model_manifest),
            "manifest_sha256": model_manifest_sha,
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir),
            "adapter_config_sha256": sha256_file(adapter_config),
            "adapter_weights_sha256": sha256_file(adapter_weights),
            "projector_path": str(projector),
            "projector_sha256": sha256_file(projector),
            "processor_dir": str(processor_dir),
            "processor_manifest_path": str(processor_manifest),
            "processor_manifest_sha256": processor_manifest_sha,
        },
        "runtime": {
            "input_modality": args.input_modality,
            "attention_implementation": args.attention_implementation,
            "sensor_projector": {
                "input_points": args.input_points,
                "hidden_size": args.hidden_size,
                "sensor_tokens": args.sensor_tokens,
                "base_channels": args.base_channels,
                "dropout": args.dropout,
            },
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
        },
        "provenance": {
            "report_path": str(report),
            "report_sha256": sha256_file(report),
            "internal_test_accessed": False,
            "validation_answers_opened": False,
            "validation_prompts_opened": False,
        },
        "contracts": {
            "user_samples_only": True,
            "internal_test_accessed": False,
            "validation_answers_opened": False,
            "validation_prompts_opened": False,
        },
    }
    manifest_path = output / "deployment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = verify_deployment_manifest(manifest_path)
    result = {
        "deployment_id": verified.deployment_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "input_modality": verified.input_modality,
        "internal_test_accessed": False,
        "validation_answers_opened": False,
        "validation_prompts_opened": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
