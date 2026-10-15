"""Run bounded image-plus-XIC Qwen3-VL LoRA/projector training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_smoke import run_fusion_smoke
from multimodal_science.qwen3vl.inference import AdapterSpec


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--fusion-bundle-root", type=Path, required=True)
    command.add_argument("--fusion-bundle-report-sha256", required=True)
    command.add_argument("--lora-bundle-root", type=Path, required=True)
    command.add_argument("--lora-bundle-report-sha256", required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--initial-adapter-root", type=Path, required=True)
    command.add_argument("--initial-adapter-report-sha256", required=True)
    command.add_argument("--initial-adapter-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--model-manifest-path", type=Path, required=True)
    command.add_argument("--model-manifest-sha256", required=True)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--max-records", type=int, default=8)
    command.add_argument("--max-steps", type=int, default=2)
    command.add_argument("--seed", type=int, default=17)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = run_fusion_smoke(
        fusion_bundle_root=arguments.fusion_bundle_root,
        fusion_bundle_report_sha256=arguments.fusion_bundle_report_sha256,
        lora_bundle_root=arguments.lora_bundle_root,
        lora_bundle_report_sha256=arguments.lora_bundle_report_sha256,
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        assets_root=arguments.assets_root,
        initial_adapter=AdapterSpec(
            root=arguments.initial_adapter_root,
            training_report_sha256=arguments.initial_adapter_report_sha256,
            manifest_sha256=arguments.initial_adapter_manifest_sha256,
        ),
        output_dir=arguments.output_dir,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        model_manifest_path=arguments.model_manifest_path,
        model_manifest_sha256=arguments.model_manifest_sha256,
        code_revision=arguments.code_revision,
        max_records=arguments.max_records,
        max_steps=arguments.max_steps,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "optimizer_updates": result.optimizer_updates,
                "development_comparison_eligible": False,
                "internal_test_accessed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
