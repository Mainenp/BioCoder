"""Run image-LoRA validation with frozen SequencePeakNet predictions in the prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import AdapterSpec, GenerationSettings
from multimodal_science.qwen3vl.sequence_prompt_inference import (
    run_sequence_prompt_inference,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--inference-bundle-root", type=Path, required=True)
    command.add_argument("--inference-bundle-report-sha256", required=True)
    command.add_argument("--fusion-bundle-root", type=Path, required=True)
    command.add_argument("--fusion-bundle-report-sha256", required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--image-adapter-root", type=Path, required=True)
    command.add_argument("--image-adapter-report-sha256", required=True)
    command.add_argument("--image-adapter-manifest-sha256", required=True)
    command.add_argument("--sequence-prompt-bundle-root", type=Path, required=True)
    command.add_argument("--sequence-prompt-bundle-report-sha256", required=True)
    command.add_argument("--sequence-prompt-bundle-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--max-new-tokens", type=int, default=64)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--dtype", default="bfloat16")
    command.add_argument("--device-map", default="auto")
    command.add_argument("--attention-implementation", choices=("sdpa", "eager"))
    command.add_argument("--max-records", type=int)
    command.add_argument("--resume", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = run_sequence_prompt_inference(
        inference_bundle_root=arguments.inference_bundle_root,
        inference_bundle_report_sha256=arguments.inference_bundle_report_sha256,
        fusion_bundle_root=arguments.fusion_bundle_root,
        fusion_bundle_report_sha256=arguments.fusion_bundle_report_sha256,
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        assets_root=arguments.assets_root,
        image_adapter=AdapterSpec(
            root=arguments.image_adapter_root,
            training_report_sha256=arguments.image_adapter_report_sha256,
            manifest_sha256=arguments.image_adapter_manifest_sha256,
        ),
        sequence_prompt_bundle_root=arguments.sequence_prompt_bundle_root,
        sequence_prompt_bundle_report_sha256=(
            arguments.sequence_prompt_bundle_report_sha256
        ),
        sequence_prompt_bundle_manifest_sha256=(
            arguments.sequence_prompt_bundle_manifest_sha256
        ),
        output_dir=arguments.output_dir,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        settings=GenerationSettings(
            batch_size=arguments.batch_size,
            max_new_tokens=arguments.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            seed=arguments.seed,
            dtype=arguments.dtype,
            device_map=arguments.device_map,
            attention_implementation=arguments.attention_implementation,
        ),
        max_records=arguments.max_records,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "predictions_path": str(result.predictions_path),
                "prediction_records": result.prediction_records,
                "complete_prompt_coverage": result.complete_prompt_coverage,
                "development_comparison_candidate": (
                    result.development_comparison_candidate
                ),
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
