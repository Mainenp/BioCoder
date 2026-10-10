"""Run answer-isolated Qwen3-VL image/XIC validation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_inference import (
    XIC_INTERVENTIONS,
    run_fusion_inference,
)
from multimodal_science.qwen3vl.inference import AdapterSpec, GenerationSettings


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validation predictions with a trained image/XIC fusion adapter"
    )
    parser.add_argument("--inference-bundle-root", type=Path, required=True)
    parser.add_argument("--inference-bundle-report-sha256", required=True)
    parser.add_argument("--fusion-bundle-root", type=Path, required=True)
    parser.add_argument("--fusion-bundle-report-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-report-sha256", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--fusion-adapter-root", type=Path, required=True)
    parser.add_argument("--fusion-training-report-sha256", required=True)
    parser.add_argument("--fusion-adapter-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="single_cuda")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"))
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--xic-intervention",
        choices=XIC_INTERVENTIONS,
        default="aligned",
    )
    parser.add_argument("--xic-intervention-seed", type=int, default=17)
    parser.add_argument(
        "--input-modality",
        choices=("image_xic", "xic_only"),
        default="image_xic",
    )
    return parser


def main() -> None:
    arguments = parser().parse_args()
    result = run_fusion_inference(
        inference_bundle_root=arguments.inference_bundle_root,
        inference_bundle_report_sha256=arguments.inference_bundle_report_sha256,
        fusion_bundle_root=arguments.fusion_bundle_root,
        fusion_bundle_report_sha256=arguments.fusion_bundle_report_sha256,
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        assets_root=arguments.assets_root,
        fusion_adapter=AdapterSpec(
            root=arguments.fusion_adapter_root,
            training_report_sha256=arguments.fusion_training_report_sha256,
            manifest_sha256=arguments.fusion_adapter_manifest_sha256,
        ),
        output_dir=arguments.output_dir,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        model_artifact_sha256=arguments.model_artifact_sha256,
        settings=GenerationSettings(
            batch_size=1,
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
        xic_intervention=arguments.xic_intervention,
        xic_intervention_seed=arguments.xic_intervention_seed,
        input_modality=arguments.input_modality,
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
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
