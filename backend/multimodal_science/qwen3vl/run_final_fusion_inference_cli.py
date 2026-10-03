"""Run the single frozen image/XIC candidate on sealed internal-test prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_inference import run_fusion_inference
from multimodal_science.qwen3vl.inference import (
    AdapterSpec,
    FinalBenchmarkAccessSpec,
    GenerationSettings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
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
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    access = FinalBenchmarkAccessSpec(
        protocol_root=args.protocol_root,
        protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        candidate_name="qwen3vl_image_xic_fusion",
    )
    result = run_fusion_inference(
        inference_bundle_root=args.inference_bundle_root,
        inference_bundle_report_sha256=args.inference_bundle_report_sha256,
        fusion_bundle_root=args.fusion_bundle_root,
        fusion_bundle_report_sha256=args.fusion_bundle_report_sha256,
        dataset_root=args.dataset_root,
        dataset_report_sha256=args.dataset_report_sha256,
        assets_root=args.assets_root,
        fusion_adapter=AdapterSpec(
            root=args.fusion_adapter_root,
            training_report_sha256=args.fusion_training_report_sha256,
            manifest_sha256=args.fusion_adapter_manifest_sha256,
        ),
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        model_artifact_sha256=args.model_artifact_sha256,
        settings=GenerationSettings(
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed,
            dtype="bfloat16",
            device_map="single_cuda",
            attention_implementation=args.attention_implementation,
        ),
        resume=args.resume,
        xic_intervention="aligned",
        xic_intervention_seed=17,
        final_benchmark_access=access,
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
                "development_comparison_candidate": False,
                "final_benchmark_candidate": result.final_benchmark_candidate,
                "internal_test_accessed": result.internal_test_accessed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
