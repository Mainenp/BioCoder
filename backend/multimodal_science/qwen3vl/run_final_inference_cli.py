"""Run a frozen zero-shot or image-LoRA candidate on the sealed test prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import (
    AdapterSpec,
    FinalBenchmarkAccessSpec,
    GenerationSettings,
    run_qwen_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=("qwen3vl_zero_shot", "qwen3vl_image_lora"),
        required=True,
    )
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-report-sha256", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--adapter-training-report-sha256")
    parser.add_argument("--adapter-manifest-sha256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention-implementation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    adapter_values = (
        args.adapter_root,
        args.adapter_training_report_sha256,
        args.adapter_manifest_sha256,
    )
    if args.candidate == "qwen3vl_image_lora":
        if not all(value is not None for value in adapter_values):
            raise SystemExit("The frozen image-LoRA candidate requires all adapter arguments")
        adapter = AdapterSpec(
            root=args.adapter_root,
            training_report_sha256=args.adapter_training_report_sha256,
            manifest_sha256=args.adapter_manifest_sha256,
        )
    else:
        if any(value is not None for value in adapter_values):
            raise SystemExit("The frozen zero-shot candidate forbids adapter arguments")
        adapter = None

    result = run_qwen_inference(
        args.bundle_root,
        args.assets_root,
        args.output_dir,
        expected_bundle_report_sha256=args.bundle_report_sha256,
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        model_artifact_sha256=args.model_artifact_sha256,
        adapter=adapter,
        settings=GenerationSettings(
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed,
            dtype=args.dtype,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
        ),
        resume=args.resume,
        final_benchmark_access=FinalBenchmarkAccessSpec(
            protocol_root=args.protocol_root,
            protocol_sha256=args.protocol_sha256,
            ledger_dir=args.ledger_dir,
            candidate_name=args.candidate,
        ),
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
