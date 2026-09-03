"""Run provenance-bound Qwen3-VL inference from a prompt-only bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import (
    GenerationSettings,
    run_qwen_inference,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--bundle-root", type=Path, required=True)
    command.add_argument("--bundle-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256")
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--max-new-tokens", type=int, default=64)
    command.add_argument("--do-sample", action="store_true")
    command.add_argument("--temperature", type=float)
    command.add_argument("--top-p", type=float)
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--dtype", default="auto")
    command.add_argument("--device-map", default="auto")
    command.add_argument("--attention-implementation")
    command.add_argument("--max-records", type=int)
    command.add_argument("--resume", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    settings = GenerationSettings(
        batch_size=arguments.batch_size,
        max_new_tokens=arguments.max_new_tokens,
        do_sample=arguments.do_sample,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        seed=arguments.seed,
        dtype=arguments.dtype,
        device_map=arguments.device_map,
        attention_implementation=arguments.attention_implementation,
    )
    result = run_qwen_inference(
        arguments.bundle_root,
        arguments.assets_root,
        arguments.output_dir,
        expected_bundle_report_sha256=arguments.bundle_report_sha256,
        model_name_or_path=arguments.model_name_or_path,
        model_revision=arguments.model_revision,
        settings=settings,
        model_artifact_sha256=arguments.model_artifact_sha256,
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
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
