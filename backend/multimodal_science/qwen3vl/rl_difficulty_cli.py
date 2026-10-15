"""Run train-only G=8 difficulty screening for v2 GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.rl_data import load_rl_training_pool
from multimodal_science.qwen3vl.rl_difficulty import run_difficulty_screening
from multimodal_science.qwen3vl.rl_rollout import (
    XicSamplingRuntime,
    verify_train_signals,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--rl-pool-root", type=Path, required=True)
    command.add_argument("--rl-pool-report-sha256", required=True)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--starting-adapter-root", type=Path, required=True)
    command.add_argument("--starting-adapter-report-sha256", required=True)
    command.add_argument("--starting-adapter-manifest-sha256", required=True)
    command.add_argument("--model-name-or-path", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--model-artifact-sha256", required=True)
    command.add_argument("--model-manifest-path", type=Path, required=True)
    command.add_argument("--model-manifest-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--max-records", type=int)
    return command


def main() -> None:
    args = parser().parse_args()
    pool = load_rl_training_pool(args.rl_pool_root, args.rl_pool_report_sha256)
    if pool.report["sources"].get("dataset_report_sha256") != args.dataset_report_sha256:
        raise ValueError("RL pool and staged Dataset disagree")
    train_signals = verify_train_signals(args.dataset_root, args.dataset_report_sha256)
    runtime = XicSamplingRuntime(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        model_artifact_sha256=args.model_artifact_sha256,
        model_manifest_path=args.model_manifest_path,
        model_manifest_sha256=args.model_manifest_sha256,
        starting_adapter=AdapterSpec(
            root=args.starting_adapter_root,
            training_report_sha256=args.starting_adapter_report_sha256,
            manifest_sha256=args.starting_adapter_manifest_sha256,
        ),
        train_signals=train_signals,
    )
    try:
        result = run_difficulty_screening(
            code_revision=args.code_revision,
            rl_pool_root=args.rl_pool_root,
            rl_pool_report_sha256=args.rl_pool_report_sha256,
            runtime=runtime,
            output_dir=args.output_dir,
            max_records=args.max_records,
            progress_callback=lambda current, total: print(
                f"[difficulty {current}/{total}]", flush=True
            ),
        )
    finally:
        runtime.close()
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "selected_pool_path": str(result.selected_pool_path),
                "rollout_path": str(result.rollout_path),
                "input_records": result.input_records,
                "selected_records": result.selected_records,
                "informative_records": result.informative_records,
                "development_screening_complete": result.development_screening_complete,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

