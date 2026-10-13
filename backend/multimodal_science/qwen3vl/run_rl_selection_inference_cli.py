"""Run answer-free greedy XIC-only inference for one v2 RL selection candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.rl_selection_inference import run_selection_inference


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--code-revision", required=True)
    result.add_argument("--bundle-root", required=True, type=Path)
    result.add_argument("--bundle-report-sha256", required=True)
    result.add_argument("--dataset-root", required=True, type=Path)
    result.add_argument("--dataset-report-sha256", required=True)
    result.add_argument("--starting-adapter-root", required=True, type=Path)
    result.add_argument("--starting-adapter-report-sha256", required=True)
    result.add_argument("--starting-adapter-manifest-sha256", required=True)
    result.add_argument("--candidate-json", required=True, type=Path)
    result.add_argument("--candidate-json-sha256", required=True)
    result.add_argument("--model-name-or-path", required=True)
    result.add_argument("--model-revision", required=True)
    result.add_argument("--model-artifact-sha256", required=True)
    result.add_argument("--model-manifest-path", required=True, type=Path)
    result.add_argument("--model-manifest-sha256", required=True)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    result = run_selection_inference(
        code_revision=args.code_revision,
        bundle_root=args.bundle_root,
        bundle_report_sha256=args.bundle_report_sha256,
        dataset_root=args.dataset_root,
        dataset_report_sha256=args.dataset_report_sha256,
        starting_adapter=AdapterSpec(
            root=args.starting_adapter_root,
            training_report_sha256=args.starting_adapter_report_sha256,
            manifest_sha256=args.starting_adapter_manifest_sha256,
        ),
        candidate_path=args.candidate_json,
        candidate_descriptor_sha256=args.candidate_json_sha256,
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        model_artifact_sha256=args.model_artifact_sha256,
        model_manifest_path=args.model_manifest_path,
        model_manifest_sha256=args.model_manifest_sha256,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.__dict__.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
