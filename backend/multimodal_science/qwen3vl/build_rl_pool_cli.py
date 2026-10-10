"""Build the train-only, source-group-isolated v2 RL pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.rl_data import build_rl_pool


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--code-revision", required=True)
    command.add_argument("--lora-bundle-root", type=Path, required=True)
    command.add_argument("--lora-bundle-report-sha256", required=True)
    command.add_argument("--fusion-bundle-root", type=Path, required=True)
    command.add_argument("--fusion-bundle-report-sha256", required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=17)
    return command


def main() -> None:
    args = parser().parse_args()
    result = build_rl_pool(
        code_revision=args.code_revision,
        lora_bundle_root=args.lora_bundle_root,
        lora_bundle_report_sha256=args.lora_bundle_report_sha256,
        fusion_bundle_root=args.fusion_bundle_root,
        fusion_bundle_report_sha256=args.fusion_bundle_report_sha256,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "training_pool_path": str(result.training_pool_path),
                "selection_set_path": str(result.selection_set_path),
                "training_records": result.training_records,
                "selection_records": result.selection_records,
                "training_source_groups": result.training_source_groups,
                "selection_source_groups": result.selection_source_groups,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

