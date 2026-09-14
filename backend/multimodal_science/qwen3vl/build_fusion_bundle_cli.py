"""Build a hash-bound image/XIC fusion bundle without validation answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.fusion_data import build_fusion_bundle


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--dataset-report-sha256", required=True)
    command.add_argument("--lora-bundle-root", type=Path, required=True)
    command.add_argument("--lora-bundle-report-sha256", required=True)
    command.add_argument("--inference-bundle-root", type=Path, required=True)
    command.add_argument("--inference-bundle-report-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_fusion_bundle(
        dataset_root=arguments.dataset_root,
        dataset_report_sha256=arguments.dataset_report_sha256,
        lora_bundle_root=arguments.lora_bundle_root,
        lora_bundle_report_sha256=arguments.lora_bundle_report_sha256,
        inference_bundle_root=arguments.inference_bundle_root,
        inference_bundle_report_sha256=arguments.inference_bundle_report_sha256,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "train_links": result.train_links,
                "validation_links": result.validation_links,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
