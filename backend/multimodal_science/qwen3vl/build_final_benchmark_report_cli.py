"""Build the five-candidate sealed-test report and final evidence registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_report import (
    build_final_benchmark_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--zero-shot-root", type=Path, required=True)
    parser.add_argument("--zero-shot-report-sha256", required=True)
    parser.add_argument("--image-lora-root", type=Path, required=True)
    parser.add_argument("--image-lora-report-sha256", required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--fusion-report-sha256", required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--sequence-report-sha256", required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--detector-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_final_benchmark_report(
        protocol_root=args.protocol_root,
        expected_protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        qwen_evaluation_roots={
            "qwen3vl_zero_shot": args.zero_shot_root,
            "qwen3vl_image_lora": args.image_lora_root,
            "qwen3vl_image_xic_fusion": args.fusion_root,
        },
        qwen_evaluation_report_sha256={
            "qwen3vl_zero_shot": args.zero_shot_report_sha256,
            "qwen3vl_image_lora": args.image_lora_report_sha256,
            "qwen3vl_image_xic_fusion": args.fusion_report_sha256,
        },
        sequence_evaluation_root=args.sequence_root,
        expected_sequence_report_sha256=args.sequence_report_sha256,
        detector_evaluation_root=args.detector_root,
        expected_detector_report_sha256=args.detector_report_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "table_path": str(result.table_path),
                "registry_path": str(result.registry_path),
                "manifest_path": str(result.manifest_path),
                "access_id": result.access_id,
                "candidates": result.candidates,
                "final_benchmark_eligible": True,
                "internal_test_accessed": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
