"""CLI for access-bound internal-test artifact materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_data import build_final_benchmark_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-report-sha256", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-image-hash-verification", action="store_true")
    args = parser.parse_args()
    result = build_final_benchmark_data(
        protocol_root=args.protocol_root,
        expected_protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        dataset_root=args.dataset_root,
        expected_dataset_report_sha256=args.dataset_report_sha256,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        verify_image_hashes=not args.skip_image_hash_verification,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "inference_root": str(result.inference_root),
                "answer_root": str(result.answer_root),
                "fusion_root": str(result.fusion_root),
                "detector_root": str(result.detector_root),
                "assets": result.assets,
                "prompts": result.prompts,
                "source_groups": result.source_groups,
                "access_id": result.access_id,
                "cached": result.cached,
                "internal_test_accessed": True,
                "final_benchmark_eligible": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
