"""Evaluate a frozen Qwen candidate against the sealed-test answer key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_evaluation import (
    evaluate_final_qwen_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=(
            "qwen3vl_zero_shot",
            "qwen3vl_image_lora",
            "qwen3vl_image_xic_fusion",
        ),
        required=True,
    )
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--inference-report-sha256", required=True)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--answer-report-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--generation-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_final_qwen_predictions(
        protocol_root=args.protocol_root,
        expected_protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        candidate_name=args.candidate,
        inference_root=args.inference_root,
        expected_inference_report_sha256=args.inference_report_sha256,
        answer_root=args.answer_root,
        expected_answer_report_sha256=args.answer_report_sha256,
        predictions_path=args.predictions,
        generation_report_path=args.generation_report,
        expected_generation_report_sha256=args.generation_report_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "prediction_records": result.prediction_records,
                "valid_json_records": result.valid_json_records,
                "schema_valid_records": result.schema_valid_records,
                "internal_test_source_groups": result.source_groups,
                "candidate_name": result.candidate_name,
                "access_id": result.access_id,
                "final_benchmark_eligible": True,
                "internal_test_accessed": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
