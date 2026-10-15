"""Run and evaluate the frozen detector on the sealed internal test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.final_detector_evaluation import (
    evaluate_final_detector_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--detector-data-root", type=Path, required=True)
    parser.add_argument("--detector-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    result = evaluate_final_detector_candidate(
        protocol_root=args.protocol_root,
        expected_protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        source_root=args.source_root,
        detector_data_root=args.detector_data_root,
        expected_detector_report_sha256=args.detector_report_sha256,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "predictions_path": str(result.predictions_path),
                "images": result.images,
                "predictions": result.predictions,
                "internal_test_source_groups": result.source_groups,
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
