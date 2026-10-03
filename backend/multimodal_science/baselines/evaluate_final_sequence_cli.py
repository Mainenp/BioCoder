"""Evaluate a frozen SequencePeakNet candidate on the sealed test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.baselines.final_sequence_evaluation import (
    evaluate_final_sequence_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=("sequence_peak_net", "sequence_peak_net_metadata"),
        required=True,
    )
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    result = evaluate_final_sequence_candidate(
        protocol_root=args.protocol_root,
        expected_protocol_sha256=args.protocol_sha256,
        ledger_dir=args.ledger_dir,
        candidate_name=args.candidate,
        dataset_root=args.dataset_root,
        expected_dataset_report_sha256=args.dataset_report_sha256,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "manifest_path": str(result.manifest_path),
                "predictions_path": str(result.predictions_path),
                "assets": result.assets,
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
