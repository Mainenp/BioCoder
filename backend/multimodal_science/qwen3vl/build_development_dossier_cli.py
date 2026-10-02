"""Build the final pre-test multimodal development dossier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.development_dossier import (
    build_development_dossier,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--cross-family-report", type=Path, required=True)
    command.add_argument("--fusion-matrix-report", type=Path, required=True)
    command.add_argument("--xic-intervention-report", type=Path, required=True)
    command.add_argument("--selected-fusion-evaluation-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_development_dossier(
        cross_family_report_path=arguments.cross_family_report,
        fusion_matrix_report_path=arguments.fusion_matrix_report,
        xic_intervention_report_path=arguments.xic_intervention_report,
        selected_fusion_evaluation_root=arguments.selected_fusion_evaluation_root,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "main_table_path": str(result.main_table_path),
                "failure_analysis_path": str(result.failure_analysis_path),
                "failure_records_path": str(result.failure_records_path),
                "manifest_path": str(result.manifest_path),
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
