from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.auxiliary_msdata import (
    build_auxiliary_msdata_dataset,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Verify converted msdata frames for auxiliary unlabeled training."
    )
    argument_parser.add_argument("--quarantine-manifest", type=Path, required=True)
    argument_parser.add_argument("--quarantine-report", type=Path, required=True)
    argument_parser.add_argument("--converted-root", type=Path, required=True)
    argument_parser.add_argument("--converter", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, required=True)
    return argument_parser


def main() -> None:
    arguments = parser().parse_args()
    result = build_auxiliary_msdata_dataset(
        quarantine_manifest_path=arguments.quarantine_manifest,
        quarantine_report_path=arguments.quarantine_report,
        converted_root=arguments.converted_root,
        converter_path=arguments.converter,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "plan_path": str(result.plan_path),
                "plan_sha256": result.plan_sha256,
                "source_groups": result.source_groups,
                "acquisition_frames": result.acquisition_frames,
                "transition_traces": result.transition_traces,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
