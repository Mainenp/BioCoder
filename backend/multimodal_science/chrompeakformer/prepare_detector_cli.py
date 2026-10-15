"""Prepare a leakage-audited ChromPeakFormer COCO training layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.chrompeakformer.detector_dataset import build_detector_dataset


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--asset-index", type=Path, required=True)
    command.add_argument("--asset-index-report", type=Path, required=True)
    command.add_argument("--assets-root", type=Path, required=True)
    command.add_argument("--train-coco", type=Path, required=True)
    command.add_argument("--validation-coco", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--verify-image-hashes", action="store_true")
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build_detector_dataset(
        asset_index_path=arguments.asset_index,
        asset_index_report_path=arguments.asset_index_report,
        assets_root=arguments.assets_root,
        train_coco_path=arguments.train_coco,
        validation_coco_path=arguments.validation_coco,
        output_dir=arguments.output_dir,
        verify_image_hashes=arguments.verify_image_hashes,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "coco_root": str(result.coco_root),
                "asset_index_sha256": result.asset_index_sha256,
                "train_assets": result.train_assets,
                "validation_assets": result.validation_assets,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
