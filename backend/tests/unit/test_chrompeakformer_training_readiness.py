from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.chrompeakformer.training_readiness import (
    build_training_readiness_report,
)
from multimodal_science.data.manifest import sha256_file

PLAN_SHA256 = "a" * 64


def asset(
    asset_id: str,
    *,
    split: str,
    job_id: str,
    source_sha256: str,
    peak_label: int,
) -> dict[str, object]:
    boxes = []
    intervals = []
    if peak_label == 1:
        boxes = [
            {
                "annotation_id": 11,
                "category_id": 0,
                "interval_index": 0,
                "rt_interval": [0.8, 1.2],
                "visible_rt_interval": [0.8, 1.2],
                "bbox": [160.0, 0.0, 80.0, 300.0],
                "area": 24000.0,
            }
        ]
        intervals = [[0.8, 1.2]]
    return {
        "schema_version": "chrompeak-asset-v1",
        "asset_id": asset_id,
        "dataset_version": "raw-test0001",
        "plan_sha256": PLAN_SHA256,
        "split": split,
        "evaluation_tier": "primary_development",
        "metrics_allowed": True,
        "job_id": job_id,
        "record_id": f"record-{asset_id}",
        "source_mzml": f"mzml/{job_id}.mzML",
        "source_artifact_hash": source_sha256,
        "job_output_sha256": "b" * 64,
        "image": {
            "id": 101 if split == "train" else 202,
            "path": f"jobs/{split}/{job_id}/{asset_id}.jpeg",
            "sha256": "c" * 64,
            "width": 400,
            "height": 300,
        },
        "xic": {
            "path": f"jobs/{split}/{job_id}/xic_matrix.npy",
            "rt_row": 0,
            "signal_row": 1,
            "point_count": 100 if split == "train" else 120,
        },
        "feature": {
            "native_id": "SRM name=compound-a",
            "q1": 100.1,
            "q3": 50.1,
            "rt": 1.0,
            "roi_window": [0.0, 2.0],
        },
        "label": {
            "component": "compound-a",
            "channel": "quantifier",
            "expected_rt": 1.0,
            "peak_label": peak_label,
            "peak_intervals": intervals,
            "coco_boxes": boxes,
        },
    }


def fixture(
    root: Path, *, partial: bool = False, crossing_source: bool = False
) -> tuple[Path, Path]:
    index = root / "asset_index.jsonl"
    train_source = "d" * 64
    validation_source = train_source if crossing_source else "e" * 64
    records = [
        asset(
            "asset-train",
            split="train",
            job_id="job-train",
            source_sha256=train_source,
            peak_label=1,
        ),
        asset(
            "asset-validation",
            split="validation",
            job_id="job-validation",
            source_sha256=validation_source,
            peak_label=0,
        ),
    ]
    index.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    report = root / "asset_index_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-asset-index-report-v1",
                "plan_sha256": PLAN_SHA256,
                "asset_index_sha256": sha256_file(index),
                "partial": partial,
                "counts": {
                    "assets": 2,
                    "positive_assets": 1,
                    "negative_assets": 1,
                    "annotations": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return index, report


class ChromPeakFormerTrainingReadinessTests(unittest.TestCase):
    def test_complete_index_produces_training_readiness_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, index_report = fixture(root)
            result = build_training_readiness_report(
                index, index_report, root / "training_readiness.json"
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            observed_report_sha256 = sha256_file(result.report_path)

        self.assertTrue(report["quality_gate"]["passed"])
        self.assertEqual(report["counts"]["assets"], 2)
        self.assertEqual(report["counts"]["splits"]["train"]["positive"], 1)
        self.assertEqual(report["counts"]["splits"]["validation"]["negative"], 1)
        self.assertEqual(report["distributions"]["xic_point_count"]["p50"], 110.0)
        self.assertEqual(report["coverage"]["validation_only_components"], [])
        self.assertEqual(result.report_sha256, observed_report_sha256)

    def test_partial_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, index_report = fixture(root, partial=True)
            with self.assertRaisesRegex(ValueError, "Partial asset indexes"):
                build_training_readiness_report(
                    index, index_report, root / "training_readiness.json"
                )

    def test_tampered_index_is_rejected_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, index_report = fixture(root)
            index.write_text(index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_training_readiness_report(
                    index, index_report, root / "training_readiness.json"
                )

    def test_source_artifact_cannot_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, index_report = fixture(root, crossing_source=True)
            with self.assertRaisesRegex(ValueError, "Source artifacts cross"):
                build_training_readiness_report(
                    index, index_report, root / "training_readiness.json"
                )

    def test_output_cannot_overwrite_the_asset_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, index_report = fixture(root)
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                build_training_readiness_report(index, index_report, index)
