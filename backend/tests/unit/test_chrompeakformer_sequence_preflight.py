from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.chrompeakformer.sequence_preflight import (
    build_sequence_preflight_report,
)
from multimodal_science.data.manifest import sha256_file


def record(
    asset_id: str,
    *,
    xic_path: str,
    point_count: int,
    roi_window: list[float],
    peak_label: int,
) -> dict[str, object]:
    return {
        "schema_version": "chrompeak-asset-v1",
        "asset_id": asset_id,
        "xic": {
            "path": xic_path,
            "rt_row": 0,
            "signal_row": 1,
            "point_count": point_count,
        },
        "feature": {"roi_window": roi_window},
        "label": {"peak_label": peak_label},
    }


def fixture(root: Path) -> tuple[Path, Path, Path]:
    assets_root = root / "assets"
    train_path = assets_root / "jobs" / "train" / "a" / "xic_matrix.npy"
    validation_path = (
        assets_root / "jobs" / "validation" / "b" / "xic_matrix.npy"
    )
    train_path.parent.mkdir(parents=True)
    validation_path.parent.mkdir(parents=True)
    np.save(
        train_path,
        np.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
        ),
    )
    np.save(
        validation_path,
        np.asarray(
            [
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
                [-1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            ]
        ),
    )
    records = [
        record(
            "asset-train",
            xic_path="jobs/train/a/xic_matrix.npy",
            point_count=6,
            roi_window=[1.0, 3.0],
            peak_label=1,
        ),
        record(
            "asset-validation",
            xic_path="jobs/validation/b/xic_matrix.npy",
            point_count=6,
            roi_window=[0.5, 1.5],
            peak_label=0,
        ),
    ]
    index = root / "asset_index.jsonl"
    index.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in records),
        encoding="utf-8",
    )
    readiness = root / "training_readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-training-readiness-v1",
                "dataset_version": "raw-test0001",
                "plan_sha256": "a" * 64,
                "asset_index_sha256": sha256_file(index),
                "quality_gate": {"passed": True},
                "counts": {"assets": 2},
            }
        ),
        encoding="utf-8",
    )
    return index, readiness, assets_root


class ChromPeakFormerSequencePreflightTests(unittest.TestCase):
    def test_profiles_each_matrix_once_and_each_roi_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, assets_root = fixture(root)
            events: list[tuple[int, int, str]] = []
            result = build_sequence_preflight_report(
                index,
                readiness,
                assets_root,
                root / "sequence_preflight.json",
                progress_callback=lambda completed, total, path: events.append(
                    (completed, total, path)
                ),
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["quality_gate"]["passed"])
        self.assertEqual(report["counts"]["assets"], 2)
        self.assertEqual(report["counts"]["matrices"], 2)
        self.assertEqual(report["counts"]["negative_value_assets"], 0)
        self.assertEqual(
            report["distributions"]["cropped_roi_point_count_per_asset"]["p50"],
            3.0,
        )
        self.assertEqual(
            report["distributions"]["median_sampling_interval_seconds_per_matrix"][
                "p50"
            ],
            45.0,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(result.matrix_count, 2)

    def test_asset_index_must_match_readiness_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, assets_root = fixture(root)
            index.write_text(index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_sequence_preflight_report(
                    index,
                    readiness,
                    assets_root,
                    root / "sequence_preflight.json",
                )

    def test_xic_path_cannot_escape_assets_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, assets_root = fixture(root)
            records = [json.loads(line) for line in index.read_text().splitlines()]
            records[0]["xic"]["path"] = "../outside.npy"
            index.write_text(
                "".join(json.dumps(value) + "\n" for value in records),
                encoding="utf-8",
            )
            readiness_payload = json.loads(readiness.read_text(encoding="utf-8"))
            readiness_payload["asset_index_sha256"] = sha256_file(index)
            readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes assets root"):
                build_sequence_preflight_report(
                    index,
                    readiness,
                    assets_root,
                    root / "sequence_preflight.json",
                )

    def test_non_monotonic_rt_axis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, assets_root = fixture(root)
            matrix_path = assets_root / "jobs" / "train" / "a" / "xic_matrix.npy"
            np.save(
                matrix_path,
                np.asarray(
                    [
                        [0.0, 1.0, 1.0, 3.0, 4.0, 5.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    ]
                ),
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                build_sequence_preflight_report(
                    index,
                    readiness,
                    assets_root,
                    root / "sequence_preflight.json",
                )
