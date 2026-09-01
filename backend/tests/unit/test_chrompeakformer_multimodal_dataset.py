from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.chrompeakformer.multimodal_dataset import (
    build_multimodal_dataset,
)
from multimodal_science.data.manifest import sha256_file

PLAN_SHA256 = "a" * 64


def asset(
    asset_id: str,
    *,
    split: str,
    job_id: str,
    matrix_path: str,
    peak_label: int,
    bbox_x: float = 100.0,
) -> dict[str, object]:
    boxes = []
    intervals = []
    if peak_label:
        boxes = [
            {
                "annotation_id": 10,
                "category_id": 0,
                "interval_index": 0,
                "rt_interval": [0.25, 0.75],
                "visible_rt_interval": [0.25, 0.75],
                "bbox": [bbox_x, 0.0, 200.0, 300.0],
                "area": 60000.0,
            }
        ]
        intervals = [[0.25, 0.75]]
    return {
        "schema_version": "chrompeak-asset-v1",
        "asset_id": asset_id,
        "dataset_version": "raw-test0001",
        "plan_sha256": PLAN_SHA256,
        "split": split,
        "job_id": job_id,
        "record_id": f"record-{asset_id}",
        "source_artifact_hash": ("b" if split == "train" else "c") * 64,
        "job_output_sha256": "d" * 64,
        "image": {
            "path": f"jobs/{split}/{job_id}/{asset_id}.jpeg",
            "sha256": "e" * 64,
            "width": 400,
            "height": 300,
        },
        "xic": {
            "path": matrix_path,
            "rt_row": 0,
            "signal_row": 1,
            "point_count": 5,
        },
        "feature": {
            "q1": 100.0 if split == "train" else 110.0,
            "q3": 50.0 if split == "train" else 55.0,
            "rt": 0.5,
            "roi_window": [0.0, 1.0],
        },
        "label": {
            "component": "compound-a",
            "channel": "quantifier",
            "expected_rt": 0.5,
            "peak_label": peak_label,
            "peak_intervals": intervals,
            "coco_boxes": boxes,
        },
    }


def fixture(root: Path, *, bbox_x: float = 100.0) -> tuple[Path, Path, Path, Path]:
    assets_root = root / "assets"
    train_matrix = assets_root / "jobs" / "train" / "a" / "xic_matrix.npy"
    validation_matrix = (
        assets_root / "jobs" / "validation" / "b" / "xic_matrix.npy"
    )
    train_matrix.parent.mkdir(parents=True)
    validation_matrix.parent.mkdir(parents=True)
    rt = np.asarray([0.1, 0.101, 0.5, 0.501, 0.9], dtype=np.float64)
    np.save(train_matrix, np.vstack([rt, [0.0, 2.0, 8.0, 8.5, 0.0]]))
    np.save(validation_matrix, np.vstack([rt, np.zeros(5)]))
    records = [
        asset(
            "asset-train",
            split="train",
            job_id="a",
            matrix_path="jobs/train/a/xic_matrix.npy",
            peak_label=1,
            bbox_x=bbox_x,
        ),
        asset(
            "asset-validation",
            split="validation",
            job_id="b",
            matrix_path="jobs/validation/b/xic_matrix.npy",
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
                "plan_sha256": PLAN_SHA256,
                "asset_index_sha256": sha256_file(index),
                "quality_gate": {"passed": True},
                "counts": {
                    "assets": 2,
                    "splits": {
                        "train": {"assets": 1},
                        "validation": {"assets": 1},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    preflight = root / "sequence_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-sequence-preflight-v2",
                "asset_index_sha256": sha256_file(index),
                "readiness_report_sha256": sha256_file(readiness),
                "quality_gate": {"passed": True},
                "counts": {"assets": 2},
            }
        ),
        encoding="utf-8",
    )
    return index, readiness, preflight, assets_root


class ChromPeakFormerMultimodalDatasetTests(unittest.TestCase):
    def test_materializes_aligned_model_ready_arrays_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, preflight, assets_root = fixture(root)
            events: list[tuple[int, int, str]] = []
            result = build_multimodal_dataset(
                index,
                readiness,
                preflight,
                assets_root,
                root / "dataset",
                target_points=32,
                progress_callback=lambda completed, total, path: events.append(
                    (completed, total, path)
                ),
            )
            train_signal = np.load(result.output_dir / "train" / "signals.npy")
            validation_signal = np.load(
                result.output_dir / "validation" / "signals.npy"
            )
            train_scalars = np.load(
                result.output_dir / "train" / "scalar_features.npy"
            )
            validation_scalars = np.load(
                result.output_dir / "validation" / "scalar_features.npy"
            )
            train_targets = np.load(result.output_dir / "train" / "targets.npy")
            validation_targets = np.load(
                result.output_dir / "validation" / "targets.npy"
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            example = json.loads(
                (result.output_dir / "train" / "examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            cached = build_multimodal_dataset(
                index,
                readiness,
                preflight,
                assets_root,
                root / "dataset",
                target_points=32,
            )

        self.assertEqual(train_signal.shape, (1, 32))
        self.assertEqual(float(train_signal.max()), 1.0)
        self.assertEqual(float(train_signal[0, 0]), 0.0)
        self.assertEqual(float(train_signal[0, -1]), 0.0)
        self.assertTrue(np.all(validation_signal == 0.0))
        self.assertEqual(train_scalars.shape, (1, 7))
        self.assertEqual(train_scalars[0, -1], 1.0)
        self.assertEqual(validation_scalars[0, -1], 0.0)
        np.testing.assert_allclose(train_targets[0], [1.0, 0.25, 0.75])
        np.testing.assert_allclose(validation_targets[0], [0.0, -1.0, -1.0])
        self.assertEqual(example["sequence"]["length"], 32)
        self.assertEqual(example["target"]["coordinate_system"], "roi_fraction_0_1")
        self.assertEqual(report["counts"]["assets"], 2)
        self.assertEqual(report["counts"]["by_split"]["validation"]["signal_unavailable"], 1)
        self.assertEqual(len(events), 2)
        self.assertFalse(result.cached)
        self.assertTrue(cached.cached)
        self.assertEqual(result.report_sha256, cached.report_sha256)

    def test_image_and_rt_boundaries_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, preflight, assets_root = fixture(root, bbox_x=0.0)
            with self.assertRaisesRegex(ValueError, "Image and RT boundaries disagree"):
                build_multimodal_dataset(
                    index,
                    readiness,
                    preflight,
                    assets_root,
                    root / "dataset",
                    target_points=32,
                )
            self.assertFalse((root / "dataset").exists())

    def test_sequence_preflight_v2_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, preflight, assets_root = fixture(root)
            payload = json.loads(preflight.read_text(encoding="utf-8"))
            payload["schema_version"] = "chrompeak-sequence-preflight-v1"
            preflight.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v2 is required"):
                build_multimodal_dataset(
                    index,
                    readiness,
                    preflight,
                    assets_root,
                    root / "dataset",
                    target_points=32,
                )

    def test_sequence_preflight_must_match_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, readiness, preflight, assets_root = fixture(root)
            payload = json.loads(readiness.read_text(encoding="utf-8"))
            payload["extra"] = "tampered"
            readiness.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_multimodal_dataset(
                    index,
                    readiness,
                    preflight,
                    assets_root,
                    root / "dataset",
                    target_points=32,
                )
