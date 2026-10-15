from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.baselines.dataset import load_sequence_split
from multimodal_science.data.manifest import sha256_file


def _write_fixture(root: Path) -> None:
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    signals = np.arange(16, dtype=np.float32).reshape(2, 8) / 15.0
    scalars = np.arange(14, dtype=np.float32).reshape(2, 7)
    targets = np.asarray([[1.0, 0.2, 0.6], [0.0, -1.0, -1.0]], dtype=np.float32)
    np.save(split_dir / "signals.npy", signals, allow_pickle=False)
    np.save(split_dir / "scalar_features.npy", scalars, allow_pickle=False)
    np.save(split_dir / "targets.npy", targets, allow_pickle=False)
    examples = [
        {
            "schema_version": "chrompeak-multimodal-example-v1",
            "row": 0,
            "asset_id": "asset-positive",
            "split": "train",
            "group_id": "source-a",
            "metadata": {"roi_window_minutes": [1.0, 2.0]},
            "target": {
                "peak_present": True,
                "start_normalized": 0.2,
                "end_normalized": 0.6,
            },
        },
        {
            "schema_version": "chrompeak-multimodal-example-v1",
            "row": 1,
            "asset_id": "asset-negative",
            "split": "train",
            "group_id": "source-b",
            "metadata": {"roi_window_minutes": [2.0, 3.5]},
            "target": {
                "peak_present": False,
                "start_normalized": None,
                "end_normalized": None,
            },
        },
    ]
    examples_path = split_dir / "examples.jsonl"
    examples_path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in examples),
        encoding="utf-8",
    )
    artifacts = {}
    for key, relative in (
        ("train_signals", "train/signals.npy"),
        ("train_scalar_features", "train/scalar_features.npy"),
        ("train_targets", "train/targets.npy"),
        ("train_examples", "train/examples.jsonl"),
    ):
        path = root / relative
        artifacts[key] = {"path": relative, "sha256": sha256_file(path)}
    report = {
        "schema_version": "chrompeak-multimodal-dataset-v1",
        "asset_index_sha256": "a" * 64,
        "target_points": 8,
        "splits": ["train"],
        "counts": {"assets": 2, "by_split": {"train": {"assets": 2}}},
        "contracts": {"scalar_features": [f"feature_{index}" for index in range(7)]},
        "artifacts": artifacts,
    }
    (root / "dataset_report.json").write_text(json.dumps(report), encoding="utf-8")


class MultimodalBaselineDatasetTests(unittest.TestCase):
    def test_loads_verified_arrays_and_source_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            split = load_sequence_split(root, "train")

            self.assertEqual(split.signals.shape, (2, 8))
            self.assertEqual(split.scalar_features.shape, (2, 7))
            self.assertEqual(split.targets.shape, (2, 3))
            self.assertEqual(split.asset_ids, ("asset-positive", "asset-negative"))
            self.assertEqual(split.group_ids, ("source-a", "source-b"))
            np.testing.assert_allclose(split.roi_width_minutes, [1.0, 1.5])
            self.assertEqual(split.asset_index_sha256, "a" * 64)
            self.assertEqual(len(split.dataset_report_sha256), 64)

    def test_rejects_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            signals_path = root / "train" / "signals.npy"
            signals = np.load(signals_path)
            signals[0, 0] = 999.0
            np.save(signals_path, signals, allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_sequence_split(root, "train")

    def test_rejects_invalid_negative_boundary_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            targets_path = root / "train" / "targets.npy"
            targets = np.load(targets_path)
            targets[1, 1:] = 0.0
            np.save(targets_path, targets, allow_pickle=False)
            report_path = root / "dataset_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifacts"]["train_targets"]["sha256"] = sha256_file(targets_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sentinel"):
                load_sequence_split(root, "train")


if __name__ == "__main__":
    unittest.main()
