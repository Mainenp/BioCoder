from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.auxiliary_pretraining import (
    AuxiliaryPretrainingSettings,
    _validate_settings,
    load_verified_pretrained_projector,
)
from multimodal_science.qwen3vl.auxiliary_pretraining_data import (
    AUXILIARY_PRETRAINING_DATASET_SCHEMA,
    _canonicalize_rt_axis,
    _canonicalize_signal,
    build_auxiliary_pretraining_dataset,
)
from multimodal_science.qwen3vl.sensor_projector import SensorProjectorSpec


def write_npy(path: Path, rows: list[list[float]]) -> None:
    shape = (len(rows), len(rows[0]))
    header_text = repr({"descr": "<f8", "fortran_order": False, "shape": shape})
    padding = (-((10 + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    values = [value for row in rows for value in row]
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header))
        + header
        + struct.pack(f"<{len(values)}d", *values)
    )


def write_jpeg(path: Path, width: int = 400, height: int = 300) -> None:
    frame = b"\x08" + struct.pack(">HH", height, width) + b"\x03" + b"\x01\x11\x00" * 3
    path.write_bytes(
        b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", len(frame) + 2) + frame + b"\xff\xd9"
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class AuxiliaryPretrainingTests(unittest.TestCase):
    def fixture(
        self, root: Path, matrix_rows: list[list[float]] | None = None
    ) -> tuple[Path, Path]:
        assets_root = root / "assets"
        matrix = assets_root / "jobs" / "aux" / "xic_matrix.npy"
        matrix.parent.mkdir(parents=True)
        write_npy(
            matrix,
            matrix_rows
            or [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        images = []
        for name in ("one.jpeg", "two.jpeg"):
            image = matrix.parent / name
            write_jpeg(image)
            images.append(image)

        index_root = root / "index"
        index_root.mkdir()
        rows = []
        for row, image in enumerate(images, start=1):
            rows.append(
                {
                    "schema_version": "chrompeak-auxiliary-asset-v1",
                    "asset_id": f"asset-{row}",
                    "split": "auxiliary_unlabeled_train",
                    "metrics_allowed": False,
                    "job_id": "job-1",
                    "source_group": "source-a",
                    "image": {
                        "path": image.relative_to(assets_root).as_posix(),
                        "sha256": sha256_file(image),
                    },
                    "xic": {
                        "path": matrix.relative_to(assets_root).as_posix(),
                        "sha256": sha256_file(matrix),
                        "signal_row": row,
                        "point_count": 4,
                    },
                    "feature": {
                        "q1": 100.0 + row,
                        "q3": 50.0 + row,
                        "rt": 1.5,
                        "roi_window": [0.0, 3.0],
                    },
                    "supervision": {
                        "label_status": "unlabeled",
                        "supervised_train_eligible": False,
                        "auxiliary_unlabeled_train_eligible": True,
                        "benchmark_eligible": False,
                        "internal_test_accessed": False,
                    },
                }
            )
        index_path = index_root / "auxiliary_asset_index.jsonl"
        index_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        report_path = index_root / "auxiliary_asset_index_report.json"
        write_json(
            report_path,
            {
                "schema_version": "chrompeak-auxiliary-asset-index-report-v1",
                "asset_index_sha256": sha256_file(index_path),
                "counts": {"extracted_trace_assets": 2},
                "contracts": {
                    "complete_plan_coverage": True,
                    "labels_present": False,
                    "metrics_allowed": False,
                    "train_supervision_present": False,
                    "auxiliary_unlabeled_train_eligible": True,
                    "validation_membership_changed": False,
                    "benchmark_membership_changed": False,
                    "internal_test_accessed": False,
                },
                "quality_gate_passed": True,
            },
        )
        manifest = index_root / "artifact_manifest.sha256"
        manifest.write_text(
            f"{sha256_file(index_path)}  {index_path.name}\n"
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        return index_root, assets_root

    def test_materializes_normalized_zero_label_signals(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required by the materializer")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_root, assets_root = self.fixture(root)
            report_path = index_root / "auxiliary_asset_index_report.json"
            manifest = index_root / "artifact_manifest.sha256"
            result = build_auxiliary_pretraining_dataset(
                index_root,
                sha256_file(report_path),
                sha256_file(manifest),
                assets_root,
                root / "output",
                target_points=160,
            )

            signals = np.load(result.output_dir / "signals.npy", allow_pickle=False)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            examples = [
                json.loads(line)
                for line in (result.output_dir / "examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(signals.shape, (2, 160))
            self.assertEqual(signals.dtype, np.float32)
            self.assertEqual(float(signals.min()), 0.0)
            self.assertEqual(float(signals.max()), 1.0)
            self.assertEqual(report["schema_version"], AUXILIARY_PRETRAINING_DATASET_SCHEMA)
            self.assertEqual(report["counts"]["labels"], 0)
            self.assertEqual(report["counts"]["signals_available"], 1)
            self.assertEqual(report["counts"]["signals_unavailable"], 1)
            self.assertFalse(report["contracts"]["metrics_allowed"])
            self.assertFalse(examples[0]["supervision"]["benchmark_eligible"])
            self.assertNotIn("label", examples[0])
            self.assertEqual(report["warnings"], [])
            self.assertEqual(
                report["rt_axis_normalization"]["matrices_already_strict"], 1
            )

    def test_normalizes_out_of_order_and_duplicate_vendor_rt_points(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required by the materializer")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_root, assets_root = self.fixture(
                root,
                matrix_rows=[
                    [2.0, 0.0, 1.0, 1.0],
                    [8.0, 0.0, 2.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            )
            report_path = index_root / "auxiliary_asset_index_report.json"
            manifest = index_root / "artifact_manifest.sha256"
            result = build_auxiliary_pretraining_dataset(
                index_root,
                sha256_file(report_path),
                sha256_file(manifest),
                assets_root,
                root / "output",
                target_points=160,
            )

            signals = np.load(result.output_dir / "signals.npy", allow_pickle=False)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            first_example = json.loads(
                (result.output_dir / "examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

            self.assertEqual(signals.shape, (2, 160))
            self.assertTrue(np.isfinite(signals).all())
            self.assertEqual(report["rt_axis_normalization"]["matrices_normalized"], 1)
            self.assertEqual(report["rt_axis_normalization"]["matrices_reordered"], 1)
            self.assertEqual(
                report["rt_axis_normalization"]["matrices_with_duplicate_rt"], 1
            )
            self.assertEqual(
                report["rt_axis_normalization"]["duplicate_points_collapsed"], 1
            )
            self.assertEqual(report["warnings"][0]["code"], "rt_axis_normalized")
            axis = first_example["signal"]["rt_axis_normalization"]
            self.assertTrue(axis["was_reordered"])
            self.assertEqual(axis["duplicate_points_collapsed"], 1)
            self.assertEqual(
                report["contracts"]["duplicate_rt_intensity_reducer"], "maximum"
            )
            canonical_rt, order, group_starts, _ = _canonicalize_rt_axis(
                np.asarray([2.0, 0.0, 1.0, 1.0]), np
            )
            canonical_signal = _canonicalize_signal(
                np.asarray([8.0, 0.0, 2.0, 6.0]),
                order,
                group_starts,
                expected_points=4,
                np=np,
            )
            np.testing.assert_array_equal(canonical_rt, [0.0, 1.0, 2.0])
            np.testing.assert_array_equal(canonical_signal, [0.0, 6.0, 8.0])

    def test_rejects_rt_axis_without_a_real_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_root, assets_root = self.fixture(
                root,
                matrix_rows=[
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 2.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            )
            report_path = index_root / "auxiliary_asset_index_report.json"
            manifest = index_root / "artifact_manifest.sha256"
            with self.assertRaisesRegex(ValueError, "fewer than two unique points"):
                build_auxiliary_pretraining_dataset(
                    index_root,
                    sha256_file(report_path),
                    sha256_file(manifest),
                    assets_root,
                    root / "output",
                )

    def test_rejects_source_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_root, assets_root = self.fixture(root)
            report_path = index_root / "auxiliary_asset_index_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["contracts"]["labels_present"] = True
            write_json(report_path, report)
            manifest = index_root / "artifact_manifest.sha256"
            index_path = index_root / "auxiliary_asset_index.jsonl"
            manifest.write_text(
                f"{sha256_file(index_path)}  {index_path.name}\n"
                f"{sha256_file(report_path)}  {report_path.name}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "labels_present"):
                build_auxiliary_pretraining_dataset(
                    index_root,
                    sha256_file(report_path),
                    sha256_file(manifest),
                    assets_root,
                    root / "output",
                )

    def test_completed_projector_verification_is_hash_and_spec_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "sensor_projector.safetensors"
            weights.write_bytes(b"projector")
            spec = SensorProjectorSpec()
            report_path = root / "auxiliary_pretraining_report.json"
            write_json(
                report_path,
                {
                    "schema_version": "chrompeak-xic-auxiliary-pretraining-v1",
                    "code_revision": "a" * 40,
                    "sources": {"dataset_report_sha256": "b" * 64},
                    "model": {
                        "sensor_projector": spec.as_dict(),
                        "persisted_projector_sha256": sha256_file(weights),
                    },
                    "contracts": {
                        "unlabeled_auxiliary_only": True,
                        "validation_opened": False,
                        "validation_answers_opened": False,
                        "internal_test_accessed": False,
                        "metrics_allowed": False,
                        "projector_weights_compatible_with_formal_fusion": True,
                    },
                    "development_training_complete": True,
                    "development_comparison_eligible": False,
                    "final_benchmark_eligible": False,
                },
            )
            manifest = root / "artifact_manifest.sha256"
            manifest.write_text(
                f"{sha256_file(weights)}  {weights.name}\n"
                f"{sha256_file(report_path)}  {report_path.name}\n",
                encoding="utf-8",
            )

            path, report = load_verified_pretrained_projector(
                root,
                report_sha256=sha256_file(report_path),
                manifest_sha256=sha256_file(manifest),
                expected_spec=spec,
            )
            self.assertEqual(path, weights.resolve())
            self.assertTrue(report["development_training_complete"])

            remapped_path, _ = load_verified_pretrained_projector(
                root,
                report_sha256=sha256_file(report_path),
                manifest_sha256=sha256_file(manifest),
                expected_spec=SensorProjectorSpec(sensor_tokens=8),
            )
            self.assertEqual(remapped_path, weights.resolve())

            report["development_training_complete"] = False
            write_json(report_path, report)
            manifest.write_text(
                f"{sha256_file(weights)}  {weights.name}\n"
                f"{sha256_file(report_path)}  {report_path.name}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Incomplete pretraining"):
                load_verified_pretrained_projector(
                    root,
                    report_sha256=sha256_file(report_path),
                    manifest_sha256=sha256_file(manifest),
                    expected_spec=spec,
                )

    def test_settings_and_cli_keep_auxiliary_stage_unlabeled(self) -> None:
        _validate_settings(AuxiliaryPretrainingSettings())
        with self.assertRaisesRegex(ValueError, "batch_size"):
            _validate_settings(AuxiliaryPretrainingSettings(batch_size=1))

        from multimodal_science.qwen3vl.build_auxiliary_pretraining_data_cli import (
            parser as data_parser,
        )
        from multimodal_science.qwen3vl.train_auxiliary_pretraining_cli import (
            parser as train_parser,
        )
        from multimodal_science.qwen3vl.train_fusion_cli import parser as fusion_parser

        for command_parser in (data_parser(), train_parser()):
            destinations = {action.dest for action in command_parser._actions}
            self.assertNotIn("validation", destinations)
            self.assertNotIn("validation_answers", destinations)
            self.assertNotIn("internal_test", destinations)
        fusion_destinations = {action.dest for action in fusion_parser()._actions}
        self.assertIn("pretrained_projector_root", fusion_destinations)
        self.assertIn("pretrained_projector_report_sha256", fusion_destinations)
        self.assertIn("pretrained_projector_manifest_sha256", fusion_destinations)

    def test_slurm_contract_is_guarded_and_never_opens_validation(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_auxiliary_pretrain.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("#SBATCH --partition=GPU-5090", script)
        self.assertIn("build_auxiliary_pretraining_data_cli", script)
        self.assertIn("train_auxiliary_pretraining_cli", script)
        self.assertIn("AUXILIARY_PRETRAINING_CONTRACT=OK", script)
        self.assertIn("QWEN3VL_AUXILIARY_PRETRAINING=OK", script)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)
        self.assertNotIn("validation_answers", script)


if __name__ == "__main__":
    unittest.main()
