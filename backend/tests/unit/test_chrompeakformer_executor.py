from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from multimodal_science.chrompeakformer.executor import (
    DependencyGateError,
    run_job,
)
from multimodal_science.chrompeakformer.outputs import read_npy_metadata
from multimodal_science.data.manifest import sha256_file


def write_npy(path: Path, shape: tuple[int, int]) -> None:
    header_text = repr(
        {"descr": "<f8", "fortran_order": False, "shape": shape}
    )
    preamble_length = 10
    padding = (-((preamble_length + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    values = [float(index) for index in range(shape[0] * shape[1])]
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header))
        + header
        + struct.pack(f"<{len(values)}d", *values)
    )


def write_success_outputs(output_dir: Path, matrix_shape: tuple[int, int] = (3, 4)) -> None:
    features = [
        {
            "Compound Name": "1",
            "native_id": "compound-a-1",
            "mz": "100.1",
            "q3": "50.1",
            "RT": "1.2",
        },
        {
            "Compound Name": "2",
            "native_id": "compound-b-1",
            "mz": "200.1",
            "q3": "70.1",
            "RT": "2.2",
        },
    ]
    with (output_dir / "feature.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)

    windows = [
        {"image": "one.jpeg", "rt_lo": "0.2", "rt_hi": "2.2"},
        {"image": "two.jpeg", "rt_lo": "1.2", "rt_hi": "3.2"},
    ]
    with (output_dir / "roi_windows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    (output_dir / "one.jpeg").write_bytes(b"\xff\xd8one\xff\xd9")
    (output_dir / "two.jpeg").write_bytes(b"\xff\xd8two\xff\xd9")
    write_npy(output_dir / "xic_matrix.npy", matrix_shape)


def make_job(source_relative: str, artifact_hash: str) -> dict[str, object]:
    job_id = "0123456789abcdef01234567"
    return {
        "job_id": job_id,
        "dataset_version": "raw-test0001",
        "derivation_mode": "label_driven",
        "split": "train",
        "evaluation_tier": "primary_development",
        "source_mzml": source_relative,
        "artifact_hash": artifact_hash,
        "output_prefix": f"jobs/train/{job_id}",
    }


class ChromPeakFormerExecutorTests(unittest.TestCase):
    def source_fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "data"
        source = data_root / "mzml" / "train" / "sample.mzML"
        source.parent.mkdir(parents=True)
        source.write_text("<mzML><chromatogram/></mzML>", encoding="utf-8")
        return data_root, source

    def test_job_is_atomically_published_and_then_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source = self.source_fixture(root)
            output_root = root / "outputs"
            job = make_job("mzml/train/sample.mzML", sha256_file(source))
            calls = []

            def extractor(job_value: dict, source_path: Path, output_dir: Path) -> dict:
                calls.append((job_value["job_id"], source_path.name))
                write_success_outputs(output_dir)
                return {"extractor_version": "synthetic-test"}

            first = run_job(
                job,
                plan_sha256="a" * 64,
                data_root=data_root,
                output_root=output_root,
                extractor=extractor,
                enforce_dependencies=False,
            )
            second = run_job(
                job,
                plan_sha256="a" * 64,
                data_root=data_root,
                output_root=output_root,
                extractor=extractor,
                enforce_dependencies=False,
            )
            provenance = json.loads(first.provenance_path.read_text(encoding="utf-8"))

        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "cached")
        self.assertEqual(len(calls), 1)
        self.assertEqual(provenance["status"], "ok")
        self.assertEqual(provenance["outputs"]["feature_count"], 2)
        self.assertEqual(first.output_sha256, second.output_sha256)

    def test_dependency_gate_blocks_before_extractor_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source = self.source_fixture(root)
            output_root = root / "outputs"
            job = make_job("mzml/train/sample.mzML", sha256_file(source))
            called = False

            def extractor(job_value: dict, source_path: Path, output_dir: Path) -> None:
                nonlocal called
                called = True

            with self.assertRaises(DependencyGateError):
                run_job(
                    job,
                    plan_sha256="b" * 64,
                    data_root=data_root,
                    output_root=output_root,
                    extractor=extractor,
                    dependency_status={},
                )
            failure = json.loads(
                (output_root / "failures" / f"{job['job_id']}.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertFalse(called)
        self.assertEqual(failure["status"], "dependency_blocked")
        self.assertEqual(failure["stage"], "dependency_gate")

    def test_invalid_matrix_shape_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source = self.source_fixture(root)
            output_root = root / "outputs"
            job = make_job("mzml/train/sample.mzML", sha256_file(source))

            def extractor(job_value: dict, source_path: Path, output_dir: Path) -> None:
                write_success_outputs(output_dir, matrix_shape=(2, 4))

            with self.assertRaisesRegex(ValueError, "Feature/sequence count mismatch"):
                run_job(
                    job,
                    plan_sha256="c" * 64,
                    data_root=data_root,
                    output_root=output_root,
                    extractor=extractor,
                    enforce_dependencies=False,
                )
            failure = json.loads(
                (output_root / "failures" / f"{job['job_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            final_dir = output_root / str(job["output_prefix"])
            staging_root = output_root / ".staging"

        self.assertEqual(failure["status"], "validation_error")
        self.assertFalse(final_dir.exists())
        self.assertFalse(staging_root.exists() and any(staging_root.iterdir()))

    def test_npy_payload_truncation_is_detected_without_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npy"
            write_npy(path, (3, 4))
            payload = path.read_bytes()
            path.write_bytes(payload[:-8])

            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                read_npy_metadata(path)


if __name__ == "__main__":
    unittest.main()
