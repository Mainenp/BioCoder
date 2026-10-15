from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from multimodal_science.chrompeakformer.asset_index import build_asset_index
from multimodal_science.chrompeakformer.executor import run_job
from multimodal_science.data.manifest import sha256_file


def write_npy(path: Path, shape: tuple[int, int]) -> None:
    header_text = repr({"descr": "<f8", "fortran_order": False, "shape": shape})
    padding = (-((10 + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    values = [float(index) for index in range(shape[0] * shape[1])]
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


def write_outputs(output_dir: Path, unmatched: bool = False) -> None:
    features = [
        {
            "Compound Name": "1",
            "native_id": "unknown" if unmatched else "compound-a-1",
            "mz": "100.1",
            "q3": "50.1",
            "RT": "1.0",
        },
        {
            "Compound Name": "2",
            "native_id": "SRM SIC Q1=200 Q3=70 name=compound-b",
            "mz": "200.1",
            "q3": "70.1",
            "RT": "2.0",
        },
    ]
    with (output_dir / "feature.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    windows = [
        {"image": "one.jpeg", "rt_lo": "0", "rt_hi": "2"},
        {"image": "two.jpeg", "rt_lo": "1", "rt_hi": "3"},
    ]
    with (output_dir / "roi_windows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    write_jpeg(output_dir / "one.jpeg")
    write_jpeg(output_dir / "two.jpeg")
    write_npy(output_dir / "xic_matrix.npy", (3, 4))


def make_job(job_id: str, source_relative: str, artifact_hash: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "dataset_version": "raw-test0001",
        "derivation_mode": "label_driven",
        "split": "train",
        "evaluation_tier": "primary_development",
        "metrics_allowed": True,
        "source_mzml": source_relative,
        "artifact_hash": artifact_hash,
        "output_prefix": f"jobs/train/{job_id}",
        "labels": [
            {
                "record_id": "111111111111111111111111",
                "component": "compound-a",
                "channel": "定量离子",
                "rt": 1.0,
                "peak_label": 1,
                "peak_intervals": [[0.5, 1.0]],
            },
            {
                "record_id": "222222222222222222222222",
                "component": "compound-b",
                "channel": "定量离子",
                "rt": 2.0,
                "peak_label": 0,
                "peak_intervals": [],
            },
        ],
    }


def write_plan(path: Path, jobs: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(job, ensure_ascii=False, separators=(",", ":")) + "\n" for job in jobs),
        encoding="utf-8",
    )


class ChromPeakFormerAssetIndexTests(unittest.TestCase):
    def fixture(
        self, root: Path, *, missing_job: bool = False, unmatched: bool = False
    ) -> tuple[Path, Path, Path]:
        data_root = root / "data"
        source = data_root / "mzml" / "sample.mzML"
        source.parent.mkdir(parents=True)
        source.write_text("<mzML/>", encoding="utf-8")
        job = make_job(
            "0123456789abcdef01234567", "mzml/sample.mzML", sha256_file(source)
        )
        jobs = [job]
        if missing_job:
            missing = dict(job)
            missing["job_id"] = "abcdef0123456789abcdef01"
            missing["output_prefix"] = "jobs/train/abcdef0123456789abcdef01"
            missing["source_mzml"] = "mzml/missing.mzML"
            jobs.append(missing)
        plan = root / "derivation_plan.jsonl"
        write_plan(plan, jobs)
        assets_root = root / "assets"

        def extractor(_job: dict, _source: Path, output_dir: Path) -> dict:
            write_outputs(output_dir, unmatched=unmatched)
            return {
                "adapter_version": "synthetic",
                "source_api": "synthetic_extract",
                "private_code_sha256": "a" * 64,
                "smooth_sigma": 1.0,
            }

        run_job(
            job,
            plan_sha256=sha256_file(plan),
            data_root=data_root,
            output_root=assets_root,
            extractor=extractor,
            enforce_dependencies=False,
        )
        return plan, assets_root, root / "index"

    def test_index_joins_images_xic_rows_labels_and_coco_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir = self.fixture(root)
            result = build_asset_index(plan, assets_root, output_dir)
            assets = [
                json.loads(line)
                for line in result.index_path.read_text(encoding="utf-8").splitlines()
            ]
            coco = json.loads((output_dir / "train_coco.json").read_text(encoding="utf-8"))
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.asset_count, 2)
        self.assertEqual(result.annotation_count, 1)
        self.assertEqual(assets[0]["xic"]["signal_row"], 1)
        self.assertEqual(assets[0]["record_id"], "111111111111111111111111")
        self.assertEqual(assets[0]["plan_sha256"], result.plan_sha256)
        self.assertEqual(assets[0]["extractor"]["private_code_sha256"], "a" * 64)
        self.assertLess(assets[0]["image"]["id"], 2**53)
        self.assertEqual(assets[0]["label"]["coco_boxes"][0]["bbox"], [100.0, 0.0, 100.0, 300.0])
        self.assertEqual(assets[1]["label"]["peak_label"], 0)
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(coco["annotations"][0]["bbox"], [100.0, 0.0, 100.0, 300.0])
        self.assertEqual(report["counts"]["positive_assets"], 1)
        self.assertEqual(report["counts"]["negative_assets"], 1)

    def test_partial_index_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir = self.fixture(root, missing_job=True)
            with self.assertRaisesRegex(FileNotFoundError, "allow_partial"):
                build_asset_index(plan, assets_root, output_dir)
            result = build_asset_index(
                plan, assets_root, output_dir, allow_partial=True
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.indexed_jobs, 1)
        self.assertEqual(result.missing_jobs, 1)
        self.assertTrue(report["partial"])

    def test_unmatched_native_id_is_rejected_without_row_order_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir = self.fixture(root, unmatched=True)
            with self.assertRaisesRegex(ValueError, "No label matches"):
                build_asset_index(plan, assets_root, output_dir)

    def test_progress_callback_reports_each_verified_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir = self.fixture(root)
            events: list[tuple[int, int, str]] = []

            build_asset_index(
                plan,
                assets_root,
                output_dir,
                progress_callback=lambda completed, total, job_id: events.append(
                    (completed, total, job_id)
                ),
            )

        self.assertEqual(events, [(1, 1, "0123456789abcdef01234567")])
