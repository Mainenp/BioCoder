from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from multimodal_science.chrompeakformer.auxiliary_asset_index import (
    build_auxiliary_asset_index,
)
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


def write_outputs(output_dir: Path) -> None:
    features = [
        {
            "Compound Name": "1",
            "native_id": "SRM SIC Q1=100.1 Q3=50.1 transition=1",
            "mz": "100.1",
            "q3": "50.1",
            "RT": "1.0",
        },
        {
            "Compound Name": "2",
            "native_id": "SRM SIC Q1=200.1 Q3=70.1 transition=2",
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


def make_job(source_hash: str) -> dict[str, object]:
    job_id = "0123456789abcdef01234567"
    return {
        "job_id": job_id,
        "dataset_version": "msdata-test0001",
        "derivation_mode": "channel_driven_inference",
        "split": "auxiliary_unlabeled_train",
        "evaluation_tier": "auxiliary_training_only",
        "metrics_allowed": False,
        "split_group": "msdata:independent-source-a",
        "source_mzml": "mzml/source/frame-1.mzML",
        "artifact_hash": source_hash,
        "record_count": 0,
        "positive": None,
        "negative": None,
        "labels": [],
        "output_prefix": f"jobs/auxiliary_unlabeled_train/{job_id}",
    }


def write_plan(path: Path, jobs: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(job, separators=(",", ":")) + "\n" for job in jobs),
        encoding="utf-8",
    )


class ChromPeakFormerAuxiliaryAssetIndexTests(unittest.TestCase):
    def test_slurm_indexer_is_cpu_only_complete_and_unlabeled(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "chrompeakformer"
            / "slurm"
            / "coder_auxiliary_index.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn("auxiliary_index_cli", script)
        self.assertIn("AUXILIARY_TRANSITION_CANDIDATES", script)
        self.assertIn("AUXILIARY_EXCLUDED_OR_DEDUPLICATED_TRACES", script)
        self.assertIn("extracted_traces > transition_candidates", script)
        self.assertIn("AUXILIARY_LABELS=0", script)
        self.assertIn("AUXILIARY_METRICS_ALLOWED=false", script)
        self.assertIn("sha256sum -c artifact_manifest.sha256", script)
        self.assertNotIn("allow-partial", script)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)

    def fixture(
        self, root: Path, *, publish: bool = True, job_update: dict[str, object] | None = None
    ) -> tuple[Path, Path, Path, dict[str, object]]:
        data_root = root / "data"
        source = data_root / "mzml" / "source" / "frame-1.mzML"
        source.parent.mkdir(parents=True)
        source.write_text("<mzML/>", encoding="utf-8")
        job = make_job(sha256_file(source))
        if job_update:
            job.update(job_update)
        plan = root / "derivation_plan.jsonl"
        write_plan(plan, [job])
        assets_root = root / "assets"
        assets_root.mkdir()

        if publish:
            def extractor(_job: dict, _source: Path, output_dir: Path) -> dict:
                write_outputs(output_dir)
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
        return plan, assets_root, root / "auxiliary-index", job

    def test_builds_unlabeled_only_trace_index_and_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir, job = self.fixture(root)
            events: list[tuple[int, int, str]] = []
            result = build_auxiliary_asset_index(
                plan,
                assets_root,
                output_dir,
                progress_callback=lambda completed, total, job_id: events.append(
                    (completed, total, job_id)
                ),
            )
            assets = [
                json.loads(line)
                for line in result.index_path.read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            manifest = result.manifest_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(result.asset_count, 2)
            self.assertEqual(result.source_group_count, 1)
            self.assertEqual(events, [(1, 1, job["job_id"])])
            self.assertNotIn("label", assets[0])
            self.assertEqual(assets[0]["xic"]["signal_row"], 1)
            self.assertEqual(assets[1]["xic"]["signal_row"], 2)
            self.assertEqual(assets[0]["xic"]["point_count"], 4)
            self.assertEqual(assets[0]["source_group"], "msdata:independent-source-a")
            self.assertFalse(assets[0]["metrics_allowed"])
            self.assertFalse(assets[0]["supervision"]["supervised_train_eligible"])
            self.assertTrue(
                assets[0]["supervision"]["auxiliary_unlabeled_train_eligible"]
            )
            self.assertEqual(report["counts"]["labels"], 0)
            self.assertEqual(report["counts"]["extracted_trace_assets"], 2)
            self.assertTrue(report["contracts"]["complete_plan_coverage"])
            self.assertEqual(len(manifest), 2)
            for line in manifest:
                digest, relative = line.split("  ", 1)
                self.assertEqual(digest, sha256_file(output_dir / relative))

    def test_rejects_any_labels_or_metrics_permission(self) -> None:
        for update, message in (
            ({"labels": [{"record_id": "1"}]}, "contains labels"),
            ({"metrics_allowed": True}, "Metrics must be disabled"),
        ):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, assets_root, output_dir, _ = self.fixture(
                    root, publish=False, job_update=update
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_auxiliary_asset_index(plan, assets_root, output_dir)

    def test_partial_auxiliary_index_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir, _ = self.fixture(root, publish=False)
            with self.assertRaisesRegex(
                FileNotFoundError, "partial auxiliary indexes are forbidden"
            ):
                build_auxiliary_asset_index(plan, assets_root, output_dir)

    def test_refuses_to_overwrite_an_existing_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, assets_root, output_dir, _ = self.fixture(root)
            build_auxiliary_asset_index(plan, assets_root, output_dir)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_auxiliary_asset_index(plan, assets_root, output_dir)


if __name__ == "__main__":
    unittest.main()
