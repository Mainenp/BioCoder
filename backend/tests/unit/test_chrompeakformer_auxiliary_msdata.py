from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.chrompeakformer.auxiliary_msdata import (
    build_auxiliary_msdata_dataset,
)
from multimodal_science.data.manifest import sha256_file


def write_mzml(path: Path, chromatograms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'<chromatogram id="c{index}"/>' for index in range(chromatograms))
    path.write_text(f"<mzML><run><chromatogramList>{body}</chromatogramList></run></mzML>", encoding="utf-8")


def fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    quarantine = root / "quarantine"
    converted = root / "converted"
    quarantine.mkdir()
    source_sha = "a" * 64
    source_stem = f"msdata-{source_sha[:12]}"
    records = [
        {
            "schema_version": "chrompeak-msdata-quarantine-record-v1",
            "record_id": "source-record",
            "source_file": f"{source_stem}.msdata",
            "source_artifact_sha256": source_sha,
            "source_group": f"msdata:{source_sha[:24]}",
            "acquisition_frames": 2,
            "transition_trace_candidates": 5,
            "annotation_like_fields_detected": [],
            "label_status": "unlabeled",
            "train_eligible": False,
            "benchmark_eligible": False,
            "frames": [
                {
                    "frame_id": "frame-0",
                    "frame_index": 0,
                    "transition_trace_candidates": 2,
                },
                {
                    "frame_id": "frame-1",
                    "frame_index": 1,
                    "transition_trace_candidates": 3,
                },
            ],
        }
    ]
    manifest = quarantine / "msdata_manifest.jsonl"
    manifest.write_text(json.dumps(records[0], separators=(",", ":")) + "\n", encoding="utf-8")
    report = quarantine / "msdata_quarantine_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "chrompeak-msdata-quarantine-report-v1",
                "dataset_version": "msdata-test0001",
                "dataset_digest_sha256": "b" * 64,
                "manifest_sha256": sha256_file(manifest),
                "counts": {
                    "source_files": 1,
                    "independent_source_groups": 1,
                    "acquisition_frames": 2,
                    "transition_trace_candidates": 5,
                },
                "contracts": {
                    "labels_present": False,
                    "annotation_like_fields_detected": False,
                    "train_eligible": False,
                    "benchmark_eligible": False,
                },
            }
        ),
        encoding="utf-8",
    )
    write_mzml(converted / source_stem / f"{source_stem}_1.mzML", 3)
    write_mzml(converted / source_stem / f"{source_stem}_2.mzML", 4)
    converter = root / "converter.exe"
    converter.write_bytes(b"converter-test")
    return manifest, report, converted, converter


class AuxiliaryMsdataTests(unittest.TestCase):
    def test_slurm_extractor_is_cpu_only_and_auxiliary_only(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "chrompeakformer"
            / "slurm"
            / "coder_auxiliary_extract.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn("--split auxiliary_unlabeled_train", script)
        self.assertIn("AUXILIARY_LABELS=0", script)
        self.assertIn("AUXILIARY_METRICS_ALLOWED=false", script)
        self.assertIn("sha256sum -c artifact_manifest.sha256", script)
        self.assertNotIn("validation", script.casefold())
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)

    def test_builds_hash_bound_unlabeled_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, report, converted, converter = fixture(root)
            result = build_auxiliary_msdata_dataset(
                quarantine_manifest_path=manifest,
                quarantine_report_path=report,
                converted_root=converted,
                converter_path=converter,
                output_dir=root / "output",
            )
            payload = json.loads(result.report_path.read_text(encoding="utf-8"))
            plan = [json.loads(line) for line in result.plan_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result.acquisition_frames, 2)
        self.assertEqual(result.transition_traces, 5)
        self.assertEqual(payload["counts"]["chromatograms"], 7)
        self.assertEqual(payload["counts"]["strict_mzml_files"], 2)
        self.assertEqual(payload["counts"]["labels"], 0)
        self.assertTrue(payload["quality_gate_passed"])
        self.assertEqual({job["split"] for job in plan}, {"auxiliary_unlabeled_train"})
        self.assertEqual({job["derivation_mode"] for job in plan}, {"channel_driven_inference"})
        self.assertEqual({job["metrics_allowed"] for job in plan}, {False})

    def test_normalizes_recoverable_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, report, converted, converter = fixture(root)
            source_stem = "msdata-aaaaaaaaaaaa"
            path = converted / source_stem / f"{source_stem}_1.mzML"
            content = path.read_bytes().replace(b'<mzML>', b'<mzML note="\xff">', 1)
            path.write_bytes(content)
            result = build_auxiliary_msdata_dataset(
                quarantine_manifest_path=manifest,
                quarantine_report_path=report,
                converted_root=converted,
                converter_path=converter,
                output_dir=root / "output",
            )
            payload = json.loads(result.report_path.read_text(encoding="utf-8"))
            normalized = result.output_dir / "mzml" / source_stem / f"{source_stem}_1.mzML"
            normalized_text = normalized.read_text(encoding="utf-8")

        self.assertEqual(payload["counts"]["normalized_invalid_utf8_mzml_files"], 1)
        self.assertIn("\ufffd", normalized_text)

    def test_rejects_missing_or_extra_converted_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, report, converted, converter = fixture(root)
            write_mzml(converted / "extra.mzML", 1)
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                build_auxiliary_msdata_dataset(
                    quarantine_manifest_path=manifest,
                    quarantine_report_path=report,
                    converted_root=converted,
                    converter_path=converter,
                    output_dir=root / "output",
                )

    def test_rejects_chromatogram_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, report, converted, converter = fixture(root)
            source_stem = "msdata-aaaaaaaaaaaa"
            write_mzml(converted / source_stem / f"{source_stem}_1.mzML", 2)
            with self.assertRaisesRegex(ValueError, "Chromatogram count mismatch"):
                build_auxiliary_msdata_dataset(
                    quarantine_manifest_path=manifest,
                    quarantine_report_path=report,
                    converted_root=converted,
                    converter_path=converter,
                    output_dir=root / "output",
                )

    def test_rejects_supervised_or_benchmark_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, report, converted, converter = fixture(root)
            report_value = json.loads(report.read_text(encoding="utf-8"))
            report_value["contracts"]["labels_present"] = True
            report.write_text(json.dumps(report_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supervised path"):
                build_auxiliary_msdata_dataset(
                    quarantine_manifest_path=manifest,
                    quarantine_report_path=report,
                    converted_root=converted,
                    converter_path=converter,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
