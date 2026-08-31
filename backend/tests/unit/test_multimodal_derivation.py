from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.derivation import build_derivation_plan, dependency_readiness
from multimodal_science.data.manifest import sha256_file


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def split_record(source_mzml: str, artifact_hash: str) -> dict[str, object]:
    return {
        "record_id": "record-1",
        "dataset_version": "raw-test0001",
        "dataset_name": "traindata3",
        "sample_id": Path(source_mzml).name,
        "source_mzml": source_mzml,
        "artifact_hash": artifact_hash,
        "split_group": source_mzml,
        "split": "train",
        "evaluation_tier": "primary_development",
        "train_eligible": True,
        "audit_bucket": False,
        "component": "compound-a",
        "channel": "quantifier",
        "expected_rt": 1.25,
        "peak_label": 1,
        "peak_intervals": [[1.2, 1.3]],
    }


class MultimodalDerivationTests(unittest.TestCase):
    def test_dependency_gate_reports_available_and_missing_modules(self) -> None:
        status = dependency_readiness(("json", "module_that_does_not_exist_12345"))

        self.assertTrue(status["json"])
        self.assertFalse(status["module_that_does_not_exist_12345"])

    def test_plan_verifies_sources_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            supervised_source = data_root / "mzml" / "traindata3" / "sample.mzML"
            unlabeled_source = data_root / "mzml" / "test2" / "unlabeled.mzML"
            supervised_source.parent.mkdir(parents=True)
            unlabeled_source.parent.mkdir(parents=True)
            supervised_source.write_text("<mzML><chromatogram/></mzML>", encoding="utf-8")
            unlabeled_source.write_text("<mzML><chromatogram/></mzML>", encoding="utf-8")

            split_manifest = root / "split_manifest.jsonl"
            source_relative = "mzml/traindata3/sample.mzML"
            write_jsonl(
                split_manifest,
                [split_record(source_relative, sha256_file(supervised_source))],
            )
            audit_report = root / "audit_report.json"
            audit_report.write_text(
                json.dumps(
                    {
                        "dataset_version": "raw-test0001",
                        "unlabeled_mzml": ["mzml/test2/unlabeled.mzML"],
                    }
                ),
                encoding="utf-8",
            )

            first = build_derivation_plan(
                split_manifest,
                data_root,
                root / "first-output",
                audit_report_path=audit_report,
            )
            second = build_derivation_plan(
                split_manifest,
                data_root,
                root / "second-output",
                audit_report_path=audit_report,
            )
            jobs = [
                json.loads(line)
                for line in first.plan_path.read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads(first.report_path.read_text(encoding="utf-8"))

        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(first.job_count, 2)
        self.assertEqual(first.source_count, 2)
        self.assertEqual({job["derivation_mode"] for job in jobs}, {
            "label_driven",
            "channel_driven_inference",
        })
        supervised = next(job for job in jobs if job["derivation_mode"] == "label_driven")
        unlabeled = next(
            job for job in jobs if job["derivation_mode"] == "channel_driven_inference"
        )
        self.assertEqual(supervised["labels"][0]["record_id"], "record-1")
        self.assertTrue(supervised["metrics_allowed"])
        self.assertFalse(unlabeled["metrics_allowed"])
        self.assertEqual(report["extractor_contract"]["public_name"], "ChromPeakFormer")
        self.assertFalse(report["dependency_gate"]["installation_attempted"])

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            source = data_root / "mzml" / "traindata3" / "sample.mzML"
            source.parent.mkdir(parents=True)
            source.write_text("<mzML/>", encoding="utf-8")
            split_manifest = root / "split_manifest.jsonl"
            write_jsonl(
                split_manifest,
                [split_record("mzml/traindata3/sample.mzML", "incorrect-hash")],
            )

            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                build_derivation_plan(split_manifest, data_root, root / "output")


if __name__ == "__main__":
    unittest.main()
