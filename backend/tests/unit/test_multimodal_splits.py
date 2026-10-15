from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from multimodal_science.data.splits import build_splits


def make_record(
    index: int,
    *,
    dataset_name: str,
    group: str,
    sample_id: str,
    peak_label: int = 1,
    artifact_hash: str | None = None,
    train_eligible: bool = True,
    audit_bucket: bool = False,
) -> dict[str, object]:
    return {
        "record_id": f"record-{index:04d}",
        "dataset_version": "raw-test0001",
        "dataset_name": dataset_name,
        "label_variant": "primary" if train_eligible else "alternate",
        "sample_id": sample_id,
        "source_mzml": group,
        "artifact_hash": artifact_hash or f"hash-{index:04d}",
        "train_eligible": train_eligible,
        "benchmark_eligible": train_eligible,
        "audit_bucket": audit_bucket,
        "exclusion_reasons": [] if train_eligible else ["alternate_label_variant"],
        "split_group": group,
        "peak_label": peak_label,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MultimodalSplitTests(unittest.TestCase):
    def primary_fixture(self) -> list[dict[str, object]]:
        records = []
        index = 0
        for stratum, group_count in (("BLANK", 9), ("QC", 10), ("SAMPLE", 90)):
            for group_index in range(group_count):
                group = f"mzml/traindata3/{stratum}-{group_index}.mzML"
                records.append(
                    make_record(
                        index,
                        dataset_name="traindata3",
                        group=group,
                        sample_id=Path(group).name,
                        peak_label=0 if stratum == "BLANK" else 1,
                        artifact_hash=f"content-{stratum}-{group_index}",
                    )
                )
                index += 1
        return records

    def test_stratified_group_allocation_is_exact_and_leak_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, self.primary_fixture())

            result = build_splits(manifest, root / "output")
            records = read_jsonl(result.split_manifest_path)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        group_splits: dict[str, set[str]] = defaultdict(set)
        for record in records:
            group_splits[str(record["split_group"])].add(str(record["split"]))
        split_counts = Counter(next(iter(splits)) for splits in group_splits.values())

        self.assertEqual(split_counts["train"], 87)
        self.assertEqual(split_counts["validation"], 11)
        self.assertEqual(split_counts["internal_test"], 11)
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        self.assertEqual(
            report["primary_group_allocation"],
            {
                "blank": {"groups": 9, "train": 7, "validation": 1, "internal_test": 1},
                "qc": {"groups": 10, "train": 8, "validation": 1, "internal_test": 1},
                "sample": {
                    "groups": 90,
                    "train": 72,
                    "validation": 9,
                    "internal_test": 9,
                },
            },
        )
        self.assertTrue(report["leakage_audit"]["passed"])

    def test_assignment_is_independent_of_manifest_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.primary_fixture()
            first_manifest = root / "first.jsonl"
            second_manifest = root / "second.jsonl"
            write_jsonl(first_manifest, records)
            write_jsonl(second_manifest, list(reversed(records)))

            first = build_splits(first_manifest, root / "first-output")
            second = build_splits(second_manifest, root / "second-output")

        self.assertEqual(first.split_manifest_sha256, second.split_manifest_sha256)

    def test_non_primary_populations_receive_explicit_evaluation_tiers(self) -> None:
        records = self.primary_fixture()[:3]
        records.extend(
            [
                make_record(
                    200,
                    dataset_name="traindata1",
                    group="mzml/traindata1/negative.mzML",
                    sample_id="negative.mzML",
                    peak_label=0,
                ),
                make_record(
                    201,
                    dataset_name="test1",
                    group="mzml/test1/legacy.mzML",
                    sample_id="legacy.mzML",
                ),
                make_record(
                    202,
                    dataset_name="test1",
                    group="mzml/test1/legacy.mzML",
                    sample_id="legacy.mzML",
                    train_eligible=False,
                    audit_bucket=True,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            audit_report = root / "audit_report.json"
            write_jsonl(manifest, records)
            audit_report.write_text(
                json.dumps(
                    {
                        "dataset_version": "raw-test0001",
                        "unlabeled_mzml": ["mzml/test2/unlabeled.mzML"],
                    }
                ),
                encoding="utf-8",
            )

            result = build_splits(
                manifest, root / "output", audit_report_path=audit_report
            )
            by_id = {
                record["record_id"]: record
                for record in read_jsonl(result.split_manifest_path)
            }
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(by_id["record-0200"]["split"], "auxiliary_train")
        self.assertEqual(by_id["record-0200"]["evaluation_tier"], "auxiliary_negative_only")
        self.assertEqual(by_id["record-0201"]["split"], "legacy_external")
        self.assertEqual(
            by_id["record-0201"]["evaluation_tier"], "legacy_external_non_pristine"
        )
        self.assertEqual(by_id["record-0202"]["split"], "audit")
        self.assertEqual(
            report["unlabeled_external"]["sources"], ["mzml/test2/unlabeled.mzML"]
        )
        self.assertFalse(report["unlabeled_external"]["metrics_allowed"])

    def test_duplicate_primary_content_across_groups_is_rejected(self) -> None:
        records = [
            make_record(
                1,
                dataset_name="traindata3",
                group="mzml/traindata3/a.mzML",
                sample_id="a.mzML",
                artifact_hash="same-content",
            ),
            make_record(
                2,
                dataset_name="traindata3",
                group="mzml/traindata3/b.mzML",
                sample_id="b.mzML",
                artifact_hash="same-content",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, records)

            with self.assertRaisesRegex(ValueError, "Duplicate primary mzML content"):
                build_splits(manifest, root / "output")


if __name__ == "__main__":
    unittest.main()
