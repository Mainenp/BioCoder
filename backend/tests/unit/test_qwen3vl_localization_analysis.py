from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_RECORD_SCHEMA,
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.localization_analysis import (
    REPORT_SCHEMA,
    _failures,
    _localization_metrics,
    _positive_grounding,
    _stratum,
    _tertile_boundaries,
    analyze_localization,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(root: Path, *names: str) -> None:
    (root / "artifact_manifest.sha256").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def _classification() -> dict:
    return {"macro_f1": 0.9, "mcc": 0.8, "false_positive_rate": 0.1, "recall": 0.95}


def _task_metrics() -> dict:
    return {
        "peak_presence": {"classification": _classification()},
        "peak_presence_metadata": {"classification": _classification()},
        "scientific_qc": {"exact_match_rate": 0.9},
        "peak_grounding": {"grounding": {"mean_bbox_iou_all": 0.5, "iou_at_0_5_rate_all": 0.5}},
    }


def _rows(label: str, *, drift: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for group in range(11):
        for language in ("en", "zh-CN"):
            asset = f"asset-{group}"
            for task in ("peak_presence", "peak_presence_metadata"):
                rows.append({"schema_version": BILINGUAL_EVALUATION_RECORD_SCHEMA, "instruction_id": f"{asset}-{language}-{task}", "pair_id": f"{asset}-{task}", "asset_id": asset, "group_id": f"group-{group}", "task": task, "language": language, "target_peak_present": True, "predicted_peak_present": True, "classification_score": 1.0, "schema_valid": True, "exact_match": True})
            rows.append({"schema_version": BILINGUAL_EVALUATION_RECORD_SCHEMA, "instruction_id": f"{asset}-{language}-scientific_qc", "pair_id": f"{asset}-scientific_qc", "asset_id": asset, "group_id": f"group-{group}", "task": "scientific_qc", "language": language, "schema_valid": True, "exact_match": True, "predicted_qc_state": "pass", "predicted_reason": "ok"})
            expected = [10.0 + group * 10.0, 0.0, 20.0 + group * 10.0, 20.0]
            if label == "c0":
                predicted = [0.0, 0.0, 5.0, 20.0]
            elif label == "c1":
                predicted = expected if group != 0 else [11.0, 0.0, 21.0, 20.0]
            else:
                predicted = expected if group != 2 else [15.0, 0.0, 25.0, 20.0]
            iou = 1.0 if predicted == expected else (0.8181818181818182 if label == "c1" else 0.3333333333333333)
            if drift == "language" and language == "zh-CN":
                asset = "different"
            rows.append({"schema_version": BILINGUAL_EVALUATION_RECORD_SCHEMA, "instruction_id": f"asset-{group}-{language}-peak_grounding", "pair_id": f"asset-{group}-peak_grounding", "asset_id": asset, "group_id": f"group-{group}", "task": "peak_grounding", "language": language, "target_peak_present": True, "image_width": 100, "image_height": 20, "expected_bbox_2d": expected, "predicted_bbox_2d": predicted, "bbox_iou": iou, "iou_at_0_5": iou >= 0.5, "schema_valid": True, "exact_match": predicted == expected})
    return rows


def _run(root: Path, label: str, *, drift: str | None = None) -> Path:
    directory = root / label
    directory.mkdir()
    rows = _rows(label, drift=drift)
    records = directory / "evaluation_records.jsonl"
    records.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "prediction_generation_provenance_verified": True,
        "counts": {"predictions": len(rows), "valid_json": len(rows)},
        "metrics": _task_metrics(),
        "metrics_by_language": {"en": _task_metrics(), "zh-CN": _task_metrics()},
        "artifacts": {"evaluation_records": {"path": records.name, "sha256": sha256_file(records), "records": len(rows)}},
    }
    _write_json(directory / "qwen_evaluation_report.json", report)
    _manifest(directory, "qwen_evaluation_report.json", "evaluation_records.jsonl")
    return directory


def _metadata(path: Path) -> str:
    rows = []
    for group in range(11):
        rows.append({"task": "peak_grounding", "pair_id": f"asset-{group}-peak_grounding", "asset_id": f"asset-{group}", "group_id": f"group-{group}", "raw_max": 10.0 * (group + 1), "roi_window_minutes": [1.0, 3.0]})
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return sha256_file(path)


class LocalizationAnalysisTests(unittest.TestCase):
    def _build(self, directory: Path, **overrides: object):
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / "metadata.jsonl"
        digest = _metadata(metadata)
        kwargs = {"c0_evaluation_root": _run(directory, "c0"), "c1_evaluation_root": _run(directory, "c1"), "rl_evaluation_root": _run(directory, "rl"), "validation_metadata_path": metadata, "expected_validation_metadata_sha256": digest, "output_dir": directory / "analysis"}
        kwargs.update(overrides)
        return analyze_localization(**kwargs)

    def test_positive_metrics_threshold_equality_and_seconds(self) -> None:
        row = {"expected_bbox_2d": [10.0, 0.0, 20.0, 10.0], "predicted_bbox_2d": [10.0, 0.0, 20.0, 10.0], "bbox_iou": 0.5, "schema_valid": True, "image_width": 100, "roi_window_minutes": [1.0, 3.0]}
        metrics = _localization_metrics([row])
        self.assertEqual(metrics["iou_at_0_5"], 1.0)
        self.assertEqual(metrics["iou_at_0_75"], 0.0)
        self.assertEqual(metrics["start_error_pixels"], 0.0)
        self.assertEqual(metrics["start_error_seconds"], 0.0)

    def test_fixed_tertiles_are_shared_and_ties_are_low(self) -> None:
        rows = []
        for raw_max in (1.0, 1.0, 1.0, 9.0, 9.0, 20.0):
            rows.append({"expected_bbox_2d": [10.0, 0.0, 20.0, 10.0], "image_width": 100, "raw_max": raw_max})
        boundaries = _tertile_boundaries(rows)
        self.assertEqual(_stratum(rows[0], "log1p_raw_max", boundaries["log1p_raw_max"]), "low")
        self.assertIn("tie_policy", boundaries["true_peak_width_pixels"])

    def test_positive_filter_excludes_negative_grounding_and_scopes_language(self) -> None:
        base = {"task": "peak_grounding", "expected_bbox_2d": [1.0, 0.0, 2.0, 1.0]}
        rows = [
            {**base, "instruction_id": "en-positive", "language": "en", "target_peak_present": True},
            {**base, "instruction_id": "zh-negative", "language": "zh-CN", "target_peak_present": False},
        ]
        self.assertEqual([row["instruction_id"] for row in _positive_grounding(rows, "overall")], ["en-positive"])
        self.assertEqual([row["instruction_id"] for row in _positive_grounding(rows, "en")], ["en-positive"])

    def test_failures_are_deterministic_and_limited(self) -> None:
        baseline = []
        rl = []
        for index in range(25):
            row = {"instruction_id": f"id-{index}", "pair_id": f"pair-{index}", "asset_id": "asset", "group_id": "group", "language": "en", "expected_bbox_2d": [1.0, 0.0, 5.0, 4.0], "image_width": 10, "predicted_bbox_2d": [1.0, 0.0, 5.0, 4.0], "bbox_iou": 1.0, "schema_valid": True}
            baseline.append(row)
            rl.append({**row, "predicted_bbox_2d": [6.0, 0.0, 9.0, 4.0], "bbox_iou": 0.0})
        first = _failures(baseline, rl, seed=29)
        self.assertEqual(first, _failures(baseline, rl, seed=29))
        self.assertEqual(len(first), 20)

    def test_builds_scoped_hash_bound_analysis_and_deterministic_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._build(root)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["paired_group_bootstrap"]["iterations"], 10_000)
            self.assertEqual(report["localization"]["rl"]["en"]["positive_grounding_rows"], 11)
            self.assertEqual(report["paired_group_bootstrap"]["comparisons"]["rl_minus_c1"]["overall"]["source_groups"], 11)
            self.assertLessEqual(report["failure_cases"]["records"], 20)
            self.assertFalse(report["internal_test_accessed"])
            self.assertIn("localization_failure_cases.jsonl", result.manifest_path.read_text(encoding="utf-8"))

    def test_rejects_metadata_hash_and_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.jsonl"
            digest = _metadata(metadata)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze_localization(c0_evaluation_root=_run(root, "c0"), c1_evaluation_root=_run(root, "c1"), rl_evaluation_root=_run(root, "rl"), validation_metadata_path=metadata, expected_validation_metadata_sha256="0" * 64, output_dir=root / "analysis")
            root2 = root / "drift"
            root2.mkdir()
            metadata2 = root2 / "metadata.jsonl"
            digest2 = _metadata(metadata2)
            with self.assertRaisesRegex(ValueError, "identity drift"):
                analyze_localization(c0_evaluation_root=_run(root2, "c0"), c1_evaluation_root=_run(root2, "c1"), rl_evaluation_root=_run(root2, "rl", drift="language"), validation_metadata_path=metadata2, expected_validation_metadata_sha256=digest2, output_dir=root2 / "analysis")

    def test_paired_bootstrap_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.jsonl"
            digest = _metadata(metadata)
            c0, c1, rl = _run(root, "c0"), _run(root, "c1"), _run(root, "rl")
            first = analyze_localization(c0_evaluation_root=c0, c1_evaluation_root=c1, rl_evaluation_root=rl, validation_metadata_path=metadata, expected_validation_metadata_sha256=digest, output_dir=root / "one")
            second = analyze_localization(c0_evaluation_root=c0, c1_evaluation_root=c1, rl_evaluation_root=rl, validation_metadata_path=metadata, expected_validation_metadata_sha256=digest, output_dir=root / "two")
            one = json.loads(first.report_path.read_text(encoding="utf-8"))
            two = json.loads(second.report_path.read_text(encoding="utf-8"))
            self.assertEqual(one["paired_group_bootstrap"], two["paired_group_bootstrap"])
            self.assertEqual(first.failure_cases_path.read_bytes(), second.failure_cases_path.read_bytes())

    def test_rejects_protected_path_and_wrong_bootstrap_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.jsonl"
            digest = _metadata(metadata)
            protected = root / "internal-test"
            protected.mkdir()
            with self.assertRaisesRegex(ValueError, "internal test"):
                analyze_localization(c0_evaluation_root=protected, c1_evaluation_root=_run(root, "c1"), rl_evaluation_root=_run(root, "rl"), validation_metadata_path=metadata, expected_validation_metadata_sha256=digest, output_dir=root / "analysis")
            with self.assertRaisesRegex(ValueError, "exactly 10,000"):
                self._build(root / "iterations", bootstrap_iterations=100)


if __name__ == "__main__":
    unittest.main()
