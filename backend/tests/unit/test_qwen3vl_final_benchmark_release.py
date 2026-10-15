import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from biocoder.cli import build_parser
from multimodal_science.baselines.final_sequence_evaluation import (
    FINAL_SEQUENCE_EVALUATION_SCHEMA,
)
from multimodal_science.chrompeakformer.final_detector_evaluation import (
    FINAL_DETECTOR_EVALUATION_SCHEMA,
)
from multimodal_science.qwen3vl.final_benchmark_evaluation import (
    FINAL_QWEN_EVALUATION_SCHEMA,
)
from multimodal_science.final_benchmark_release import (
    _render_table,
    build_final_benchmark_release,
    verify_final_benchmark_evidence,
    verify_public_release,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, relatives: list[str]) -> Path:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(f"{_sha256(root / relative)}  {relative}\n" for relative in relatives),
        encoding="utf-8",
    )
    return manifest


class _ReleaseFixture:
    candidates = (
        "qwen3vl_zero_shot",
        "qwen3vl_image_lora",
        "qwen3vl_image_xic_fusion",
        "sequence_peak_net",
        "chrompeakformer",
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        self.protocol_root = root / "protocol"
        self.ledger = root / "ledger"
        self.report_root = root / "report"
        self.protocol_sha256 = ""
        self.evaluation_roots: dict[str, Path] = {}
        self._build()

    def _build_protocol(self) -> None:
        self.protocol_root.mkdir()
        _write_json(self.protocol_root / "candidate_lock.json", {"fixture": True})
        _write_json(self.protocol_root / "metrics_lock.json", {"fixture": True})
        protocol = self.protocol_root / "final_benchmark_protocol.json"
        _write_json(
            protocol,
            {
                "schema_version": "chrompeak-final-benchmark-protocol-v1",
                "pre_internal_test_ready": True,
                "internal_test_accessed": False,
                "sealed_split": {
                    "name": "internal_test",
                    "expected_assets": 1815,
                    "expected_source_groups": 11,
                },
            },
        )
        _write_manifest(
            self.protocol_root,
            ["final_benchmark_protocol.json", "candidate_lock.json", "metrics_lock.json"],
        )
        self.protocol_sha256 = _sha256(protocol)

    def _build_evaluations(self) -> dict[str, dict[str, str]]:
        registry: dict[str, dict[str, str]] = {}
        for candidate in self.candidates:
            root = self.root / f"evaluation-{candidate}"
            root.mkdir()
            if candidate.startswith("qwen3vl_"):
                filename = "qwen_evaluation_report.json"
                schema = FINAL_QWEN_EVALUATION_SCHEMA
                provenance_name = "inputs"
            elif candidate == "sequence_peak_net":
                filename = "sequence_evaluation_report.json"
                schema = FINAL_SEQUENCE_EVALUATION_SCHEMA
                provenance_name = "provenance"
            else:
                filename = "detector_evaluation_report.json"
                schema = FINAL_DETECTOR_EVALUATION_SCHEMA
                provenance_name = "provenance"
            report = root / filename
            _write_json(
                report,
                {
                    "schema_version": schema,
                    "candidate_name": candidate,
                    provenance_name: {
                        "protocol_sha256": self.protocol_sha256,
                        "access_id": "a" * 24,
                    },
                    "development_comparison_eligible": False,
                    "final_benchmark_eligible": True,
                    "internal_test_accessed": True,
                },
            )
            manifest = _write_manifest(root, [filename])
            self.evaluation_roots[candidate] = root
            registry[candidate] = {
                "root": str(root.resolve()),
                "report_path": str(report.resolve()),
                "report_sha256": _sha256(report),
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": _sha256(manifest),
            }
        return registry

    def _result_row(self, candidate: str, scope: str) -> dict:
        labels = {
            "qwen3vl_zero_shot": "Qwen3-VL zero-shot",
            "qwen3vl_image_lora": "Qwen3-VL image-only LoRA",
            "qwen3vl_image_xic_fusion": "Qwen3-VL image + aligned XIC",
            "sequence_peak_net": "SequencePeakNet",
            "chrompeakformer": "ChromPeakFormer",
        }
        specialist = candidate in {"sequence_peak_net", "chrompeakformer"}
        return {
            "candidate_name": candidate,
            "model": labels[candidate],
            "scope": scope,
            "classification": {
                "balanced_accuracy": 0.9,
                "macro_f1": 0.89,
                "mcc": 0.8,
                "false_positive_rate": 0.1,
            },
            "localization": {"mean_iou": 0.75, "iou_at_0_5_rate": 0.8},
            "scientific_qc_exact_match": None if specialist else 0.95,
            "valid_json_rate": None if specialist else 1.0,
            "schema_valid_rate": None if specialist else 0.99,
            "coco": (
                {"ap_50_95": 0.5, "ap_50": 0.8, "ap_75": 0.6}
                if candidate == "chrompeakformer"
                else None
            ),
        }

    def _build_report(self, evaluations: dict[str, dict[str, str]]) -> None:
        self.report_root.mkdir()
        registry = self.report_root / "final_evidence_registry.json"
        _write_json(
            registry,
            {
                "schema_version": "chrompeak-final-benchmark-registry-v1",
                "protocol_sha256": self.protocol_sha256,
                "access_id": "a" * 24,
                "candidate_order": list(self.candidates),
                "evaluations": evaluations,
                "internal_test_accessed": True,
            },
        )
        rows = [
            self._result_row(candidate, language)
            for candidate in self.candidates[:3]
            for language in ("en", "zh-CN")
        ]
        rows.extend(
            [
                self._result_row("sequence_peak_net", "language-neutral"),
                self._result_row("chrompeakformer", "language-neutral"),
            ]
        )
        overall_rows = [
            self._result_row(candidate, "overall") for candidate in self.candidates[:3]
        ]
        table = self.report_root / "final_benchmark_table.md"
        table.write_text(_render_table(rows), encoding="utf-8")
        report = self.report_root / "final_benchmark_report.json"
        _write_json(
            report,
            {
                "schema_version": "chrompeak-final-benchmark-report-v1",
                "protocol_sha256": self.protocol_sha256,
                "access_id": "a" * 24,
                "candidate_order": list(self.candidates),
                "language_separated_rows": rows,
                "qwen_overall_rows": overall_rows,
                "uncertainty": {
                    "bootstrap_iterations": 10_000,
                    "seed": 17,
                    "bootstrap_unit": "complete_source_mzml_group",
                },
                "artifacts": {
                    "main_table": {"path": table.name, "sha256": _sha256(table)},
                    "evidence_registry": {
                        "path": registry.name,
                        "sha256": _sha256(registry),
                    },
                },
                "contracts": {
                    "all_primary_candidates_present": True,
                    "same_one_time_access_event": True,
                    "candidate_selection_performed_after_test": False,
                    "threshold_selection_performed_after_test": False,
                    "combined_cross_task_score_reported": False,
                    "qwen_languages_reported_separately": True,
                    "source_group_uncertainty_reported_in_bound_inputs": True,
                    "additional_internal_test_access_authorized": False,
                    "biocoder_agent_promotion_claimed": False,
                },
                "development_comparison_eligible": False,
                "final_benchmark_eligible": True,
                "internal_test_accessed": True,
            },
        )
        _write_manifest(
            self.report_root,
            [report.name, table.name, registry.name],
        )

    def _build_ledger(self) -> None:
        self.ledger.mkdir()
        started = self.ledger / "access_started.json"
        _write_json(
            started,
            {
                "schema_version": "chrompeak-final-benchmark-access-v1",
                "access_id": "a" * 24,
                "access_sequence": 1,
                "protocol_sha256": self.protocol_sha256,
                "internal_test_accessed": True,
                "completed": False,
            },
        )
        (self.ledger / "access_started.sha256").write_text(
            f"{_sha256(started)}  access_started.json\n", encoding="utf-8"
        )
        final_manifest = self.report_root / "artifact_manifest.sha256"
        completed = self.ledger / "access_completed.json"
        _write_json(
            completed,
            {
                "schema_version": "chrompeak-final-benchmark-access-completion-v1",
                "access_id": "a" * 24,
                "protocol_sha256": self.protocol_sha256,
                "final_evidence_manifest_path": str(final_manifest.resolve()),
                "final_evidence_manifest_sha256": _sha256(final_manifest),
                "access_sequence": 1,
                "additional_test_access_authorized": False,
                "completed": True,
                "internal_test_accessed": True,
            },
        )
        _write_manifest(self.ledger, [started.name, completed.name])

    def _build(self) -> None:
        self._build_protocol()
        evaluations = self._build_evaluations()
        self._build_report(evaluations)
        self._build_ledger()


class FinalBenchmarkReleaseTests(unittest.TestCase):
    def test_verifies_and_builds_deterministic_path_free_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ReleaseFixture(Path(temporary))
            verified, _ = verify_final_benchmark_evidence(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha256,
                ledger_dir=fixture.ledger,
                report_root=fixture.report_root,
            )
            self.assertTrue(verified.access_completed)
            self.assertEqual(verified.candidates, 5)
            self.assertEqual(verified.rows, 8)

            first = build_final_benchmark_release(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha256,
                ledger_dir=fixture.ledger,
                report_root=fixture.report_root,
                output_dir=fixture.root / "public-v1-a",
            )
            second = build_final_benchmark_release(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha256,
                ledger_dir=fixture.ledger,
                report_root=fixture.report_root,
                output_dir=fixture.root / "public-v1-b",
            )
            self.assertEqual(first.archive_sha256, second.archive_sha256)
            release = verify_public_release(
                release_root=first.output_dir,
                archive_path=first.archive_path,
            )
            self.assertEqual(release.protocol_sha256, fixture.protocol_sha256)
            self.assertEqual(release.candidates, 5)
            for path in first.output_dir.iterdir():
                if path.is_file():
                    self.assertNotIn(str(fixture.root), path.read_text(encoding="utf-8"))

    def test_rejects_source_and_public_release_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ReleaseFixture(Path(temporary))
            release = build_final_benchmark_release(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha256,
                ledger_dir=fixture.ledger,
                report_root=fixture.report_root,
                output_dir=fixture.root / "public-v1",
            )
            (release.output_dir / "benchmark_table.md").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                verify_public_release(release_root=release.output_dir)

            evaluation = (
                fixture.evaluation_roots["qwen3vl_zero_shot"]
                / "qwen_evaluation_report.json"
            )
            evaluation.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                verify_final_benchmark_evidence(
                    protocol_root=fixture.protocol_root,
                    expected_protocol_sha256=fixture.protocol_sha256,
                    ledger_dir=fixture.ledger,
                    report_root=fixture.report_root,
                )

    def test_biocoder_cli_exposes_multimodal_release_commands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["multimodal", "verify-release", "--release-root", "release"]
        )
        self.assertEqual(args.command, "multimodal")
        self.assertEqual(args.multimodal_command, "verify-release")


if __name__ == "__main__":
    unittest.main()
