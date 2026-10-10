from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.run_sequence_prompt_inference_cli import parser
from multimodal_science.qwen3vl.sequence_prompt_data import (
    SEQUENCE_PROMPT_BUNDLE_SCHEMA,
    build_sequence_prompt_bundle,
)
from multimodal_science.qwen3vl.sequence_prompt_inference import (
    _verify_sequence_expert,
    sequence_evidence_prompt,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _source_sequence_run(
    root: Path,
    *,
    include_report_in_manifest: bool = True,
    manifest_dot_prefix: bool = False,
) -> tuple[str, str]:
    rows = [
        {
            "schema_version": "chrompeak-sequence-prediction-v1",
            "row": index,
            "asset_id": f"asset-{index}",
            "group_id": f"group-{index}",
            "presence_probability": 0.25 + index * 0.5,
            "start_normalized": 0.1,
            "end_normalized": 0.4,
            "roi_width_minutes": 3.0,
            "target_peak_present": bool(index),
            "target_start_normalized": 0.2 if index else None,
            "target_end_normalized": 0.3 if index else None,
        }
        for index in range(2)
    ]
    predictions = root / "validation_predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = root / "scientific_report.json"
    _write_json(
        report,
        {
            "schema_version": "chrompeak-sequence-baseline-report-v1",
            "development_comparison_eligible": True,
            "final_benchmark_eligible": False,
            "internal_test_accessed": False,
            "config": {"modality": "sequence"},
            "artifacts": {
                "validation_predictions": {
                    "path": predictions.name,
                    "sha256": sha256_file(predictions),
                    "records": len(rows),
                }
            },
        },
    )
    manifest = root / "artifact_manifest.sha256"
    manifest_paths = [predictions]
    if include_report_in_manifest:
        manifest_paths.insert(0, report)
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{'./' if manifest_dot_prefix else ''}{path.name}\n"
            for path in manifest_paths
        ),
        encoding="utf-8",
    )
    return sha256_file(report), sha256_file(manifest)


class SequencePromptContractTests(unittest.TestCase):
    def test_builder_normalizes_recovery_manifest_dot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            report_sha, manifest_sha = _source_sequence_run(
                source,
                include_report_in_manifest=False,
                manifest_dot_prefix=True,
            )

            result = build_sequence_prompt_bundle(
                sequence_run_root=source,
                sequence_report_sha256=report_sha,
                sequence_manifest_sha256=manifest_sha,
                output_dir=root / "bundle",
                code_revision="a" * 40,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["contracts"]["source_predictions_manifest_bound"])

    def test_builder_rejects_duplicate_normalized_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            report_sha, _ = _source_sequence_run(
                source,
                include_report_in_manifest=False,
            )
            predictions = source / "validation_predictions.jsonl"
            manifest = source / "artifact_manifest.sha256"
            digest = sha256_file(predictions)
            manifest.write_text(
                f"{digest}  {predictions.name}\n{digest}  ./{predictions.name}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                build_sequence_prompt_bundle(
                    sequence_run_root=source,
                    sequence_report_sha256=report_sha,
                    sequence_manifest_sha256=sha256_file(manifest),
                    output_dir=root / "bundle",
                    code_revision="a" * 40,
                )

    def test_builder_accepts_report_pinned_separately_from_recovery_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            report_sha, manifest_sha = _source_sequence_run(
                source, include_report_in_manifest=False
            )

            result = build_sequence_prompt_bundle(
                sequence_run_root=source,
                sequence_report_sha256=report_sha,
                sequence_manifest_sha256=manifest_sha,
                output_dir=root / "bundle",
                code_revision="a" * 40,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["contracts"]["sequence_report_sha256_pinned"])
            self.assertTrue(report["contracts"]["source_predictions_manifest_bound"])

    def test_builder_removes_targets_and_binds_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            report_sha, manifest_sha = _source_sequence_run(source)
            output = root / "bundle"

            result = build_sequence_prompt_bundle(
                sequence_run_root=source,
                sequence_report_sha256=report_sha,
                sequence_manifest_sha256=manifest_sha,
                output_dir=output,
                code_revision="a" * 40,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], SEQUENCE_PROMPT_BUNDLE_SCHEMA)
            self.assertTrue(report["contracts"]["target_fields_excluded"])
            self.assertFalse(report["contracts"]["instruction_answers_opened"])
            self.assertIn("target_peak_present", report["sanitization"]["discarded_source_fields"])
            text = result.predictions_path.read_text(encoding="utf-8")
            self.assertNotIn("target_", text)
            self.assertNotIn("roi_width_minutes", text)
            self.assertEqual(result.prediction_records, 2)

            links = {
                f"instruction-{index}": {
                    "asset_id": f"asset-{index}",
                    "group_id": f"group-{index}",
                }
                for index in range(2)
            }
            expert = _verify_sequence_expert(
                root=output,
                report_sha256=result.report_sha256,
                manifest_sha256=sha256_file(output / "artifact_manifest.sha256"),
                validation_links=links,
            )
            self.assertEqual(set(expert.predictions_by_asset), {"asset-0", "asset-1"})
            self.assertFalse(expert.metadata["target_fields_exposed"])

    def test_expert_bundle_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            report_sha, manifest_sha = _source_sequence_run(source)
            output = root / "bundle"
            result = build_sequence_prompt_bundle(
                sequence_run_root=source,
                sequence_report_sha256=report_sha,
                sequence_manifest_sha256=manifest_sha,
                output_dir=output,
                code_revision="a" * 40,
            )
            manifest_digest = sha256_file(output / "artifact_manifest.sha256")
            result.predictions_path.write_text("tampered\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _verify_sequence_expert(
                    root=output,
                    report_sha256=result.report_sha256,
                    manifest_sha256=manifest_digest,
                    validation_links={},
                )

    def test_prompt_is_language_matched_and_prediction_only(self) -> None:
        prediction = {
            "presence_probability": 0.75,
            "start_normalized": 0.1,
            "end_normalized": 0.4,
            "target_peak_present": "MUST_NOT_APPEAR",
        }

        english = sequence_evidence_prompt("<image> Return JSON.", "en", prediction)
        chinese = sequence_evidence_prompt("<image> 返回 JSON。", "zh-CN", prediction)

        self.assertIn("not ground truth", english)
        self.assertIn("不是标准答案", chinese)
        self.assertIn('"presence_probability":0.75', english)
        self.assertNotIn("MUST_NOT_APPEAR", english + chinese)
        self.assertEqual(english.count("<image>"), 1)

    def test_cli_and_slurm_have_no_answer_input_during_generation(self) -> None:
        destinations = {action.dest for action in parser()._actions}
        self.assertNotIn("answers", destinations)
        self.assertNotIn("internal_test", destinations)
        self.assertIn("sequence_prompt_bundle_root", destinations)

        script = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_sequence_prompt_evaluate.sbatch"
        ).read_text(encoding="utf-8")
        generation_start = script.index("run_sequence_prompt_inference_cli")
        evaluation_start = script.index("evaluate_predictions_cli")
        generation_block = script[generation_start:evaluation_start]
        self.assertIn("SEQUENCE_PROMPT_BUNDLE_CONTRACT=OK", script)
        self.assertIn('report["code_revision"] == sys.argv[2]', script)
        self.assertIn('report["sources"]["sequence_report_sha256"]', script)
        self.assertIn('report["contracts"]["sequence_report_sha256_pinned"]', script)
        self.assertIn('report["contracts"]["source_predictions_manifest_bound"]', script)
        self.assertNotIn("--instruction-root", generation_block)
        self.assertNotIn("--instruction-report-sha256", generation_block)
        self.assertIn('target_fields_exposed"] is False', script)
        self.assertIn("QWEN3VL_SEQUENCE_PROMPT_EVALUATION=OK", script)


if __name__ == "__main__":
    unittest.main()
