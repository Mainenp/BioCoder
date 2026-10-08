from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FINAL_BENCHMARK_PROTOCOL_SCHEMA,
    _requested_model_revision,
    complete_final_benchmark_access,
    freeze_final_benchmark_protocol,
    open_final_benchmark_access,
    verify_final_benchmark_access,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_bytes(path: Path, payload: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_manifest(root: Path, relatives: list[str]) -> str:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(f"{_sha256(root / relative)}  {relative}\n" for relative in relatives),
        encoding="utf-8",
    )
    return _sha256(manifest)


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


class _ProtocolFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset_root = root / "dataset"
        self.instruction_root = root / "instructions"
        self.model_manifest = root / "model.files.sha256"
        self.split_manifest = root / "split_manifest.jsonl"
        self.split_report = root / "split_report.json"
        self.derivation_plan = root / "derivation_plan.jsonl"
        self.derivation_report = root / "derivation_report.json"
        self.dossier_root = root / "dossier"
        self.output = root / "protocol"
        self._build()

    def _build_sequence(self, name: str, threshold: float) -> Path:
        sequence_root = self.root / name
        checkpoint = sequence_root / "best_model.pt"
        frozen_threshold = sequence_root / "frozen_threshold.json"
        _write_bytes(checkpoint, name.encode("utf-8"))
        _write_json(
            frozen_threshold,
            {
                "schema_version": "chrompeak-frozen-threshold-v1",
                "selected_on_split": "validation",
                "objective": "macro_f1",
                "threshold": threshold,
                "internal_test_accessed": False,
            },
        )
        report_path = sequence_root / "scientific_report.json"
        _write_json(
            report_path,
            {
                "schema_version": "chrompeak-sequence-baseline-report-v1",
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
                "dataset": {"dataset_report_sha256": self.dataset_sha256},
                "artifacts": {
                    "checkpoint": {
                        "path": checkpoint.name,
                        "sha256": _sha256(checkpoint),
                    },
                    "frozen_threshold": {
                        "path": frozen_threshold.name,
                        "sha256": _sha256(frozen_threshold),
                    },
                },
            },
        )
        _write_manifest(
            sequence_root,
            [report_path.name, checkpoint.name, frozen_threshold.name],
        )
        return report_path

    def _build(self) -> None:
        dataset_report = self.dataset_root / "dataset_report.json"
        instruction_report = self.instruction_root / "instruction_dataset_report.json"
        normalization_path = self.dataset_root / "scalar_normalization.json"
        _write_json(
            normalization_path,
            {
                "schema_version": "chrompeak-scalar-normalization-v1",
                "fit_split": "train",
            },
        )
        _write_json(
            dataset_report,
            {
                "schema_version": "chrompeak-multimodal-dataset-v1",
                "artifacts": {
                    "scalar_normalization": {
                        "path": normalization_path.name,
                        "sha256": _sha256(normalization_path),
                    }
                },
            },
        )
        _write_json(instruction_report, {"schema_version": "fixture-instructions"})
        self.dataset_sha256 = _sha256(dataset_report)
        self.instruction_sha256 = _sha256(instruction_report)
        _write_bytes(self.model_manifest, b"base model manifest\n")
        self.model_manifest_sha256 = _sha256(self.model_manifest)
        self.model_artifact_sha256 = "a" * 64

        self.lora_root = self.root / "image_lora"
        lora_weights = self.lora_root / "adapter" / "adapter_model.safetensors"
        _write_bytes(lora_weights, b"lora")
        lora_report_path = self.lora_root / "lora_training_report.json"
        _write_json(
            lora_report_path,
            {
                "schema_version": "chrompeak-qwen3vl-lora-training-v1",
                "development_training_complete": True,
                "internal_test_accessed": False,
                "source": {"dataset_report_sha256": self.dataset_sha256},
                "model": {"artifact_sha256": self.model_artifact_sha256},
            },
        )
        lora_manifest_sha = _write_manifest(
            self.lora_root,
            ["lora_training_report.json", "adapter/adapter_model.safetensors"],
        )

        fusion_root = self.root / "fusion"
        fusion_adapter = fusion_root / "adapter" / "adapter_model.safetensors"
        fusion_projector = fusion_root / "sensor_projector.safetensors"
        _write_bytes(fusion_adapter, b"fusion adapter")
        _write_bytes(fusion_projector, b"fusion projector")
        fusion_report_path = fusion_root / "fusion_training_report.json"
        _write_json(
            fusion_report_path,
            {
                "schema_version": "chrompeak-qwen3vl-xic-fusion-training-v1",
                "development_training_complete": True,
                "internal_test_accessed": False,
                "sources": {"dataset_report_sha256": self.dataset_sha256},
                "training": {"seed": 17, "attention_implementation": "sdpa"},
                "model": {
                    "revision": "modelscope-master",
                    "base_artifact_sha256": self.model_artifact_sha256,
                    "verification_manifest_sha256": self.model_manifest_sha256,
                    "sensor_projector": {"sensor_tokens": 4},
                },
            },
        )
        fusion_manifest_sha = _write_manifest(
            fusion_root,
            [
                "fusion_training_report.json",
                "adapter/adapter_model.safetensors",
                "sensor_projector.safetensors",
            ],
        )

        zero_generation_path = self.root / "zero" / "generation_report.json"
        _write_json(
            zero_generation_path,
            {
                "internal_test_accessed": False,
                "model": {
                    "name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
                    "requested_revision": "modelscope-master",
                    "artifact_sha256": self.model_artifact_sha256,
                },
            },
        )
        lora_generation_path = self.root / "lora_generation" / "generation_report.json"
        _write_json(
            lora_generation_path,
            {
                "internal_test_accessed": False,
                "model": {
                    "name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
                    "requested_revision": "modelscope-master",
                    "artifact_sha256": self.model_artifact_sha256,
                    "adapter": {
                        "training_report_sha256": _sha256(lora_report_path),
                        "manifest_sha256": lora_manifest_sha,
                    },
                },
            },
        )

        sequence_path = self._build_sequence("sequence", 0.32)
        sequence_metadata_path = self._build_sequence("sequence_metadata", 0.41)

        detector_root = self.root / "detector"
        detector_checkpoint = detector_root / "training" / "checkpoint.pth"
        _write_bytes(detector_checkpoint, b"detector")
        detector_training_path = detector_root / "training" / "detector_training_report.json"
        _write_json(
            detector_training_path,
            {
                "schema_version": "chrompeak-detector-training-v1",
                "development_comparison_eligible": True,
                "internal_test_accessed": False,
                "checkpoint_sha256": _sha256(detector_checkpoint),
            },
        )
        detector_evaluation_path = (
            detector_root / "evaluation" / "detector_evaluation_report.json"
        )
        _write_json(
            detector_evaluation_path,
            {
                "schema_version": "chrompeak-detector-evaluation-v1",
                "development_comparison_eligible": True,
                "internal_test_accessed": False,
                "classification": {
                    "validation_selected_threshold": {"threshold": 0.61}
                },
            },
        )
        _write_manifest(
            detector_root,
            [
                "training/detector_training_report.json",
                "training/checkpoint.pth",
                "evaluation/detector_evaluation_report.json",
            ],
        )

        specialist_path = self.root / "specialist" / "development_ablation_report.json"
        _write_json(
            specialist_path,
            {
                "sources": {
                    "sequence_report": _descriptor(sequence_path),
                    "sequence_metadata_report": _descriptor(sequence_metadata_path),
                    "detector_evaluation": _descriptor(detector_evaluation_path),
                }
            },
        )

        cross_root = self.root / "cross"
        cross_path = cross_root / "cross_family_development_report.json"
        _write_json(
            cross_path,
            {
                "sources": {
                    "zero_shot_generation": _descriptor(zero_generation_path),
                    "lora_generation": _descriptor(lora_generation_path),
                    "specialist_comparison": _descriptor(specialist_path),
                }
            },
        )
        cross_manifest_sha = _write_manifest(
            cross_root, ["cross_family_development_report.json"]
        )

        matrix_root = self.root / "matrix"
        matrix_path = matrix_root / "fusion_matrix_analysis.json"
        _write_json(
            matrix_path,
            {
                "matrix": {
                    "primary-seed17": {
                        "sensor_tokens": 4,
                        "training_seed": 17,
                        "paths": {"training_root": str(fusion_root)},
                        "sha256": {
                            "training_report": _sha256(fusion_report_path),
                            "training_manifest": fusion_manifest_sha,
                        },
                    }
                }
            },
        )
        matrix_manifest_sha = _write_manifest(matrix_root, ["fusion_matrix_analysis.json"])

        dossier_path = self.dossier_root / "development_dossier.json"
        readiness = {
            "bilingual_failure_analysis_complete": True,
            "five_cell_matrix_complete": True,
            "hash_bound_source_inputs_complete": True,
            "localization_failure_analysis_complete": True,
            "selected_checkpoint_intervention_complete": True,
            "three_seed_statistics_complete": True,
            "ready": True,
        }
        _write_json(
            dossier_path,
            {
                "schema_version": "chrompeak-multimodal-development-dossier-v1",
                "development_comparison_eligible": True,
                "final_benchmark_eligible": False,
                "internal_test_accessed": False,
                "pre_internal_test_readiness": readiness,
                "sources": {
                    "cross_family": {
                        "path": str(cross_path),
                        "report_sha256": _sha256(cross_path),
                        "manifest_sha256": cross_manifest_sha,
                    },
                    "fusion_matrix": {
                        "path": str(matrix_path),
                        "report_sha256": _sha256(matrix_path),
                        "manifest_sha256": matrix_manifest_sha,
                    },
                    "shared_evidence": {
                        "dataset_report_sha256": self.dataset_sha256,
                        "instruction_report_sha256": self.instruction_sha256,
                        "base_model_artifact_sha256": self.model_artifact_sha256,
                    },
                },
            },
        )
        _write_manifest(self.dossier_root, ["development_dossier.json"])

        _write_bytes(self.split_manifest, b"sealed split manifest\n")
        _write_json(
            self.split_report,
            {
                "schema_version": "chrompeak-split-v1",
                "dataset_version": "fixture-v1",
                "split_manifest_sha256": _sha256(self.split_manifest),
                "leakage_audit": {"passed": True},
                "splits": [
                    {
                        "split": "internal_test",
                        "records": 1815,
                        "groups": 11,
                        "audit_records": 0,
                    }
                ],
            },
        )
        _write_bytes(self.derivation_plan, b"sealed derivation plan\n")
        _write_json(
            self.derivation_report,
            {
                "schema_version": "chrompeak-derivation-v1",
                "derivation_plan_sha256": _sha256(self.derivation_plan),
                "source_split_manifest_sha256": _sha256(self.split_manifest),
            },
        )

    def freeze(self, *, image_lora_root: Path | None = None):
        return freeze_final_benchmark_protocol(
            development_dossier_root=self.dossier_root,
            split_manifest_path=self.split_manifest,
            split_report_path=self.split_report,
            derivation_plan_path=self.derivation_plan,
            derivation_report_path=self.derivation_report,
            dataset_root=self.dataset_root,
            instruction_root=self.instruction_root,
            image_lora_root=image_lora_root or self.lora_root,
            base_model_manifest_path=self.model_manifest,
            output_dir=self.output,
        )


class FinalBenchmarkProtocolTests(unittest.TestCase):
    def test_reads_current_and_legacy_model_revision_without_ambiguity(self) -> None:
        self.assertEqual(
            _requested_model_revision(
                {"requested_revision": "modelscope-master"}, "current"
            ),
            "modelscope-master",
        )
        self.assertEqual(
            _requested_model_revision({"revision": "legacy-revision"}, "legacy"),
            "legacy-revision",
        )
        with self.assertRaisesRegex(ValueError, "fields disagree"):
            _requested_model_revision(
                {
                    "requested_revision": "modelscope-master",
                    "revision": "different-revision",
                },
                "conflicting",
            )

    def test_freezes_candidates_thresholds_and_metrics_without_opening_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ProtocolFixture(Path(temporary))
            result = fixture.freeze()

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            candidates = json.loads(result.candidate_lock_path.read_text(encoding="utf-8"))
            metrics = json.loads(result.metrics_lock_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], FINAL_BENCHMARK_PROTOCOL_SCHEMA)
            self.assertTrue(report["pre_internal_test_ready"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertEqual(
                report["benchmark_scope"], "frozen_multimodal_model_candidates"
            )
            self.assertFalse(report["biocoder_agent_promotion_claimed"])
            self.assertFalse(report["sealed_split"]["labels_opened_while_freezing_protocol"])
            self.assertEqual(
                candidates["shared"]["base_model"]["revision"], "modelscope-master"
            )
            self.assertEqual(
                candidates["models"]["sequence_peak_net"]["frozen_threshold"]["value"],
                0.32,
            )
            self.assertEqual(
                candidates["models"]["chrompeakformer"]["primary_threshold"], 0.61
            )
            self.assertEqual(metrics["uncertainty"]["bootstrap_iterations"], 10_000)
            self.assertTrue(metrics["post_access_model_selection_forbidden"])

    def test_rejects_incomplete_pre_internal_test_dossier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ProtocolFixture(Path(temporary))
            dossier_path = fixture.dossier_root / "development_dossier.json"
            dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
            dossier["pre_internal_test_readiness"]["three_seed_statistics_complete"] = False
            _write_json(dossier_path, dossier)
            _write_manifest(fixture.dossier_root, ["development_dossier.json"])

            with self.assertRaisesRegex(ValueError, "three_seed_statistics_complete"):
                fixture.freeze()

    def test_explicit_lora_root_must_match_hash_bound_generation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ProtocolFixture(Path(temporary))
            decoy = fixture.root / "decoy_lora"
            _write_bytes(decoy / "adapter" / "adapter_model.safetensors", b"decoy")
            _write_json(
                decoy / "lora_training_report.json",
                {"schema_version": "chrompeak-qwen3vl-lora-training-v1"},
            )
            _write_manifest(
                decoy,
                ["lora_training_report.json", "adapter/adapter_model.safetensors"],
            )

            with self.assertRaisesRegex(ValueError, "Image-LoRA manifest drift"):
                fixture.freeze(image_lora_root=decoy)

    def test_one_time_access_is_exclusive_and_exact_protocol_resume_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _ProtocolFixture(root)
            protocol = fixture.freeze()
            ledger = root / "ledger"

            opened = open_final_benchmark_access(
                protocol_root=protocol.output_dir,
                expected_protocol_sha256=protocol.report_sha256,
                ledger_dir=ledger,
            )
            self.assertFalse(opened.completed)
            resumed = verify_final_benchmark_access(
                protocol_root=protocol.output_dir,
                expected_protocol_sha256=protocol.report_sha256,
                ledger_dir=ledger,
            )
            self.assertEqual(resumed.access_id, opened.access_id)
            with self.assertRaisesRegex(ValueError, "already exists"):
                open_final_benchmark_access(
                    protocol_root=protocol.output_dir,
                    expected_protocol_sha256=protocol.report_sha256,
                    ledger_dir=ledger,
                )
            with self.assertRaisesRegex(ValueError, "Protocol hash mismatch"):
                verify_final_benchmark_access(
                    protocol_root=protocol.output_dir,
                    expected_protocol_sha256="0" * 64,
                    ledger_dir=ledger,
                )

            final_manifest = root / "final-evidence.sha256"
            _write_bytes(final_manifest, b"final evidence\n")
            completed = complete_final_benchmark_access(
                protocol_root=protocol.output_dir,
                expected_protocol_sha256=protocol.report_sha256,
                ledger_dir=ledger,
                final_evidence_manifest_path=final_manifest,
            )
            self.assertTrue(completed.completed)
            self.assertTrue((ledger / "access_completed.json").is_file())
            (ledger / "artifact_manifest.sha256").unlink()
            resumed_completion = complete_final_benchmark_access(
                protocol_root=protocol.output_dir,
                expected_protocol_sha256=protocol.report_sha256,
                ledger_dir=ledger,
                final_evidence_manifest_path=final_manifest,
            )
            self.assertTrue(resumed_completion.completed)
            self.assertTrue((ledger / "artifact_manifest.sha256").is_file())

            different_manifest = root / "different-evidence.sha256"
            _write_bytes(different_manifest, b"different evidence\n")
            with self.assertRaisesRegex(ValueError, "evidence path drift"):
                complete_final_benchmark_access(
                    protocol_root=protocol.output_dir,
                    expected_protocol_sha256=protocol.report_sha256,
                    ledger_dir=ledger,
                    final_evidence_manifest_path=different_manifest,
                )

    def test_slurm_scripts_keep_label_access_behind_all_preflight_gates(self) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        slurm_root = backend_root / "multimodal_science" / "qwen3vl" / "slurm"
        freeze_script = (
            slurm_root / "coder_final_benchmark_freeze.sbatch"
        ).read_text(encoding="utf-8")
        run_script = (slurm_root / "coder_final_benchmark_run.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertTrue(freeze_script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("freeze_final_benchmark_cli", freeze_script)
        self.assertIn("BIOCODER_IMAGE_LORA_ROOT", freeze_script)
        self.assertIn('--image-lora-root "$BIOCODER_IMAGE_LORA_ROOT"', freeze_script)
        self.assertNotIn("final_benchmark_access_cli open", freeze_script)
        self.assertNotIn("build_final_benchmark_data_cli", freeze_script)

        access_open = run_script.index("final_benchmark_access_cli open")
        preflight = run_script.index("FROZEN_RUNTIME_INPUTS=OK")
        gpu_guard = run_script.index("=== GPU STARTUP GUARD ===")
        data_materialization = run_script.index(
            "=== MATERIALIZE INTERNAL TEST WITH FROZEN TRAIN NORMALIZATION ==="
        )
        report = run_script.index("build_final_benchmark_report_cli")
        access_complete = run_script.rindex("final_benchmark_access_cli complete")
        self.assertLess(preflight, gpu_guard)
        self.assertLess(gpu_guard, access_open)
        self.assertLess(access_open, data_materialization)
        self.assertLess(data_materialization, report)
        self.assertLess(report, access_complete)
        self.assertEqual(run_script.count("final_benchmark_access_cli open"), 1)
        self.assertEqual(run_script.count("final_benchmark_access_cli complete"), 2)
        for candidate in (
            "qwen3vl_zero_shot",
            "qwen3vl_image_lora",
            "qwen3vl_image_xic_fusion",
            "sequence_peak_net",
            "chrompeakformer",
        ):
            self.assertIn(candidate, run_script)


if __name__ == "__main__":
    unittest.main()
