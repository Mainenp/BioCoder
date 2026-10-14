from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_inference import (
    _FusionAdapterVerifier,
    _RLCandidateFusionAdapterVerifier,
    _build_xic_intervention_plan,
    _eos_token_ids,
    RLCandidateAdapterSpec,
)
from multimodal_science.qwen3vl.fusion_training import (
    FUSION_TRAINING_REPORT_SCHEMA,
    XIC_ONLY_TRAINING_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.run_fusion_inference_cli import parser


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_fusion_adapter(root: Path, input_modality: str = "image_xic") -> AdapterSpec:
    for relative, contents in (
        ("adapter/adapter_config.json", "{}\n"),
        ("adapter/adapter_model.safetensors", "adapter"),
        ("sensor_projector.safetensors", "projector"),
        ("processor/preprocessor_config.json", "{}\n"),
        ("processor/tokenizer_config.json", "{}\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    report = {
        "schema_version": (
            FUSION_TRAINING_REPORT_SCHEMA
            if input_modality == "image_xic"
            else XIC_ONLY_TRAINING_REPORT_SCHEMA
        ),
        "code_revision": "a" * 40,
        "sources": {"fusion_bundle_report_sha256": "b" * 64},
        "model": {
            "name_or_path": "Qwen/test",
            "revision": "modelscope-master",
            "base_artifact_sha256": "c" * 64,
            "sensor_projector": {
                "input_points": 160,
                "hidden_size": 2560,
                "sensor_tokens": 4,
                "base_channels": 32,
                "dropout": 0.1,
            },
        },
        "training": {
            "optimizer_updates": 3396,
            "training_records": 54335,
            "vision_forward_calls": 54335 if input_modality == "image_xic" else 0,
        },
        "contracts": {
            "base_weights_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "assistant_tokens_only_supervision": True,
            "input_modality": input_modality,
            "image_and_xic_forward": input_modality == "image_xic",
            "xic_only_forward": input_modality == "xic_only",
            "image_pixels_forwarded": input_modality == "image_xic",
            "images_opened_for_provenance_only": input_modality == "xic_only",
            "native_multimodal_rope_positions": input_modality == "image_xic",
            "native_qwen_rope_positions": True,
            "lora_and_projector_backward": True,
            "parameter_updates_verified": True,
            "train_split_only": True,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
        "development_training_complete": True,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    report_path = root / "fusion_training_report.json"
    _write_json(report_path, report)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest_path = root / "artifact_manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return AdapterSpec(
        root=root,
        training_report_sha256=sha256_file(report_path),
        manifest_sha256=sha256_file(manifest_path),
    )


class FusionInferenceContractTests(unittest.TestCase):
    @staticmethod
    def intervention_links() -> dict[str, dict[str, object]]:
        return {
            f"instruction-{row}-{language}": {
                "signal": {"row": row, "available": row != 2},
            }
            for row in range(6)
            for language in ("en", "zh-CN")
        }

    def test_formal_slurm_evaluation_separates_generation_from_answers(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_fusion_evaluate.sbatch"
        ).read_text(encoding="utf-8")

        generation_start = script.index("=== RUN ANSWER-ISOLATED FUSED GENERATION ===")
        answer_start = script.index("=== OPEN ANSWERS ONLY AFTER GENERATION IS IMMUTABLE ===")
        generation_block = script[generation_start:answer_start]
        self.assertIn("env -i", generation_block)
        self.assertIn("run_fusion_inference_cli", generation_block)
        self.assertNotIn("BIOCODER_INSTRUCTION_ROOT", generation_block)
        self.assertNotIn("instruction_root", generation_block)
        self.assertNotIn("validation_answers", generation_block)
        self.assertNotIn("--max-records", script)
        self.assertIn('seed" == "17"', script)
        self.assertIn('max_new_tokens" == "64"', script)
        self.assertIn('bootstrap_iterations" == "1000"', script)
        self.assertIn('validation_links = copy_artifact(', script)
        self.assertNotIn('expected = row["image_sha256"]\nwith prompts.open', script)
        self.assertIn("evaluate_predictions_cli", script[answer_start:])
        self.assertIn("FUSION_FULL_GENERATION_CONTRACT=OK", script)
        self.assertIn("FUSION_FULL_EVALUATION_CONTRACT=OK", script)
        self.assertIn("QWEN3VL_XIC_FUSION_EVALUATION=OK", script)
        self.assertIn("BIOCODER_XIC_INTERVENTION", script)
        self.assertIn('--xic-intervention "$xic_intervention"', script)

    def test_rl_validation_rechecks_open_hash_bound_ledger_before_answers(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_fusion_evaluate.sbatch"
        ).read_text(encoding="utf-8")

        answer_start = script.index("=== OPEN ANSWERS ONLY AFTER GENERATION IS IMMUTABLE ===")
        self.assertIn("verify_open_validation_access", script)
        self.assertIn("Validation requires the exact 16-candidate matrix", script)
        self.assertIn("RL_VALIDATION_LEDGER=OPEN_AND_HASH_BOUND", script)
        self.assertIn("verify_rl_validation_access >/dev/null", script[answer_start:])
        self.assertIn('${rl_candidate_sha:0:12}', script)
        self.assertIn("QWEN3VL_RL_VALIDATION_EVALUATION=OK", script)

    def test_shuffled_xic_is_seeded_deranged_and_asset_stable(self) -> None:
        links = self.intervention_links()

        first = _build_xic_intervention_plan(links, mode="shuffled", seed=29)
        second = _build_xic_intervention_plan(links, mode="shuffled", seed=29)

        self.assertEqual(first.row_mapping, second.row_mapping)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(set(first.row_mapping), set(first.row_mapping.values()))
        self.assertTrue(
            all(source != donor for source, donor in first.row_mapping.items())
        )
        self.assertEqual(first.metadata["changed_signal_rows"], 6)
        self.assertFalse(first.metadata["answer_key_used"])
        self.assertTrue(
            first.metadata["language_variants_share_one_asset_intervention"]
        )

    def test_zero_and_availability_off_are_distinct_interventions(self) -> None:
        links = self.intervention_links()

        zero = _build_xic_intervention_plan(links, mode="zero", seed=17)
        unavailable = _build_xic_intervention_plan(
            links, mode="availability-off", seed=17
        )

        self.assertTrue(zero.metadata["signal_values_zeroed"])
        self.assertFalse(zero.metadata["availability_forced_off"])
        self.assertTrue(unavailable.metadata["signal_values_zeroed"])
        self.assertTrue(unavailable.metadata["availability_forced_off"])
        self.assertEqual(zero.row_mapping, unavailable.row_mapping)

    def test_completed_fusion_adapter_is_hash_bound_and_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specification = _make_fusion_adapter(root)
            verifier = _FusionAdapterVerifier()

            adapter_dir, metadata = verifier(
                specification,
                model_name_or_path="Qwen/test",
                model_revision="modelscope-master",
                model_artifact_sha256="c" * 64,
            )

            self.assertTrue(adapter_dir.samefile(root / "adapter"))
            self.assertEqual(metadata["kind"], "image_xic_fusion")
            self.assertTrue(metadata["development_training_complete"])
            self.assertEqual(metadata["optimizer_updates"], 3396)
            self.assertIsNotNone(verifier.verified)

    def test_fusion_adapter_rejects_tampered_projector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specification = _make_fusion_adapter(root)
            (root / "sensor_projector.safetensors").write_text(
                "tampered", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _FusionAdapterVerifier()(
                    specification,
                    model_name_or_path="Qwen/test",
                    model_revision="modelscope-master",
                    model_artifact_sha256="c" * 64,
                )

    def test_xic_only_adapter_requires_zero_visual_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specification = _make_fusion_adapter(root, input_modality="xic_only")
            verifier = _FusionAdapterVerifier("xic_only")

            _, metadata = verifier(
                specification,
                model_name_or_path="Qwen/test",
                model_revision="modelscope-master",
                model_artifact_sha256="c" * 64,
            )

            self.assertEqual(metadata["kind"], "xic_only_qwen")
            self.assertEqual(metadata["input_modality"], "xic_only")
            self.assertEqual(verifier.verified.report["training"]["vision_forward_calls"], 0)

    def test_rl_candidate_verifier_binds_checkpoint_and_sealed_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            starting = _make_fusion_adapter(root / "starting", input_modality="xic_only")
            pool_sha = "d" * 64
            checkpoint_root = root / "candidate" / "checkpoints" / "step-00000300"
            (checkpoint_root / "adapter").mkdir(parents=True)
            (checkpoint_root / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint_root / "adapter" / "adapter_model.safetensors").write_text("rl", encoding="utf-8")
            (checkpoint_root / "sensor_projector.safetensors").write_text("projector", encoding="utf-8")
            checkpoint_path = checkpoint_root / "checkpoint.json"
            _write_json(checkpoint_path, {"global_step": 300})
            checkpoint_files = sorted(
                path for path in checkpoint_root.rglob("*") if path.is_file()
            )
            (checkpoint_root / "artifact_manifest.sha256").write_text(
                "".join(
                    f"{sha256_file(path)}  {path.relative_to(checkpoint_root).as_posix()}\n"
                    for path in checkpoint_files
                ),
                encoding="utf-8",
            )
            training_root = root / "candidate"
            training_report = training_root / "fusion_training_report.json"
            _write_json(
                training_report,
                {
                    "sources": {
                        "rl_pool_report_sha256": pool_sha,
                        "starting_adapter_report_sha256": starting.training_report_sha256,
                        "starting_adapter_manifest_sha256": starting.manifest_sha256,
                    }
                },
            )
            (training_root / "artifact_manifest.sha256").write_text(
                f"{sha256_file(training_report)}  fusion_training_report.json\n",
                encoding="utf-8",
            )
            candidate = {
                "family": "R2",
                "config": "lora-lr-5e-6-projector-frozen",
                "checkpoint": {
                    "step": 300,
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                },
                "training_report": {
                    "path": str(training_report),
                    "sha256": sha256_file(training_report),
                },
                "sources": {
                    "rl_pool_report_sha256": pool_sha,
                    "starting_adapter_report_sha256": starting.training_report_sha256,
                    "starting_adapter_manifest_sha256": starting.manifest_sha256,
                },
                "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0},
            }
            candidate_path = root / "candidate.json"
            _write_json(candidate_path, candidate)
            verifier = _RLCandidateFusionAdapterVerifier(
                RLCandidateAdapterSpec(
                    candidate_path=candidate_path,
                    candidate_descriptor_sha256=sha256_file(candidate_path),
                    expected_pool_report_sha256=pool_sha,
                    selection_result_sha256="e" * 64,
                    validation_access_id="f" * 24,
                ),
                {"mode": "aligned"},
            )

            adapter_dir, metadata = verifier(
                starting,
                model_name_or_path="Qwen/test",
                model_revision="modelscope-master",
                model_artifact_sha256="c" * 64,
            )

            self.assertTrue(adapter_dir.samefile(checkpoint_root / "adapter"))
            self.assertEqual(metadata["rl_candidate"]["checkpoint_step"], 300)
            self.assertEqual(metadata["rl_candidate"]["validation_access_id"], "f" * 24)
            self.assertEqual(metadata["training_report_sha256"], sha256_file(training_report))
            self.assertEqual(
                metadata["manifest_sha256"],
                sha256_file(checkpoint_root / "artifact_manifest.sha256"),
            )

    def test_eos_contract_accepts_scalar_or_sequence(self) -> None:
        self.assertEqual(_eos_token_ids(7), {7})
        self.assertEqual(_eos_token_ids([7, 8]), {7, 8})
        with self.assertRaisesRegex(ValueError, "EOS"):
            _eos_token_ids(None)

    def test_cli_is_greedy_batch_one_and_exposes_no_answer_or_test_path(self) -> None:
        command = parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            [
                "--inference-bundle-root", "inference",
                "--inference-bundle-report-sha256", "a" * 64,
                "--fusion-bundle-root", "fusion",
                "--fusion-bundle-report-sha256", "b" * 64,
                "--dataset-root", "dataset",
                "--dataset-report-sha256", "c" * 64,
                "--assets-root", "assets",
                "--fusion-adapter-root", "adapter",
                "--fusion-training-report-sha256", "d" * 64,
                "--fusion-adapter-manifest-sha256", "e" * 64,
                "--output-dir", "output",
                "--model-name-or-path", "Qwen/test",
                "--model-revision", "modelscope-master",
                "--model-artifact-sha256", "f" * 64,
            ]
        )

        self.assertEqual(arguments.max_new_tokens, 64)
        self.assertEqual(arguments.xic_intervention, "aligned")
        self.assertEqual(arguments.xic_intervention_seed, 17)
        self.assertEqual(arguments.input_modality, "image_xic")
        self.assertNotIn("batch_size", destinations)
        self.assertNotIn("do_sample", destinations)
        self.assertNotIn("answer", destinations)
        self.assertNotIn("answers", destinations)
        self.assertNotIn("internal_test", destinations)
        self.assertIn("input_modality", destinations)
        self.assertIn("rl_candidate_json", destinations)
        self.assertIn("rl_selection_result_sha256", destinations)


if __name__ == "__main__":
    unittest.main()
