from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_inference import (
    _FusionAdapterVerifier,
    _build_xic_intervention_plan,
    _eos_token_ids,
)
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.inference import AdapterSpec
from multimodal_science.qwen3vl.run_fusion_inference_cli import parser


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_fusion_adapter(root: Path) -> AdapterSpec:
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
        "schema_version": FUSION_TRAINING_REPORT_SCHEMA,
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
        "training": {"optimizer_updates": 3396, "training_records": 54335},
        "contracts": {
            "base_weights_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "assistant_tokens_only_supervision": True,
            "image_and_xic_forward": True,
            "native_multimodal_rope_positions": True,
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
        self.assertNotIn("batch_size", destinations)
        self.assertNotIn("do_sample", destinations)
        self.assertNotIn("answer", destinations)
        self.assertNotIn("answers", destinations)
        self.assertNotIn("internal_test", destinations)


if __name__ == "__main__":
    unittest.main()
