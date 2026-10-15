from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multimodal_science.qwen3vl.fusion_training import (
    FUSION_TRAINING_CONFIG_SCHEMA,
    FUSION_TRAINING_REPORT_SCHEMA,
    XIC_ONLY_TRAINING_CONFIG_SCHEMA,
    XIC_ONLY_TRAINING_REPORT_SCHEMA,
    FusionTrainingSettings,
    _xic_only_messages,
    _latest_checkpoint,
    _validate_settings,
)
from multimodal_science.qwen3vl.train_fusion_cli import parser


class FusionTrainingContractTests(unittest.TestCase):
    def test_defaults_are_uncapped_formal_single_gpu_training(self) -> None:
        settings = FusionTrainingSettings()

        self.assertEqual(settings.epochs, 1)
        self.assertIsNone(settings.max_steps)
        self.assertEqual(settings.batch_size, 1)
        self.assertEqual(settings.gradient_accumulation_steps, 16)
        self.assertEqual(settings.save_steps, 250)
        self.assertEqual(settings.attention_implementation, "sdpa")
        self.assertTrue(settings.gradient_checkpointing)
        self.assertTrue(settings.deterministic_warn_only)
        self.assertEqual(settings.sensor_tokens, 4)
        self.assertEqual(settings.input_modality, "image_xic")
        _validate_settings(settings)

    def test_training_rejects_unsafe_or_nonpositive_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            _validate_settings(FusionTrainingSettings(batch_size=2))
        with self.assertRaisesRegex(ValueError, "max_steps must be positive"):
            _validate_settings(FusionTrainingSettings(max_steps=0))
        with self.assertRaisesRegex(ValueError, "warmup_ratio"):
            _validate_settings(FusionTrainingSettings(warmup_ratio=1.0))
        with self.assertRaisesRegex(ValueError, "sensor_tokens"):
            _validate_settings(FusionTrainingSettings(sensor_tokens=2))
        with self.assertRaisesRegex(ValueError, "input_modality"):
            _validate_settings(FusionTrainingSettings(input_modality="blank_image"))
        for sensor_tokens in (1, 4, 8):
            _validate_settings(FusionTrainingSettings(sensor_tokens=sensor_tokens))

    def test_resume_selects_only_the_highest_well_formed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "step-00000002").mkdir()
            (checkpoints / "step-00000017").mkdir()
            (checkpoints / "step-bad").mkdir()
            (checkpoints / "step-00000100").write_text("not a directory")

            self.assertEqual(
                _latest_checkpoint(root),
                checkpoints / "step-00000017",
            )

    def test_cli_has_no_validation_answer_or_internal_test_surface(self) -> None:
        destinations = {action.dest for action in parser()._actions}

        self.assertIn("resume", destinations)
        self.assertIn("max_steps", destinations)
        self.assertIn("sensor_tokens", destinations)
        self.assertIn("input_modality", destinations)
        self.assertNotIn("validation_answers", destinations)
        self.assertNotIn("internal_test", destinations)

    def test_schema_versions_are_distinct_from_the_bounded_smoke(self) -> None:
        self.assertEqual(
            FUSION_TRAINING_CONFIG_SCHEMA,
            "chrompeak-qwen3vl-xic-fusion-training-config-v1",
        )
        self.assertEqual(
            FUSION_TRAINING_REPORT_SCHEMA,
            "chrompeak-qwen3vl-xic-fusion-training-v1",
        )
        self.assertEqual(
            XIC_ONLY_TRAINING_CONFIG_SCHEMA,
            "chrompeak-qwen3vl-xic-only-training-config-v1",
        )
        self.assertEqual(
            XIC_ONLY_TRAINING_REPORT_SCHEMA,
            "chrompeak-qwen3vl-xic-only-training-v1",
        )

    def test_xic_only_messages_are_text_only_and_language_matched(self) -> None:
        prompt, full = _xic_only_messages(
            {
                "conversations": [
                    {"from": "human", "value": "<image>只返回 JSON。"},
                    {"from": "gpt", "value": '{"peak_present":true}'},
                ],
            },
            language="zh-CN",
        )

        self.assertEqual(prompt[0]["content"][0]["type"], "text")
        self.assertNotIn("<image>", prompt[0]["content"][0]["text"])
        self.assertIn("不提供任何图像像素", prompt[0]["content"][0]["text"])
        self.assertEqual(full[-1]["role"], "assistant")

    def test_slurm_launcher_is_uncapped_resumable_and_train_only(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_fusion_train.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("#SBATCH --partition=GPU-5090", script)
        self.assertIn('max_steps="${BIOCODER_MAX_STEPS:-}"', script)
        self.assertNotIn("--max-steps 2", script)
        self.assertIn('resume_arguments+=(--resume)', script)
        self.assertIn(".incomplete", script)
        self.assertIn('archive "$code_revision"', script)
        self.assertIn("staged_train_artifacts_only", script)
        self.assertIn('("train_xic_links",)', script)
        self.assertIn('("train_qwen", "selection_manifest")', script)
        self.assertIn('("train_signals", "train_examples")', script)
        self.assertNotIn('("validation_xic_links",)', script)
        self.assertNotIn("--validation-answers", script)
        self.assertNotIn("internal-test", script)
        self.assertIn("QWEN3VL_XIC_FUSION_TRAINING=OK", script)
        self.assertIn('sensor_tokens="${BIOCODER_SENSOR_TOKENS:-4}"', script)
        self.assertIn('--sensor-tokens "$sensor_tokens"', script)
        self.assertIn('input_modality="${BIOCODER_INPUT_MODALITY:-image_xic}"', script)
        self.assertIn('--input-modality "$input_modality"', script)
        self.assertIn("QWEN3VL_XIC_ONLY_TRAINING=OK", script)
        self.assertIn('tokens${sensor_tokens}', script)

    def test_development_matrix_has_three_primary_seeds_and_token_ablations(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_fusion_development_matrix.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("#SBATCH --array=0-4%2", script)
        for configuration in (
            "sensor_tokens=4; training_seed=17",
            "sensor_tokens=4; training_seed=29",
            "sensor_tokens=4; training_seed=43",
            "sensor_tokens=1; training_seed=17",
            "sensor_tokens=8; training_seed=17",
        ):
            self.assertIn(configuration, script)
        self.assertIn('export BIOCODER_MAX_STEPS=""', script)
        self.assertIn("unset BIOCODER_PRETRAINED_PROJECTOR_ROOT", script)
        self.assertIn("INITIALIZATION=random", script)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)
        self.assertIn('exec bash "$training_script"', script)

    def test_evaluation_matrix_reuses_the_same_five_training_cells(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_fusion_evaluation_matrix.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("#SBATCH --array=0-4%2", script)
        self.assertIn("BIOCODER_MATRIX_TRAINING_ROOT", script)
        self.assertIn("BIOCODER_MATRIX_TRAINING_REVISION", script)
        for configuration in (
            "sensor_tokens=4; training_seed=17",
            "sensor_tokens=4; training_seed=29",
            "sensor_tokens=4; training_seed=43",
            "sensor_tokens=1; training_seed=17",
            "sensor_tokens=8; training_seed=17",
        ):
            self.assertIn(configuration, script)
        self.assertIn("MATRIX_TRAINING_INPUT_CONTRACT=OK", script)
        self.assertIn("report[\"training\"][\"optimizer_updates\"] == 3396", script)
        self.assertIn("export BIOCODER_XIC_INTERVENTION=aligned", script)
        self.assertIn("export BIOCODER_SEED=17", script)
        self.assertIn("export BIOCODER_BOOTSTRAP_ITERATIONS=1000", script)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", script)
        self.assertIn('exec bash "$evaluation_script"', script)


if __name__ == "__main__":
    unittest.main()
