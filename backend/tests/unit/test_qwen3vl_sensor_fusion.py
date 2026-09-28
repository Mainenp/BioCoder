from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_smoke import (
    _bounded_training_rows,
    _verify_model_manifest,
    _verify_mrope_insertion,
)
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings


class SensorFusionTests(unittest.TestCase):
    def test_bounded_training_rows_rejects_fewer_rows_than_updates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested 2 updates"):
            _bounded_training_rows([{"row": 1}], 2)
        self.assertEqual(_bounded_training_rows([1, 2, 3], 2), [1, 2])

    def test_model_manifest_binds_every_local_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            manifest = root / "model.files.sha256"
            manifest.write_text(
                "".join(
                    f"{sha256_file(path)}  /source/cache/{path.name}\n"
                    for path in sorted(model.iterdir())
                ),
                encoding="utf-8",
            )

            inventory = _verify_model_manifest(model, manifest, sha256_file(manifest))

            self.assertEqual(inventory["files"], 2)
            (model / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Model file hash mismatch"):
                _verify_model_manifest(model, manifest, sha256_file(manifest))

            (model / "model.safetensors").write_bytes(b"weights")
            (model / "unlisted.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not cover"):
                _verify_model_manifest(model, manifest, sha256_file(manifest))

    def test_insertion_preserves_vocabulary_and_masks_sensor_targets(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is verified in the server training environment")
        import torch

        class Model:
            @staticmethod
            def get_input_embeddings():
                return torch.nn.Embedding.from_pretrained(
                    torch.arange(40, dtype=torch.float32).reshape(10, 4)
                )

        model = Model()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        sensors = torch.tensor([[[101.0] * 4, [102.0] * 4]])
        fused, targets, attention, shadow_ids = insert_sensor_embeddings(
            model, input_ids, labels, sensors, [2]
        )

        original = model.get_input_embeddings()(input_ids)
        self.assertEqual(tuple(fused.shape), (1, 6, 4))
        self.assertTrue(torch.equal(fused[0, :2], original[0, :2]))
        self.assertTrue(torch.equal(fused[0, 2:4], sensors[0]))
        self.assertTrue(torch.equal(fused[0, 4:], original[0, 2:]))
        self.assertEqual(targets.tolist(), [[-100, -100, -100, -100, 3, 4]])
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 1, 1]])
        self.assertEqual(shadow_ids.tolist(), [[1, 2, 0, 0, 3, 4]])
        self.assertEqual(model.get_input_embeddings().num_embeddings, 10)

    def test_mrope_insertion_preserves_prefix_and_shifts_text_suffix(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is verified in the server training environment")
        import torch

        original = torch.tensor(
            [
                [[0, 1, 2, 8, 9]],
                [[0, 1, 4, 8, 9]],
                [[0, 1, 6, 8, 9]],
            ]
        )
        inserted = torch.tensor([8, 9, 10, 11])
        fused = torch.cat(
            (
                original[:, :, :3],
                inserted.view(1, 1, -1).expand(3, 1, -1),
                original[:, :, 3:] + 4,
            ),
            dim=2,
        )

        evidence = _verify_mrope_insertion(
            original,
            fused,
            boundary=3,
            sensor_tokens=4,
        )

        self.assertEqual(evidence["axes"], 3)
        self.assertEqual(evidence["fused_length"], 9)

    def test_smoke_cli_has_no_validation_answer_or_internal_test_surface(self) -> None:
        from multimodal_science.qwen3vl.run_fusion_smoke_cli import parser

        destinations = {action.dest for action in parser()._actions}
        self.assertNotIn("validation_answers", destinations)
        self.assertNotIn("internal_test", destinations)

    def test_slurm_smoke_is_guarded_bounded_and_named_coder(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_fusion_smoke.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("--max-records 8", script)
        self.assertIn("--max-steps 2", script)
        self.assertIn("--seed 17", script)
        self.assertIn("GPU failed startup guard", script)
        self.assertIn("status --porcelain --untracked-files=all", script)
        self.assertIn('archive "$code_revision"', script)
        self.assertIn("BIOCODER_MODEL_MANIFEST_PATH", script)
        self.assertIn("BIOCODER_MODEL_MANIFEST_SHA256", script)
        self.assertIn("manual_physical_index_guard_no_slurm_gres", script)
        self.assertIn("BIOCODER_TRAIN_INPUT_SCOPE", script)
        self.assertIn("staged_train_artifacts_only", script)
        self.assertIn("BIOCODER_VERIFIED_CODE_REVISION", script)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", script)
        self.assertIn('cd "$scratch/code/backend"', script)
        self.assertIn("env -i", script)
        self.assertIn("PYTHONNOUSERSITE=1", script)
        self.assertIn('for variable in "${!BIOCODER_@}"', script)
        self.assertIn("publish_staging", script)
        self.assertIn("verify_exact_manifest", script)
        self.assertNotIn("assert report", script)
        self.assertNotIn("assert listed", script)
        self.assertIn("FUSION_OUTPUT_EXACT_MANIFEST=OK", script)
        self.assertIn('(\"train_xic_links\",)', script)
        self.assertIn('(\"train_qwen\", \"selection_manifest\")', script)
        self.assertIn('(\"train_signals\", \"train_examples\")', script)
        self.assertNotIn('(\"validation_xic_links\",)', script)
        self.assertNotIn("--validation-answers", script)
        self.assertNotIn("BIOCODER_VALIDATION_ANSWERS", script)
        self.assertNotIn("internal-test", script)


if __name__ == "__main__":
    unittest.main()
