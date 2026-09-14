from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_data import build_fusion_bundle
from multimodal_science.qwen3vl.inference_bundle import BUNDLE_SCHEMA
from multimodal_science.qwen3vl.lora_data import LORA_BUNDLE_SCHEMA
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)


class QwenFusionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def dataset(self) -> tuple[Path, str]:
        root = self.root / "dataset"
        artifacts = {}
        for split, group in (("train", "train-group"), ("validation", "val-group")):
            split_root = root / split
            split_root.mkdir(parents=True)
            signals = split_root / "signals.npy"
            scalars = split_root / "scalar_features.npy"
            np.save(signals, np.ones((1, 160), dtype=np.float32), allow_pickle=False)
            np.save(scalars, np.ones((1, 7), dtype=np.float32), allow_pickle=False)
            example = {
                "schema_version": "chrompeak-multimodal-example-v1",
                "row": 0,
                "split": split,
                "asset_id": f"{split}-asset",
                "group_id": group,
                "image": {
                    "path": f"jobs/{split}/roi.jpeg",
                    "sha256": "a" * 64,
                },
                "sequence": {
                    "array": f"{split}/signals.npy",
                    "row": 0,
                    "length": 160,
                    "signal_available": True,
                },
                "scalar_features": {
                    "array": f"{split}/scalar_features.npy",
                    "row": 0,
                },
            }
            examples = split_root / "examples.jsonl"
            self.write_jsonl(examples, [example])
            for key, path in (
                (f"{split}_signals", signals),
                (f"{split}_scalar_features", scalars),
                (f"{split}_examples", examples),
            ):
                artifacts[key] = {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    **({"records": 1} if key.endswith("examples") else {}),
                    **(
                        {"shape": [1, 160]}
                        if key.endswith("signals")
                        else {"shape": [1, 7]}
                        if key.endswith("scalar_features")
                        else {}
                    ),
                }
        report_path = root / "dataset_report.json"
        self.write_json(
            report_path,
            {
                "schema_version": "chrompeak-multimodal-dataset-v1",
                "asset_index_sha256": "b" * 64,
                "target_points": 160,
                "artifacts": artifacts,
            },
        )
        return root, sha256_file(report_path)

    def bundles(self, dataset_hash: str) -> tuple[Path, str, Path, str]:
        lora_root = self.root / "lora"
        inference_root = self.root / "inference"
        lora_root.mkdir()
        inference_root.mkdir()
        selection = lora_root / "selection_manifest.jsonl"
        self.write_jsonl(
            selection,
            [
                {
                    "instruction_id": "1" * 24,
                    "asset_id": "train-asset",
                    "group_id": "train-group",
                    "image": "jobs/train/roi.jpeg",
                    "image_sha256": "a" * 64,
                    "task": "peak_presence",
                    "language": "zh-CN",
                }
            ],
        )
        lora_report = lora_root / "lora_bundle_report.json"
        self.write_json(
            lora_report,
            {
                "schema_version": LORA_BUNDLE_SCHEMA,
                "source": {"dataset_report_sha256": dataset_hash},
                "artifacts": {
                    "selection_manifest": {
                        "path": selection.name,
                        "sha256": sha256_file(selection),
                        "records": 1,
                    }
                },
                "internal_test_accessed": False,
            },
        )
        prompts = inference_root / "inference_prompts.jsonl"
        self.write_jsonl(
            prompts,
            [
                {
                    "instruction_id": "2" * 24,
                    "pair_id": "3" * 24,
                    "task": "peak_presence",
                    "language": "en",
                    "image": "jobs/validation/roi.jpeg",
                }
            ],
        )
        inference_report = inference_root / "inference_bundle_report.json"
        self.write_json(
            inference_report,
            {
                "schema_version": BUNDLE_SCHEMA,
                "source": {"source_dataset_report_sha256": dataset_hash},
                "artifacts": {
                    "inference_prompts": {
                        "path": prompts.name,
                        "sha256": sha256_file(prompts),
                        "records": 1,
                    }
                },
                "internal_test_accessed": False,
            },
        )
        return (
            lora_root,
            sha256_file(lora_report),
            inference_root,
            sha256_file(inference_report),
        )

    def test_builds_answer_isolated_asset_to_xic_links(self) -> None:
        dataset_root, dataset_hash = self.dataset()
        lora_root, lora_hash, inference_root, inference_hash = self.bundles(dataset_hash)

        result = build_fusion_bundle(
            dataset_root=dataset_root,
            dataset_report_sha256=dataset_hash,
            lora_bundle_root=lora_root,
            lora_bundle_report_sha256=lora_hash,
            inference_bundle_root=inference_root,
            inference_bundle_report_sha256=inference_hash,
            output_dir=self.root / "fusion",
        )

        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["train_instruction_links"], 1)
        self.assertEqual(report["counts"]["validation_independent_assets"], 1)
        self.assertFalse(report["contracts"]["validation_answer_key_opened"])
        link = json.loads(result.train_links_path.read_text(encoding="utf-8"))
        self.assertEqual(
            link["signal"],
            {
                "array": "train/signals.npy",
                "available": True,
                "length": 160,
                "row": 0,
            },
        )

    def test_preserves_unavailable_xic_as_a_masked_sensor_input(self) -> None:
        dataset_root, _ = self.dataset()
        examples = dataset_root / "train" / "examples.jsonl"
        record = json.loads(examples.read_text(encoding="utf-8"))
        record["sequence"]["signal_available"] = False
        self.write_jsonl(examples, [record])
        report_path = dataset_root / "dataset_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifacts"]["train_examples"]["sha256"] = sha256_file(examples)
        self.write_json(report_path, report)
        dataset_hash = sha256_file(report_path)
        lora_root, lora_hash, inference_root, inference_hash = self.bundles(dataset_hash)

        result = build_fusion_bundle(
            dataset_root=dataset_root,
            dataset_report_sha256=dataset_hash,
            lora_bundle_root=lora_root,
            lora_bundle_report_sha256=lora_hash,
            inference_bundle_root=inference_root,
            inference_bundle_report_sha256=inference_hash,
            output_dir=self.root / "fusion",
        )

        link = json.loads(result.train_links_path.read_text(encoding="utf-8"))
        self.assertFalse(link["signal"]["available"])

    def test_rejects_train_validation_group_leakage(self) -> None:
        dataset_root, _ = self.dataset()
        examples = dataset_root / "validation" / "examples.jsonl"
        record = json.loads(examples.read_text(encoding="utf-8"))
        record["group_id"] = "train-group"
        self.write_jsonl(examples, [record])
        report_path = dataset_root / "dataset_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifacts"]["validation_examples"]["sha256"] = sha256_file(examples)
        self.write_json(report_path, report)
        dataset_hash = sha256_file(report_path)
        lora_root, lora_hash, inference_root, inference_hash = self.bundles(dataset_hash)

        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            build_fusion_bundle(
                dataset_root=dataset_root,
                dataset_report_sha256=dataset_hash,
                lora_bundle_root=lora_root,
                lora_bundle_report_sha256=lora_hash,
                inference_bundle_root=inference_root,
                inference_bundle_report_sha256=inference_hash,
                output_dir=self.root / "fusion",
            )

    def test_sensor_projector_produces_short_qwen_width_token_sequence(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is verified in the server training environment")
        import torch

        spec = SensorProjectorSpec(hidden_size=128, sensor_tokens=4)
        model = build_sensor_projector(spec)
        signals = torch.ones((2, 160), dtype=torch.float32)

        tokens = model(signals)

        self.assertEqual(tuple(tokens.shape), (2, 4, 128))
        self.assertGreater(float(tokens.abs().sum()), 0.0)
        self.assertAlmostEqual(float(model.gate_logit), -4.0)

    def test_slurm_builder_is_cpu_only_answer_isolated_and_hash_bound(self) -> None:
        script = (
            Path(__file__).parents[2]
            / "multimodal_science"
            / "qwen3vl"
            / "slurm"
            / "coder_build_fusion_bundle.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertNotIn("--gres", script)
        self.assertNotIn("validation_answers", script)
        self.assertIn("BIOCODER_DATASET_REPORT_SHA256", script)
        self.assertIn("BIOCODER_LORA_BUNDLE_REPORT_SHA256", script)
        self.assertIn("BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256", script)
        self.assertIn('== 54335', script)
        self.assertIn('== 13708', script)


if __name__ == "__main__":
    unittest.main()
