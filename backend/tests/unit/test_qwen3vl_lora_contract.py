from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.build_lora_bundle_cli import parser as bundle_parser
from multimodal_science.qwen3vl.instruction_data import BILINGUAL_DATASET_SCHEMA
from multimodal_science.qwen3vl.inference import AdapterSpec, _verify_adapter
from multimodal_science.qwen3vl.lora_data import (
    LORA_BUNDLE_SCHEMA,
    build_lora_training_bundle,
)
from multimodal_science.qwen3vl.lora_training import (
    assistant_supervision_labels,
    epoch_sample_indices,
    reconcile_history,
)
from multimodal_science.qwen3vl.train_lora_cli import parser as training_parser
from multimodal_science.qwen3vl.run_inference_cli import parser as inference_parser


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def make_instruction_root(root: Path) -> tuple[Path, Path, str]:
    instruction_root = root / "instructions"
    assets_root = root / "assets"
    instruction_root.mkdir()
    assets_root.mkdir()
    train = []
    manifest = []
    tasks = ("peak_presence", "scientific_qc")
    languages = ("en", "zh-CN")
    for row in range(8):
        target_present = row >= 4
        image = f"jobs/train/group-{row % 3}/roi-{row}.jpeg"
        image_path = assets_root / image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"jpeg-{row}".encode("ascii"))
        response = json.dumps({"peak_present": target_present}, separators=(",", ":"))
        prompt = "<image>\nAnalyze this ROI."
        train.append(
            {
                "image": image,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response},
                ],
            }
        )
        manifest.append(
            {
                "schema_version": "chrompeak-qwen3vl-instruction-manifest-v2",
                "instruction_id": f"{row + 1:024x}",
                "split": "train",
                "task": tasks[row % len(tasks)],
                "output_artifact": "train_qwen.jsonl",
                "output_row": row,
                "asset_id": f"asset-{row}",
                "group_id": f"group-{row % 3}",
                "image_path": image,
                "image_sha256": sha256_file(image_path),
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "target_peak_present": target_present,
                "language": languages[(row // 2) % len(languages)],
            }
        )
    # This row is deliberately not a valid training row. The builder must stop
    # parsing at the declared train prefix and never consume validation labels.
    manifest.append({"split": "validation", "target_peak_present": "sealed"})
    train_path = instruction_root / "train_qwen.jsonl"
    manifest_path = instruction_root / "instruction_manifest.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(manifest_path, manifest)
    report = {
        "schema_version": BILINGUAL_DATASET_SCHEMA,
        "source_dataset": {
            "dataset_report_sha256": "b" * 64,
            "asset_index_sha256": "a" * 64,
        },
        "counts": {"by_split": {"train": {"source_groups": 3}}},
        "contracts": {
            "one_image_token_per_train_record": True,
            "visual_tokens_forbidden_in_answers": True,
            "validation_answers_separated_from_prompts": True,
            "image_paths_relative_to_external_assets_root": True,
            "train_has_one_language_per_semantic_instruction": True,
        },
        "internal_test_accessed": False,
        "final_benchmark_eligible": False,
        "artifacts": {
            "train_qwen": {
                "path": train_path.name,
                "sha256": sha256_file(train_path),
                "records": len(train),
            },
            "instruction_manifest": {
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "records": len(manifest),
            },
            "validation_answers": {
                "path": "intentionally-absent-validation-answers.jsonl",
                "sha256": "f" * 64,
            },
        },
    }
    report_path = instruction_root / "instruction_dataset_report.json"
    write_json(report_path, report)
    return instruction_root, assets_root, sha256_file(report_path)


def make_adapter(root: Path, *, development_complete: bool = True) -> AdapterSpec:
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    report = {
        "schema_version": "chrompeak-qwen3vl-lora-training-v1",
        "code_revision": "d" * 40,
        "model": {
            "name_or_path": "Qwen/test-model",
            "revision": "a" * 40,
            "artifact_sha256": "b" * 64,
            "adapter_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {"training_records": 128},
        "contracts": {
            "base_weights_frozen": True,
            "vision_tower_frozen": True,
            "vision_merger_frozen": True,
            "assistant_tokens_only_supervision": True,
            "train_split_only": True,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
        "development_training_complete": development_complete,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }
    report_path = root / "lora_training_report.json"
    write_json(report_path, report)
    artifacts = [
        report_path,
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
    ]
    manifest_path = root / "artifact_manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in artifacts
        ),
        encoding="utf-8",
    )
    return AdapterSpec(
        root=root,
        training_report_sha256=sha256_file(report_path),
        manifest_sha256=sha256_file(manifest_path),
    )


class Qwen3VlLoraContractTests(unittest.TestCase):
    def test_formal_evaluation_slurm_script_is_hash_bound_and_resumable(self) -> None:
        script_path = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_lora_evaluate.sbatch"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("BIOCODER_ADAPTER_TRAINING_REPORT_SHA256", script)
        self.assertIn("BIOCODER_ADAPTER_MANIFEST_SHA256", script)
        self.assertIn("LOCAL_VALIDATION_IMAGE_COPIES=OK", script)
        self.assertIn("RESUME_GENERATION=YES", script)
        self.assertIn('generation_resume="$run_root/qwen3vl/runs/.${run_name}.work"', script)
        self.assertIn('--output-dir "$generation_final"', script)
        self.assertIn("ensure_artifact_manifest", script)
        self.assertIn("ARTIFACT_MANIFEST_CREATED", script)
        self.assertIn("generation_records.jsonl", script)
        self.assertIn("evaluation_records.jsonl", script)
        self.assertIn("evaluate_predictions_cli", script)
        self.assertIn('generation["counts"]["predictions"]', script)
        self.assertIn('generation["scope"]["complete_prompt_coverage"]', script)
        self.assertIn("prediction_generation_provenance_verified", script)
        self.assertIn("LORA_FULL_EVALUATION_CONTRACT=OK", script)
        self.assertNotIn("--max-records", script)

    def test_formal_slurm_script_is_guarded_resumable_and_uncapped_by_default(self) -> None:
        script_path = (
            Path(__file__).parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_lora_train.sbatch"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn('BIOCODER_GPU_INDEX', script)
        self.assertIn('RESUME_FROM_PERSISTENT_CHECKPOINT=YES', script)
        self.assertIn('run_lock="$output_root/.${run_name}.lock"', script)
        self.assertIn('flock -n 7', script)
        self.assertIn('Another job already owns this exact LoRA run', script)
        self.assertIn('LOCAL_TRAIN_IMAGE_HASHES=OK', script)
        self.assertIn('development_training_complete', script)
        self.assertIn('max_records="${BIOCODER_MAX_RECORDS:-}"', script)
        self.assertIn('max_steps="${BIOCODER_MAX_STEPS:-}"', script)

    def test_verifies_complete_adapter_and_base_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = make_adapter(Path(directory))
            adapter_dir, metadata = _verify_adapter(
                specification,
                model_name_or_path="/different/local/model-cache",
                model_revision="a" * 40,
                model_artifact_sha256="b" * 64,
            )

            self.assertEqual(adapter_dir, specification.root.resolve() / "adapter")
            self.assertTrue(metadata["development_training_complete"])
            self.assertEqual(metadata["training_records"], 128)
            self.assertEqual(metadata["trained_base_name_or_path"], "Qwen/test-model")

    def test_rejects_adapter_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = make_adapter(Path(directory))
            (specification.root / "adapter" / "adapter_model.safetensors").write_bytes(
                b"tampered"
            )

            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                _verify_adapter(
                    specification,
                    model_name_or_path="Qwen/test-model",
                    model_revision="a" * 40,
                    model_artifact_sha256="b" * 64,
                )

    def test_builds_stratified_train_only_bundle_without_validation_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instruction_root, assets_root, report_hash = make_instruction_root(root)
            result = build_lora_training_bundle(
                instruction_root=instruction_root,
                instruction_report_sha256=report_hash,
                assets_root=assets_root,
                output_dir=root / "bundle",
                max_records=4,
                seed=17,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], LORA_BUNDLE_SCHEMA)
            self.assertEqual(report["counts"]["selected_train_instructions"], 4)
            self.assertTrue(report["contracts"]["train_split_only"])
            self.assertFalse(report["contracts"]["validation_answers_opened"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertEqual(len((result.training_path).read_text().splitlines()), 4)
            self.assertEqual(len(report["counts"]["by_stratum"]), 4)
            selections = [
                json.loads(line)
                for line in (result.output_dir / "selection_manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(all("image_sha256" in row for row in selections))

    def test_rejects_response_that_no_longer_matches_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instruction_root, assets_root, _ = make_instruction_root(root)
            train_path = instruction_root / "train_qwen.jsonl"
            records = [json.loads(line) for line in train_path.read_text().splitlines()]
            records[0]["conversations"][1]["value"] = '{"peak_present":true}'
            write_jsonl(train_path, records)
            report_path = instruction_root / "instruction_dataset_report.json"
            report = json.loads(report_path.read_text())
            report["artifacts"]["train_qwen"]["sha256"] = sha256_file(train_path)
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "Response/manifest mismatch"):
                build_lora_training_bundle(
                    instruction_root=instruction_root,
                    instruction_report_sha256=sha256_file(report_path),
                    assets_root=assets_root,
                    output_dir=root / "bundle",
                )

    def test_rejects_tampered_selected_training_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instruction_root, assets_root, report_hash = make_instruction_root(root)
            (assets_root / "jobs/train/group-0/roi-0.jpeg").write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
                build_lora_training_bundle(
                    instruction_root=instruction_root,
                    instruction_report_sha256=report_hash,
                    assets_root=assets_root,
                    output_dir=root / "bundle",
                )

    def test_assistant_supervision_masks_exact_prompt_prefix(self) -> None:
        self.assertEqual(
            assistant_supervision_labels([10, 20, 30, 40], [10, 20]),
            [-100, -100, 30, 40],
        )
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            assistant_supervision_labels([10, 99, 30], [10, 20])

    def test_resume_discards_only_uncheckpointed_history_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "training_history.jsonl"
            write_jsonl(
                history,
                [{"global_step": step, "loss": 1.0 / step} for step in range(1, 5)],
            )
            self.assertEqual(reconcile_history(history, 2), 2)
            remaining = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual([record["global_step"] for record in remaining], [1, 2])

    def test_resume_uses_only_unseen_deterministic_epoch_indices(self) -> None:
        full_order = epoch_sample_indices(11, batch_size=3, seed=17)
        resumed_order = epoch_sample_indices(11, batch_size=3, seed=17, start_batch=2)

        self.assertEqual(resumed_order, full_order[6:])
        self.assertEqual(sorted(full_order), list(range(11)))
        with self.assertRaisesRegex(ValueError, "exceeds the epoch"):
            epoch_sample_indices(11, batch_size=3, seed=17, start_batch=5)

    def test_cli_exposes_bounded_smoke_and_training_controls(self) -> None:
        bundle = bundle_parser().parse_args(
            [
                "--instruction-root",
                "instructions",
                "--instruction-report-sha256",
                "a" * 64,
                "--assets-root",
                "assets",
                "--output-dir",
                "bundle",
                "--max-records",
                "128",
            ]
        )
        self.assertEqual(bundle.max_records, 128)
        training = training_parser().parse_args(
            [
                "--bundle-root",
                "bundle",
                "--bundle-report-sha256",
                "b" * 64,
                "--assets-root",
                "assets",
                "--output-dir",
                "run",
                "--model-name-or-path",
                "model",
                "--model-revision",
                "revision",
                "--model-artifact-sha256",
                "c" * 64,
                "--code-revision",
                "d" * 40,
                "--max-steps",
                "2",
                "--gradient-accumulation-steps",
                "2",
            ]
        )
        self.assertEqual(training.max_steps, 2)
        self.assertEqual(training.gradient_accumulation_steps, 2)
        self.assertEqual(training.attention_implementation, "sdpa")
        inference = inference_parser().parse_args(
            [
                "--bundle-root",
                "inference-bundle",
                "--bundle-report-sha256",
                "e" * 64,
                "--assets-root",
                "assets",
                "--output-dir",
                "predictions",
                "--model-name-or-path",
                "Qwen/test-model",
                "--model-revision",
                "a" * 40,
                "--model-artifact-sha256",
                "b" * 64,
                "--adapter-root",
                "adapter-run",
                "--adapter-training-report-sha256",
                "c" * 64,
                "--adapter-manifest-sha256",
                "d" * 64,
            ]
        )
        self.assertEqual(inference.adapter_root, Path("adapter-run"))


if __name__ == "__main__":
    unittest.main()
