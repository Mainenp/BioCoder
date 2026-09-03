from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.build_inference_bundle_cli import (
    parser as inference_bundle_parser,
)
from multimodal_science.qwen3vl.build_instruction_cli import parser
from multimodal_science.qwen3vl.evaluate_predictions_cli import parser as evaluation_parser
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
    evaluate_qwen_predictions,
)
from multimodal_science.qwen3vl.instruction_data import (
    BILINGUAL_DATASET_SCHEMA,
    DATASET_SCHEMA,
    TASKS,
    build_instruction_dataset,
)
from multimodal_science.qwen3vl.inference_bundle import (
    BUNDLE_PROMPT_SCHEMA,
    BUNDLE_SCHEMA,
    build_inference_bundle,
)
from multimodal_science.qwen3vl.inference import (
    GENERATION_REPORT_SCHEMA,
    GenerationSettings,
    PromptRequest,
    run_qwen_inference,
)
from multimodal_science.qwen3vl.run_inference_cli import parser as inference_parser


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def example(split: str, row: int, *, present: bool, group: str) -> dict[str, object]:
    image_sha256 = (
        hashlib.sha256(b"test-jpeg-" + str(row).encode("ascii")).hexdigest()
        if split == "validation"
        else f"{row + 1:x}" * 64
    )
    return {
        "schema_version": "chrompeak-multimodal-example-v1",
        "row": row,
        "asset_id": f"{split}-asset-{row}",
        "record_id": f"{split}-record-{row}",
        "split": split,
        "group_id": group,
        "image": {
            "path": f"jobs/{split}/{group}/roi-{row}.jpeg",
            "sha256": image_sha256,
            "width": 400,
            "height": 300,
        },
        "metadata": {
            "component": f"Analyte {row}",
            "channel": "quantifier",
            "q1": 100.1 + row,
            "q3": 80.2 + row,
            "expected_rt_minutes": 2.5 + row,
            "roi_window_minutes": [2.0 + row, 3.5 + row],
        },
        "target": {
            "peak_present": present,
            "start_normalized": 0.2 if present else None,
            "end_normalized": 0.4 if present else None,
            "coordinate_system": "roi_fraction_0_1",
            "supervision_source": "human",
        },
        "provenance": {"asset_index_sha256": "a" * 64},
    }


def make_dataset(root: Path, *, leak_group: bool = False) -> Path:
    dataset = root / "dataset"
    train = [
        example("train", 0, present=True, group="train-source-0"),
        example("train", 1, present=False, group="train-source-1"),
    ]
    validation = [
        example(
            "validation",
            0,
            present=True,
            group="train-source-0" if leak_group else "validation-source-0",
        ),
        example("validation", 1, present=False, group="validation-source-1"),
    ]
    write_jsonl(dataset / "train" / "examples.jsonl", train)
    write_jsonl(dataset / "validation" / "examples.jsonl", validation)
    artifacts = {}
    for split in ("train", "validation"):
        split_examples = train if split == "train" else validation
        signals = np.zeros((2, 160), dtype=np.float32)
        scalars = np.zeros((2, 7), dtype=np.float32)
        targets = np.asarray(
            [
                [
                    1.0 if item["target"]["peak_present"] else 0.0,
                    item["target"]["start_normalized"]
                    if item["target"]["peak_present"]
                    else -1.0,
                    item["target"]["end_normalized"]
                    if item["target"]["peak_present"]
                    else -1.0,
                ]
                for item in split_examples
            ],
            dtype=np.float32,
        )
        for name, array in (
            ("signals", signals),
            ("scalar_features", scalars),
            ("targets", targets),
        ):
            path = dataset / split / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            artifacts[f"{split}_{name}"] = {
                "path": f"{split}/{name}.npy",
                "sha256": sha256_file(path),
            }
        path = dataset / split / "examples.jsonl"
        artifacts[f"{split}_examples"] = {
            "path": f"{split}/examples.jsonl",
            "sha256": sha256_file(path),
            "records": 2,
        }
    normalization_path = dataset / "scalar_normalization.json"
    write_json(normalization_path, {"fit_split": "train"})
    artifacts["scalar_normalization"] = {
        "path": normalization_path.name,
        "sha256": sha256_file(normalization_path),
    }
    write_json(
        dataset / "dataset_report.json",
        {
            "schema_version": "chrompeak-multimodal-dataset-v1",
            "dataset_version": "fixture-v1",
            "asset_index_sha256": "a" * 64,
            "target_points": 160,
            "splits": ["train", "validation"],
            "counts": {
                "assets": 4,
                "by_split": {
                    "train": {"assets": 2},
                    "validation": {"assets": 2},
                },
            },
            "contracts": {
                "scalar_features": [
                    "log1p_raw_maximum",
                    "log1p_dynamic_range",
                    "roi_width_minutes",
                    "expected_rt_minutes",
                    "q1",
                    "q3",
                    "signal_available",
                ]
            },
            "artifacts": artifacts,
        },
    )
    return dataset


def make_validation_assets(root: Path) -> Path:
    assets = root / "assets"
    for row, group in enumerate(("validation-source-0", "validation-source-1")):
        path = assets / "jobs" / "validation" / group / f"roi-{row}.jpeg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-jpeg-" + str(row).encode("ascii"))
    return assets


class FakeQwenGenerator:
    def __init__(
        self,
        *,
        fail_after_calls: int | None = None,
        backend: str = "test-double",
    ) -> None:
        self.calls = 0
        self.fail_after_calls = fail_after_calls
        self._metadata = {
            "backend": backend,
            "transformers_version": "4.57.1",
            "torch_version": "test",
            "resolved_model_revision": "a" * 40,
            "model_class": "FakeQwenGenerator",
        }

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def generate(
        self,
        requests: list[PromptRequest],
        settings: GenerationSettings,
    ) -> list[str]:
        del settings
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("simulated interruption")
        return [
            json.dumps(
                {"instruction_id_seen": request.instruction_id},
                separators=(",", ":"),
            )
            for request in requests
        ]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Qwen3VLInstructionDataTests(unittest.TestCase):
    def test_builds_official_train_records_and_answer_separated_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            output = root / "instructions"

            result = build_instruction_dataset(dataset, output)
            train = read_jsonl(output / "train_qwen.jsonl")
            prompts = read_jsonl(output / "validation_prompts.jsonl")
            answers = read_jsonl(output / "validation_answers.jsonl")
            manifest = read_jsonl(output / "instruction_manifest.jsonl")
            report = json.loads(
                (output / "instruction_dataset_report.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result.source_assets, 4)
            self.assertEqual(result.train_instructions, 7)
            self.assertEqual(result.validation_instructions, 7)
            self.assertEqual(len(manifest), 14)
            self.assertEqual(report["schema_version"], DATASET_SCHEMA)
            self.assertTrue(all(set(record) == {"image", "conversations"} for record in train))
            self.assertTrue(
                all(record["conversations"][0]["value"].count("<image>") == 1 for record in train)
            )
            self.assertTrue(all(len(record["conversations"]) == 1 for record in prompts))
            self.assertTrue(all("expected_response" not in record for record in prompts))
            self.assertTrue(all("language" not in record for record in prompts))
            self.assertTrue(all("pair_id" not in record for record in manifest))
            self.assertEqual(
                {record["instruction_id"] for record in prompts},
                {record["instruction_id"] for record in answers},
            )
            grounding = json.loads(train[2]["conversations"][1]["value"])
            self.assertEqual(grounding["bbox_2d"], [80, 0, 160, 300])

    def test_artifacts_are_deterministic_across_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            first = root / "first"
            second = root / "second"

            build_instruction_dataset(dataset, first, tasks=TASKS)
            build_instruction_dataset(dataset, second, tasks=TASKS)

            for name in (
                "builder_config.json",
                "train_qwen.jsonl",
                "validation_prompts.jsonl",
                "validation_answers.jsonl",
                "instruction_manifest.jsonl",
            ):
                self.assertEqual(sha256_file(first / name), sha256_file(second / name))

    def test_builds_deterministic_bilingual_train_and_parallel_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            first = root / "bilingual-first"
            second = root / "bilingual-second"

            result = build_instruction_dataset(
                dataset,
                first,
                language_profile="bilingual",
            )
            build_instruction_dataset(
                dataset,
                second,
                language_profile="bilingual",
            )

            train = read_jsonl(first / "train_qwen.jsonl")
            prompts = read_jsonl(first / "validation_prompts.jsonl")
            answers = read_jsonl(first / "validation_answers.jsonl")
            manifest = read_jsonl(first / "instruction_manifest.jsonl")
            report = json.loads(
                (first / "instruction_dataset_report.json").read_text(encoding="utf-8")
            )
            validation_manifest = [
                record for record in manifest if record["split"] == "validation"
            ]
            pairs: dict[str, list[dict[str, object]]] = {}
            for record in validation_manifest:
                pairs.setdefault(str(record["pair_id"]), []).append(record)

            self.assertEqual(result.train_instructions, 7)
            self.assertEqual(result.validation_instructions, 14)
            self.assertEqual(len(train), 7)
            self.assertEqual(len(prompts), 14)
            self.assertEqual(len(answers), 14)
            self.assertEqual(len(manifest), 21)
            self.assertTrue(all(set(record) == {"image", "conversations"} for record in train))
            self.assertEqual(len(pairs), 7)
            self.assertTrue(
                all(
                    {item["language"] for item in pair} == {"en", "zh-CN"}
                    for pair in pairs.values()
                )
            )
            self.assertTrue(
                any("请" in record["conversations"][0]["value"] for record in prompts)
            )
            self.assertEqual(report["schema_version"], BILINGUAL_DATASET_SCHEMA)
            self.assertEqual(
                report["counts"]["by_split"]["validation"]["by_language"],
                {"en": 7, "zh-CN": 7},
            )
            self.assertTrue(
                report["contracts"]["language_variants_are_not_independent_source_assets"]
            )
            for name in (
                "builder_config.json",
                "train_qwen.jsonl",
                "validation_prompts.jsonl",
                "validation_answers.jsonl",
                "instruction_manifest.jsonl",
            ):
                self.assertEqual(sha256_file(first / name), sha256_file(second / name))

    def test_rejects_tampered_source_dataset_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            with (dataset / "train" / "examples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_instruction_dataset(dataset, root / "output")

    def test_rejects_source_group_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root, leak_group=True)

            with self.assertRaisesRegex(ValueError, "Source-group leakage"):
                build_instruction_dataset(dataset, root / "output")

    def test_rejects_dataset_that_exposes_internal_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            report_path = dataset / "dataset_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["splits"].append("internal_test")
            write_json(report_path, report)

            with self.assertRaisesRegex(ValueError, "train/validation-only"):
                build_instruction_dataset(dataset, root / "output")

    def test_cli_has_no_internal_test_surface(self) -> None:
        command = parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            ["--dataset-root", "dataset", "--output-dir", "instructions"]
        )

        self.assertIsNone(arguments.task)
        self.assertEqual(arguments.language_profile, "english")
        bilingual = command.parse_args(
            [
                "--dataset-root",
                "dataset",
                "--output-dir",
                "instructions",
                "--language-profile",
                "bilingual",
                "--chinese-train-ratio",
                "0.7",
            ]
        )
        self.assertEqual(bilingual.language_profile, "bilingual")
        self.assertEqual(bilingual.chinese_train_ratio, 0.7)
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)


class Qwen3VLInferenceBundleTests(unittest.TestCase):
    def test_builds_prompt_only_bilingual_bundle_without_reading_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "instructions"
            build_instruction_dataset(
                make_dataset(root),
                instructions,
                language_profile="bilingual",
            )
            instruction_report_hash = sha256_file(
                instructions / "instruction_dataset_report.json"
            )
            with (instructions / "validation_answers.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("tampered but deliberately unopened\n")

            result = build_inference_bundle(
                instructions,
                root / "bundle",
                expected_instruction_report_sha256=instruction_report_hash,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            prompts = read_jsonl(root / "bundle" / "inference_prompts.jsonl")
            bundle_text = (root / "bundle" / "inference_prompts.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(report["schema_version"], BUNDLE_SCHEMA)
            self.assertEqual(result.prompt_records, 14)
            self.assertEqual(report["counts"]["by_language"], {"en": 7, "zh-CN": 7})
            self.assertTrue(report["contracts"]["prompt_only"])
            self.assertFalse(report["contracts"]["answer_key_opened"])
            self.assertFalse(report["contracts"]["answer_key_materialized"])
            self.assertEqual(set(report["artifacts"]), {"inference_prompts"})
            self.assertTrue(
                all(record["schema_version"] == BUNDLE_PROMPT_SCHEMA for record in prompts)
            )
            self.assertNotIn("expected_response", bundle_text)
            self.assertNotIn("response_sha256", bundle_text)

    def test_bundle_cli_exposes_no_answer_or_internal_test_path(self) -> None:
        command = inference_bundle_parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            [
                "--instruction-root",
                "instructions",
                "--output-dir",
                "bundle",
                "--instruction-report-sha256",
                "a" * 64,
            ]
        )

        self.assertEqual(arguments.output_dir, Path("bundle"))
        self.assertNotIn("answer", destinations)
        self.assertNotIn("answers", destinations)
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)


class Qwen3VLInferenceRunnerTests(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, str]:
        instructions = root / "instructions"
        build_instruction_dataset(
            make_dataset(root),
            instructions,
            language_profile="bilingual",
        )
        bundle = root / "bundle"
        result = build_inference_bundle(
            instructions,
            bundle,
            expected_instruction_report_sha256=sha256_file(
                instructions / "instruction_dataset_report.json"
            ),
        )
        return bundle, result.report_sha256

    def test_runs_prompt_only_generation_and_emits_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, bundle_hash = self._bundle(root)
            assets = make_validation_assets(root)
            fake = FakeQwenGenerator()

            result = run_qwen_inference(
                bundle,
                assets,
                root / "run",
                expected_bundle_report_sha256=bundle_hash,
                model_name_or_path="Qwen/test-model",
                model_revision="a" * 40,
                settings=GenerationSettings(batch_size=2),
                max_records=3,
                generator_factory=lambda *_: fake,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            predictions = read_jsonl(result.predictions_path)
            evidence = read_jsonl(result.output_dir / "generation_records.jsonl")
            self.assertEqual(report["schema_version"], GENERATION_REPORT_SCHEMA)
            self.assertEqual(result.prediction_records, 3)
            self.assertFalse(result.complete_prompt_coverage)
            self.assertFalse(result.development_comparison_candidate)
            self.assertEqual(len(predictions), 3)
            self.assertEqual(len(evidence), 3)
            self.assertEqual(report["counts"]["unique_images_opened"], 1)
            self.assertTrue(report["contracts"]["input_is_prompt_only_bundle"])
            self.assertFalse(report["contracts"]["answer_key_available_to_runner"])
            self.assertFalse(report["internal_test_accessed"])

    def test_resumes_from_a_verified_prompt_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, bundle_hash = self._bundle(root)
            assets = make_validation_assets(root)
            output = root / "resumed-run"
            settings = GenerationSettings(batch_size=1)
            interrupted = FakeQwenGenerator(fail_after_calls=1)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_qwen_inference(
                    bundle,
                    assets,
                    output,
                    expected_bundle_report_sha256=bundle_hash,
                    model_name_or_path="Qwen/test-model",
                    model_revision="a" * 40,
                    settings=settings,
                    max_records=3,
                    generator_factory=lambda *_: interrupted,
                )

            journal = root / ".resumed-run.work" / "generation_records.jsonl"
            self.assertEqual(len(read_jsonl(journal)), 1)
            resumed = FakeQwenGenerator()
            result = run_qwen_inference(
                bundle,
                assets,
                output,
                expected_bundle_report_sha256=bundle_hash,
                model_name_or_path="Qwen/test-model",
                model_revision="a" * 40,
                settings=settings,
                max_records=3,
                resume=True,
                generator_factory=lambda *_: resumed,
            )

            self.assertEqual(result.prediction_records, 3)
            self.assertEqual(resumed.calls, 2)
            self.assertFalse((root / ".resumed-run.work").exists())
            self.assertTrue(output.is_dir())

    def test_inference_cli_has_no_answer_instruction_or_internal_test_path(self) -> None:
        command = inference_parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            [
                "--bundle-root",
                "bundle",
                "--bundle-report-sha256",
                "a" * 64,
                "--assets-root",
                "assets",
                "--output-dir",
                "run",
                "--model-name-or-path",
                "Qwen/model",
                "--model-revision",
                "b" * 40,
            ]
        )

        self.assertEqual(arguments.batch_size, 1)
        self.assertNotIn("instruction_root", destinations)
        self.assertNotIn("answer", destinations)
        self.assertNotIn("answers", destinations)
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)


class Qwen3VLEvaluationTests(unittest.TestCase):
    def _perfect_predictions(self, instructions: Path, path: Path) -> list[dict[str, object]]:
        answers = read_jsonl(instructions / "validation_answers.jsonl")
        predictions = [
            {
                "schema_version": PREDICTION_SCHEMA,
                "instruction_id": answer["instruction_id"],
                "response": answer["expected_response"],
            }
            for answer in answers
        ]
        write_jsonl(path, predictions)
        return predictions

    def test_scores_complete_predictions_across_all_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "instructions"
            build_instruction_dataset(make_dataset(root), instructions)
            predictions = root / "predictions.jsonl"
            self._perfect_predictions(instructions, predictions)

            result = evaluate_qwen_predictions(
                instructions,
                predictions,
                root / "evaluation",
                expected_instruction_report_sha256=sha256_file(
                    instructions / "instruction_dataset_report.json"
                ),
                bootstrap_iterations=20,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(result.prediction_records, 7)
            self.assertEqual(result.valid_json_records, 7)
            self.assertEqual(result.schema_valid_records, 7)
            self.assertEqual(result.validation_source_groups, 2)
            self.assertEqual(report["metrics"]["peak_presence"]["exact_match_rate"], 1.0)
            self.assertEqual(
                report["metrics"]["peak_presence_metadata"]["classification"]["macro_f1"],
                1.0,
            )
            self.assertEqual(
                report["metrics"]["peak_grounding"]["grounding"]["mean_bbox_iou_all"],
                1.0,
            )
            self.assertEqual(report["metrics"]["scientific_qc"]["exact_match_rate"], 1.0)
            self.assertFalse(report["development_comparison_eligible"])
            self.assertFalse(report["final_benchmark_eligible"])
            self.assertFalse(report["internal_test_accessed"])
            self.assertFalse(report["prediction_generation_provenance_verified"])

    def test_qualifies_only_complete_hash_bound_transformers_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "instructions"
            build_instruction_dataset(
                make_dataset(root),
                instructions,
                language_profile="bilingual",
            )
            instruction_report_hash = sha256_file(
                instructions / "instruction_dataset_report.json"
            )
            bundle = root / "bundle"
            bundle_result = build_inference_bundle(
                instructions,
                bundle,
                expected_instruction_report_sha256=instruction_report_hash,
            )
            generation = run_qwen_inference(
                bundle,
                make_validation_assets(root),
                root / "generation",
                expected_bundle_report_sha256=bundle_result.report_sha256,
                model_name_or_path="Qwen/test-model",
                model_revision="a" * 40,
                settings=GenerationSettings(batch_size=3),
                generator_factory=lambda *_: FakeQwenGenerator(
                    backend="transformers"
                ),
            )

            result = evaluate_qwen_predictions(
                instructions,
                generation.predictions_path,
                root / "evaluation",
                expected_instruction_report_sha256=instruction_report_hash,
                generation_report_path=generation.report_path,
                expected_generation_report_sha256=generation.report_sha256,
                bootstrap_iterations=20,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertTrue(result.generation_provenance_verified)
            self.assertTrue(result.development_comparison_eligible)
            self.assertTrue(report["prediction_generation_provenance_verified"])
            self.assertTrue(report["development_comparison_eligible"])
            self.assertEqual(report["counts"]["predictions"], 14)
            self.assertEqual(report["counts"]["schema_valid"], 0)
            self.assertFalse(report["final_benchmark_eligible"])

            evidence_path = generation.output_dir / "generation_records.jsonl"
            evidence = read_jsonl(evidence_path)
            evidence[0]["image_sha256"] = "f" * 64
            write_jsonl(evidence_path, evidence)
            generation_report = json.loads(
                generation.report_path.read_text(encoding="utf-8")
            )
            generation_report["artifacts"]["generation_records"]["sha256"] = (
                sha256_file(evidence_path)
            )
            write_json(generation.report_path, generation_report)
            with self.assertRaisesRegex(ValueError, "image_sha256 mismatch"):
                evaluate_qwen_predictions(
                    instructions,
                    generation.predictions_path,
                    root / "tampered-evaluation",
                    expected_instruction_report_sha256=instruction_report_hash,
                    generation_report_path=generation.report_path,
                    expected_generation_report_sha256=sha256_file(
                        generation.report_path
                    ),
                    bootstrap_iterations=20,
                )

    def test_malformed_json_is_retained_as_a_scored_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "instructions"
            build_instruction_dataset(make_dataset(root), instructions)
            predictions_path = root / "predictions.jsonl"
            predictions = self._perfect_predictions(instructions, predictions_path)
            prompts = read_jsonl(instructions / "validation_prompts.jsonl")
            invalid_id = next(
                record["instruction_id"]
                for record in prompts
                if record["task"] == "peak_presence"
            )
            for prediction in predictions:
                if prediction["instruction_id"] == invalid_id:
                    prediction["response"] = "```json\n{\"peak_present\": true}\n```"
            write_jsonl(predictions_path, predictions)

            result = evaluate_qwen_predictions(
                instructions,
                predictions_path,
                root / "evaluation",
                expected_instruction_report_sha256=sha256_file(
                    instructions / "instruction_dataset_report.json"
                ),
                bootstrap_iterations=20,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(result.valid_json_records, 6)
            self.assertEqual(result.schema_valid_records, 6)
            self.assertLess(report["metrics"]["peak_presence"]["exact_match_rate"], 1.0)
            records = read_jsonl(root / "evaluation" / "evaluation_records.jsonl")
            invalid = next(record for record in records if record["instruction_id"] == invalid_id)
            self.assertFalse(invalid["valid_json"])
            self.assertFalse(invalid["schema_valid"])

    def test_reports_language_metrics_and_cross_language_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "bilingual-instructions"
            build_instruction_dataset(
                make_dataset(root),
                instructions,
                language_profile="bilingual",
            )
            predictions_path = root / "bilingual-predictions.jsonl"
            predictions = self._perfect_predictions(instructions, predictions_path)

            result = evaluate_qwen_predictions(
                instructions,
                predictions_path,
                root / "perfect-evaluation",
                expected_instruction_report_sha256=sha256_file(
                    instructions / "instruction_dataset_report.json"
                ),
                bootstrap_iterations=20,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertEqual(result.prediction_records, 14)
            self.assertEqual(report["schema_version"], BILINGUAL_EVALUATION_REPORT_SCHEMA)
            self.assertEqual(report["counts"]["by_language"], {"en": 7, "zh-CN": 7})
            self.assertEqual(
                report["metrics_by_language"]["zh-CN"]["peak_presence"][
                    "exact_match_rate"
                ],
                1.0,
            )
            self.assertEqual(
                report["cross_language_consistency"]["peak_grounding"][
                    "mean_prediction_bbox_iou_all"
                ],
                1.0,
            )

            answers = read_jsonl(instructions / "validation_answers.jsonl")
            changed_id = next(
                answer["instruction_id"]
                for answer in answers
                if answer["task"] == "peak_presence"
                and answer["language"] == "zh-CN"
                and json.loads(answer["expected_response"])["peak_present"]
            )
            for prediction in predictions:
                if prediction["instruction_id"] == changed_id:
                    prediction["response"] = '{"peak_present":false}'
            write_jsonl(predictions_path, predictions)
            changed = evaluate_qwen_predictions(
                instructions,
                predictions_path,
                root / "changed-evaluation",
                expected_instruction_report_sha256=sha256_file(
                    instructions / "instruction_dataset_report.json"
                ),
                bootstrap_iterations=20,
            )
            changed_report = json.loads(changed.report_path.read_text(encoding="utf-8"))

            self.assertEqual(
                changed_report["metrics_by_language"]["en"]["peak_presence"][
                    "exact_match_rate"
                ],
                1.0,
            )
            self.assertLess(
                changed_report["metrics_by_language"]["zh-CN"]["peak_presence"][
                    "exact_match_rate"
                ],
                1.0,
            )
            self.assertLess(
                changed_report["cross_language_consistency"]["peak_presence"][
                    "exact_prediction_consistency_rate_all"
                ],
                1.0,
            )

    def test_rejects_incomplete_or_tampered_evaluation_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instructions = root / "instructions"
            build_instruction_dataset(make_dataset(root), instructions)
            predictions_path = root / "predictions.jsonl"
            predictions = self._perfect_predictions(instructions, predictions_path)

            with self.assertRaisesRegex(ValueError, "Instruction report hash mismatch"):
                evaluate_qwen_predictions(
                    instructions,
                    predictions_path,
                    root / "wrong-report",
                    expected_instruction_report_sha256="0" * 64,
                    bootstrap_iterations=20,
                )

            write_jsonl(predictions_path, predictions[:-1])

            with self.assertRaisesRegex(ValueError, "Prediction IDs"):
                evaluate_qwen_predictions(
                    instructions,
                    predictions_path,
                    root / "incomplete",
                    expected_instruction_report_sha256=sha256_file(
                        instructions / "instruction_dataset_report.json"
                    ),
                    bootstrap_iterations=20,
                )

            self._perfect_predictions(instructions, predictions_path)
            with (instructions / "validation_answers.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_qwen_predictions(
                    instructions,
                    predictions_path,
                    root / "tampered",
                    expected_instruction_report_sha256=sha256_file(
                        instructions / "instruction_dataset_report.json"
                    ),
                    bootstrap_iterations=20,
                )

    def test_evaluation_cli_has_no_internal_test_surface(self) -> None:
        command = evaluation_parser()
        destinations = {action.dest for action in command._actions}
        arguments = command.parse_args(
            [
                "--instruction-root",
                "instructions",
                "--predictions",
                "predictions.jsonl",
                "--output-dir",
                "evaluation",
                "--instruction-report-sha256",
                "a" * 64,
            ]
        )

        self.assertEqual(arguments.bootstrap_iterations, 1000)
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)


if __name__ == "__main__":
    unittest.main()
