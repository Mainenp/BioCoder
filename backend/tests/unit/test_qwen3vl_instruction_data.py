from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.build_instruction_cli import parser
from multimodal_science.qwen3vl.instruction_data import (
    TASKS,
    build_instruction_dataset,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def example(split: str, row: int, *, present: bool, group: str) -> dict[str, object]:
    return {
        "schema_version": "chrompeak-multimodal-example-v1",
        "row": row,
        "asset_id": f"{split}-asset-{row}",
        "record_id": f"{split}-record-{row}",
        "split": split,
        "group_id": group,
        "image": {
            "path": f"jobs/{split}/{group}/roi-{row}.jpeg",
            "sha256": f"{row + 1:x}" * 64,
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

            self.assertEqual(result.source_assets, 4)
            self.assertEqual(result.train_instructions, 7)
            self.assertEqual(result.validation_instructions, 7)
            self.assertEqual(len(manifest), 14)
            self.assertTrue(all(set(record) == {"image", "conversations"} for record in train))
            self.assertTrue(
                all(record["conversations"][0]["value"].count("<image>") == 1 for record in train)
            )
            self.assertTrue(all(len(record["conversations"]) == 1 for record in prompts))
            self.assertTrue(all("expected_response" not in record for record in prompts))
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
        self.assertNotIn("test", destinations)
        self.assertNotIn("internal_test", destinations)


if __name__ == "__main__":
    unittest.main()
