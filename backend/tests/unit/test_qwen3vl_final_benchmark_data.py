from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.baselines.final_sequence_evaluation import (
    FINAL_SEQUENCE_EVALUATION_SCHEMA,
    evaluate_final_sequence_candidate,
)
from multimodal_science.chrompeakformer.final_detector_evaluation import (
    FINAL_DETECTOR_EVALUATION_SCHEMA,
    _manifest_entries as _detector_manifest_entries,
    _source_tree_sha256,
    evaluate_final_detector_candidate,
)
from multimodal_science.qwen3vl.final_benchmark_data import (
    FINAL_BENCHMARK_DATA_SCHEMA,
    FINAL_INFERENCE_BUNDLE_SCHEMA,
    build_final_benchmark_data,
)
from multimodal_science.qwen3vl.final_benchmark_evaluation import (
    FINAL_QWEN_EVALUATION_SCHEMA,
    evaluate_final_qwen_predictions,
)
from multimodal_science.qwen3vl.final_benchmark_report import (
    FINAL_BENCHMARK_REPORT_SCHEMA,
    build_final_benchmark_report,
)
from multimodal_science.qwen3vl.inference import (
    FinalBenchmarkAccessSpec,
    GenerationSettings,
    run_qwen_inference,
)
from multimodal_science.qwen3vl.fusion_inference import _load_validation_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_manifest(root: Path, relatives: list[str]) -> None:
    (root / "artifact_manifest.sha256").write_text(
        "".join(f"{_sha256(root / relative)}  {relative}\n" for relative in relatives),
        encoding="utf-8",
    )


class _FinalDataFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.protocol_root = root / "protocol"
        self.ledger = root / "ledger"
        self.dataset_root = root / "dataset"
        self.assets_root = root / "assets"
        self.output = root / "final-data"
        self.normalization_sha = ""
        self.protocol_sha = ""
        self.dataset_sha = ""
        self._build()

    def _example(self, row: int, *, present: bool) -> dict:
        relative_image = f"jobs/internal_test/group-{row}/roi.jpeg"
        image_path = self.assets_root / relative_image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{row}".encode("utf-8"))
        return {
            "schema_version": "chrompeak-multimodal-example-v1",
            "row": row,
            "asset_id": f"asset-{row}",
            "record_id": f"record-{row}",
            "split": "internal_test",
            "group_id": f"group-{row}",
            "image": {
                "path": relative_image,
                "sha256": _sha256(image_path),
                "width": 400,
                "height": 300,
            },
            "sequence": {
                "array": "internal_test/signals.npy",
                "row": row,
                "length": 160,
                "signal_available": True,
            },
            "scalar_features": {
                "array": "internal_test/scalar_features.npy",
                "row": row,
            },
            "metadata": {
                "component": "fixture",
                "channel": "fixture-channel",
                "q1": 100.0,
                "q3": 50.0,
                "expected_rt_minutes": 1.5,
                "roi_window_minutes": [1.0, 2.0],
            },
            "target": {
                "array": "internal_test/targets.npy",
                "row": row,
                "peak_present": present,
                "start_normalized": 0.25 if present else None,
                "end_normalized": 0.75 if present else None,
                "coordinate_system": "roi_fraction_0_1",
                "supervision_source": "human",
            },
            "provenance": {"asset_index_sha256": "a" * 64},
        }

    def _build(self) -> None:
        normalization = self.dataset_root / "scalar_normalization.json"
        _write_json(
            normalization,
            {
                "schema_version": "chrompeak-scalar-normalization-v1",
                "fit_split": "train",
            },
        )
        self.normalization_sha = _sha256(normalization)
        examples = [self._example(0, present=True), self._example(1, present=False)]
        arrays = {
            "internal_test/signals.npy": np.vstack(
                [
                    np.linspace(0.0, 1.0, 160, dtype=np.float32),
                    np.linspace(1.0, 0.0, 160, dtype=np.float32),
                ]
            ),
            "internal_test/scalar_features.npy": np.zeros((2, 7), dtype=np.float32),
            "internal_test/targets.npy": np.asarray(
                [[1.0, 0.25, 0.75], [0.0, -1.0, -1.0]], dtype=np.float32
            ),
        }
        for relative, content in arrays.items():
            path = self.dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, content, allow_pickle=False)
        examples_path = self.dataset_root / "internal_test/examples.jsonl"
        _write_jsonl(examples_path, examples)
        report_path = self.dataset_root / "dataset_report.json"
        _write_json(
            report_path,
            {
                "schema_version": "chrompeak-multimodal-dataset-v1",
                "asset_index_sha256": "a" * 64,
                "target_points": 160,
                "splits": ["internal_test"],
                "frozen_scalar_normalization_sha256": self.normalization_sha,
                "contracts": {
                    "scalar_features": [f"feature_{index}" for index in range(7)]
                },
                "counts": {
                    "assets": 2,
                    "by_split": {
                        "internal_test": {
                            "assets": 2,
                            "positive": 1,
                            "negative": 1,
                        }
                    },
                },
                "artifacts": {
                    "scalar_normalization": {
                        "path": "scalar_normalization.json",
                        "sha256": self.normalization_sha,
                    },
                    "internal_test_signals": {
                        "path": "internal_test/signals.npy",
                        "sha256": _sha256(self.dataset_root / "internal_test/signals.npy"),
                        "shape": [2, 160],
                    },
                    "internal_test_scalar_features": {
                        "path": "internal_test/scalar_features.npy",
                        "sha256": _sha256(
                            self.dataset_root / "internal_test/scalar_features.npy"
                        ),
                        "shape": [2, 7],
                    },
                    "internal_test_targets": {
                        "path": "internal_test/targets.npy",
                        "sha256": _sha256(self.dataset_root / "internal_test/targets.npy"),
                        "shape": [2, 3],
                    },
                    "internal_test_examples": {
                        "path": "internal_test/examples.jsonl",
                        "sha256": _sha256(examples_path),
                        "records": 2,
                    },
                },
            },
        )
        self.dataset_sha = _sha256(report_path)

        sequence_root = self.root / "sequence-candidate"
        sequence_root.mkdir(parents=True)
        sequence_checkpoint = sequence_root / "best_model.pt"
        sequence_checkpoint.write_bytes(b"fixture-sequence-checkpoint")
        sequence_threshold = sequence_root / "frozen_threshold.json"
        _write_json(
            sequence_threshold,
            {
                "schema_version": "chrompeak-frozen-threshold-v1",
                "selected_on_split": "validation",
                "objective": "macro_f1",
                "threshold": 0.5,
                "internal_test_accessed": False,
            },
        )
        sequence_report = sequence_root / "scientific_report.json"
        _write_json(
            sequence_report,
            {
                "schema_version": "chrompeak-sequence-baseline-report-v1",
                "development_comparison_eligible": True,
                "internal_test_accessed": False,
                "config": {"modality": "sequence"},
            },
        )
        _write_manifest(
            sequence_root,
            [
                sequence_checkpoint.name,
                sequence_threshold.name,
                sequence_report.name,
            ],
        )

        detector_source = self.root / "detector-source"
        (detector_source / "models").mkdir(parents=True)
        (detector_source / "framework").mkdir()
        (detector_source / "models/model.py").write_text(
            "MODEL = 'fixture'\n", encoding="utf-8"
        )
        (detector_source / "framework/data.py").write_text(
            "DATA = 'fixture'\n", encoding="utf-8"
        )
        self.detector_source = detector_source
        detector_root = self.root / "detector-candidate"
        detector_training = detector_root / "training"
        detector_training.mkdir(parents=True)
        detector_checkpoint = detector_training / "checkpoint.pth"
        detector_checkpoint.write_bytes(b"fixture-detector-checkpoint")
        detector_report = detector_training / "detector_training_report.json"
        _write_json(
            detector_report,
            {
                "schema_version": "chrompeak-detector-training-v1",
                "model_family": "ChromPeakFormer",
                "development_comparison_eligible": True,
                "internal_test_accessed": False,
                "source_tree_sha256": _source_tree_sha256(detector_source),
            },
        )
        _write_manifest(
            detector_root,
            [
                "./training/checkpoint.pth",
                "./training/detector_training_report.json",
            ],
        )

        self.protocol_root.mkdir(parents=True)
        candidate_path = self.protocol_root / "candidate_lock.json"
        metrics_path = self.protocol_root / "metrics_lock.json"
        _write_json(
            candidate_path,
            {
                "schema_version": "chrompeak-final-benchmark-candidate-lock-v1",
                "primary_models": [
                    "qwen3vl_zero_shot",
                    "qwen3vl_image_lora",
                    "qwen3vl_image_xic_fusion",
                    "sequence_peak_net",
                    "chrompeakformer",
                ],
                "shared": {
                    "base_model": {
                        "name_or_path": "fixture-model",
                        "revision": "fixture-revision",
                        "artifact_sha256": "c" * 64,
                    },
                    "inference_policy": {
                        "batch_size": 1,
                        "max_new_tokens": 64,
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "seed": 17,
                        "dtype": "bfloat16",
                        "zero_and_lora_device_map": "auto",
                        "zero_and_lora_attention_implementation": None,
                        "fusion_device_map": "single_cuda",
                        "fusion_attention_implementation": "sdpa",
                    },
                    "scalar_normalization": {
                        "path": "/frozen/train/scalar_normalization.json",
                        "sha256": self.normalization_sha,
                        "fit_split": "train",
                    }
                },
                "models": {
                    "qwen3vl_zero_shot": {"adapter": None},
                    "qwen3vl_image_lora": {"adapter": "fixture"},
                    "qwen3vl_image_xic_fusion": {"adapter": "fixture"},
                    "sequence_peak_net": {
                        "root": str(sequence_root),
                        "report_sha256": _sha256(sequence_report),
                        "manifest_sha256": _sha256(
                            sequence_root / "artifact_manifest.sha256"
                        ),
                        "checkpoint": {
                            "path": str(sequence_checkpoint),
                            "sha256": _sha256(sequence_checkpoint),
                        },
                        "frozen_threshold": {
                            "path": str(sequence_threshold),
                            "sha256": _sha256(sequence_threshold),
                            "value": 0.5,
                            "selected_on_split": "validation",
                            "objective": "macro_f1",
                        },
                    },
                    "chrompeakformer": {
                        "root": str(detector_root),
                        "training_report_sha256": _sha256(detector_report),
                        "evaluation_report_sha256": "d" * 64,
                        "manifest_sha256": _sha256(
                            detector_root / "artifact_manifest.sha256"
                        ),
                        "checkpoint_sha256": _sha256(detector_checkpoint),
                        "primary_threshold": 0.5,
                        "primary_threshold_selected_on_split": "validation",
                        "secondary_fixed_threshold": 0.5,
                    },
                },
            },
        )
        _write_json(
            metrics_path,
            {
                "schema_version": "chrompeak-final-benchmark-metrics-lock-v1",
                "threshold_selection_forbidden": True,
                "post_access_model_selection_forbidden": True,
                "post_access_threshold_tuning_forbidden": True,
                "threshold_policy": {
                    "primary": "validation_frozen_operating_threshold",
                    "secondary": "fixed_0.5",
                    "internal_test_threshold_selection_forbidden": True,
                },
                "uncertainty": {
                    "bootstrap_iterations": 10_000,
                    "seed": 17,
                },
            },
        )
        protocol_path = self.protocol_root / "final_benchmark_protocol.json"
        _write_json(
            protocol_path,
            {
                "schema_version": "chrompeak-final-benchmark-protocol-v1",
                "pre_internal_test_ready": True,
                "internal_test_accessed": False,
                "sealed_split": {
                    "name": "internal_test",
                    "expected_assets": 2,
                    "expected_source_groups": 2,
                },
                "locks": {
                    "candidate": {
                        "path": candidate_path.name,
                        "sha256": _sha256(candidate_path),
                    },
                    "metrics": {
                        "path": metrics_path.name,
                        "sha256": _sha256(metrics_path),
                    },
                },
            },
        )
        _write_manifest(
            self.protocol_root,
            [protocol_path.name, candidate_path.name, metrics_path.name],
        )
        self.protocol_sha = _sha256(protocol_path)

        self.ledger.mkdir(parents=True)
        access_path = self.ledger / "access_started.json"
        _write_json(
            access_path,
            {
                "schema_version": "chrompeak-final-benchmark-access-v1",
                "access_id": "b" * 24,
                "access_sequence": 1,
                "protocol_sha256": self.protocol_sha,
                "internal_test_accessed": True,
                "completed": False,
            },
        )
        (self.ledger / "access_started.sha256").write_text(
            f"{_sha256(access_path)}  {access_path.name}\n", encoding="utf-8"
        )

    def build(self):
        return build_final_benchmark_data(
            protocol_root=self.protocol_root,
            expected_protocol_sha256=self.protocol_sha,
            ledger_dir=self.ledger,
            dataset_root=self.dataset_root,
            expected_dataset_report_sha256=self.dataset_sha,
            assets_root=self.assets_root,
            output_dir=self.output,
        )


class FinalBenchmarkDataTests(unittest.TestCase):
    def test_builds_disjoint_hash_bound_internal_test_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            result = fixture.build()

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            inference = json.loads(
                (result.inference_root / "inference_bundle_report.json").read_text(
                    encoding="utf-8"
                )
            )
            answers_text = (result.answer_root / "answers.jsonl").read_text(
                encoding="utf-8"
            )
            prompts_text = (result.inference_root / "inference_prompts.jsonl").read_text(
                encoding="utf-8"
            )

            self.assertEqual(report["schema_version"], FINAL_BENCHMARK_DATA_SCHEMA)
            self.assertEqual(inference["schema_version"], FINAL_INFERENCE_BUNDLE_SCHEMA)
            self.assertEqual(result.assets, 2)
            self.assertEqual(result.source_groups, 2)
            self.assertEqual(result.prompts, 14)
            self.assertNotIn("expected_response", prompts_text)
            self.assertIn("expected_response", answers_text)
            self.assertFalse((result.inference_root / "answers.jsonl").exists())
            self.assertTrue((result.detector_root / "coco/val/val_coco.json").is_file())
            self.assertTrue(report["contracts"]["prompt_answer_roots_disjoint"])
            for root in (
                result.output_dir,
                result.inference_root,
                result.answer_root,
                result.fusion_root,
                result.detector_root,
            ):
                self.assertTrue((root / "artifact_manifest.sha256").is_file())

            cached = fixture.build()
            self.assertTrue(cached.cached)
            self.assertEqual(cached.report_sha256, result.report_sha256)

    def test_rejects_internal_dataset_with_nonfrozen_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            report_path = fixture.dataset_root / "dataset_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["frozen_scalar_normalization_sha256"] = "0" * 64
            _write_json(report_path, report)
            fixture.dataset_sha = _sha256(report_path)

            with self.assertRaisesRegex(ValueError, "frozen train fit"):
                fixture.build()

    def test_final_zero_shot_inference_requires_the_frozen_access_and_full_scope(self) -> None:
        class Generator:
            metadata = {
                "backend": "transformers",
                "resolved_model_revision": "d" * 40,
            }

            def generate(self, requests, settings):
                return ['{"peak_present":true}' for _ in requests]

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            data = fixture.build()
            access = FinalBenchmarkAccessSpec(
                protocol_root=fixture.protocol_root,
                protocol_sha256=fixture.protocol_sha,
                ledger_dir=fixture.ledger,
                candidate_name="qwen3vl_zero_shot",
            )
            result = run_qwen_inference(
                data.inference_root,
                fixture.assets_root,
                fixture.root / "zero-shot-final",
                expected_bundle_report_sha256=_sha256(
                    data.inference_root / "inference_bundle_report.json"
                ),
                model_name_or_path="fixture-model",
                model_revision="fixture-revision",
                model_artifact_sha256="c" * 64,
                settings=GenerationSettings(batch_size=1, dtype="bfloat16"),
                generator_factory=lambda *args: Generator(),
                final_benchmark_access=access,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertTrue(result.internal_test_accessed)
            self.assertTrue(result.final_benchmark_candidate)
            self.assertFalse(result.development_comparison_candidate)
            self.assertTrue(report["contracts"]["answer_key_available_to_runner"] is False)
            self.assertEqual(report["source"]["access_id"], "b" * 24)

            with self.assertRaisesRegex(ValueError, "complete prompt coverage"):
                run_qwen_inference(
                    data.inference_root,
                    fixture.assets_root,
                    fixture.root / "partial-final",
                    expected_bundle_report_sha256=_sha256(
                        data.inference_root / "inference_bundle_report.json"
                    ),
                    model_name_or_path="fixture-model",
                    model_revision="fixture-revision",
                    model_artifact_sha256="c" * 64,
                    settings=GenerationSettings(batch_size=1, dtype="bfloat16"),
                    max_records=1,
                    generator_factory=lambda *args: Generator(),
                    final_benchmark_access=access,
                )

            with self.assertRaisesRegex(ValueError, "Final generation setting drift"):
                run_qwen_inference(
                    data.inference_root,
                    fixture.assets_root,
                    fixture.root / "settings-drift-final",
                    expected_bundle_report_sha256=_sha256(
                        data.inference_root / "inference_bundle_report.json"
                    ),
                    model_name_or_path="fixture-model",
                    model_revision="fixture-revision",
                    model_artifact_sha256="c" * 64,
                    settings=GenerationSettings(
                        batch_size=1,
                        dtype="bfloat16",
                        max_new_tokens=32,
                    ),
                    generator_factory=lambda *args: Generator(),
                    final_benchmark_access=access,
                )

    def test_final_fusion_loader_binds_prompts_signals_and_access_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            data = fixture.build()
            loaded = _load_validation_inputs(
                fusion_bundle_root=data.fusion_root,
                fusion_bundle_report_sha256=_sha256(
                    data.fusion_root / "fusion_bundle_report.json"
                ),
                dataset_root=fixture.dataset_root,
                dataset_report_sha256=fixture.dataset_sha,
                inference_bundle_root=data.inference_root,
                inference_bundle_report_sha256=_sha256(
                    data.inference_root / "inference_bundle_report.json"
                ),
                final_access={
                    "protocol_sha256": fixture.protocol_sha,
                    "access_id": "b" * 24,
                    "candidate_name": "qwen3vl_image_xic_fusion",
                },
            )
            self.assertEqual(loaded.signal_shape, (2, 160))
            self.assertEqual(len(loaded.links_by_instruction_id), 14)
            self.assertEqual(loaded.dataset_report_sha256, fixture.dataset_sha)

    def test_final_sequence_evaluation_uses_only_the_frozen_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            fixture.build()

            def load_checkpoint(path):
                self.assertEqual(path.name, "best_model.pt")
                return {
                    "schema_version": "chrompeak-sequence-baseline-checkpoint-v1",
                    "asset_index_sha256": "a" * 64,
                    "model_spec": {
                        "input_points": 160,
                        "scalar_features": 7,
                        "modality": "sequence",
                        "base_channels": 32,
                        "position_bins": 10,
                        "dropout": 0.15,
                    },
                    "model_state_dict": {},
                }

            def predict(split, checkpoint, device, batch_size):
                self.assertEqual(split.split, "internal_test")
                self.assertEqual(device, "cpu")
                self.assertEqual(batch_size, 2)
                return (
                    np.asarray([0.9, 0.1], dtype=np.float64),
                    np.asarray([[0.25, 0.75], [0.1, 0.2]], dtype=np.float64),
                )

            result = evaluate_final_sequence_candidate(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha,
                ledger_dir=fixture.ledger,
                candidate_name="sequence_peak_net",
                dataset_root=fixture.dataset_root,
                expected_dataset_report_sha256=fixture.dataset_sha,
                output_dir=fixture.root / "sequence-final",
                device="cpu",
                batch_size=2,
                checkpoint_loader=load_checkpoint,
                predictor=predict,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], FINAL_SEQUENCE_EVALUATION_SCHEMA)
            self.assertEqual(
                report["classification"]["validation_frozen_threshold"]["accuracy"],
                1.0,
            )
            self.assertEqual(report["localization"]["mean_iou"], 1.0)
            self.assertFalse(report["evaluation"]["threshold_selection_performed"])
            self.assertEqual(report["evaluation"]["bootstrap_iterations"], 10_000)
            self.assertTrue(report["final_benchmark_eligible"])

    def test_final_detector_evaluation_uses_the_frozen_candidate_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            data = fixture.build()

            def predict(source, checkpoint, coco_root, device, batch, workers, amp):
                self.assertEqual(source, fixture.detector_source.resolve())
                self.assertEqual(checkpoint.name, "checkpoint.pth")
                self.assertEqual(coco_root.name, "coco")
                self.assertEqual(device, "cpu")
                return (
                    [
                        {
                            "image_id": 1,
                            "category_id": 0,
                            "bbox": [100.0, 0.0, 200.0, 300.0],
                            "score": 0.9,
                        },
                        {
                            "image_id": 2,
                            "category_id": 0,
                            "bbox": [20.0, 0.0, 30.0, 300.0],
                            "score": 0.1,
                        },
                    ],
                    [1, 2],
                )

            result = evaluate_final_detector_candidate(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha,
                ledger_dir=fixture.ledger,
                source_root=fixture.detector_source,
                detector_data_root=data.detector_root,
                expected_detector_report_sha256=_sha256(
                    data.detector_root / "detector_dataset_report.json"
                ),
                output_dir=fixture.root / "detector-final",
                device="cpu",
                batch_size=2,
                num_workers=0,
                amp=False,
                predictor=predict,
                coco_metric_function=lambda truth, predictions: {
                    "ap_50_95": 1.0,
                    "ap_50": 1.0,
                    "ap_75": 1.0,
                    "ar_at_1": 1.0,
                    "ar_at_10": 1.0,
                    "ar_at_100": 1.0,
                },
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], FINAL_DETECTOR_EVALUATION_SCHEMA)
            self.assertEqual(
                report["classification"]["validation_frozen_threshold"]["accuracy"],
                1.0,
            )
            self.assertEqual(
                report["localization"]["validation_frozen_threshold"]["mean_best_iou"],
                1.0,
            )
            self.assertFalse(report["evaluation"]["threshold_selection_performed"])
            self.assertEqual(report["evaluation"]["bootstrap_iterations"], 10_000)
            self.assertTrue(report["final_benchmark_eligible"])

    def test_detector_manifest_rejects_duplicate_normalized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "training" / "checkpoint.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            digest = _sha256(checkpoint)
            manifest = root / "artifact_manifest.sha256"
            manifest.write_text(
                f"{digest}  training/checkpoint.pth\n"
                f"{digest}  ./training/checkpoint.pth\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "Duplicate normalized detector manifest path"
            ):
                _detector_manifest_entries(root, _sha256(manifest))

    def test_final_report_requires_all_five_candidates_from_one_access_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))

            def task_metrics():
                classification = {
                    "balanced_accuracy": 0.9,
                    "macro_f1": 0.9,
                    "mcc": 0.8,
                    "false_positive_rate": 0.1,
                }
                return {
                    "peak_presence": {
                        "instructions": 2,
                        "valid_json_rate": 1.0,
                        "schema_valid_rate": 1.0,
                        "exact_match_rate": 0.9,
                        "classification": classification,
                    },
                    "peak_presence_metadata": {
                        "instructions": 2,
                        "valid_json_rate": 1.0,
                        "schema_valid_rate": 1.0,
                        "exact_match_rate": 0.9,
                        "classification": classification,
                    },
                    "peak_grounding": {
                        "instructions": 2,
                        "valid_json_rate": 1.0,
                        "schema_valid_rate": 1.0,
                        "exact_match_rate": 0.8,
                        "grounding": {
                            "mean_bbox_iou_all": 0.75,
                            "iou_at_0_5_rate_all": 0.8,
                        },
                    },
                    "scientific_qc": {
                        "instructions": 2,
                        "valid_json_rate": 1.0,
                        "schema_valid_rate": 1.0,
                        "exact_match_rate": 0.95,
                    },
                }

            qwen_roots = {}
            qwen_hashes = {}
            for candidate in (
                "qwen3vl_zero_shot",
                "qwen3vl_image_lora",
                "qwen3vl_image_xic_fusion",
            ):
                root = fixture.root / candidate
                root.mkdir()
                report_path = root / "qwen_evaluation_report.json"
                _write_json(
                    report_path,
                    {
                        "schema_version": "chrompeak-qwen3vl-final-evaluation-v1",
                        "candidate_name": candidate,
                        "inputs": {
                            "protocol_sha256": fixture.protocol_sha,
                            "access_id": "b" * 24,
                        },
                        "counts": {"predictions": 16},
                        "metrics": task_metrics(),
                        "metrics_by_language": {
                            "en": task_metrics(),
                            "zh-CN": task_metrics(),
                        },
                        "evaluation": {
                            "bootstrap_iterations": 10_000,
                            "seed": 17,
                            "bootstrap_unit": "complete_source_mzml_group",
                        },
                        "development_comparison_eligible": False,
                        "final_benchmark_eligible": True,
                        "internal_test_accessed": True,
                    },
                )
                _write_manifest(root, [report_path.name])
                qwen_roots[candidate] = root
                qwen_hashes[candidate] = _sha256(report_path)

            sequence_root = fixture.root / "sequence-evaluation"
            sequence_root.mkdir()
            sequence_report = sequence_root / "sequence_evaluation_report.json"
            _write_json(
                sequence_report,
                {
                    "schema_version": "chrompeak-sequence-final-evaluation-v1",
                    "candidate_name": "sequence_peak_net",
                    "classification": {
                        "validation_frozen_threshold": {
                            "balanced_accuracy": 0.95,
                            "macro_f1": 0.95,
                            "mcc": 0.9,
                            "false_positive_rate": 0.05,
                        }
                    },
                    "localization": {"mean_iou": 0.8, "iou_at_0_5_rate": 0.9},
                    "provenance": {
                        "protocol_sha256": fixture.protocol_sha,
                        "access_id": "b" * 24,
                    },
                    "evaluation": {
                        "bootstrap_iterations": 10_000,
                        "seed": 17,
                        "bootstrap_unit": "complete_source_mzml_group",
                    },
                    "development_comparison_eligible": False,
                    "final_benchmark_eligible": True,
                    "internal_test_accessed": True,
                },
            )
            _write_manifest(sequence_root, [sequence_report.name])

            detector_root = fixture.root / "detector-evaluation"
            detector_root.mkdir()
            detector_report = detector_root / "detector_evaluation_report.json"
            _write_json(
                detector_report,
                {
                    "schema_version": "chrompeak-detector-final-evaluation-v1",
                    "candidate_name": "chrompeakformer",
                    "classification": {
                        "validation_frozen_threshold": {
                            "balanced_accuracy": 0.85,
                            "macro_f1": 0.84,
                            "mcc": 0.7,
                            "false_positive_rate": 0.2,
                        }
                    },
                    "localization": {
                        "validation_frozen_threshold": {
                            "mean_best_iou": 0.78,
                            "iou_at_0_5_rate": 0.9,
                        }
                    },
                    "coco": {"ap_50_95": 0.5, "ap_50": 0.8, "ap_75": 0.6},
                    "provenance": {
                        "protocol_sha256": fixture.protocol_sha,
                        "access_id": "b" * 24,
                    },
                    "evaluation": {
                        "bootstrap_iterations": 10_000,
                        "seed": 17,
                        "bootstrap_unit": "complete_source_mzml_group",
                    },
                    "development_comparison_eligible": False,
                    "final_benchmark_eligible": True,
                    "internal_test_accessed": True,
                },
            )
            _write_manifest(detector_root, [detector_report.name])

            result = build_final_benchmark_report(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha,
                ledger_dir=fixture.ledger,
                qwen_evaluation_roots=qwen_roots,
                qwen_evaluation_report_sha256=qwen_hashes,
                sequence_evaluation_root=sequence_root,
                expected_sequence_report_sha256=_sha256(sequence_report),
                detector_evaluation_root=detector_root,
                expected_detector_report_sha256=_sha256(detector_report),
                output_dir=fixture.root / "final-report",
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], FINAL_BENCHMARK_REPORT_SCHEMA)
            self.assertEqual(
                report["benchmark_scope"], "frozen_multimodal_model_candidates"
            )
            self.assertEqual(result.candidates, 5)
            self.assertEqual(len(report["language_separated_rows"]), 8)
            self.assertTrue(report["contracts"]["same_one_time_access_event"])
            self.assertFalse(
                report["contracts"]["candidate_selection_performed_after_test"]
            )
            self.assertFalse(
                report["contracts"]["biocoder_agent_promotion_claimed"]
            )
            self.assertIn(
                "Qwen3-VL image + aligned XIC",
                result.table_path.read_text(encoding="utf-8"),
            )

            drift_candidate = "qwen3vl_zero_shot"
            drift_report = qwen_roots[drift_candidate] / "qwen_evaluation_report.json"
            drift_payload = json.loads(drift_report.read_text(encoding="utf-8"))
            drift_payload["evaluation"]["bootstrap_iterations"] = 9_999
            _write_json(drift_report, drift_payload)
            _write_manifest(qwen_roots[drift_candidate], [drift_report.name])
            qwen_hashes[drift_candidate] = _sha256(drift_report)
            with self.assertRaisesRegex(ValueError, "uncertainty drift"):
                build_final_benchmark_report(
                    protocol_root=fixture.protocol_root,
                    expected_protocol_sha256=fixture.protocol_sha,
                    ledger_dir=fixture.ledger,
                    qwen_evaluation_roots=qwen_roots,
                    qwen_evaluation_report_sha256=qwen_hashes,
                    sequence_evaluation_root=sequence_root,
                    expected_sequence_report_sha256=_sha256(sequence_report),
                    detector_evaluation_root=detector_root,
                    expected_detector_report_sha256=_sha256(detector_report),
                    output_dir=fixture.root / "final-report-drift",
                )

    def test_final_qwen_evaluation_closes_the_separated_answer_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _FinalDataFixture(Path(temporary))
            data = fixture.build()
            answers = {
                row["instruction_id"]: row["expected_response"]
                for row in (
                    json.loads(line)
                    for line in (data.answer_root / "answers.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }

            class Generator:
                metadata = {
                    "backend": "transformers",
                    "resolved_model_revision": "d" * 40,
                }

                def generate(self, requests, settings):
                    return [answers[request.instruction_id] for request in requests]

            generation = run_qwen_inference(
                data.inference_root,
                fixture.assets_root,
                fixture.root / "perfect-generation",
                expected_bundle_report_sha256=_sha256(
                    data.inference_root / "inference_bundle_report.json"
                ),
                model_name_or_path="fixture-model",
                model_revision="fixture-revision",
                model_artifact_sha256="c" * 64,
                settings=GenerationSettings(batch_size=1, dtype="bfloat16"),
                generator_factory=lambda *args: Generator(),
                final_benchmark_access=FinalBenchmarkAccessSpec(
                    protocol_root=fixture.protocol_root,
                    protocol_sha256=fixture.protocol_sha,
                    ledger_dir=fixture.ledger,
                    candidate_name="qwen3vl_zero_shot",
                ),
            )
            evaluation = evaluate_final_qwen_predictions(
                protocol_root=fixture.protocol_root,
                expected_protocol_sha256=fixture.protocol_sha,
                ledger_dir=fixture.ledger,
                candidate_name="qwen3vl_zero_shot",
                inference_root=data.inference_root,
                expected_inference_report_sha256=_sha256(
                    data.inference_root / "inference_bundle_report.json"
                ),
                answer_root=data.answer_root,
                expected_answer_report_sha256=_sha256(
                    data.answer_root / "answer_key_report.json"
                ),
                predictions_path=generation.predictions_path,
                generation_report_path=generation.report_path,
                expected_generation_report_sha256=generation.report_sha256,
                output_dir=fixture.root / "perfect-evaluation",
            )
            report = json.loads(evaluation.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], FINAL_QWEN_EVALUATION_SCHEMA)
            self.assertEqual(evaluation.prediction_records, 14)
            self.assertEqual(evaluation.valid_json_records, 14)
            self.assertEqual(evaluation.schema_valid_records, 14)
            self.assertEqual(evaluation.source_groups, 2)
            self.assertTrue(report["final_benchmark_eligible"])
            self.assertTrue(report["internal_test_accessed"])
            self.assertEqual(report["evaluation"]["bootstrap_iterations"], 10_000)
            self.assertTrue(
                report["prediction_generation_provenance_verified"]
            )


if __name__ == "__main__":
    unittest.main()
