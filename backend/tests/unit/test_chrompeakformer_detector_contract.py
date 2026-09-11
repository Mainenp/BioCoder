from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multimodal_science.chrompeakformer.detector_dataset import build_detector_dataset
from multimodal_science.chrompeakformer.detector_evaluation import (
    _box_iou_xywh,
    _official_coco_metrics,
    evaluate_detector_predictions,
)
from multimodal_science.chrompeakformer.detector_training import (
    _training_mean_box_width,
    run_detector_training,
)
from multimodal_science.data.manifest import sha256_file


class ChromPeakDetectorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.index_dir = self.root / "index"
        self.assets.mkdir()
        self.index_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_fixture(self, *, leak: bool = False) -> tuple[Path, Path, Path, Path]:
        records = []
        coco_by_split = {}
        image_id = 100
        for split in ("train", "validation"):
            images = []
            annotations = []
            for row in range(2):
                image_id += 1
                relative = Path("jobs") / split / f"roi-{row}.jpeg"
                image_path = self.assets / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"jpeg-{split}-{row}".encode())
                positive = row == 0
                boxes = []
                if positive:
                    boxes.append(
                        {
                            "annotation_id": image_id + 1000,
                            "category_id": 0,
                            "bbox": [100.0, 0.0, 80.0, 300.0],
                        }
                    )
                    annotations.append(
                        {
                            "id": image_id + 1000,
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": [100.0, 0.0, 80.0, 300.0],
                            "area": 24000.0,
                            "iscrowd": 0,
                        }
                    )
                group = "shared" if leak else f"{split}-group"
                records.append(
                    {
                        "asset_id": f"{split}-{row}",
                        "split": split,
                        "job_id": group,
                        "image": {
                            "id": image_id,
                            "path": relative.as_posix(),
                            "sha256": sha256_file(image_path),
                            "width": 400,
                            "height": 300,
                        },
                        "label": {
                            "peak_label": int(positive),
                            "coco_boxes": boxes,
                        },
                    }
                )
                images.append(
                    {
                        "id": image_id,
                        "file_name": relative.as_posix(),
                        "width": 400,
                        "height": 300,
                    }
                )
            coco_by_split[split] = {
                "info": {"asset_index_sha256": "pending", "partial": False},
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 0, "name": "peak"}],
            }
        index_path = self.index_dir / "asset_index.jsonl"
        index_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        index_sha = sha256_file(index_path)
        report_path = self.index_dir / "asset_index_report.json"
        report_path.write_text(
            json.dumps({"asset_index_sha256": index_sha, "partial": False}), encoding="utf-8"
        )
        coco_paths = []
        for split in ("train", "validation"):
            coco_by_split[split]["info"]["asset_index_sha256"] = index_sha
            path = self.index_dir / f"{split}_coco.json"
            path.write_text(json.dumps(coco_by_split[split]), encoding="utf-8")
            coco_paths.append(path)
        return index_path, report_path, coco_paths[0], coco_paths[1]

    def _build_dataset(self) -> Path:
        index, report, train_coco, validation_coco = self._write_fixture()
        output = self.root / "detector"
        result = build_detector_dataset(
            asset_index_path=index,
            asset_index_report_path=report,
            assets_root=self.assets,
            train_coco_path=train_coco,
            validation_coco_path=validation_coco,
            output_dir=output,
            verify_image_hashes=True,
        )
        self.assertEqual(result.train_assets, 2)
        self.assertEqual(result.validation_assets, 2)
        return output

    def test_prepares_train_val_layout_with_absolute_verified_images(self) -> None:
        output = self._build_dataset()

        train = json.loads(
            (output / "coco" / "train" / "train_coco.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (output / "coco" / "val" / "val_coco.json").read_text(encoding="utf-8")
        )
        report = json.loads((output / "detector_dataset_report.json").read_text(encoding="utf-8"))

        self.assertTrue(Path(train["images"][0]["file_name"]).is_absolute())
        self.assertEqual(len(validation["images"]), 2)
        self.assertTrue(report["leakage_audit"]["passed"])
        self.assertTrue(report["splits"]["train"]["image_hashes_verified"])
        self.assertTrue((output / "artifact_manifest.sha256").is_file())

    def test_rejects_source_group_leakage(self) -> None:
        index, report, train_coco, validation_coco = self._write_fixture(leak=True)

        with self.assertRaisesRegex(ValueError, "leakage"):
            build_detector_dataset(
                asset_index_path=index,
                asset_index_report_path=report,
                assets_root=self.assets,
                train_coco_path=train_coco,
                validation_coco_path=validation_coco,
                output_dir=self.root / "detector",
            )

    def test_unified_evaluation_reports_coco_classification_and_iou(self) -> None:
        output = self._build_dataset()
        validation_coco = output / "coco" / "val" / "val_coco.json"
        validation = json.loads(validation_coco.read_text(encoding="utf-8"))
        positive_id = validation["annotations"][0]["image_id"]
        negative_id = next(
            image["id"] for image in validation["images"] if image["id"] != positive_id
        )
        predictions = [
            {
                "image_id": positive_id,
                "category_id": 0,
                "bbox": [100.0, 0.0, 80.0, 300.0],
                "score": 0.9,
            },
            {
                "image_id": negative_id,
                "category_id": 0,
                "bbox": [20.0, 0.0, 10.0, 300.0],
                "score": 0.1,
            },
        ]
        predictions_path = self.root / "predictions.json"
        predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
        dataset_report = output / "detector_dataset_report.json"
        inference_report = self.root / "inference_report.json"
        inference_report.write_text(
            json.dumps(
                {
                    "model_family": "ChromPeakFormer",
                    "complete_validation_coverage": True,
                    "predictions_sha256": sha256_file(predictions_path),
                    "detector_dataset_report_sha256": sha256_file(dataset_report),
                    "validation_coco_sha256": sha256_file(validation_coco),
                    "counts": {"images": 2, "predictions": 2},
                    "development_comparison_candidate": True,
                }
            ),
            encoding="utf-8",
        )
        evaluation_output = self.root / "evaluation"

        with patch(
            "multimodal_science.chrompeakformer.detector_evaluation._official_coco_metrics",
            return_value={"ap_50_95": 1.0, "ap_50": 1.0, "ap_75": 1.0},
        ):
            evaluate_detector_predictions(
                validation_coco_path=validation_coco,
                predictions_path=predictions_path,
                inference_report_path=inference_report,
                expected_inference_report_sha256=sha256_file(inference_report),
                detector_dataset_report_path=dataset_report,
                expected_dataset_report_sha256=sha256_file(dataset_report),
                output_dir=evaluation_output,
            )
        report = json.loads(
            (evaluation_output / "detector_evaluation_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["coco"]["ap_50"], 1.0)
        self.assertEqual(report["classification"]["fixed_threshold"]["accuracy"], 1.0)
        self.assertEqual(report["localization"]["fixed_threshold"]["mean_best_iou"], 1.0)
        self.assertFalse(report["internal_test_accessed"])
        self.assertTrue(report["development_comparison_eligible"])

    def test_iou_uses_full_box_geometry(self) -> None:
        self.assertEqual(_box_iou_xywh([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]), 1.0)
        self.assertAlmostEqual(
            _box_iou_xywh([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 10.0, 10.0]),
            1.0 / 3.0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools") is not None,
        "pycocotools is an optional detector dependency",
    )
    def test_official_coco_metrics_accept_category_zero_and_perfect_box(self) -> None:
        output = self._build_dataset()
        validation_coco = output / "coco" / "val" / "val_coco.json"
        validation = json.loads(validation_coco.read_text(encoding="utf-8"))
        annotation = validation["annotations"][0]
        predictions = self.root / "official_predictions.json"
        predictions.write_text(
            json.dumps(
                [
                    {
                        "image_id": annotation["image_id"],
                        "category_id": 0,
                        "bbox": annotation["bbox"],
                        "score": 0.99,
                    }
                ]
            ),
            encoding="utf-8",
        )

        metrics = _official_coco_metrics(validation_coco, predictions)

        self.assertAlmostEqual(metrics["ap_50_95"], 1.0)
        self.assertAlmostEqual(metrics["ap_50"], 1.0)

    def test_peak_width_statistic_uses_training_boxes_only(self) -> None:
        output = self._build_dataset()

        value = _training_mean_box_width(output / "coco" / "train" / "train_coco.json")

        self.assertAlmostEqual(value, 0.2)

    def test_training_wrapper_marks_smoke_results_ineligible(self) -> None:
        output = self._build_dataset()
        dataset_report = output / "detector_dataset_report.json"
        source = self.root / "authorized_source"
        source.mkdir()
        (source / "train.py").write_text(
            """\
import argparse
import json
import os
from pathlib import Path

command = argparse.ArgumentParser()
command.add_argument("--config", required=True)
arguments = command.parse_args()
config = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
destination = Path(config["output_dir"])
(destination / "launcher_environment.json").write_text(
    json.dumps(
        {
            key: os.environ[key]
            for key in (
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "SLURM_LOCALID",
                "SLURM_NTASKS",
                "SLURM_PROCID",
                "WORLD_SIZE",
            )
            if key in os.environ
        }
    ),
    encoding="utf-8",
)
(destination / "checkpoint.pth").write_bytes(b"checkpoint")
record = {"epoch": 0, "test_coco_eval_bbox": [1.0] * 12}
(destination / "log.txt").write_text(json.dumps(record) + "\\n", encoding="utf-8")
""",
            encoding="utf-8",
        )
        source_config = source / "config.json"
        source_config.write_text(
            json.dumps({"dec_layers": 3, "num_fdr_bins": 33, "epochs": 30}),
            encoding="utf-8",
        )
        training_output = self.root / "training"

        with patch.dict(
            os.environ,
            {
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "scheduler.example",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NTASKS": "1",
                "SLURM_PROCID": "0",
                "WORLD_SIZE": "1",
            },
        ):
            result = run_detector_training(
                source_root=source,
                source_config_path=source_config,
                detector_dataset_root=output / "coco",
                detector_dataset_report_path=dataset_report,
                expected_dataset_report_sha256=sha256_file(dataset_report),
                output_dir=training_output,
                device="cpu",
                epochs=1,
                smoke_test=True,
            )
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        contract = json.loads(
            (training_output / "training_contract.json").read_text(encoding="utf-8")
        )
        launcher_environment = json.loads(
            (training_output / "launcher_environment.json").read_text(encoding="utf-8")
        )

        self.assertTrue(report["smoke_test"])
        self.assertEqual(report["run_scope"], "smoke")
        self.assertEqual(report["execution_mode"], "single_process")
        self.assertFalse(report["development_comparison_eligible"])
        self.assertEqual(launcher_environment, {})
        self.assertEqual(
            contract["distributed_launcher_environment_removed"],
            [
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "SLURM_LOCALID",
                "SLURM_NTASKS",
                "SLURM_PROCID",
                "WORLD_SIZE",
            ],
        )
        self.assertTrue((training_output / "artifact_manifest.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
