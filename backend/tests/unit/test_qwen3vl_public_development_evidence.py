from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.auxiliary_pretraining import (
    AUXILIARY_PRETRAINING_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.development_dossier import REPORT_SCHEMA
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_training import FUSION_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.lora_training import LORA_TRAINING_REPORT_SCHEMA
from multimodal_science.qwen3vl.public_development_evidence import (
    PUBLIC_DEVELOPMENT_EVIDENCE_SCHEMA,
    build_public_development_evidence,
)


METRICS = (
    ("peak_presence", "balanced_accuracy"),
    ("peak_presence", "macro_f1"),
    ("peak_presence", "mcc"),
    ("peak_presence", "false_positive_rate"),
    ("peak_presence_metadata", "macro_f1"),
    ("peak_presence_metadata", "mcc"),
    ("peak_grounding", "mean_bbox_iou_all"),
    ("peak_grounding", "iou_at_0_5_rate_all"),
    ("scientific_qc", "exact_match_rate"),
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _root(parent: Path, name: str, report_name: str, payload: dict) -> Path:
    root = parent / name
    root.mkdir()
    report = root / report_name
    _write_json(report, payload)
    (root / "artifact_manifest.sha256").write_text(
        f"{sha256_file(report)}  {report.name}\n", encoding="utf-8"
    )
    return root


def _tasks(offset: float = 0.0) -> dict:
    result: dict[str, dict[str, float]] = {}
    for index, (task, metric) in enumerate(METRICS, start=1):
        result.setdefault(task, {})[metric] = 0.5 + offset + index / 100.0
    return result


def _statistics(value: float) -> dict:
    return {
        "values_by_seed": {"17": value - 0.01, "29": value, "43": value + 0.01},
        "n": 3,
        "mean": value,
        "sample_standard_deviation": 0.01,
        "standard_deviation_ddof": 1,
        "minimum": value - 0.01,
        "maximum": value + 0.01,
    }


def _dossier() -> dict:
    aggregate = {}
    for scope_offset, scope in enumerate(("overall", "en", "zh-CN")):
        tasks: dict[str, dict[str, dict]] = {}
        for index, (task, metric) in enumerate(METRICS, start=1):
            tasks.setdefault(task, {})[metric] = _statistics(
                0.5 + scope_offset / 100.0 + index / 100.0
            )
        aggregate[scope] = tasks
    token_runs = {}
    for token, offset in (("1", -0.02), ("4", 0.0), ("8", 0.01)):
        token_runs[token] = {
            "sensor_tokens": int(token),
            "training_seed": 17,
            "metrics": {"overall": _tasks(offset)},
            "sensor_gate": {"final_probability": 0.018 + offset / 10.0},
        }
    interventions = {
        name: _tasks(offset)
        for name, offset in (
            ("aligned", 0.1),
            ("shuffled", -0.1),
            ("zero", -0.05),
            ("availability-off", -0.08),
        )
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "fusion_reproducibility": aggregate,
        "sensor_token_ablation": {"runs": token_runs},
        "causal_xic_intervention_evidence": {
            "selected_checkpoint_match": True,
            "observed_metrics": {"overall": interventions},
        },
        "pre_internal_test_readiness": {"ready": True},
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }


def _training(schema: str, seconds: float) -> dict:
    return {
        "schema_version": schema,
        "wall_time_seconds": seconds,
        "training": {
            "max_steps": None,
            "training_records": 54335,
            "optimizer_updates": 3396,
            "effective_batch_size": 16,
            "resumed_from": None,
        },
        "development_training_complete": True,
        "development_comparison_eligible": False,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }


def _evaluation(offset: float) -> dict:
    return {
        "schema_version": BILINGUAL_EVALUATION_REPORT_SCHEMA,
        "metrics": _tasks(offset),
        "metrics_by_language": {
            "en": _tasks(offset - 0.01),
            "zh-CN": _tasks(offset + 0.01),
        },
        "counts": {"predictions": 13708},
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }


class PublicDevelopmentEvidenceTests(unittest.TestCase):
    def _fixture(self, parent: Path) -> dict:
        return {
            "development_dossier_root": _root(
                parent, "dossier", "development_dossier.json", _dossier()
            ),
            "lora_training_root": _root(
                parent,
                "lora",
                "lora_training_report.json",
                _training(LORA_TRAINING_REPORT_SCHEMA, 5890.844),
            ),
            "fusion_training_root": _root(
                parent,
                "fusion",
                "fusion_training_report.json",
                _training(FUSION_TRAINING_REPORT_SCHEMA, 22209.544),
            ),
            "auxiliary_pretraining_root": _root(
                parent,
                "auxiliary-pretraining",
                "auxiliary_pretraining_report.json",
                {
                    "schema_version": AUXILIARY_PRETRAINING_REPORT_SCHEMA,
                    "wall_time_seconds": 12.5,
                    "development_training_complete": True,
                    "development_comparison_eligible": False,
                    "final_benchmark_eligible": False,
                    "internal_test_accessed": False,
                },
            ),
            "random_projector_evaluation_root": _root(
                parent,
                "random-evaluation",
                "qwen_evaluation_report.json",
                _evaluation(0.1),
            ),
            "auxiliary_projector_evaluation_root": _root(
                parent,
                "auxiliary-evaluation",
                "qwen_evaluation_report.json",
                _evaluation(0.05),
            ),
            "xic_only_evaluation_root": _root(
                parent,
                "xic-only-evaluation",
                "qwen_evaluation_report.json",
                _evaluation(0.02),
            ),
            "sequence_prompt_evaluation_root": _root(
                parent,
                "sequence-prompt-evaluation",
                "qwen_evaluation_report.json",
                _evaluation(0.03),
            ),
        }

    def test_exports_public_tables_without_machine_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._fixture(root)
            result = build_public_development_evidence(
                **inputs, output_dir=root / "public"
            )
            report_text = result.report_path.read_text(encoding="utf-8")
            report = json.loads(report_text)
            self.assertEqual(report["schema_version"], PUBLIC_DEVELOPMENT_EVIDENCE_SCHEMA)
            self.assertEqual(
                report["three_seed_statistics"]["overall"]["Presence Macro-F1"][
                    "sample_standard_deviation"
                ],
                0.01,
            )
            self.assertEqual(set(report["sensor_token_ablation"]), {"1", "4", "8"})
            self.assertIn("availability-off", report["xic_interventions"])
            self.assertEqual(
                set(report["post_seal_controls"]),
                {"xic_only_qwen", "image_lora_sequence_prompt"},
            )
            self.assertEqual(
                report["post_seal_controls"]["xic_only_qwen"]["prediction_records"],
                13708,
            )
            self.assertEqual(
                report["training_wall_clock"]["image_lora"]["wall_time_seconds"],
                5890.844,
            )
            self.assertFalse(report["internal_test_accessed"])
            self.assertNotIn(str(root), report_text)
            markdown = result.markdown_path.read_text(encoding="utf-8")
            self.assertIn("Mean ± SD", markdown)
            self.assertIn("availability-off", markdown)
            self.assertIn("06:10:10", markdown)
            self.assertIn("did not improve", markdown)
            self.assertIn("XIC-only Qwen", markdown)
            self.assertIn("Image LoRA + SequencePeakNet prompt", markdown)
            for line in result.manifest_path.read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", maxsplit=1)
                self.assertEqual(sha256_file(result.output_dir / name), digest)

    def test_rejects_step_capped_training_and_partial_auxiliary_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._fixture(root)
            lora_report = inputs["lora_training_root"] / "lora_training_report.json"
            payload = json.loads(lora_report.read_text(encoding="utf-8"))
            payload["training"]["max_steps"] = 4
            _write_json(lora_report, payload)
            (inputs["lora_training_root"] / "artifact_manifest.sha256").write_text(
                f"{sha256_file(lora_report)}  {lora_report.name}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "step-capped"):
                build_public_development_evidence(
                    **inputs, output_dir=root / "public"
                )

            second = root / "second"
            second.mkdir()
            inputs = self._fixture(second)
            inputs["auxiliary_projector_evaluation_root"] = None
            with self.assertRaisesRegex(ValueError, "supplied together"):
                build_public_development_evidence(
                    **inputs, output_dir=root / "public-second"
                )

            third = root / "third"
            third.mkdir()
            inputs = self._fixture(third)
            inputs["sequence_prompt_evaluation_root"] = None
            with self.assertRaisesRegex(ValueError, "supplied together"):
                build_public_development_evidence(
                    **inputs, output_dir=root / "public-third"
                )

    def test_slurm_publisher_is_cpu_only_and_never_opens_internal_test(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "multimodal_science/qwen3vl/slurm/coder_publish_development_evidence.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=coder", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertNotIn("--gres", script)
        self.assertIn("build_public_development_evidence_cli", script)
        self.assertIn("sha256sum -c artifact_manifest.sha256", script)
        self.assertIn("SEALED_INTERNAL_TEST_REOPENED=false", script)
        self.assertNotIn("BIOCODER_INTERNAL_TEST", script)
        self.assertIn("PUBLIC_DEVELOPMENT_EVIDENCE=OK", script)
        self.assertIn("BIOCODER_XIC_ONLY_EVALUATION_ROOT", script)
        self.assertIn("BIOCODER_SEQUENCE_PROMPT_EVALUATION_ROOT", script)


if __name__ == "__main__":
    unittest.main()
