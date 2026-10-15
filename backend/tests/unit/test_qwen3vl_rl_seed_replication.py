import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_seed_replication import (
    analyze_seed_replication,
    build_seed_replication_candidates,
)


REVISION = "a" * 40
POOL_SHA = "b" * 64
C0_REPORT_SHA = "c" * 64
C0_MANIFEST_SHA = "d" * 64
BUNDLE_SHA = "e" * 64


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(root: Path, paths: list[Path]) -> Path:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return manifest


def _training_run(root: Path, family: str, *, seed: int = 29) -> Path:
    root.mkdir(parents=True)
    report_name = (
        "fusion_training_report.json"
        if family == "R2"
        else "sft_continue_training_report.json"
    )
    report = root / report_name
    config = root / "training_config.json"
    settings = {"seed": seed, "run_mode": "formal"}
    if family == "R2":
        settings.update(
            {"experiment": "R2", "train_projector": False, "lora_learning_rate": 5e-6}
        )
    else:
        settings["learning_rate_choice"] = "1e-6"
    _json(config, {"settings": settings})
    _json(
        report,
        {
            "training": (
                {"experiment": "R2", "lora_learning_rate": 5e-6}
                if family == "R2"
                else {}
            ),
            "sources": {
                "rl_pool_report_sha256": POOL_SHA,
                "starting_adapter_report_sha256": C0_REPORT_SHA,
                "starting_adapter_manifest_sha256": C0_MANIFEST_SHA,
            },
            "contracts": {
                "selection_groups_trained": False,
                "validation_prompts_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
                "projector_trainable": False,
            },
            "development_training_complete": True,
        },
    )
    _manifest(root, [report, config])
    checkpoint = root / "checkpoints" / "step-00000300"
    checkpoint.mkdir(parents=True)
    state = checkpoint / "checkpoint.json"
    _json(state, {"global_step": 300})
    _manifest(checkpoint, [state])
    return root


def _evaluation(
    root: Path,
    *,
    family: str,
    seed: int,
    mean_iou: float,
    bundle_sha: str = BUNDLE_SHA,
    validation_answers_opened: bool = False,
) -> Path:
    root.mkdir(parents=True)
    config = "lora-lr-5e-6-projector-frozen" if family == "R2" else "lora-lr-1e-6"
    if seed == 29:
        config += "-seed29"
    report = root / "selection_evaluation_report.json"
    _json(
        report,
        {
            "schema_version": "chrompeak-qwen3vl-rl-selection-evaluation-v1",
            "candidate": {
                "family": family,
                "config": config,
                "checkpoint": {"step": 300},
            },
            "sources": {"selection_bundle_report_sha256": bundle_sha},
            "metrics": {
                "grounding_mean_iou": mean_iou,
                "json_valid_rate": 1.0,
                "peak_presence_macro_f1": 0.95,
            },
            "contracts": {
                "selection_groups_only": True,
                "validation_answers_opened": validation_answers_opened,
                "internal_test_accessed": False,
            },
        },
    )
    _manifest(root, [report])
    return report


class SeedReplicationTests(unittest.TestCase):
    def test_builds_exact_seed29_r2_and_c1_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_seed_replication_candidates(
                code_revision=REVISION,
                rl_pool_report_sha256=POOL_SHA,
                c0_report_sha256=C0_REPORT_SHA,
                c0_manifest_sha256=C0_MANIFEST_SHA,
                r2_root=_training_run(root / "r2", "R2"),
                c1_root=_training_run(root / "c1", "C1"),
                output_dir=root / "candidates",
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"candidates": 2, "checkpoint_step": 300, "seed": 29})
            self.assertTrue(report["contracts"]["selection_set_evaluation_only"])
            self.assertFalse(report["contracts"]["validation_answers_opened"])
            rows = [
                json.loads(line)
                for line in (result.output_dir / "candidate_list.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual({row["family"] for row in rows}, {"R2", "C1"})
            self.assertTrue(all(row["seed"] == 29 for row in rows))

    def test_candidate_builder_rejects_wrong_training_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Seed drift"):
                build_seed_replication_candidates(
                    code_revision=REVISION,
                    rl_pool_report_sha256=POOL_SHA,
                    c0_report_sha256=C0_REPORT_SHA,
                    c0_manifest_sha256=C0_MANIFEST_SHA,
                    r2_root=_training_run(root / "r2", "R2", seed=17),
                    c1_root=_training_run(root / "c1", "C1"),
                    output_dir=root / "candidates",
                )

    def test_analysis_is_selection_only_and_reports_paired_seed_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = analyze_seed_replication(
                seed17_r2_evaluation=_evaluation(root / "r2-17", family="R2", seed=17, mean_iou=0.75),
                seed17_c1_evaluation=_evaluation(root / "c1-17", family="C1", seed=17, mean_iou=0.72),
                seed29_r2_evaluation=_evaluation(root / "r2-29", family="R2", seed=29, mean_iou=0.74),
                seed29_c1_evaluation=_evaluation(root / "c1-29", family="C1", seed=29, mean_iou=0.71),
                output_dir=root / "analysis",
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["per_seed"]["17"]["r2_minus_c1"]["grounding_mean_iou"], 0.03)
            self.assertAlmostEqual(report["per_seed"]["29"]["r2_minus_c1"]["grounding_mean_iou"], 0.03)
            self.assertAlmostEqual(report["across_seed_summary"]["r2"]["grounding_mean_iou"]["sample_standard_deviation"], 0.007071067811865481)
            self.assertFalse(report["interpretation"]["inferential_claim_allowed"])
            self.assertFalse(report["contracts"]["validation_answers_opened"])
            self.assertFalse(report["contracts"]["internal_test_accessed"])

    def test_analysis_rejects_validation_or_selection_set_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "seed17_r2_evaluation": _evaluation(root / "r2-17", family="R2", seed=17, mean_iou=0.75),
                "seed17_c1_evaluation": _evaluation(root / "c1-17", family="C1", seed=17, mean_iou=0.72),
                "seed29_r2_evaluation": _evaluation(root / "r2-29", family="R2", seed=29, mean_iou=0.74, bundle_sha="f" * 64),
                "seed29_c1_evaluation": _evaluation(root / "c1-29", family="C1", seed=29, mean_iou=0.71),
            }
            with self.assertRaisesRegex(ValueError, "different selection sets"):
                analyze_seed_replication(**common, output_dir=root / "analysis")

            protected = root / "protected"
            with self.assertRaisesRegex(ValueError, "protection contract failed"):
                analyze_seed_replication(
                    seed17_r2_evaluation=_evaluation(protected / "r2-17", family="R2", seed=17, mean_iou=0.75, validation_answers_opened=True),
                    seed17_c1_evaluation=common["seed17_c1_evaluation"],
                    seed29_r2_evaluation=_evaluation(protected / "r2-29", family="R2", seed=29, mean_iou=0.74),
                    seed29_c1_evaluation=common["seed29_c1_evaluation"],
                    output_dir=protected / "analysis",
                )


if __name__ == "__main__":
    unittest.main()
