import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_science.qwen3vl.rl_grpo_training import (
    GRPOSettings,
    _repeat_cache,
    _select_cache,
    configure_trainable_parameters,
    kl_k3_values,
    normalize_group_advantages,
    numpy_causal_answer_log_probs,
    prompt_batch,
    starting_processor_pixel_bounds,
    validate_settings,
)
from multimodal_science.qwen3vl.rl_rollout import verify_train_signals
from multimodal_science.qwen3vl.train_rl_cli import parser


class _Parameter:
    def __init__(self, value: float) -> None:
        self.value = value
        self.requires_grad = True

    def requires_grad_(self, enabled: bool):
        self.requires_grad = enabled
        return self


class _Cache:
    def __init__(self, rows: list[int]) -> None:
        self.rows = rows

    def batch_repeat_interleave(self, repeats: int):
        self.rows = [value for value in self.rows for _ in range(repeats)]

    def batch_select_indices(self, indices):
        self.rows = [self.rows[int(index)] for index in indices]


def _record(index: int, task: str) -> dict:
    return {
        "instruction_id": f"instruction-{index}",
        "task": task,
        "group_id": f"group-{index}",
    }


class GRPOTrainingTests(unittest.TestCase):
    def _train_signal_view(
        self, root: Path, *, internal_test_accessed: bool | None
    ) -> tuple[Path, str]:
        signals = root / "train_signals.npy"
        np.save(signals, np.zeros((2, 160), dtype=np.float32), allow_pickle=False)
        report = {
            "schema_version": "chrompeak-multimodal-dataset-v1",
            "splits": ["train", "validation"],
            "target_points": 160,
            "artifacts": {
                "train_signals": {
                    "path": signals.name,
                    "sha256": hashlib.sha256(signals.read_bytes()).hexdigest(),
                    "shape": [2, 160],
                }
            },
        }
        if internal_test_accessed is not None:
            report["internal_test_accessed"] = internal_test_accessed
        report_path = root / "dataset_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report_path, hashlib.sha256(report_path.read_bytes()).hexdigest()

    def test_train_signal_view_accepts_legacy_train_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, report_sha256 = self._train_signal_view(
                root, internal_test_accessed=None
            )
            verified = verify_train_signals(root, report_sha256)
            self.assertEqual(verified.shape, (2, 160))

    def test_train_signal_view_rejects_explicit_internal_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, report_sha256 = self._train_signal_view(
                root, internal_test_accessed=True
            )
            with self.assertRaisesRegex(ValueError, "used internal test"):
                verify_train_signals(root, report_sha256)

    def test_legacy_train_signal_view_rejects_internal_test_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path, _ = self._train_signal_view(
                root, internal_test_accessed=None
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifacts"]["internal_test_signals"] = {
                "path": "not-opened.npy",
                "sha256": "0" * 64,
                "shape": [1, 160],
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "references internal test"):
                verify_train_signals(root, report_sha256)

    def test_group_advantages_skip_zero_std_and_center_nonzero_groups(self) -> None:
        self.assertIsNone(normalize_group_advantages([0.5] * 8))
        advantages = normalize_group_advantages([0.0, 1.0] * 4)
        self.assertIsNotNone(advantages)
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0, places=12)

    def test_full_forward_answer_log_probs_match_incremental_mock(self) -> None:
        logits = np.array(
            [
                [
                    [0.2, 0.1, -0.4],
                    [-0.5, 0.3, 1.1],
                    [0.7, 1.2, -0.8],
                    [0.1, 0.1, 0.1],
                ]
            ],
            dtype=np.float64,
        )
        labels = np.array([[-100, -100, 2, 1]], dtype=np.int64)
        full, mask = numpy_causal_answer_log_probs(logits, labels)

        def log_probability(row: np.ndarray, target: int) -> float:
            shifted = row - row.max()
            return float(shifted[target] - np.log(np.exp(shifted).sum()))

        incremental = np.array(
            [log_probability(logits[0, 1], 2), log_probability(logits[0, 2], 1)]
        )
        np.testing.assert_allclose(full, incremental, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(mask, np.array([[False, True, True]]))

    def test_k3_is_zero_for_identical_policy_and_reference(self) -> None:
        identical = kl_k3_values([-0.1, -1.2, -3.0], [-0.1, -1.2, -3.0])
        self.assertEqual(identical, (0.0, 0.0, 0.0))
        shifted = kl_k3_values([-0.4], [-0.2])
        self.assertGreater(shifted[0], 0.0)

    def test_parameter_scope_keeps_ref_and_base_frozen(self) -> None:
        model = [
            ("base.weight", _Parameter(1.0)),
            ("layer.lora_A.policy.weight", _Parameter(2.0)),
            ("layer.lora_B.policy.weight", _Parameter(3.0)),
            ("layer.lora_A.ref.weight", _Parameter(2.0)),
            ("layer.lora_B.ref.weight", _Parameter(3.0)),
        ]
        projector = [("gate_logit", _Parameter(-4.0))]
        groups = configure_trainable_parameters(model, projector, experiment="R1")
        by_name = dict(model)
        self.assertTrue(by_name["layer.lora_A.policy.weight"].requires_grad)
        self.assertFalse(by_name["layer.lora_A.ref.weight"].requires_grad)
        self.assertFalse(by_name["base.weight"].requires_grad)
        self.assertFalse(projector[0][1].requires_grad)
        for name in groups["policy"]:
            by_name[name].value += 0.25
        self.assertEqual(by_name["layer.lora_A.ref.weight"].value, 2.0)
        self.assertEqual(by_name["base.weight"].value, 1.0)

        configure_trainable_parameters(model, projector, experiment="R3")
        self.assertTrue(projector[0][1].requires_grad)
        self.assertFalse(by_name["layer.lora_A.ref.weight"].requires_grad)

    def test_parallel_cache_is_repeated_then_compacted_without_padding(self) -> None:
        cache = _repeat_cache(_Cache([7]), 8)
        self.assertEqual(cache.rows, [7] * 8)
        cache = _select_cache(cache, [0, 2, 5])
        self.assertEqual(cache.rows, [7, 7, 7])

    def test_prompt_batch_uses_preregistered_task_mix(self) -> None:
        records = [_record(index, "peak_grounding") for index in range(40)]
        for task_index, task in enumerate(
            ("peak_presence", "peak_presence_metadata", "scientific_qc")
        ):
            records.extend(_record(100 + task_index * 10 + index, task) for index in range(6))
        first = prompt_batch(records, step=3, prompts_per_step=16, seed=17)
        second = prompt_batch(records, step=3, prompts_per_step=16, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["task"] == "peak_grounding" for row in first), 13)
        reduced = prompt_batch(records, step=3, prompts_per_step=8, seed=17)
        self.assertEqual(sum(row["task"] == "peak_grounding" for row in reduced), 6)

    def test_only_preregistered_budgets_and_learning_rates_are_exposed(self) -> None:
        expected = {
            "smoke": (16, 2),
            "calibration": (16, 2),
            "formal": (16, 300),
            "reduced-prompts": (8, 300),
            "reduced-steps": (8, 200),
        }
        for mode, budget in expected.items():
            settings = GRPOSettings(experiment="R1", run_mode=mode)
            validate_settings(settings)
            self.assertEqual((settings.prompts_per_step, settings.total_steps), budget)
            self.assertEqual(settings.lora_learning_rate, 1e-6)
        self.assertEqual(
            GRPOSettings(experiment="R2", run_mode="formal").lora_learning_rate,
            5e-6,
        )
        self.assertEqual(
            GRPOSettings(experiment="R3", run_mode="formal").projector_learning_rate,
            1e-5,
        )

    def test_processor_pixel_bounds_are_inherited_and_validated(self) -> None:
        report = {"training": {"min_pixels": 3136, "max_pixels": 1003520}}
        self.assertEqual(starting_processor_pixel_bounds(report), (3136, 1003520))
        with self.assertRaisesRegex(ValueError, "pixel bounds"):
            starting_processor_pixel_bounds(
                {"training": {"min_pixels": 100, "max_pixels": 99}}
            )

    def test_training_cli_has_no_validation_or_internal_test_surface(self) -> None:
        destinations = {action.dest for action in parser()._actions}
        self.assertFalse(any("validation" in name for name in destinations))
        self.assertFalse(any("internal" in name or "test" in name for name in destinations))
        with self.assertRaisesRegex(ValueError, "Protected input path"):
            from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths

            assert_no_protected_input_paths([Path("answers/internal_test/key.json")])


if __name__ == "__main__":
    unittest.main()
