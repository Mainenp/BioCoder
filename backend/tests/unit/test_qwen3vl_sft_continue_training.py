import unittest
from pathlib import Path

from multimodal_science.qwen3vl.rl_data import assert_training_pool_isolation
from multimodal_science.qwen3vl.sft_continue_training import (
    C1Settings,
    configure_c1_trainable_parameters,
    resolve_lora_learning_rate,
    validate_settings,
)
from multimodal_science.qwen3vl.train_sft_continue_cli import parser


class _Parameter:
    def __init__(self, value: float) -> None:
        self.value = value
        self.requires_grad = True

    def requires_grad_(self, enabled: bool):
        self.requires_grad = enabled
        return self


def _record(index: int, task: str) -> dict:
    return {
        "instruction_id": f"instruction-{index}",
        "task": task,
        "group_id": f"group-{index}",
    }


class ContinueSFTTests(unittest.TestCase):
    def test_budgets_exactly_mirror_preregistered_grpo_reductions(self) -> None:
        expected = {
            "smoke": (16, 2),
            "calibration": (16, 2),
            "formal": (16, 300),
            "reduced-prompts": (8, 300),
            "reduced-steps": (8, 200),
        }
        for mode, budget in expected.items():
            settings = C1Settings(run_mode=mode, learning_rate_choice="original")
            validate_settings(settings)
            self.assertEqual((settings.prompts_per_step, settings.total_steps), budget)
        formal = C1Settings(run_mode="formal", learning_rate_choice="1e-6")
        self.assertEqual(formal.prompts_per_step * formal.total_steps, 4800)

    def test_learning_rate_is_limited_to_c0_original_or_one_e_minus_six(self) -> None:
        self.assertEqual(resolve_lora_learning_rate("original", 2e-7), 2e-7)
        self.assertEqual(resolve_lora_learning_rate("1e-6", 2e-7), 1e-6)
        with self.assertRaisesRegex(ValueError, "Unknown C1"):
            resolve_lora_learning_rate("5e-6", 2e-7)
        with self.assertRaisesRegex(ValueError, "valid LoRA"):
            resolve_lora_learning_rate("original", None)

    def test_sampling_uses_same_deterministic_eighty_twenty_batch(self) -> None:
        from multimodal_science.qwen3vl.rl_grpo_training import prompt_batch

        records = [_record(index, "peak_grounding") for index in range(40)]
        for offset, task in enumerate(
            ("peak_presence", "peak_presence_metadata", "scientific_qc")
        ):
            records.extend(_record(100 + offset * 10 + index, task) for index in range(6))
        first = prompt_batch(records, step=11, prompts_per_step=16, seed=17)
        second = prompt_batch(records, step=11, prompts_per_step=16, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["task"] == "peak_grounding" for row in first), 13)
        self.assertEqual(len(first), 16)

    def test_selection_and_protected_paths_are_rejected(self) -> None:
        rows = [_record(1, "peak_grounding")]
        rows[0]["group_id"] = "selection-group"
        with self.assertRaisesRegex(ValueError, "Selection-set"):
            assert_training_pool_isolation(rows, {"selection-group"})
        from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths

        with self.assertRaisesRegex(ValueError, "Protected input path"):
            assert_no_protected_input_paths([Path("answers/validation_answers/key.json")])
        with self.assertRaisesRegex(ValueError, "Protected input path"):
            assert_no_protected_input_paths([Path("answers/internal_test/key.json")])

    def test_parameter_scope_trains_only_policy_lora(self) -> None:
        model = [
            ("base.weight", _Parameter(1.0)),
            ("layer.lora_A.policy.weight", _Parameter(2.0)),
            ("layer.lora_B.policy.weight", _Parameter(3.0)),
            ("layer.lora_A.ref.weight", _Parameter(4.0)),
        ]
        projector = [("gate_logit", _Parameter(-4.0))]
        groups = configure_c1_trainable_parameters(model, projector)
        by_name = dict(model)
        self.assertTrue(by_name["layer.lora_A.policy.weight"].requires_grad)
        self.assertFalse(by_name["layer.lora_A.ref.weight"].requires_grad)
        self.assertFalse(by_name["base.weight"].requires_grad)
        self.assertFalse(projector[0][1].requires_grad)
        self.assertEqual(groups["projector"], ("gate_logit",))

    def test_cli_has_no_validation_internal_or_arbitrary_lr_surface(self) -> None:
        command = parser()
        destinations = {action.dest for action in command._actions}
        self.assertFalse(any("validation" in name for name in destinations))
        self.assertFalse(any("internal" in name or "test" in name for name in destinations))
        self.assertIn("learning_rate_choice", destinations)
        choices = next(
            action.choices for action in command._actions if action.dest == "learning_rate_choice"
        )
        self.assertEqual(tuple(choices), ("original", "1e-6"))


if __name__ == "__main__":
    unittest.main()
