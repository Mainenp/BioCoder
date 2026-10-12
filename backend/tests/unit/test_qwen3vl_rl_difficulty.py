import unittest
from pathlib import Path

import numpy as np

from multimodal_science.qwen3vl.rl_difficulty import (
    GROUP_SIZE,
    _difficulty_work_dir,
    select_difficulty_records,
)
from multimodal_science.qwen3vl.rl_rollout import (
    increment_mrope_position,
    sample_seed,
)


def _record(index: int, task: str, language: str = "en") -> dict:
    return {
        "instruction_id": f"instruction-{index}",
        "group_id": f"group-{index}",
        "task": task,
        "language": language,
        "target_peak_present": index % 2 == 0,
    }


def _rollout(record: dict, rewards: list[float], exact: list[bool]) -> dict:
    from multimodal_science.qwen3vl.rl_difficulty import _canonical_sha256

    return {
        "instruction_id": record["instruction_id"],
        "source_record_sha256": _canonical_sha256(record),
        "rewards": rewards,
        "exact_matches": exact,
    }


class RLDifficultyTests(unittest.TestCase):
    def test_resume_journal_matches_slurm_cleanup_for_dot_prefixed_output(self) -> None:
        self.assertEqual(
            _difficulty_work_dir(Path("/runs/.formal.incomplete")),
            Path("/runs/.formal.incomplete.work"),
        )
        self.assertEqual(
            _difficulty_work_dir(Path("/runs/formal")),
            Path("/runs/.formal.work"),
        )

    def test_filters_by_preregistered_task_specific_rules(self) -> None:
        records = [
            _record(0, "peak_grounding", "zh-CN"),
            _record(1, "peak_grounding"),
            _record(2, "peak_presence"),
            _record(3, "scientific_qc"),
        ]
        rollouts = [
            _rollout(records[0], [0.0, 0.2] * 4, [False] * GROUP_SIZE),
            _rollout(records[1], [0.5] * GROUP_SIZE, [False] * GROUP_SIZE),
            _rollout(records[2], [1.0, 0.0] * 4, [True, False] * 4),
            _rollout(records[3], [1.0] * GROUP_SIZE, [True] * GROUP_SIZE),
        ]
        selected, summary = select_difficulty_records(records, rollouts)
        selected_ids = {record["instruction_id"] for record in selected}
        self.assertIn("instruction-0", selected_ids)
        self.assertIn("instruction-2", selected_ids)
        self.assertEqual(summary["informative_by_task"]["peak_grounding"]["informative"], 1)
        self.assertEqual(summary["grounding_reward_mean_distribution"]["median"], 0.3)

    def test_uninformative_reinjection_is_fixed_seed_deterministic(self) -> None:
        records = [_record(index, "peak_presence") for index in range(20)]
        rollouts = [
            _rollout(record, [0.0] * GROUP_SIZE, [False] * GROUP_SIZE)
            for record in records
        ]
        first, first_summary = select_difficulty_records(records, rollouts)
        second, second_summary = select_difficulty_records(
            records, list(reversed(rollouts))
        )
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            first_summary["grounding_reward_mean_distribution"]["records"], 0
        )
        self.assertIsNone(
            first_summary["grounding_reward_mean_distribution"]["mean"]
        )
        self.assertIsNone(
            first_summary["grounding_reward_std_distribution"][
                "zero_variance_fraction"
            ]
        )
        self.assertEqual(len(first), 2)

    def test_rollout_seed_and_three_axis_positions_are_reproducible(self) -> None:
        self.assertEqual(sample_seed(17, "abc", 3), sample_seed(17, "abc", 3))
        self.assertNotEqual(sample_seed(17, "abc", 3), sample_seed(17, "abc", 4))
        positions = np.array([[[4]], [[4]], [[4]]], dtype=np.int64)
        incremented = increment_mrope_position(positions)
        np.testing.assert_array_equal(incremented, np.array([[[5]], [[5]], [[5]]]))


if __name__ == "__main__":
    unittest.main()
