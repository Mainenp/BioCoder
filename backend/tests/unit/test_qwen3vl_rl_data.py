import unittest

from multimodal_science.qwen3vl.build_rl_pool_cli import parser
from multimodal_science.qwen3vl.rl_data import (
    AUXILIARY_TASKS,
    GROUNDING_TASK,
    SourceGroup,
    assert_no_protected_input_paths,
    assert_training_pool_isolation,
    construct_training_pool,
    select_source_groups,
    selection_group_key,
)


def _group(index: int, stratum: str) -> SourceGroup:
    marker = "blank" if stratum == "blank" else "qc" if stratum == "qc" else "sample"
    return SourceGroup(
        group_id=f"group-{index:03d}",
        source_path=f"mzml/train/{marker}-{index:03d}.mzML",
        stratum=stratum,
        provenance_sha256=f"{index:064x}",
    )


def _record(index: int, group_id: str, task: str) -> dict:
    return {
        "instruction_id": f"instruction-{index:04d}",
        "group_id": group_id,
        "task": task,
        "language": "zh-CN" if index % 5 < 3 else "en",
        "target_peak_present": index % 2 == 0,
    }


class RLDataTests(unittest.TestCase):
    def test_selection_is_deterministic_disjoint_and_covers_three_strata(self) -> None:
        groups = [*(_group(i, "sample") for i in range(40))]
        groups.extend((_group(100, "blank"), _group(101, "qc")))
        first = select_source_groups(groups)
        second = select_source_groups(list(reversed(groups)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual({group.stratum for group in first}, {"blank", "qc", "sample"})
        selected_ids = {group.group_id for group in first}
        self.assertFalse(selected_ids & ({group.group_id for group in groups} - selected_ids))
        self.assertEqual(
            [selection_group_key(group.source_path) for group in first],
            sorted(selection_group_key(group.source_path) for group in first),
        )

    def test_training_pool_keeps_all_grounding_and_twenty_percent_auxiliary(self) -> None:
        selected = {"group-selected"}
        records = []
        for index in range(40):
            records.append(_record(index, f"group-{index % 9}", GROUNDING_TASK))
        for index in range(90):
            records.append(
                _record(
                    100 + index,
                    f"group-{index % 9}",
                    AUXILIARY_TASKS[index % len(AUXILIARY_TASKS)],
                )
            )
        records.append(_record(999, "group-selected", GROUNDING_TASK))
        pool = construct_training_pool(records, selected)
        grounding = [record for record in pool if record["task"] == GROUNDING_TASK]
        auxiliary = [record for record in pool if record["task"] != GROUNDING_TASK]
        self.assertEqual(len(grounding), 40)
        self.assertEqual(len(auxiliary), 10)
        self.assertEqual({record["task"] for record in auxiliary}, set(AUXILIARY_TASKS))
        self.assertAlmostEqual(len(grounding) / len(pool), 0.8)
        self.assertFalse({record["group_id"] for record in pool} & selected)

    def test_training_process_rejects_selection_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "Selection-set sample"):
            assert_training_pool_isolation(
                [_record(1, "held-out-group", GROUNDING_TASK)],
                {"held-out-group"},
            )

    def test_protected_inputs_and_cli_surfaces_are_absent(self) -> None:
        for path in (
            "answer/validation_answers.jsonl",
            "jobs/internal_test/source.npy",
        ):
            with self.assertRaisesRegex(ValueError, "Protected input"):
                assert_no_protected_input_paths([path])
        options = {option for action in parser()._actions for option in action.option_strings}
        self.assertFalse(
            options
            & {
                "--validation-answers",
                "--validation-root",
                "--internal-test-root",
                "--selection-set",
            }
        )


if __name__ == "__main__":
    unittest.main()
