import json
import unittest

from multimodal_science.qwen3vl.rl_rewards import (
    INVALID_GROUNDING_REWARD,
    interval_iou_giou,
    score_rl_response,
)


class RLRewardTests(unittest.TestCase):
    prompt = {
        "conversations": [
            {"from": "human", "value": "Return a box for a 100x20 image"}
        ]
    }
    manifest = {"image_width": 100, "image_height": 20}

    def _grounding(self, predicted, *, eos=True):
        return score_rl_response(
            task="peak_grounding",
            prompt=self.prompt,
            manifest=self.manifest,
            expected_response=json.dumps({"bbox_2d": [20, 0, 40, 20]}),
            predicted_response=(
                json.dumps({"bbox_2d": predicted})
                if isinstance(predicted, list)
                else predicted
            ),
            terminated_with_eos=eos,
        )

    def test_interval_giou_geometry(self) -> None:
        self.assertEqual(interval_iou_giou([0, 2], [0, 2]), (1.0, 1.0))

        near_iou, near_giou = interval_iou_giou([0, 1], [2, 3])
        far_iou, far_giou = interval_iou_giou([0, 1], [10, 11])
        self.assertEqual(near_iou, 0.0)
        self.assertEqual(far_iou, 0.0)
        self.assertLess(near_giou, 0.0)
        self.assertLess(far_giou, near_giou)

        contained_iou, contained_giou = interval_iou_giou([0, 6], [2, 4])
        self.assertAlmostEqual(contained_iou, 1 / 3)
        self.assertAlmostEqual(contained_giou, 1 / 3)
        self.assertEqual(interval_iou_giou([0, 1], [1, 2]), (0.0, 0.0))

        narrow_iou, _ = interval_iou_giou([1, 5], [1, 5])
        wide_iou, _ = interval_iou_giou([0, 10], [1, 5])
        self.assertLess(wide_iou, narrow_iou)

    def test_grounding_threshold_bonuses_are_exact(self) -> None:
        at_half = self._grounding([20, 0, 60, 20])
        self.assertAlmostEqual(at_half.interval_iou, 0.5)
        self.assertEqual(at_half.iou_at_0_5_bonus, 0.2)
        self.assertEqual(at_half.iou_at_0_75_bonus, 0.0)

        at_three_quarters = self._grounding([20, 0, 140 / 3, 20])
        self.assertAlmostEqual(at_three_quarters.interval_iou, 0.75)
        self.assertEqual(at_three_quarters.iou_at_0_5_bonus, 0.2)
        self.assertEqual(at_three_quarters.iou_at_0_75_bonus, 0.2)

    def test_grounding_invalid_or_non_full_height_is_strictly_worse(self) -> None:
        malformed = self._grounding("not-json")
        no_eos = self._grounding([20, 0, 40, 20], eos=False)
        non_full_height = self._grounding([20, 1, 40, 20])
        for result in (malformed, no_eos, non_full_height):
            self.assertEqual(result.reward, INVALID_GROUNDING_REWARD)

        legal_examples = (
            self._grounding([0, 0, 1, 20]),
            self._grounding([99, 0, 100, 20]),
            self._grounding([1, 0, 99, 20]),
            self._grounding([20, 0, 40, 20]),
        )
        self.assertTrue(
            all(INVALID_GROUNDING_REWARD < result.reward for result in legal_examples)
        )

    def test_auxiliary_rewards_preserve_capabilities(self) -> None:
        correct = score_rl_response(
            task="peak_presence",
            prompt={},
            manifest={},
            expected_response='{"peak_present":true}',
            predicted_response='{"peak_present":true}',
            terminated_with_eos=True,
        )
        wrong = score_rl_response(
            task="peak_presence_metadata",
            prompt={},
            manifest={},
            expected_response='{"peak_present":true}',
            predicted_response='{"peak_present":false}',
            terminated_with_eos=True,
        )
        invalid = score_rl_response(
            task="scientific_qc",
            prompt={},
            manifest={},
            expected_response='{"qc_state":"pass","reason":"clean"}',
            predicted_response='{"qc_state":"pass","reason":"clean"}',
            terminated_with_eos=False,
        )
        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(wrong.reward, 0.0)
        self.assertEqual(invalid.reward, -1.0)


if __name__ == "__main__":
    unittest.main()
