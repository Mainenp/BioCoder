import json
import unittest

from multimodal_science.qwen3vl.task_scoring import (
    bbox_iou,
    grounding_dimensions,
    score_task_response,
    strict_json_object,
    valid_bbox,
)


class TaskScoringTests(unittest.TestCase):
    def test_strict_json_and_bbox_helpers_preserve_evaluator_semantics(self) -> None:
        self.assertEqual(strict_json_object(' {"peak_present": true} '), ({"peak_present": True}, True))
        self.assertEqual(strict_json_object("[]"), (None, True))
        self.assertEqual(strict_json_object("not json"), (None, False))
        self.assertEqual(valid_bbox([1, 0, 9, 10], 10, 10), [1.0, 0.0, 9.0, 10.0])
        self.assertIsNone(valid_bbox([1, 0, True, 10], 10, 10))
        self.assertAlmostEqual(bbox_iou([0, 0, 5, 10], [2, 0, 7, 10]), 3 / 7)

    def test_grounding_dimensions_prefer_manifest_then_prompt(self) -> None:
        prompt = {
            "conversations": [
                {"from": "human", "value": "Return a box for a 100x20 image"}
            ]
        }
        self.assertEqual(grounding_dimensions(prompt, {"image_width": 12, "image_height": 8}), (12, 8))
        self.assertEqual(grounding_dimensions(prompt, {}), (100, 20))

    def test_scores_all_registered_task_schemas(self) -> None:
        prompt = {
            "conversations": [
                {"from": "human", "value": "Return a box for a 100x20 image"}
            ]
        }
        presence = score_task_response(
            task="peak_presence",
            prompt=prompt,
            manifest={},
            expected_response=json.dumps({"peak_present": True}),
            predicted_response=json.dumps({"peak_present": False}),
        )
        self.assertTrue(presence["schema_valid"])
        self.assertFalse(presence["exact_match"])
        self.assertEqual(presence["classification_score"], 0.0)

        grounding = score_task_response(
            task="peak_grounding",
            prompt=prompt,
            manifest={},
            expected_response=json.dumps({"bbox_2d": [10, 0, 40, 20]}),
            predicted_response=json.dumps({"bbox_2d": [20, 0, 50, 20]}),
        )
        self.assertTrue(grounding["schema_valid"])
        self.assertTrue(grounding["full_height"])
        self.assertAlmostEqual(grounding["bbox_iou"], 0.5)

        qc = score_task_response(
            task="scientific_qc",
            prompt=prompt,
            manifest={},
            expected_response=json.dumps({"qc_state": "pass", "reason": "clean"}),
            predicted_response=json.dumps({"qc_state": "pass", "reason": "clean"}),
            bilingual=True,
        )
        self.assertTrue(qc["exact_match"])
        self.assertEqual(qc["predicted_qc_state"], "pass")

    def test_invalid_predictions_remain_scored_failures(self) -> None:
        result = score_task_response(
            task="peak_presence_metadata",
            prompt={},
            manifest={},
            expected_response='{"peak_present": false}',
            predicted_response="broken",
        )
        self.assertFalse(result["valid_json"])
        self.assertFalse(result["schema_valid"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["classification_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
