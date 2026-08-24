import json

from training.grpo.rewards import (
    EfficiencyReward,
    FormatReward,
    TaskSuccessReward,
    ToolUseReward,
    default_reward,
)
from training.grpo.train import load_config, run


def successful_trajectory() -> dict:
    return {
        "success": True,
        "final_answer": '{"answer": "EGFR"}',
        "metrics": {"score": 0.9},
        "steps": [
            {
                "action": {"type": "tool_call", "tool": "search_pubmed"},
                "error": None,
            },
            {"action": {"type": "final_answer"}, "error": None},
        ],
    }


def test_rewards_are_modular_and_bounded() -> None:
    trajectory = successful_trajectory()

    assert TaskSuccessReward()(trajectory) == 1
    assert ToolUseReward()(trajectory) == 1
    assert FormatReward("json")(trajectory) == 1
    assert EfficiencyReward(target_steps=1, maximum_steps=3)(trajectory) == 0.5
    assert 0 <= default_reward()(trajectory) <= 1


def test_grpo_dry_run_scores_trajectory_dataset(tmp_path) -> None:
    dataset = tmp_path / "trajectories.jsonl"
    dataset.write_text(json.dumps(successful_trajectory()) + "\n", encoding="utf-8")
    config = tmp_path / "grpo.yaml"
    config.write_text(
        f"""
model_name: local/test-model
dataset_path: {dataset}
dataset_version: v1
output_root: {tmp_path / 'models'}
rollout_adapter: null
rewards: {{task_success: 0.5, tool_use: 0.2, format: 0.2, efficiency: 0.1}}
training:
  learning_rate: 0.000005
  epochs: 1
  num_generations: 2
  max_prompt_length: 128
  max_completion_length: 128
""",
        encoding="utf-8",
    )

    result = run(load_config(config))

    assert result["status"] == "dry_run_validated"
    assert result["experimental"] is True
    assert result["reward_summary"]["records"] == 1
    assert result["checkpoint_metadata"] == {}
