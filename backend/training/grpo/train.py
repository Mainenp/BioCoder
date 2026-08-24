from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from training.common import dataset_metadata, git_commit, load_yaml, new_run_dir, write_run_metadata
from training.grpo.rewards import default_reward

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


class GRPOTrainingSettings(BaseModel):
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    num_generations: int = Field(default=4, ge=2)
    max_prompt_length: int = Field(default=2048, ge=64)
    max_completion_length: int = Field(default=2048, ge=64)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    bf16: bool = True


class GRPOJobConfig(BaseModel):
    model_name: str = Field(min_length=1)
    dataset_path: Path
    dataset_version: str = "unset"
    output_root: Path = Path("models")
    rollout_adapter: str | None = None
    rewards: dict[str, float]
    training: GRPOTrainingSettings


def load_config(path: Path, dataset_override: Path | None = None) -> GRPOJobConfig:
    config = GRPOJobConfig.model_validate(load_yaml(path))
    if dataset_override:
        config.dataset_path = dataset_override
    for field_name in ("dataset_path", "output_root"):
        value = getattr(config, field_name)
        if not value.is_absolute():
            setattr(config, field_name, BACKEND_ROOT / value)
    default_reward(config.rewards)
    return config


def _reward_summary(dataset_path: Path, weights: dict[str, float]) -> dict[str, Any]:
    reward = default_reward(weights)
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    scores = [reward(row) for row in rows]
    return {
        "records": len(rows),
        "mean_reward": round(sum(scores) / len(scores), 4),
        "minimum_reward": min(scores),
        "maximum_reward": max(scores),
    }


def run(config: GRPOJobConfig, *, execute: bool = False) -> dict[str, Any]:
    dataset = dataset_metadata(config.dataset_path)
    run_dir = new_run_dir(config.output_root, "grpo")
    reward_summary = _reward_summary(config.dataset_path, config.rewards)
    metadata: dict[str, Any] = {
        "training_method": "grpo_agentic_rl",
        "status": "dry_run_validated",
        "dry_run": not execute,
        "experimental": True,
        "model_name": config.model_name,
        "dataset_version": config.dataset_version,
        "dataset": dataset,
        "reward_summary": reward_summary,
        "training_config": config.model_dump(mode="json"),
        "git_commit": git_commit(BACKEND_ROOT.parent),
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_metadata": {},
    }
    if execute:
        if not config.rollout_adapter:
            metadata["status"] = "blocked_missing_rollout_adapter"
            write_run_metadata(run_dir, metadata)
            raise RuntimeError(
                "GRPO execution needs a rollout_adapter that converts prompts/completions into verifiable Agent trajectories."
            )
        try:
            import trl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("GRPO execution requires the optional TRL training stack.") from exc
        metadata["status"] = "scaffold_requires_project_specific_rollout_adapter"
    write_run_metadata(run_dir, metadata)
    metadata["run_dir"] = str(run_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the experimental BioCoder GRPO setup.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(load_config(args.config, args.dataset), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
