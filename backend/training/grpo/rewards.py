from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class RewardFunction(ABC):
    name: str

    @abstractmethod
    def __call__(self, trajectory: dict[str, Any]) -> float:
        """Return a bounded reward in [0, 1]."""


class TaskSuccessReward(RewardFunction):
    name = "task_success"

    def __call__(self, trajectory: dict[str, Any]) -> float:
        if trajectory.get("success") is True:
            return 1.0
        score = float(trajectory.get("metrics", {}).get("score", 0) or 0)
        return max(0.0, min(1.0, score))


class ToolUseReward(RewardFunction):
    name = "tool_use"

    def __call__(self, trajectory: dict[str, Any]) -> float:
        tool_steps = [
            step
            for step in trajectory.get("steps", [])
            if step.get("action", {}).get("type") == "tool_call"
        ]
        if not tool_steps:
            required = trajectory.get("metrics", {}).get("required_tools", [])
            return 0.0 if required else 1.0
        successful = sum(not step.get("error") for step in tool_steps)
        return successful / len(tool_steps)


class FormatReward(RewardFunction):
    name = "format"

    def __init__(self, expected_format: str = "text") -> None:
        self.expected_format = expected_format

    def __call__(self, trajectory: dict[str, Any]) -> float:
        answer = trajectory.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        if self.expected_format == "json":
            try:
                json.loads(answer)
            except json.JSONDecodeError:
                return 0.0
        return 1.0


@dataclass(slots=True)
class EfficiencyReward(RewardFunction):
    name = "efficiency"
    target_steps: int = 6
    maximum_steps: int = 20

    def __call__(self, trajectory: dict[str, Any]) -> float:
        steps = len(trajectory.get("steps", []))
        if steps <= self.target_steps:
            return 1.0
        if steps >= self.maximum_steps:
            return 0.0
        return 1 - (steps - self.target_steps) / (self.maximum_steps - self.target_steps)


@dataclass(slots=True)
class WeightedReward:
    rewards: dict[str, RewardFunction]
    weights: dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(self.rewards)
        if unknown:
            raise ValueError(f"Weights reference unknown rewards: {sorted(unknown)}")
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Reward weights must be non-negative with a positive sum")

    def components(self, trajectory: dict[str, Any]) -> dict[str, float]:
        return {name: round(reward(trajectory), 4) for name, reward in self.rewards.items()}

    def __call__(self, trajectory: dict[str, Any]) -> float:
        values = self.components(trajectory)
        total = sum(self.weights.values())
        return round(sum(values[name] * weight for name, weight in self.weights.items()) / total, 4)


def default_reward(weights: dict[str, float] | None = None) -> WeightedReward:
    rewards: dict[str, RewardFunction] = {
        "task_success": TaskSuccessReward(),
        "tool_use": ToolUseReward(),
        "format": FormatReward(),
        "efficiency": EfficiencyReward(),
    }
    return WeightedReward(
        rewards=rewards,
        weights=weights
        or {"task_success": 0.45, "tool_use": 0.25, "format": 0.15, "efficiency": 0.15},
    )
