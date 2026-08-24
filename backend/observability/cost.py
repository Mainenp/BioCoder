from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biocoder.state import TokenUsage


@dataclass(frozen=True, slots=True)
class ModelPrice:
    prompt_per_million: float = 0.0
    completion_per_million: float = 0.0


def estimate_cost(usage: TokenUsage, price: ModelPrice) -> float:
    return round(
        usage.prompt_tokens * price.prompt_per_million / 1_000_000
        + usage.completion_tokens * price.completion_per_million / 1_000_000,
        8,
    )


def token_usage_from_message(message: Any) -> TokenUsage:
    raw = getattr(message, "usage_metadata", None) or {}
    if not raw:
        metadata = getattr(message, "response_metadata", None) or {}
        raw = metadata.get("token_usage", {}) or metadata.get("usage", {})
    prompt = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    completion = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total = int(raw.get("total_tokens", prompt + completion) or prompt + completion)
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
