from __future__ import annotations

from typing import Any


def quality_score(record: dict[str, Any]) -> float:
    """Deterministic quality score used as a gate, never as a clinical truth metric."""
    score = float(record.get("evaluation_score", 0.0) or 0.0)
    if score == 0 and record.get("success"):
        score = 0.7
    answer = str(record.get("answer", "")).strip()
    if not answer:
        score -= 0.25
    if record.get("failure_type"):
        score -= 0.25
    tool_errors = sum(1 for step in record.get("trajectory", {}).get("steps", []) if step.get("error"))
    score -= min(0.3, tool_errors * 0.1)
    feedback = record.get("feedback", [])
    for item in feedback:
        kind = item.get("feedback_type")
        if kind == "thumbs_up" or (kind == "rating" and (item.get("rating") or 0) >= 4):
            score += 0.15
        if kind == "thumbs_down" or (kind == "rating" and (item.get("rating") or 5) <= 2):
            score -= 0.35
        if item.get("corrected_answer"):
            score = min(score, 0.4)
    return round(max(0.0, min(1.0, score)), 4)
