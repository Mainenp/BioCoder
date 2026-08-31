from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

NUMERIC_PREFIX = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")


class MatchStrategy(StrEnum):
    EXACT_SAMPLE_ID = "exact_sample_id"
    UNMATCHED_SAMPLE_ID = "unmatched_sample_id"
    AMBIGUOUS_SAMPLE_ID = "ambiguous_sample_id"


class AuditReason(StrEnum):
    ALTERNATE_LABEL_VARIANT = "alternate_label_variant"
    AMBIGUOUS_SAMPLE_ID = "ambiguous_sample_id"
    INVALID_PEAK_LABEL = "invalid_peak_label"
    INVALID_POSITIVE_BOUNDARY = "invalid_positive_boundary"
    INVALID_SOURCE_SIGNAL = "invalid_source_signal"
    MISSING_SAMPLE_ID = "missing_sample_id"
    UNMATCHED_SAMPLE_ID = "unmatched_sample_id"


@dataclass(frozen=True)
class EligibilityDecision:
    match_strategy: MatchStrategy
    fallback_order: int | None
    train_eligible: bool
    benchmark_eligible: bool
    audit_bucket: bool
    reasons: tuple[str, ...]


def normalize_headers(row: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "comonent": "component",
        "compound name": "component",
    }
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        canonical = aliases.get(str(key).strip().casefold(), str(key).strip().casefold())
        normalized[canonical] = value
    return normalized


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        match = NUMERIC_PREFIX.match(str(value))
        if not match:
            return None
        result = float(match.group(1))
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    result = as_float(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def peak_intervals(row: dict[str, Any]) -> tuple[list[list[float]], bool]:
    intervals: list[list[float]] = []
    incomplete = False
    for index in range(1, 4):
        start = as_float(row.get(f"peak_start{index}"))
        end = as_float(row.get(f"peak_end{index}"))
        if start is None and end is None:
            continue
        if start is None or end is None or end <= start:
            incomplete = True
            continue
        intervals.append([start, end])
    return intervals, incomplete


def decide_eligibility(
    row: dict[str, Any],
    *,
    matched_source_count: int,
    alternate_label_variant: bool,
    source_signal_valid: bool = True,
) -> EligibilityDecision:
    reasons: list[str] = []
    sample_id = as_text(row.get("sample_id"))
    if not sample_id:
        strategy = MatchStrategy.UNMATCHED_SAMPLE_ID
        fallback_order = None
        reasons.append(AuditReason.MISSING_SAMPLE_ID)
    elif matched_source_count == 0:
        strategy = MatchStrategy.UNMATCHED_SAMPLE_ID
        fallback_order = None
        reasons.append(AuditReason.UNMATCHED_SAMPLE_ID)
    elif matched_source_count > 1:
        strategy = MatchStrategy.AMBIGUOUS_SAMPLE_ID
        fallback_order = None
        reasons.append(AuditReason.AMBIGUOUS_SAMPLE_ID)
    else:
        strategy = MatchStrategy.EXACT_SAMPLE_ID
        fallback_order = 0

    if alternate_label_variant:
        reasons.append(AuditReason.ALTERNATE_LABEL_VARIANT)
    if matched_source_count == 1 and not source_signal_valid:
        reasons.append(AuditReason.INVALID_SOURCE_SIGNAL)

    peak_label = as_int(row.get("peak_label"))
    if peak_label not in {0, 1}:
        reasons.append(AuditReason.INVALID_PEAK_LABEL)
    elif peak_label == 1:
        intervals, incomplete = peak_intervals(row)
        peak_count = as_int(row.get("peak_count"))
        if incomplete or peak_count is None or peak_count < 1 or len(intervals) < peak_count:
            reasons.append(AuditReason.INVALID_POSITIVE_BOUNDARY)

    unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
    eligible = not unique_reasons
    return EligibilityDecision(
        match_strategy=strategy,
        fallback_order=fallback_order,
        train_eligible=eligible,
        benchmark_eligible=eligible,
        audit_bucket=not eligible,
        reasons=unique_reasons,
    )
