"""Read-only summary of sealed v2 RL development evidence.

This module never opens prompts, answers, model weights, validation datasets, or
internal-test artifacts.  It reduces already completed, manifest-bound reports
and Slurm completion logs to the values needed by the public result document.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CANDIDATE = re.compile(
    r"^(?:RL_VALIDATION_CANDIDATE_ID|CANDIDATE_ID)=(.+)$", re.MULTILINE
)
_EVALUATION = re.compile(r"^PERSISTENT_EVALUATION=(.+)$", re.MULTILINE)
_MODELS = ("c0", "c1", "rl")
_SCOPES = ("overall", "en", "zh-CN")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{label} not found: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(isinstance(value, dict), f"{label} row {number} must be an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entries(root: Path) -> dict[str, str]:
    path = root / "artifact_manifest.sha256"
    _require(path.is_file(), f"Artifact manifest not found: {path}")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        _require(bool(_HEX_64.fullmatch(digest)), "Invalid artifact digest")
        _require(relative not in entries, f"Duplicate manifest path: {relative}")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"Manifest path escapes root: {relative}") from error
        _require(candidate.is_file(), f"Manifest artifact not found: {relative}")
        _require(_sha256(candidate) == digest, f"Manifest artifact drift: {relative}")
        entries[relative] = digest
    return entries


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _find_first(value: Any, names: Iterable[str]) -> Any:
    wanted = set(names)
    if isinstance(value, dict):
        for name in wanted:
            if name in value:
                return value[name]
        for child in value.values():
            found = _find_first(child, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first(child, wanted)
            if found is not None:
                return found
    return None


def _hypotheses(report: dict[str, Any]) -> dict[str, Any]:
    bootstrap = _object(
        _object(report.get("paired_group_bootstrap"), "bootstrap").get("comparisons"),
        "bootstrap comparisons",
    )
    h1_metric = _object(
        _object(_object(bootstrap.get("rl_minus_c1"), "RL/C1 bootstrap").get("overall"), "RL/C1 overall").get("mean_iou"),
        "H1 metric",
    )
    h2_metric = _object(
        _object(_object(bootstrap.get("rl_minus_c0"), "RL/C0 bootstrap").get("overall"), "RL/C0 overall").get("iou_at_0_75"),
        "H2 metric",
    )
    h1_ci = h1_metric.get("rl_minus_reference_ci_95")
    h2_ci = h2_metric.get("rl_minus_reference_ci_95")
    _require(isinstance(h1_ci, list) and len(h1_ci) == 2, "Invalid H1 interval")
    _require(isinstance(h2_ci, list) and len(h2_ci) == 2, "Invalid H2 interval")

    constraints = _object(report.get("constraints"), "constraints")
    c0_constraints = _object(
        _object(constraints.get("c0"), "C0 constraints").get("overall"),
        "C0 overall constraints",
    )
    rl_constraints = _object(
        _object(constraints.get("rl"), "RL constraints").get("overall"),
        "RL overall constraints",
    )
    c0_presence = _number(c0_constraints.get("presence_macro_f1"), "C0 presence macro-F1")
    rl_presence = _number(rl_constraints.get("presence_macro_f1"), "RL presence macro-F1")
    rl_json = _number(rl_constraints.get("json_valid_rate"), "RL JSON-valid rate")
    presence_drop = c0_presence - rl_presence

    consistency = _object(report.get("cross_language_consistency"), "consistency")
    c0_grounding = _object(
        _object(consistency.get("c0"), "C0 consistency").get("peak_grounding"),
        "C0 grounding consistency",
    )
    rl_grounding = _object(
        _object(consistency.get("rl"), "RL consistency").get("peak_grounding"),
        "RL grounding consistency",
    )
    consistency_metric = "mean_prediction_bbox_iou_all"
    c0_consistency = _number(c0_grounding.get(consistency_metric), "C0 consistency")
    rl_consistency = _number(rl_grounding.get(consistency_metric), "RL consistency")

    return {
        "h1": {
            "passed": _number(h1_ci[0], "H1 lower bound") > 0.0,
            "metric": "rl_minus_c1_mean_iou",
            "delta": _number(h1_metric.get("rl_minus_reference"), "H1 delta"),
            "ci_95": [_number(value, "H1 interval") for value in h1_ci],
        },
        "h2": {
            "passed": _number(h2_ci[0], "H2 lower bound") > 0.0,
            "metric": "rl_minus_c0_iou_at_0_75",
            "delta": _number(h2_metric.get("rl_minus_reference"), "H2 delta"),
            "ci_95": [_number(value, "H2 interval") for value in h2_ci],
        },
        "h3": {
            "passed": presence_drop <= 0.01 and rl_json >= 0.999,
            "presence_macro_f1_c0": c0_presence,
            "presence_macro_f1_rl": rl_presence,
            "presence_drop": presence_drop,
            "rl_json_valid_rate": rl_json,
        },
        "h4": {
            "passed": rl_consistency >= c0_consistency,
            "metric": consistency_metric,
            "c0": c0_consistency,
            "rl": rl_consistency,
            "rl_minus_c0": rl_consistency - c0_consistency,
        },
    }


def _training_summaries(run_root: Path) -> list[dict[str, Any]]:
    reports = sorted(
        (run_root / "qwen3vl" / "rl" / "runs").glob(
            "v2-*-formal-seed17-*/*training_report.json"
        )
    )
    _require(len(reports) == 5, f"Expected five formal training reports, found {len(reports)}")
    fields = (
        "global_steps",
        "optimizer_updates",
        "wall_time_seconds",
        "seconds_per_update",
        "peak_cuda_gib",
        "final_loss",
        "final_kl_mean",
        "final_grounding_mean_iou",
        "final_grounding_mean_giou",
        "final_zero_variance_group_fraction",
        "stopped_early",
        "stop_reason",
    )
    output = []
    for path in reports:
        report = _read_json(path, "formal training report")
        output.append(
            {
                "run": path.parent.name,
                "report_sha256": _sha256(path),
                "schema_version": report.get("schema_version"),
                **{field: _find_first(report, (field,)) for field in fields},
            }
        )
    return output


def _candidate_logs(slurm_root: Path) -> list[tuple[str, Path]]:
    candidates: dict[str, Path] = {}
    for path in sorted(slurm_root.glob("*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        candidate_match = _CANDIDATE.search(text)
        evaluation_match = _EVALUATION.search(text)
        if candidate_match is None or evaluation_match is None:
            continue
        candidate_id = candidate_match.group(1).strip()
        evaluation_root = Path(evaluation_match.group(1).strip()).resolve()
        previous = candidates.get(candidate_id)
        if previous is not None:
            _require(previous == evaluation_root, f"Candidate evaluation drift: {candidate_id}")
        candidates[candidate_id] = evaluation_root
    _require(len(candidates) == 16, f"Expected 16 validation candidates, found {len(candidates)}")
    return sorted(candidates.items())


def _candidate_summaries(slurm_root: Path) -> list[dict[str, Any]]:
    output = []
    for candidate_id, root in _candidate_logs(slurm_root):
        entries = _manifest_entries(root)
        report_path = root / "qwen_evaluation_report.json"
        _require(
            entries.get("qwen_evaluation_report.json") == _sha256(report_path),
            f"Evaluation report is not manifest-bound: {candidate_id}",
        )
        report = _read_json(report_path, "validation evaluation report")
        _require(report.get("internal_test_accessed") is False, "Candidate used internal test")
        _require(report.get("final_benchmark_eligible") is False, "V2 candidate claims final eligibility")
        metrics = _object(report.get("metrics"), "candidate metrics")
        presence = _object(
            _object(metrics.get("peak_presence"), "presence task").get("classification"),
            "presence classification",
        )
        grounding = _object(
            _object(metrics.get("peak_grounding"), "grounding task").get("grounding"),
            "grounding metrics",
        )
        counts = _object(report.get("counts"), "candidate counts")
        records = _number(counts.get("predictions"), "prediction records")
        valid = _number(counts.get("valid_json"), "valid JSON records")
        output.append(
            {
                "candidate_id": candidate_id,
                "evaluation_report_sha256": _sha256(report_path),
                "grounding_mean_iou": _number(
                    grounding.get("mean_bbox_iou_all"), "grounding mean IoU"
                ),
                "grounding_iou_at_0_5": _number(
                    grounding.get("iou_at_0_5_rate_all"), "grounding IoU@0.5"
                ),
                "presence_macro_f1": _number(
                    presence.get("macro_f1"), "presence macro-F1"
                ),
                "presence_mcc": _number(presence.get("mcc"), "presence MCC"),
                "json_valid_rate": valid / records,
                "transparent_only": candidate_id
                not in {
                    "C0:xic-only-sft-seed17:step-00000000",
                    "R2:lora-lr-5e-6-projector-frozen:step-00000300",
                    "C1:lora-lr-1e-6:step-00000300",
                },
            }
        )
    return output


def summarize_v2_rl_evidence(
    *, analysis_root: Path, run_root: Path, slurm_root: Path
) -> dict[str, Any]:
    """Verify and summarize completed reports without reopening protected data."""

    analysis_root = analysis_root.resolve()
    run_root = run_root.resolve()
    slurm_root = slurm_root.resolve()
    entries = _manifest_entries(analysis_root)
    expected = {
        "localization_analysis.json",
        "localization_analysis.md",
        "localization_failure_cases.jsonl",
    }
    _require(set(entries) == expected, "Localization artifact set drift")
    report_path = analysis_root / "localization_analysis.json"
    report = _read_json(report_path, "localization analysis")
    _require(
        report.get("schema_version") == "chrompeak-qwen3vl-v2-localization-analysis-v1",
        "Wrong localization analysis schema",
    )
    _require(report.get("evaluation_scope") == "development_validation", "Wrong evaluation scope")
    _require(report.get("internal_test_accessed") is False, "Analysis used internal test")
    _require(report.get("final_benchmark_eligible") is False, "Analysis claims final eligibility")
    contracts = _object(report.get("contracts"), "analysis contracts")
    _require(contracts.get("internal_test_accessed") is False, "Analysis contract used internal test")

    failures = _read_jsonl(
        analysis_root / "localization_failure_cases.jsonl", "failure cases"
    )
    failure_lock = _object(report.get("failure_cases"), "failure-case lock")
    _require(failure_lock.get("records") == len(failures), "Failure-case count drift")
    _require(len(failures) <= 20, "Too many failure cases")

    return {
        "schema_version": "chrompeak-qwen3vl-v2-evidence-summary-v1",
        "evaluation_scope": "development_validation",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "localization_analysis_sha256": _sha256(report_path),
        "hypotheses": _hypotheses(report),
        "localization": report["localization"],
        "constraints": report["constraints"],
        "cross_language_consistency": report["cross_language_consistency"],
        "stratification": report["stratification"],
        "paired_group_bootstrap": report["paired_group_bootstrap"],
        "failure_cases": failures,
        "formal_training": _training_summaries(run_root),
        "validation_candidates": _candidate_summaries(slurm_root),
    }

