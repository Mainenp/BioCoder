"""Paired, provenance-bound analysis of XIC inference interventions."""

from __future__ import annotations

import json
import math
import random
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.fusion_inference import XIC_INTERVENTIONS
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA


REPORT_SCHEMA = "chrompeak-qwen3vl-xic-intervention-analysis-v1"
LANGUAGES = ("en", "zh-CN")
SCOPES = ("overall", *LANGUAGES)
CLASSIFICATION_TASKS = ("peak_presence", "peak_presence_metadata")
CLASSIFICATION_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "false_positive_rate",
)
GROUNDING_METRICS = ("mean_bbox_iou_all", "iou_at_0_5_rate_all")
QC_METRICS = ("exact_match_rate",)
HIGHER_IS_BETTER = {
    "balanced_accuracy": True,
    "macro_f1": True,
    "mcc": True,
    "false_positive_rate": False,
    "mean_bbox_iou_all": True,
    "iou_at_0_5_rate_all": True,
    "exact_match_rate": True,
}


@dataclass(frozen=True)
class XicInterventionRun:
    generation_root: Path
    evaluation_root: Path


@dataclass(frozen=True)
class XicInterventionAnalysisResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class _LoadedRun:
    generation_root: Path
    evaluation_root: Path
    generation_report: dict[str, Any]
    evaluation_report: dict[str, Any]
    generation_sha256: str
    evaluation_sha256: str
    records_sha256: str
    records: tuple[dict[str, Any], ...]
    intervention: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _object(payload, label)


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(_object(value, f"{label} line {line_number}"))
    _require(rows, f"{label} is empty")
    return tuple(rows)


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_from_counts(counts: Iterable[float]) -> dict[str, float]:
    tp, fp, tn, fn = (float(value) for value in counts)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    positive_f1 = _divide(2.0 * tp, 2.0 * tp + fp + fn)
    negative_f1 = _divide(2.0 * tn, 2.0 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "balanced_accuracy": (recall + specificity) / 2.0,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "mcc": _divide(tp * tn - fp * fn, denominator),
        "false_positive_rate": _divide(fp, fp + tn),
    }


def _scope_rows(rows: Iterable[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    _require(scope in SCOPES, f"Unsupported metric scope: {scope}")
    if scope == "overall":
        return list(rows)
    return [row for row in rows if row.get("language") == scope]


def _group_statistics(
    rows: Iterable[dict[str, Any]], scope: str
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in _scope_rows(rows, scope):
        group_id = str(row.get("group_id"))
        task = str(row.get("task"))
        _require(group_id and group_id != "None", "Evaluation record has no source group")
        if task in CLASSIFICATION_TASKS:
            stats = grouped.setdefault(group_id, {}).setdefault(task, [0.0] * 4)
            truth = bool(row.get("target_peak_present"))
            predicted = float(row.get("classification_score")) >= 0.5
            if truth and predicted:
                stats[0] += 1.0
            elif not truth and predicted:
                stats[1] += 1.0
            elif not truth and not predicted:
                stats[2] += 1.0
            else:
                stats[3] += 1.0
        elif task == "peak_grounding":
            stats = grouped.setdefault(group_id, {}).setdefault(task, [0.0] * 3)
            stats[0] += float(row.get("bbox_iou"))
            stats[1] += float(bool(row.get("iou_at_0_5")))
            stats[2] += 1.0
        elif task == "scientific_qc":
            stats = grouped.setdefault(group_id, {}).setdefault(task, [0.0] * 2)
            stats[0] += float(bool(row.get("exact_match")))
            stats[1] += 1.0
        else:
            raise ValueError(f"Unsupported evaluation task: {task}")
    _require(grouped, f"No records found for scope {scope}")
    expected_tasks = {*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc"}
    observed_tasks = {task for tasks in grouped.values() for task in tasks}
    _require(
        observed_tasks == expected_tasks,
        f"Scope {scope} does not contain all required evaluation tasks",
    )
    return grouped


def _task_groups(
    grouped: dict[str, dict[str, list[float]]], task: str
) -> list[str]:
    return sorted(group_id for group_id, tasks in grouped.items() if task in tasks)


def _sum_stats(
    grouped: dict[str, dict[str, list[float]]], sampled_groups: Iterable[str]
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for group_id in sampled_groups:
        for task, values in grouped[group_id].items():
            target = result.setdefault(task, [0.0] * len(values))
            for index, value in enumerate(values):
                target[index] += value
    return result


def _task_metrics_from_stats(task: str, stats: list[float]) -> dict[str, float]:
    if task in CLASSIFICATION_TASKS:
        return _classification_from_counts(stats)
    if task == "peak_grounding":
        return {
            "mean_bbox_iou_all": _divide(stats[0], stats[2]),
            "iou_at_0_5_rate_all": _divide(stats[1], stats[2]),
        }
    if task == "scientific_qc":
        return {"exact_match_rate": _divide(stats[0], stats[1])}
    raise ValueError(f"Unsupported evaluation task: {task}")


def _metrics_from_stats(stats: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    expected_tasks = (*CLASSIFICATION_TASKS, "peak_grounding", "scientific_qc")
    return {
        task: _task_metrics_from_stats(task, stats[task])
        for task in expected_tasks
    }


def _selected_metrics(rows: Iterable[dict[str, Any]], scope: str) -> dict[str, Any]:
    grouped = _group_statistics(rows, scope)
    return _metrics_from_stats(_sum_stats(grouped, sorted(grouped)))


def _reported_selected_metrics(
    report: dict[str, Any], scope: str
) -> dict[str, dict[str, float]]:
    if scope == "overall":
        tasks = _object(report.get("metrics"), "combined evaluation metrics")
    else:
        tasks = _object(
            _object(report.get("metrics_by_language"), "metrics by language").get(scope),
            f"{scope} evaluation metrics",
        )
    result: dict[str, dict[str, float]] = {}
    for task in CLASSIFICATION_TASKS:
        values = _object(
            _object(tasks.get(task), f"{scope} {task}").get("classification"),
            f"{scope} {task} classification",
        )
        result[task] = {name: float(values[name]) for name in CLASSIFICATION_METRICS}
    grounding = _object(
        _object(tasks.get("peak_grounding"), f"{scope} grounding").get("grounding"),
        f"{scope} grounding metrics",
    )
    result["peak_grounding"] = {
        name: float(grounding[name]) for name in GROUNDING_METRICS
    }
    qc = _object(tasks.get("scientific_qc"), f"{scope} scientific QC")
    result["scientific_qc"] = {"exact_match_rate": float(qc["exact_match_rate"])}
    return result


def _assert_metrics_match(
    observed: dict[str, Any], reported: dict[str, Any], label: str
) -> None:
    for task, values in observed.items():
        for metric, value in values.items():
            _require(
                math.isclose(value, reported[task][metric], rel_tol=0.0, abs_tol=1e-12),
                f"{label} recomputed {task}/{metric} does not match its report",
            )


def _record_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("instruction_id"),
        row.get("pair_id"),
        row.get("asset_id"),
        row.get("group_id"),
        row.get("task"),
        row.get("language"),
        row.get("target_peak_present"),
        tuple(row.get("expected_bbox_2d") or ()),
    )


def _load_run(label: str, run: XicInterventionRun) -> _LoadedRun:
    generation_root = run.generation_root.resolve()
    evaluation_root = run.evaluation_root.resolve()
    generation_path = generation_root / "generation_report.json"
    evaluation_path = evaluation_root / "qwen_evaluation_report.json"
    generation = _read_object(generation_path, f"{label} generation report")
    evaluation = _read_object(evaluation_path, f"{label} evaluation report")
    generation_sha256 = sha256_file(generation_path)
    evaluation_sha256 = sha256_file(evaluation_path)

    _require(
        generation.get("schema_version") == GENERATION_REPORT_SCHEMA,
        f"Unexpected {label} generation schema",
    )
    _require(
        evaluation.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        f"Unexpected {label} evaluation schema",
    )
    _require(
        generation.get("development_comparison_candidate") is True
        and evaluation.get("development_comparison_eligible") is True,
        f"{label} is not a complete development comparison",
    )
    for report_name, report in (("generation", generation), ("evaluation", evaluation)):
        _require(
            report.get("internal_test_accessed") is False
            and report.get("final_benchmark_eligible") is False,
            f"{label} {report_name} has an invalid evaluation scope",
        )

    inputs = _object(evaluation.get("inputs"), f"{label} evaluation inputs")
    _require(
        inputs.get("generation_report_sha256") == generation_sha256,
        f"{label} evaluation is not bound to its generation report",
    )
    provenance = _object(
        evaluation.get("generation_provenance"), f"{label} generation provenance"
    )
    _require(
        provenance.get("report_sha256") == generation_sha256
        and evaluation.get("prediction_generation_provenance_verified") is True,
        f"{label} generation provenance was not verified",
    )

    runtime = _object(generation.get("runtime"), f"{label} runtime")
    intervention = _object(runtime.get("xic_intervention"), f"{label} intervention")
    _require(intervention.get("mode") == label, f"{label} intervention mode mismatch")
    adapter = _object(
        _object(generation.get("model"), f"{label} model").get("adapter"),
        f"{label} fusion adapter",
    )
    _require(
        adapter.get("xic_intervention") == intervention,
        f"{label} adapter does not bind its intervention",
    )

    artifacts = _object(evaluation.get("artifacts"), f"{label} evaluation artifacts")
    records_artifact = _object(
        artifacts.get("evaluation_records"), f"{label} evaluation records artifact"
    )
    records_path = evaluation_root / str(records_artifact.get("path"))
    records_sha256 = sha256_file(records_path)
    _require(
        records_sha256 == records_artifact.get("sha256"),
        f"{label} evaluation-record hash mismatch",
    )
    records = _read_jsonl(records_path, f"{label} evaluation records")
    counts = _object(evaluation.get("counts"), f"{label} evaluation counts")
    _require(
        counts.get("predictions") == len(records)
        and records_artifact.get("records") == len(records),
        f"{label} evaluation-record count mismatch",
    )
    _require(
        len({str(row.get("instruction_id")) for row in records}) == len(records),
        f"{label} instruction IDs are not unique",
    )
    for scope in SCOPES:
        recomputed = _selected_metrics(records, scope)
        _assert_metrics_match(
            recomputed,
            _reported_selected_metrics(evaluation, scope),
            f"{label}/{scope}",
        )
    return _LoadedRun(
        generation_root=generation_root,
        evaluation_root=evaluation_root,
        generation_report=generation,
        evaluation_report=evaluation,
        generation_sha256=generation_sha256,
        evaluation_sha256=evaluation_sha256,
        records_sha256=records_sha256,
        records=records,
        intervention=intervention,
    )


def _shared_contract(loaded: dict[str, _LoadedRun]) -> dict[str, Any]:
    aligned = loaded["aligned"]
    generation_source = _object(aligned.generation_report.get("source"), "generation source")
    generation_settings = _object(
        aligned.generation_report.get("generation"), "generation settings"
    )
    model = _object(aligned.generation_report.get("model"), "generation model")
    adapter = _object(model.get("adapter"), "fusion adapter")
    evaluation_inputs = _object(
        aligned.evaluation_report.get("inputs"), "evaluation inputs"
    )
    shared_input_fields = (
        "instruction_report_sha256",
        "source_dataset_report_sha256",
        "validation_prompts_sha256",
        "validation_answers_sha256",
        "instruction_manifest_sha256",
    )
    identity = sorted(_record_identity(row) for row in aligned.records)
    seed = aligned.intervention.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "Bad intervention seed")

    for label, run in loaded.items():
        source = _object(run.generation_report.get("source"), f"{label} source")
        settings = _object(run.generation_report.get("generation"), f"{label} settings")
        candidate_model = _object(run.generation_report.get("model"), f"{label} model")
        candidate_adapter = _object(candidate_model.get("adapter"), f"{label} adapter")
        candidate_inputs = _object(
            run.evaluation_report.get("inputs"), f"{label} evaluation inputs"
        )
        _require(source == generation_source, f"{label} generation source drift")
        _require(settings == generation_settings, f"{label} generation-setting drift")
        _require(
            candidate_model.get("artifact_sha256") == model.get("artifact_sha256"),
            f"{label} base-model drift",
        )
        for field in (
            "training_report_sha256",
            "manifest_sha256",
            "code_revision",
            "sensor_projector",
        ):
            _require(
                candidate_adapter.get(field) == adapter.get(field),
                f"{label} fusion-adapter {field} drift",
            )
        for field in shared_input_fields:
            _require(
                candidate_inputs.get(field) == evaluation_inputs.get(field),
                f"{label} evaluation {field} drift",
            )
        _require(run.intervention.get("seed") == seed, f"{label} intervention-seed drift")
        _require(
            sorted(_record_identity(row) for row in run.records) == identity,
            f"{label} evaluation-record identity drift",
        )

    return {
        "base_model_artifact_sha256": model.get("artifact_sha256"),
        "fusion_training_report_sha256": adapter.get("training_report_sha256"),
        "fusion_manifest_sha256": adapter.get("manifest_sha256"),
        "sensor_projector": adapter.get("sensor_projector"),
        "instruction_report_sha256": evaluation_inputs.get("instruction_report_sha256"),
        "source_dataset_report_sha256": evaluation_inputs.get(
            "source_dataset_report_sha256"
        ),
        "validation_prompts_sha256": evaluation_inputs.get("validation_prompts_sha256"),
        "validation_answers_sha256": evaluation_inputs.get("validation_answers_sha256"),
        "intervention_seed": seed,
        "validation_records": len(identity),
        "validation_source_groups": len(
            {str(row.get("group_id")) for row in aligned.records}
        ),
    }


def _comparison_bootstrap(
    aligned: _LoadedRun,
    candidate: _LoadedRun,
    *,
    candidate_label: str,
    scope: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    aligned_groups = _group_statistics(aligned.records, scope)
    candidate_groups = _group_statistics(candidate.records, scope)
    groups = sorted(aligned_groups)
    _require(set(candidate_groups) == set(groups), f"{candidate_label}/{scope} group drift")
    observed_aligned = _metrics_from_stats(_sum_stats(aligned_groups, groups))
    observed_candidate = _metrics_from_stats(_sum_stats(candidate_groups, groups))
    deltas: dict[tuple[str, str], list[float]] = {}
    task_group_counts: dict[str, int] = {}
    for task in observed_aligned:
        aligned_task_groups = _task_groups(aligned_groups, task)
        candidate_task_groups = _task_groups(candidate_groups, task)
        _require(
            candidate_task_groups == aligned_task_groups,
            f"{candidate_label}/{scope}/{task} task-group drift",
        )
        _require(
            bool(aligned_task_groups),
            f"{candidate_label}/{scope}/{task} has no source groups",
        )
        task_group_counts[task] = len(aligned_task_groups)
        rng = random.Random(f"{seed}:{candidate_label}:{scope}:{task}")
        for _ in range(iterations):
            sampled = [
                rng.choice(aligned_task_groups) for _ in aligned_task_groups
            ]
            aligned_stats = _sum_stats(aligned_groups, sampled)[task]
            candidate_stats = _sum_stats(candidate_groups, sampled)[task]
            aligned_metrics = _task_metrics_from_stats(task, aligned_stats)
            candidate_metrics = _task_metrics_from_stats(task, candidate_stats)
            for metric, value in aligned_metrics.items():
                deltas.setdefault((task, metric), []).append(
                    value - candidate_metrics[metric]
                )

    result: dict[str, Any] = {}
    for task, aligned_values in observed_aligned.items():
        result[task] = {"source_groups": task_group_counts[task]}
        for metric, aligned_value in aligned_values.items():
            raw = np.asarray(deltas[(task, metric)], dtype=np.float64)
            low, high = np.quantile(raw, [0.025, 0.975]).tolist()
            direction = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
            benefit = raw * direction
            result[task][metric] = {
                "aligned": aligned_value,
                "intervention": observed_candidate[task][metric],
                "aligned_minus_intervention": (
                    aligned_value - observed_candidate[task][metric]
                ),
                "aligned_minus_intervention_ci_95": [float(low), float(high)],
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "direction_adjusted_aligned_benefit": (
                    aligned_value - observed_candidate[task][metric]
                )
                * direction,
                "probability_aligned_better": float(
                    np.mean(benefit > 0.0) + 0.5 * np.mean(benefit == 0.0)
                ),
            }
    return result


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _markdown(report: dict[str, Any]) -> str:
    observed = report["observed_metrics"]["overall"]
    lines = [
        "# XIC intervention analysis",
        "",
        "All rows use the same fusion checkpoint, prompts, answer key, and leakage-safe "
        "validation records. English and Chinese prompts remain paired views, not independent "
        "scientific samples.",
        "",
        "## Overall metrics",
        "",
        "| Intervention | Presence Macro-F1 | Presence MCC | Presence FPR | "
        "Metadata Macro-F1 | Mean bbox IoU | IoU >= 0.5 | QC exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in XIC_INTERVENTIONS:
        row = observed[label]
        lines.append(
            f"| {label} | {_fmt(row['peak_presence']['macro_f1'])} | "
            f"{_fmt(row['peak_presence']['mcc'])} | "
            f"{_fmt(row['peak_presence']['false_positive_rate'])} | "
            f"{_fmt(row['peak_presence_metadata']['macro_f1'])} | "
            f"{_fmt(row['peak_grounding']['mean_bbox_iou_all'])} | "
            f"{_fmt(row['peak_grounding']['iou_at_0_5_rate_all'])} | "
            f"{_fmt(row['scientific_qc']['exact_match_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired source-group bootstrap",
            "",
            "Positive direction-adjusted benefit favors aligned XIC. FPR is sign-inverted for "
            "this interpretation because lower is better.",
            "",
            "| Intervention | Task | Metric | Aligned - intervention | 95% CI | "
            "P(aligned better) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    comparisons = report["paired_group_bootstrap"]["comparisons"]
    primary = {
        "peak_presence": ("macro_f1", "mcc", "false_positive_rate"),
        "peak_presence_metadata": ("macro_f1", "mcc"),
        "peak_grounding": GROUNDING_METRICS,
        "scientific_qc": QC_METRICS,
    }
    for label in XIC_INTERVENTIONS[1:]:
        tasks = comparisons[label]["overall"]
        for task, metrics in primary.items():
            for metric in metrics:
                row = tasks[task][metric]
                low, high = row["aligned_minus_intervention_ci_95"]
                lines.append(
                    f"| {label} | {task} | {metric} | "
                    f"{_fmt(row['aligned_minus_intervention'])} | "
                    f"[{_fmt(low)}, {_fmt(high)}] | "
                    f"{_fmt(row['probability_aligned_better'])} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Confidence intervals resample source groups, not prompts or language variants.",
            "- Each task resamples only source groups containing that task; groups without "
            "grounding labels are not converted into zero-IoU observations.",
            "- This is a single training seed on development validation data.",
            "- The sealed internal-test split remains unopened.",
            "- Intervention effects establish model reliance, not clinical or deployment validity.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_xic_interventions(
    *,
    runs: dict[str, XicInterventionRun],
    output_dir: Path,
    bootstrap_iterations: int = 2000,
    seed: int = 17,
) -> XicInterventionAnalysisResult:
    """Build a paired source-group bootstrap analysis for all four XIC interventions."""

    _require(set(runs) == set(XIC_INTERVENTIONS), "Exactly four XIC interventions are required")
    _require(bootstrap_iterations >= 100, "At least 100 bootstrap iterations are required")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "Bootstrap seed is invalid")
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Analysis output already exists: {output_dir}")

    loaded = {label: _load_run(label, runs[label]) for label in XIC_INTERVENTIONS}
    shared = _shared_contract(loaded)
    observed = {
        scope: {
            label: _selected_metrics(run.records, scope)
            for label, run in loaded.items()
        }
        for scope in SCOPES
    }
    comparisons = {
        label: {
            scope: _comparison_bootstrap(
                loaded["aligned"],
                loaded[label],
                candidate_label=label,
                scope=scope,
                iterations=bootstrap_iterations,
                seed=seed,
            )
            for scope in SCOPES
        }
        for label in XIC_INTERVENTIONS[1:]
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "development_validation_xic_counterfactual",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "shared_provenance": shared,
        "sources": {
            label: {
                "generation_root": str(run.generation_root),
                "generation_report_sha256": run.generation_sha256,
                "evaluation_root": str(run.evaluation_root),
                "evaluation_report_sha256": run.evaluation_sha256,
                "evaluation_records_sha256": run.records_sha256,
                "xic_intervention": run.intervention,
            }
            for label, run in loaded.items()
        },
        "observed_metrics": observed,
        "cross_language_consistency": {
            label: run.evaluation_report.get("cross_language_consistency")
            for label, run in loaded.items()
        },
        "paired_group_bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "confidence_level": 0.95,
            "resampling_unit": "source_group",
            "independent_units": shared["validation_source_groups"],
            "task_specific_independent_units_recorded_with_each_task": True,
            "comparisons": comparisons,
        },
        "contracts": {
            "same_base_model": True,
            "same_fusion_checkpoint": True,
            "same_prompt_and_answer_artifacts": True,
            "same_validation_record_identities": True,
            "language_variants_not_independent": True,
            "paired_source_group_bootstrap": True,
            "task_specific_source_group_eligibility": True,
            "internal_test_accessed": False,
        },
        "interpretation_limits": [
            "This is a single-training-seed development validation analysis.",
            "English and Chinese prompts are paired views of the same scientific assets.",
            "Task-specific bootstrap populations exclude groups with no labels for that task.",
            "The sealed internal-test split remains unopened.",
            "Intervention effects measure reliance on XIC under this checkpoint, not deployment validity.",
        ],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        report_path = staging / "xic_intervention_analysis.json"
        markdown_path = staging / "xic_intervention_analysis.md"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (report_path, markdown_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "xic_intervention_analysis.json"
    return XicInterventionAnalysisResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        markdown_path=output_dir / "xic_intervention_analysis.md",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
