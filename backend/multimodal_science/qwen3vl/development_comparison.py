"""Build a provenance-bound comparison of Qwen and specialist development runs."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.evaluation import BILINGUAL_EVALUATION_REPORT_SCHEMA
from multimodal_science.qwen3vl.inference import GENERATION_REPORT_SCHEMA


REPORT_SCHEMA = "chrompeak-cross-family-development-comparison-v1"
LANGUAGES = ("en", "zh-CN")
CLASSIFICATION_TASKS = ("peak_presence", "peak_presence_metadata")
CLASSIFICATION_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "macro_f1",
    "mcc",
    "auroc",
    "auprc",
    "false_positive_rate",
)


@dataclass(frozen=True)
class CrossFamilyComparisonResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    markdown_path: Path
    manifest_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Report must contain an object: {path}")
    return payload


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    return float(value)


def _classification(metrics: dict[str, Any], label: str) -> dict[str, float]:
    return {
        field: _number(metrics.get(field), f"{label} {field}")
        for field in CLASSIFICATION_FIELDS
    }


def _validate_development(report: dict[str, Any], label: str) -> None:
    _require(
        report.get("development_comparison_eligible") is True,
        f"{label} is not development-comparison eligible",
    )
    _require(
        report.get("final_benchmark_eligible") is False,
        f"{label} must not claim final-benchmark eligibility",
    )
    _require(
        report.get("internal_test_accessed") is False,
        f"{label} accessed the sealed internal test",
    )


def _validate_qwen_pair(
    generation: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    generation_sha256: str,
    dataset_report_sha256: str,
    validation_assets: int,
    validation_source_groups: int,
    adapter_expected: bool,
    label: str,
) -> None:
    _require(
        generation.get("schema_version") == GENERATION_REPORT_SCHEMA,
        f"Unexpected {label} generation schema",
    )
    _require(
        evaluation.get("schema_version") == BILINGUAL_EVALUATION_REPORT_SCHEMA,
        f"Unexpected {label} evaluation schema",
    )
    _validate_development(evaluation, label)
    _require(
        generation.get("development_comparison_candidate") is True,
        f"{label} generation is not a complete comparison candidate",
    )
    _require(
        generation.get("final_benchmark_eligible") is False
        and generation.get("internal_test_accessed") is False,
        f"{label} generation has an invalid evaluation scope",
    )
    _require(
        evaluation.get("prediction_generation_provenance_verified") is True,
        f"{label} generation provenance was not verified",
    )
    _require(
        evaluation.get("language_variants_are_not_independent_source_assets") is True,
        f"{label} does not protect against bilingual duplicate counting",
    )

    inputs = _object(evaluation.get("inputs"), f"{label} evaluation inputs")
    counts = _object(evaluation.get("counts"), f"{label} evaluation counts")
    source = _object(generation.get("source"), f"{label} generation source")
    model = _object(generation.get("model"), f"{label} generation model")
    generation_counts = _object(
        generation.get("counts"), f"{label} generation counts"
    )
    _require(model.get("identity_immutable") is True, f"{label} base model is mutable")
    adapter = model.get("adapter")
    _require(
        (adapter is not None) is adapter_expected,
        f"{label} adapter identity does not match its comparison role",
    )
    if adapter_expected:
        adapter_payload = _object(adapter, f"{label} adapter")
        _require(
            adapter_payload.get("development_training_complete") is True,
            f"{label} adapter training is incomplete",
        )
    _require(
        inputs.get("generation_report_sha256") == generation_sha256,
        f"{label} evaluation does not bind the supplied generation report",
    )
    _require(
        generation_counts.get("predictions") == counts.get("predictions"),
        f"{label} generation/evaluation prediction count mismatch",
    )
    for field in ("instruction_report_sha256", "validation_prompts_sha256"):
        _require(
            source.get(field) == inputs.get(field),
            f"{label} generation/evaluation {field} mismatch",
        )
    _require(
        inputs.get("source_dataset_report_sha256") == dataset_report_sha256
        and source.get("source_dataset_report_sha256") == dataset_report_sha256,
        f"{label} does not share the specialist Dataset",
    )
    _require(
        counts.get("independent_validation_assets") == validation_assets,
        f"{label} validation asset count mismatch",
    )
    _require(
        counts.get("validation_source_groups") == validation_source_groups,
        f"{label} validation source-group count mismatch",
    )
    by_language = _object(counts.get("by_language"), f"{label} language counts")
    _require(
        set(by_language) == set(LANGUAGES)
        and by_language["en"] == by_language["zh-CN"],
        f"{label} bilingual prompt coverage is not paired",
    )


def _qwen_rows(evaluation: dict[str, Any], label: str) -> dict[str, Any]:
    metrics_by_language = _object(
        evaluation.get("metrics_by_language"), f"{label} metrics by language"
    )
    rows: dict[str, Any] = {}
    for language in LANGUAGES:
        tasks = _object(metrics_by_language.get(language), f"{label} {language} tasks")
        classification = {}
        for task in CLASSIFICATION_TASKS:
            task_metrics = _object(tasks.get(task), f"{label} {language} {task}")
            classification[task] = _classification(
                _object(task_metrics.get("classification"), f"{label} classification"),
                f"{label} {language} {task}",
            )
        grounding = _object(
            _object(tasks.get("peak_grounding"), f"{label} grounding").get("grounding"),
            f"{label} grounding metrics",
        )
        qc = _object(tasks.get("scientific_qc"), f"{label} scientific QC")
        rows[language] = {
            "classification": classification,
            "grounding": {
                "mean_bbox_iou_all": _number(
                    grounding.get("mean_bbox_iou_all"), f"{label} grounding mean IoU"
                ),
                "iou_at_0_5_rate_all": _number(
                    grounding.get("iou_at_0_5_rate_all"), f"{label} grounding IoU@0.5"
                ),
                "x_boundary_mae_pixels_schema_valid": _number(
                    grounding.get("x_boundary_mae_pixels_schema_valid"),
                    f"{label} grounding boundary MAE",
                ),
            },
            "scientific_qc_exact_match": _number(
                qc.get("exact_match_rate"), f"{label} QC exact match"
            ),
        }
    return rows


def _deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for language in LANGUAGES:
        classification = {}
        for task in CLASSIFICATION_TASKS:
            classification[task] = {
                field: candidate[language]["classification"][task][field]
                - reference[language]["classification"][task][field]
                for field in CLASSIFICATION_FIELDS
            }
        result[language] = {
            "classification": classification,
            "grounding": {
                field: candidate[language]["grounding"][field]
                - reference[language]["grounding"][field]
                for field in candidate[language]["grounding"]
            },
            "scientific_qc_exact_match": (
                candidate[language]["scientific_qc_exact_match"]
                - reference[language]["scientific_qc_exact_match"]
            ),
        }
    return result


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-family development comparison",
        "",
        "Qwen rows are reported per language. English and Chinese prompts describe the same "
        "1,815 assets and are never counted as independent scientific samples.",
        "",
        "## Qwen classification",
        "",
        "| Model | Language | Task | Accuracy | Balanced accuracy | Macro-F1 | "
        "MCC | Recall | Specificity | FPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, model_label in (("zero_shot", "Qwen3-VL zero-shot"), ("lora", "Qwen3-VL LoRA")):
        for language in LANGUAGES:
            for task in CLASSIFICATION_TASKS:
                row = report["qwen"][model_key][language]["classification"][task]
                lines.append(
                    f"| {model_label} | {language} | {task} | {_fmt(row['accuracy'])} | "
                    f"{_fmt(row['balanced_accuracy'])} | {_fmt(row['macro_f1'])} | "
                    f"{_fmt(row['mcc'])} | {_fmt(row['recall'])} | "
                    f"{_fmt(row['specificity'])} | {_fmt(row['false_positive_rate'])} |"
                )
    lines.extend(
        [
            "",
            "## Qwen grounding and QC",
            "",
            "| Model | Language | Mean bbox IoU | IoU >= 0.5 | "
            "X-boundary MAE (px) | QC exact match |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_key, model_label in (("zero_shot", "Qwen3-VL zero-shot"), ("lora", "Qwen3-VL LoRA")):
        for language in LANGUAGES:
            row = report["qwen"][model_key][language]
            grounding = row["grounding"]
            lines.append(
                f"| {model_label} | {language} | {_fmt(grounding['mean_bbox_iou_all'])} | "
                f"{_fmt(grounding['iou_at_0_5_rate_all'])} | "
                f"{_fmt(grounding['x_boundary_mae_pixels_schema_valid'])} | "
                f"{_fmt(row['scientific_qc_exact_match'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Specialist rows and Qwen rows share the same leakage-safe assets, "
            "but their output contracts differ.",
            "- COCO AP is detector-only; Qwen bbox IoU must not be presented as COCO AP.",
            "- This is a single-seed validation comparison, not a sealed final benchmark.",
            "- Validation results may diagnose failure modes but must not be used "
            "repeatedly to tune the protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def build_cross_family_development_comparison(
    *,
    specialist_comparison_path: Path,
    zero_shot_generation_path: Path,
    zero_shot_evaluation_path: Path,
    lora_generation_path: Path,
    lora_evaluation_path: Path,
    output_dir: Path,
) -> CrossFamilyComparisonResult:
    """Compare zero-shot and LoRA without mixing prompts with scientific samples."""

    paths = {
        "specialist_comparison": specialist_comparison_path.resolve(),
        "zero_shot_generation": zero_shot_generation_path.resolve(),
        "zero_shot_evaluation": zero_shot_evaluation_path.resolve(),
        "lora_generation": lora_generation_path.resolve(),
        "lora_evaluation": lora_evaluation_path.resolve(),
    }
    payloads = {name: _read_object(path) for name, path in paths.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}

    specialist = payloads["specialist_comparison"]
    _require(
        specialist.get("schema_version") == "chrompeak-development-ablation-v1",
        "Unexpected specialist comparison schema",
    )
    _validate_development(specialist, "specialist comparison")
    dataset = _object(specialist.get("dataset"), "specialist Dataset")
    dataset_sha256 = str(dataset.get("dataset_report_sha256"))
    validation_assets = int(dataset.get("validation_assets", 0))
    validation_groups = int(dataset.get("validation_source_groups", 0))
    _require(len(dataset_sha256) == 64, "Invalid specialist Dataset hash")
    _require(validation_assets > 0 and validation_groups > 0, "Invalid validation scope")

    for prefix, adapter_expected in (("zero_shot", False), ("lora", True)):
        _validate_qwen_pair(
            payloads[f"{prefix}_generation"],
            payloads[f"{prefix}_evaluation"],
            generation_sha256=hashes[f"{prefix}_generation"],
            dataset_report_sha256=dataset_sha256,
            validation_assets=validation_assets,
            validation_source_groups=validation_groups,
            adapter_expected=adapter_expected,
            label=prefix.replace("_", " "),
        )

    zero_inputs = _object(payloads["zero_shot_evaluation"].get("inputs"), "zero-shot inputs")
    lora_inputs = _object(payloads["lora_evaluation"].get("inputs"), "LoRA inputs")
    for field in (
        "instruction_report_sha256",
        "validation_prompts_sha256",
        "validation_answers_sha256",
        "instruction_manifest_sha256",
    ):
        _require(
            zero_inputs.get(field) == lora_inputs.get(field),
            f"Qwen runs do not share {field}",
        )
    zero_model = _object(payloads["zero_shot_generation"].get("model"), "zero-shot model")
    lora_model = _object(payloads["lora_generation"].get("model"), "LoRA model")
    _require(
        zero_model.get("artifact_sha256") == lora_model.get("artifact_sha256"),
        "Qwen runs do not share one immutable base model",
    )

    zero_rows = _qwen_rows(payloads["zero_shot_evaluation"], "zero shot")
    lora_rows = _qwen_rows(payloads["lora_evaluation"], "LoRA")
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "validation_cross_family_development_comparison",
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
        "dataset": {
            "dataset_report_sha256": dataset_sha256,
            "asset_index_sha256": dataset.get("asset_index_sha256"),
            "validation_assets": validation_assets,
            "validation_source_groups": validation_groups,
            "bilingual_prompts_are_paired_views": True,
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "specialist_context": {
            "fixed_threshold_0_5": specialist.get("fixed_threshold_0_5"),
            "localization": specialist.get("localization"),
            "detector_only_coco": specialist.get("detector_only_coco"),
        },
        "qwen": {"zero_shot": zero_rows, "lora": lora_rows},
        "lora_minus_zero_shot": _deltas(zero_rows, lora_rows),
        "cross_language_consistency": {
            "zero_shot": payloads["zero_shot_evaluation"].get("cross_language_consistency"),
            "lora": payloads["lora_evaluation"].get("cross_language_consistency"),
        },
        "interpretation_limits": [
            "English and Chinese prompts are paired views of the same assets, not "
            "independent samples.",
            "Specialist and Qwen output contracts differ; only like-for-like metrics "
            "are shown together.",
            "COCO AP is detector-only and is not inferred for Qwen grounding.",
            "This single-seed validation result is not a final benchmark.",
            "The sealed internal-test split remains unopened.",
        ],
    }

    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Comparison output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        report_path = staging / "cross_family_development_report.json"
        markdown_path = staging / "cross_family_development_table.md"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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

    final_report = output_dir / "cross_family_development_report.json"
    return CrossFamilyComparisonResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        markdown_path=output_dir / "cross_family_development_table.md",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
