"""Build the provenance-bound multimodal development dossier."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.development_comparison import (
    REPORT_SCHEMA as CROSS_FAMILY_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.evaluation import (
    BILINGUAL_EVALUATION_RECORD_SCHEMA,
    BILINGUAL_EVALUATION_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.fusion_matrix_analysis import (
    MATRIX_CONFIGURATIONS,
    REPORT_SCHEMA as FUSION_MATRIX_REPORT_SCHEMA,
)
from multimodal_science.qwen3vl.xic_intervention_analysis import (
    REPORT_SCHEMA as XIC_INTERVENTION_REPORT_SCHEMA,
)


REPORT_SCHEMA = "chrompeak-multimodal-development-dossier-v1"
FAILURE_RECORD_SCHEMA = "chrompeak-fusion-failure-case-v1"
LANGUAGES = ("en", "zh-CN")
TASKS = (
    "peak_presence",
    "peak_presence_metadata",
    "peak_grounding",
    "scientific_qc",
)
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
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DevelopmentDossierResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    main_table_path: Path
    failure_analysis_path: Path
    failure_records_path: Path
    manifest_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), label)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                rows.append(_object(json.loads(line), f"{label} line {line_number}"))
    _require(rows, f"{label} is empty")
    return rows


def _safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    _require(
        value
        and not pure.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"Unsafe {label}: {value}",
    )
    return Path(*pure.parts)


def _resolve_artifact(root: Path, relative: Path, label: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its artifact root") from error
    _require(path.is_file(), f"Missing {label}: {path}")
    return path


def _verify_manifest(root: Path, required: set[str], label: str) -> str:
    root = root.resolve()
    manifest_path = root / "artifact_manifest.sha256"
    _require(manifest_path.is_file(), f"Missing {label} manifest")
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"Malformed {label} manifest line {line_number}")
        digest, relative_text = parts
        relative_text = relative_text.lstrip(" *")
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad {label} manifest digest")
        relative = _safe_relative(relative_text, f"{label} manifest path")
        relative_posix = relative.as_posix()
        _require(relative_posix not in listed, f"Duplicate {label} manifest path")
        artifact = _resolve_artifact(root, relative, f"{label} artifact {relative_posix}")
        _require(
            sha256_file(artifact) == digest,
            f"{label} artifact hash mismatch: {relative_posix}",
        )
        listed.add(relative_posix)
    _require(required <= listed, f"{label} manifest lacks {sorted(required - listed)}")
    return sha256_file(manifest_path)


def _validate_development_report(
    report: dict[str, Any], *, schema: str, label: str
) -> None:
    _require(report.get("schema_version") == schema, f"Unexpected {label} schema")
    _require(
        report.get("development_comparison_eligible") is True,
        f"{label} is not development-comparison eligible",
    )
    _require(
        report.get("final_benchmark_eligible") is False,
        f"{label} improperly claims final-benchmark eligibility",
    )
    _require(
        report.get("internal_test_accessed") is False,
        f"{label} accessed the sealed internal test",
    )


def _statistics(payload: Any, label: str) -> dict[str, float]:
    values = _object(payload, label)
    return {
        "mean": _number(values.get("mean"), f"{label} mean"),
        "sample_standard_deviation": _number(
            values.get("sample_standard_deviation"), f"{label} sample SD"
        ),
    }


def _validate_metric_statistics(payload: Any, label: str) -> None:
    values = _object(payload, label)
    values_by_seed = _object(values.get("values_by_seed"), f"{label} values by seed")
    _require(
        set(values_by_seed) == {"17", "29", "43"},
        f"{label} does not contain the three declared training seeds",
    )
    observed = [
        _number(values_by_seed[seed], f"{label} seed {seed}")
        for seed in ("17", "29", "43")
    ]
    _require(values.get("n") == 3, f"{label} replicate count is not three")
    _require(
        values.get("standard_deviation_ddof") == 1,
        f"{label} does not use sample standard deviation",
    )
    expected = {
        "mean": statistics.fmean(observed),
        "sample_standard_deviation": statistics.stdev(observed),
        "minimum": min(observed),
        "maximum": max(observed),
    }
    for field, expected_value in expected.items():
        actual = _number(values.get(field), f"{label} {field}")
        _require(
            math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12),
            f"{label} {field} does not match its per-seed values",
        )


def _validate_matrix_contract(matrix: dict[str, Any]) -> None:
    rows = _object(matrix.get("matrix"), "fusion matrix")
    _require(
        set(rows) == set(MATRIX_CONFIGURATIONS),
        "Fusion matrix does not contain exactly the five declared cells",
    )
    required_hashes = {
        "training_report",
        "training_manifest",
        "generation_report",
        "generation_manifest",
        "evaluation_report",
        "evaluation_manifest",
        "evaluation_records",
    }
    for label, (tokens, seed) in MATRIX_CONFIGURATIONS.items():
        row = _object(rows.get(label), f"matrix row {label}")
        _require(
            row.get("sensor_tokens") == tokens and row.get("training_seed") == seed,
            f"Matrix configuration drift: {label}",
        )
        hashes = _object(row.get("sha256"), f"matrix row {label} hashes")
        _require(required_hashes <= set(hashes), f"Matrix row {label} lacks hashes")
        for name in required_hashes:
            digest = hashes.get(name)
            _require(
                isinstance(digest, str) and bool(_HEX_64.fullmatch(digest)),
                f"Invalid matrix row {label} {name} hash",
            )

    aggregate = _object(
        matrix.get("primary_seed_aggregate"), "primary seed aggregate"
    )
    metric_contract = {
        "peak_presence": (
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "false_positive_rate",
        ),
        "peak_presence_metadata": (
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "false_positive_rate",
        ),
        "peak_grounding": ("mean_bbox_iou_all", "iou_at_0_5_rate_all"),
        "scientific_qc": ("exact_match_rate",),
    }
    _require(set(aggregate) == {"overall", *LANGUAGES}, "Matrix scopes are incomplete")
    for scope, tasks in aggregate.items():
        tasks = _object(tasks, f"matrix aggregate {scope}")
        for task, metrics in metric_contract.items():
            task_metrics = _object(tasks.get(task), f"matrix aggregate {scope}/{task}")
            for metric in metrics:
                _validate_metric_statistics(
                    task_metrics.get(metric), f"matrix aggregate {scope}/{task}/{metric}"
                )

    ablation = _object(matrix.get("token_ablation"), "token ablation")
    _require(
        ablation.get("training_seed") == 17
        and ablation.get("baseline_sensor_tokens") == 4,
        "Token-ablation baseline drift",
    )
    ablation_runs = _object(ablation.get("runs"), "token-ablation runs")
    _require(set(ablation_runs) == {"1", "4", "8"}, "Token ablation is incomplete")
    for token_text in ("1", "4", "8"):
        row = _object(ablation_runs[token_text], f"token-ablation row {token_text}")
        _require(
            row.get("sensor_tokens") == int(token_text)
            and row.get("training_seed") == 17,
            f"Token-ablation row drift: {token_text}",
        )

    contracts = _object(matrix.get("contracts"), "matrix contracts")
    for field in (
        "same_leakage_safe_validation",
        "same_base_model",
        "same_training_data_and_initial_adapter",
        "three_independent_primary_training_seeds",
        "token_ablation_uses_seed17",
        "evaluation_metrics_recomputed_from_records",
        "evaluation_record_identities_match_across_runs",
    ):
        _require(contracts.get(field) is True, f"Matrix contract failed: {field}")


def _bound_cross_family_source(
    cross: dict[str, Any], name: str, label: str
) -> dict[str, Any]:
    source = _object(
        _object(cross.get("sources"), "cross-family sources").get(name),
        f"cross-family source {name}",
    )
    path_text = source.get("path")
    expected_sha256 = source.get("sha256")
    _require(isinstance(path_text, str) and path_text, f"Missing {label} path")
    _require(
        isinstance(expected_sha256, str) and bool(_HEX_64.fullmatch(expected_sha256)),
        f"Invalid {label} hash",
    )
    path = Path(path_text).resolve()
    _require(path.is_file(), f"Missing bound {label}: {path}")
    _require(sha256_file(path) == expected_sha256, f"Bound {label} hash mismatch")
    return _read_object(path, label)


def _qwen_source_rows(evaluation: dict[str, Any], label: str) -> dict[str, Any]:
    metrics_by_language = _object(
        evaluation.get("metrics_by_language"), f"{label} metrics by language"
    )
    result: dict[str, Any] = {}
    for language in LANGUAGES:
        tasks = _object(metrics_by_language.get(language), f"{label} {language}")
        classification = {}
        for task in ("peak_presence", "peak_presence_metadata"):
            metrics = _object(
                _object(tasks.get(task), f"{label} {language} {task}").get(
                    "classification"
                ),
                f"{label} {language} {task} classification",
            )
            classification[task] = {
                field: _number(
                    metrics.get(field), f"{label} {language} {task} {field}"
                )
                for field in CLASSIFICATION_FIELDS
            }
        grounding = _object(
            _object(tasks.get("peak_grounding"), f"{label} grounding").get(
                "grounding"
            ),
            f"{label} grounding metrics",
        )
        qc = _object(tasks.get("scientific_qc"), f"{label} scientific QC")
        result[language] = {
            "classification": classification,
            "grounding": {
                field: _number(
                    grounding.get(field), f"{label} {language} grounding {field}"
                )
                for field in (
                    "mean_bbox_iou_all",
                    "iou_at_0_5_rate_all",
                    "x_boundary_mae_pixels_schema_valid",
                )
            },
            "scientific_qc_exact_match": _number(
                qc.get("exact_match_rate"), f"{label} {language} QC"
            ),
        }
    return result


def _validate_cross_family_bindings(
    cross: dict[str, Any],
    *,
    dataset_sha256: str,
    matrix_shared: dict[str, Any],
    selected_evaluation_inputs: dict[str, Any],
) -> dict[str, str]:
    specialist = _bound_cross_family_source(
        cross, "specialist_comparison", "specialist comparison"
    )
    _validate_development_report(
        specialist,
        schema="chrompeak-development-ablation-v1",
        label="specialist comparison",
    )
    specialist_dataset = _object(specialist.get("dataset"), "specialist Dataset")
    _require(
        specialist_dataset.get("dataset_report_sha256") == dataset_sha256,
        "Bound specialist comparison uses a different Dataset",
    )
    specialist_context = _object(cross.get("specialist_context"), "specialist context")
    for source_field, copied_field in (
        ("fixed_threshold_0_5", "fixed_threshold_0_5"),
        ("localization", "localization"),
        ("detector_only_coco", "detector_only_coco"),
    ):
        expected = (
            specialist.get("detector_only_coco")
            if source_field == "detector_only_coco"
            else specialist.get(source_field)
        )
        _require(
            specialist_context.get(copied_field) == expected,
            f"Copied specialist field drift: {copied_field}",
        )

    payloads = {
        name: _bound_cross_family_source(cross, name, name.replace("_", " "))
        for name in (
            "zero_shot_generation",
            "zero_shot_evaluation",
            "lora_generation",
            "lora_evaluation",
        )
    }
    cross_qwen = _object(cross.get("qwen"), "cross-family Qwen metrics")
    source_descriptors = _object(cross.get("sources"), "cross-family sources")
    for prefix in ("zero_shot", "lora"):
        evaluation = payloads[f"{prefix}_evaluation"]
        generation = payloads[f"{prefix}_generation"]
        _validate_development_report(
            evaluation,
            schema=BILINGUAL_EVALUATION_REPORT_SCHEMA,
            label=f"{prefix} evaluation",
        )
        _require(
            evaluation.get("prediction_generation_provenance_verified") is True,
            f"{prefix} evaluation generation provenance is unverified",
        )
        _require(
            cross_qwen.get(prefix)
            == _qwen_source_rows(evaluation, prefix.replace("_", " ")),
            f"Cross-family copied {prefix} metrics drift from their bound source",
        )
        inputs = _object(evaluation.get("inputs"), f"{prefix} evaluation inputs")
        generation_descriptor = _object(
            source_descriptors.get(f"{prefix}_generation"),
            f"{prefix} generation descriptor",
        )
        _require(
            inputs.get("generation_report_sha256")
            == generation_descriptor.get("sha256"),
            f"{prefix} generation/evaluation binding drift",
        )
        generation_source = _object(
            generation.get("source"), f"{prefix} generation source"
        )
        _require(
            generation_source.get("source_dataset_report_sha256") == dataset_sha256,
            f"{prefix} generation Dataset drift",
        )
        for field in ("instruction_report_sha256", "validation_prompts_sha256"):
            _require(
                generation_source.get(field) == matrix_shared.get(field),
                f"{prefix} generation {field} drift",
            )

    comparison_fields = (
        "instruction_report_sha256",
        "validation_prompts_sha256",
        "validation_answers_sha256",
    )
    for field in comparison_fields:
        expected = matrix_shared.get(field)
        _require(
            isinstance(expected, str) and bool(_HEX_64.fullmatch(expected)),
            f"Invalid matrix {field}",
        )
        _require(
            selected_evaluation_inputs.get(field) == expected,
            f"Selected fusion evaluation {field} drift",
        )
        for prefix in ("zero_shot", "lora"):
            inputs = _object(
                payloads[f"{prefix}_evaluation"].get("inputs"),
                f"{prefix} evaluation inputs",
            )
            _require(inputs.get(field) == expected, f"{prefix} {field} drift")
            _require(
                inputs.get("source_dataset_report_sha256") == dataset_sha256,
                f"{prefix} Dataset drift",
            )

    instruction_manifest = selected_evaluation_inputs.get(
        "instruction_manifest_sha256"
    )
    _require(
        isinstance(instruction_manifest, str)
        and bool(_HEX_64.fullmatch(instruction_manifest)),
        "Invalid selected instruction-manifest hash",
    )
    for prefix in ("zero_shot", "lora"):
        inputs = _object(
            payloads[f"{prefix}_evaluation"].get("inputs"),
            f"{prefix} evaluation inputs",
        )
        _require(
            inputs.get("instruction_manifest_sha256") == instruction_manifest,
            f"{prefix} instruction-manifest drift",
        )

    model_hashes = []
    for prefix in ("zero_shot", "lora"):
        model = _object(
            payloads[f"{prefix}_generation"].get("model"),
            f"{prefix} generation model",
        )
        model_hashes.append(model.get("artifact_sha256"))
    expected_model = matrix_shared.get("base_model_artifact_sha256")
    _require(
        isinstance(expected_model, str) and bool(_HEX_64.fullmatch(expected_model)),
        "Invalid matrix base-model hash",
    )
    _require(
        model_hashes == [expected_model, expected_model],
        "Qwen comparison runs do not share the matrix base model",
    )
    return {
        "dataset_report_sha256": dataset_sha256,
        "base_model_artifact_sha256": expected_model,
        "instruction_report_sha256": str(matrix_shared["instruction_report_sha256"]),
        "validation_prompts_sha256": str(
            matrix_shared["validation_prompts_sha256"]
        ),
        "validation_answers_sha256": str(
            matrix_shared["validation_answers_sha256"]
        ),
        "instruction_manifest_sha256": str(instruction_manifest),
    }


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


def _record_identity_sha256(rows: Iterable[dict[str, Any]]) -> str:
    identities = [_record_identity(row) for row in rows]
    _require(
        len(set(identities)) == len(identities),
        "Evaluation-record identities are not unique",
    )
    identities.sort()
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_intervention_binding(
    intervention: dict[str, Any],
    *,
    selected_records_identity_sha256: str,
    selected_evaluation_inputs: dict[str, Any],
    matrix_shared: dict[str, Any],
) -> dict[str, str]:
    source = _object(
        _object(intervention.get("sources"), "intervention sources").get("aligned"),
        "aligned intervention source",
    )
    root_text = source.get("evaluation_root")
    _require(isinstance(root_text, str) and root_text, "Missing aligned evaluation root")
    root = Path(root_text).resolve()
    manifest_sha256 = _verify_manifest(
        root,
        {"qwen_evaluation_report.json", "evaluation_records.jsonl"},
        "aligned intervention evaluation",
    )
    report_path = root / "qwen_evaluation_report.json"
    records_path = root / "evaluation_records.jsonl"
    _require(
        sha256_file(report_path) == source.get("evaluation_report_sha256"),
        "Aligned intervention evaluation-report hash mismatch",
    )
    _require(
        sha256_file(records_path) == source.get("evaluation_records_sha256"),
        "Aligned intervention evaluation-record hash mismatch",
    )
    aligned = _read_object(report_path, "aligned intervention evaluation")
    aligned_records = _read_jsonl(records_path, "aligned intervention records")
    _validate_development_report(
        aligned,
        schema=BILINGUAL_EVALUATION_REPORT_SCHEMA,
        label="aligned intervention evaluation",
    )
    _require(
        _record_identity_sha256(aligned_records)
        == selected_records_identity_sha256,
        "Intervention and matrix evaluation-record identities differ",
    )
    aligned_inputs = _object(aligned.get("inputs"), "aligned intervention inputs")
    for field in (
        "instruction_report_sha256",
        "source_dataset_report_sha256",
        "validation_prompts_sha256",
        "validation_answers_sha256",
        "instruction_manifest_sha256",
    ):
        _require(
            aligned_inputs.get(field) == selected_evaluation_inputs.get(field),
            f"Aligned intervention {field} drift",
        )
    intervention_shared = _object(
        intervention.get("shared_provenance"), "intervention provenance"
    )
    for field in (
        "base_model_artifact_sha256",
        "instruction_report_sha256",
        "validation_prompts_sha256",
        "validation_answers_sha256",
    ):
        _require(
            intervention_shared.get(field) == matrix_shared.get(field),
            f"Intervention/matrix {field} drift",
        )
    return {
        "evaluation_report_sha256": sha256_file(report_path),
        "evaluation_records_sha256": sha256_file(records_path),
        "evaluation_manifest_sha256": manifest_sha256,
        "evaluation_record_identity_sha256": selected_records_identity_sha256,
    }


def _single(value: Any, label: str) -> dict[str, float | None]:
    return {"mean": _number(value, label), "sample_standard_deviation": None}


def _qwen_row(
    model: dict[str, Any], language: str, *, model_key: str, model_label: str
) -> dict[str, Any]:
    language_row = _object(model.get(language), f"{model_label} {language}")
    classification = _object(
        _object(language_row.get("classification"), f"{model_label} classification").get(
            "peak_presence"
        ),
        f"{model_label} peak presence",
    )
    grounding = _object(language_row.get("grounding"), f"{model_label} grounding")
    return {
        "model_key": model_key,
        "model": model_label,
        "scope": language,
        "replicates": 1,
        "classification": {
            metric: _single(classification.get(metric), f"{model_label} {metric}")
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "grounding": {
            "mean_iou": _single(
                grounding.get("mean_bbox_iou_all"), f"{model_label} mean IoU"
            ),
            "iou_at_0_5": _single(
                grounding.get("iou_at_0_5_rate_all"), f"{model_label} IoU@0.5"
            ),
        },
        "scientific_qc_exact_match": _single(
            language_row.get("scientific_qc_exact_match"), f"{model_label} QC"
        ),
    }


def _fusion_row(
    matrix: dict[str, Any], language: str
) -> dict[str, Any]:
    aggregate = _object(
        _object(matrix.get("primary_seed_aggregate"), "fusion primary aggregate").get(
            language
        ),
        f"fusion aggregate {language}",
    )
    presence = _object(aggregate.get("peak_presence"), "fusion peak presence")
    grounding = _object(aggregate.get("peak_grounding"), "fusion grounding")
    qc = _object(aggregate.get("scientific_qc"), "fusion scientific QC")
    return {
        "model_key": "image_xic_fusion",
        "model": "Qwen3-VL image + XIC fusion",
        "scope": language,
        "replicates": 3,
        "classification": {
            metric: _statistics(presence.get(metric), f"fusion {language} {metric}")
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "grounding": {
            "mean_iou": _statistics(
                grounding.get("mean_bbox_iou_all"), f"fusion {language} mean IoU"
            ),
            "iou_at_0_5": _statistics(
                grounding.get("iou_at_0_5_rate_all"), f"fusion {language} IoU@0.5"
            ),
        },
        "scientific_qc_exact_match": _statistics(
            qc.get("exact_match_rate"), f"fusion {language} QC"
        ),
    }


def _specialist_row(
    specialist: dict[str, Any], model_key: str, model_label: str
) -> dict[str, Any]:
    fixed = _object(
        _object(specialist.get("fixed_threshold_0_5"), "specialist fixed metrics").get(
            model_key
        ),
        f"{model_label} fixed metrics",
    )
    localization = _object(
        _object(specialist.get("localization"), "specialist localization").get(
            model_key
        ),
        f"{model_label} localization",
    )
    return {
        "model_key": model_key,
        "model": model_label,
        "scope": "language-neutral",
        "replicates": 1,
        "classification": {
            metric: _single(fixed.get(metric), f"{model_label} {metric}")
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "false_positive_rate",
            )
        },
        "grounding": {
            "mean_iou": _single(
                localization.get("mean_iou"), f"{model_label} mean IoU"
            ),
            "iou_at_0_5": None,
            "metric": localization.get("metric"),
        },
        "scientific_qc_exact_match": None,
    }


def _bbox_iou(first: list[float] | None, second: list[float] | None) -> float:
    if first is None or second is None:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _prediction_signature(row: dict[str, Any]) -> Any:
    if not row.get("schema_valid"):
        return None
    task = row.get("task")
    if task in {"peak_presence", "peak_presence_metadata"}:
        return row.get("predicted_peak_present")
    if task == "peak_grounding":
        return tuple(row.get("predicted_bbox_2d") or ())
    return row.get("predicted_qc_state"), row.get("predicted_reason")


def _row_correct(row: dict[str, Any]) -> bool:
    if not row.get("schema_valid"):
        return False
    if row.get("task") == "peak_grounding":
        return _number(row.get("bbox_iou"), "grounding bbox IoU") >= 0.5
    return bool(row.get("exact_match"))


def _pair_records(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        _require(
            row.get("schema_version") == BILINGUAL_EVALUATION_RECORD_SCHEMA,
            "Unexpected evaluation-record schema",
        )
        task = row.get("task")
        language = row.get("language")
        pair_id = row.get("pair_id")
        _require(task in TASKS, f"Unexpected task: {task}")
        _require(language in LANGUAGES, f"Unexpected language: {language}")
        _require(isinstance(pair_id, str) and pair_id, "Missing pair_id")
        pair = pairs.setdefault(pair_id, {})
        _require(language not in pair, f"Duplicate {language} row for pair {pair_id}")
        pair[language] = row
    for pair_id, pair in pairs.items():
        _require(set(pair) == set(LANGUAGES), f"Incomplete bilingual pair: {pair_id}")
        _require(
            pair["en"].get("task") == pair["zh-CN"].get("task"),
            f"Task mismatch for pair {pair_id}",
        )
        for field in ("asset_id", "group_id", "target_peak_present", "expected_bbox_2d"):
            _require(
                pair["en"].get(field) == pair["zh-CN"].get(field),
                f"Target identity mismatch for pair {pair_id}: {field}",
            )
    return pairs


def _failure_case(pair_id: str, pair: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    en = pair["en"]
    zh = pair["zh-CN"]
    task = str(en["task"])
    issues: list[str] = []
    if not en.get("schema_valid") or not zh.get("schema_valid"):
        issues.append("schema_invalid")
    signatures_match = (
        bool(en.get("schema_valid"))
        and bool(zh.get("schema_valid"))
        and _prediction_signature(en) == _prediction_signature(zh)
    )
    if not signatures_match:
        issues.append("bilingual_prediction_mismatch")
    en_correct = _row_correct(en)
    zh_correct = _row_correct(zh)
    if not en_correct or not zh_correct:
        issues.append("at_least_one_language_incorrect")

    payload: dict[str, Any] = {
        "schema_version": FAILURE_RECORD_SCHEMA,
        "pair_id": pair_id,
        "task": task,
        "asset_id": en.get("asset_id"),
        "group_id": en.get("group_id"),
        "issues": issues,
        "en": {
            "instruction_id": en.get("instruction_id"),
            "schema_valid": en.get("schema_valid"),
            "exact_match": en.get("exact_match"),
            "correct_under_task_metric": en_correct,
        },
        "zh-CN": {
            "instruction_id": zh.get("instruction_id"),
            "schema_valid": zh.get("schema_valid"),
            "exact_match": zh.get("exact_match"),
            "correct_under_task_metric": zh_correct,
        },
    }
    severity = float(len(issues))
    if task in {"peak_presence", "peak_presence_metadata"}:
        payload["target_peak_present"] = en.get("target_peak_present")
        payload["en"]["predicted_peak_present"] = en.get("predicted_peak_present")
        payload["zh-CN"]["predicted_peak_present"] = zh.get(
            "predicted_peak_present"
        )
    elif task == "peak_grounding":
        en_iou = _number(en.get("bbox_iou"), "English bbox IoU")
        zh_iou = _number(zh.get("bbox_iou"), "Chinese bbox IoU")
        prediction_iou = _bbox_iou(
            en.get("predicted_bbox_2d"), zh.get("predicted_bbox_2d")
        )
        if en_iou < 0.5 or zh_iou < 0.5:
            issues.append("localization_iou_below_0_5")
            severity += 1.0
        payload.update(
            {
                "expected_bbox_2d": en.get("expected_bbox_2d"),
                "en_bbox_iou": en_iou,
                "zh_cn_bbox_iou": zh_iou,
                "cross_language_prediction_bbox_iou": prediction_iou,
            }
        )
        payload["en"]["predicted_bbox_2d"] = en.get("predicted_bbox_2d")
        payload["zh-CN"]["predicted_bbox_2d"] = zh.get("predicted_bbox_2d")
        severity += 2.0 - en_iou - zh_iou + 1.0 - prediction_iou
    else:
        payload["en"].update(
            {
                "predicted_qc_state": en.get("predicted_qc_state"),
                "predicted_reason": en.get("predicted_reason"),
            }
        )
        payload["zh-CN"].update(
            {
                "predicted_qc_state": zh.get("predicted_qc_state"),
                "predicted_reason": zh.get("predicted_reason"),
            }
        )
    payload["severity"] = severity
    return payload if issues else None


def _failure_summary(
    pairs: dict[str, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_task: dict[str, list[tuple[str, dict[str, dict[str, Any]]]]] = {
        task: [] for task in TASKS
    }
    cases = []
    for pair_id, pair in pairs.items():
        task = str(pair["en"]["task"])
        by_task[task].append((pair_id, pair))
        case = _failure_case(pair_id, pair)
        if case is not None:
            cases.append(case)
    summary: dict[str, Any] = {}
    for task, task_pairs in by_task.items():
        count = len(task_pairs)
        _require(count > 0, f"No bilingual pairs for {task}")
        both_schema = sum(
            bool(pair["en"].get("schema_valid"))
            and bool(pair["zh-CN"].get("schema_valid"))
            for _, pair in task_pairs
        )
        consistent = sum(
            bool(pair["en"].get("schema_valid"))
            and bool(pair["zh-CN"].get("schema_valid"))
            and _prediction_signature(pair["en"]) == _prediction_signature(pair["zh-CN"])
            for _, pair in task_pairs
        )
        both_correct = sum(
            _row_correct(pair["en"])
            and _row_correct(pair["zh-CN"])
            for _, pair in task_pairs
        )
        task_summary: dict[str, Any] = {
            "pairs": count,
            "both_schema_valid_rate": both_schema / count,
            "exact_prediction_consistency_rate_all": consistent / count,
            "both_languages_correct_rate": both_correct / count,
            "either_language_failure_rate": 1.0 - both_correct / count,
        }
        if task == "peak_grounding":
            en_ious = [_number(pair["en"].get("bbox_iou"), "English IoU") for _, pair in task_pairs]
            zh_ious = [
                _number(pair["zh-CN"].get("bbox_iou"), "Chinese IoU")
                for _, pair in task_pairs
            ]
            cross = [
                _bbox_iou(
                    pair["en"].get("predicted_bbox_2d"),
                    pair["zh-CN"].get("predicted_bbox_2d"),
                )
                for _, pair in task_pairs
            ]
            task_summary["localization"] = {
                "english_mean_bbox_iou": statistics.fmean(en_ious),
                "chinese_mean_bbox_iou": statistics.fmean(zh_ious),
                "mean_cross_language_prediction_bbox_iou": statistics.fmean(cross),
                "either_language_iou_below_0_5_rate": sum(
                    en_iou < 0.5 or zh_iou < 0.5
                    for en_iou, zh_iou in zip(en_ious, zh_ious)
                )
                / count,
            }
        summary[task] = task_summary
    cases.sort(key=lambda row: (-float(row["severity"]), str(row["pair_id"])))
    return summary, cases


def _fmt_stat(value: dict[str, Any] | None) -> str:
    if value is None:
        return "—"
    mean = _number(value.get("mean"), "table mean")
    sample_sd = value.get("sample_standard_deviation")
    return f"{mean:.4f}" if sample_sd is None else f"{mean:.4f} ± {float(sample_sd):.4f}"


def _main_table_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Multimodal development main table",
        "",
        "All models use the same leakage-safe validation assets. English and Chinese "
        "rows are paired views of the same 1,815 assets, not independent samples. "
        "Fusion values are mean ± sample SD across training seeds 17/29/43; other "
        "rows are single completed runs.",
        "",
        "| Model | Scope | Presence balanced accuracy | Presence Macro-F1 | "
        "Presence MCC | Presence FPR | Mean IoU | IoU ≥ 0.5 | QC exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["main_table"]:
        classification = row["classification"]
        grounding = row["grounding"]
        lines.append(
            f"| {row['model']} | {row['scope']} | "
            f"{_fmt_stat(classification['balanced_accuracy'])} | "
            f"{_fmt_stat(classification['macro_f1'])} | "
            f"{_fmt_stat(classification['mcc'])} | "
            f"{_fmt_stat(classification['false_positive_rate'])} | "
            f"{_fmt_stat(grounding['mean_iou'])} | "
            f"{_fmt_stat(grounding.get('iou_at_0_5'))} | "
            f"{_fmt_stat(row.get('scientific_qc_exact_match'))} |"
        )
    lines.extend(
        [
            "",
            "Specialist localization contracts differ: ChromPeakFormer uses best-box "
            "IoU, SequencePeakNet uses interval IoU, and Qwen uses generated bbox IoU. "
            "COCO AP remains detector-only.",
            "",
            "The sealed internal-test split was not accessed. This is development "
            "evidence, not a final benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_markdown(report: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Fusion bilingual and localization failure analysis",
        "",
        "This page audits the predeclared four-token seed-17 fusion candidate on the "
        "leakage-safe validation set. It does not open the sealed internal test.",
        "For grounding, correctness means target IoU >= 0.5; coordinate-exact "
        "agreement remains a separate bilingual-consistency diagnostic.",
        "",
        "## Paired-language outcomes",
        "",
        "| Task | Pairs | Both schema-valid | Exact prediction consistency | "
        "Both languages correct | Either-language failure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = report["failure_summary"][task]
        lines.append(
            f"| {task} | {row['pairs']} | {row['both_schema_valid_rate']:.4f} | "
            f"{row['exact_prediction_consistency_rate_all']:.4f} | "
            f"{row['both_languages_correct_rate']:.4f} | "
            f"{row['either_language_failure_rate']:.4f} |"
        )
    grounding = report["failure_summary"]["peak_grounding"]["localization"]
    lines.extend(
        [
            "",
            "## Localization error profile",
            "",
            f"- English mean target IoU: `{grounding['english_mean_bbox_iou']:.4f}`.",
            f"- Chinese mean target IoU: `{grounding['chinese_mean_bbox_iou']:.4f}`.",
            "- Mean cross-language prediction IoU: "
            f"`{grounding['mean_cross_language_prediction_bbox_iou']:.4f}`.",
            "- Either-language IoU below 0.5: "
            f"`{grounding['either_language_iou_below_0_5_rate']:.4f}`.",
            "",
            "## Highest-severity examples",
            "",
            "| Pair | Task | Asset | Group | Issues | EN IoU | ZH IoU | Cross-language IoU |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for case in cases[:12]:
        lines.append(
            f"| {case['pair_id']} | {case['task']} | {case.get('asset_id')} | "
            f"{case.get('group_id')} | {', '.join(case['issues'])} | "
            f"{case.get('en_bbox_iou', '—')} | {case.get('zh_cn_bbox_iou', '—')} | "
            f"{case.get('cross_language_prediction_bbox_iou', '—')} |"
        )
    lines.extend(
        [
            "",
            "Cases are ranked deterministically by schema failure, bilingual disagreement, "
            "correctness, and localization severity. The full hash-bound case list is "
            "stored in `fusion_failure_cases.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_development_dossier(
    *,
    cross_family_report_path: Path,
    fusion_matrix_report_path: Path,
    xic_intervention_report_path: Path,
    selected_fusion_evaluation_root: Path,
    output_dir: Path,
) -> DevelopmentDossierResult:
    """Consolidate validated development evidence without opening internal test."""

    cross_family_report_path = cross_family_report_path.resolve()
    fusion_matrix_report_path = fusion_matrix_report_path.resolve()
    xic_intervention_report_path = xic_intervention_report_path.resolve()
    selected_fusion_evaluation_root = selected_fusion_evaluation_root.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"Dossier output already exists: {output_dir}")

    cross_manifest = _verify_manifest(
        cross_family_report_path.parent,
        {cross_family_report_path.name},
        "cross-family comparison",
    )
    matrix_manifest = _verify_manifest(
        fusion_matrix_report_path.parent,
        {fusion_matrix_report_path.name},
        "fusion matrix analysis",
    )
    intervention_manifest = _verify_manifest(
        xic_intervention_report_path.parent,
        {xic_intervention_report_path.name},
        "XIC intervention analysis",
    )
    evaluation_manifest = _verify_manifest(
        selected_fusion_evaluation_root,
        {"qwen_evaluation_report.json", "evaluation_records.jsonl"},
        "selected fusion evaluation",
    )

    cross = _read_object(cross_family_report_path, "cross-family comparison")
    matrix = _read_object(fusion_matrix_report_path, "fusion matrix analysis")
    intervention = _read_object(xic_intervention_report_path, "XIC intervention analysis")
    evaluation_path = selected_fusion_evaluation_root / "qwen_evaluation_report.json"
    evaluation = _read_object(evaluation_path, "selected fusion evaluation")
    records_path = selected_fusion_evaluation_root / "evaluation_records.jsonl"
    records = _read_jsonl(records_path, "selected fusion evaluation records")

    _validate_development_report(
        cross, schema=CROSS_FAMILY_REPORT_SCHEMA, label="cross-family comparison"
    )
    _validate_development_report(
        matrix, schema=FUSION_MATRIX_REPORT_SCHEMA, label="fusion matrix analysis"
    )
    _validate_matrix_contract(matrix)
    _validate_development_report(
        intervention,
        schema=XIC_INTERVENTION_REPORT_SCHEMA,
        label="XIC intervention analysis",
    )
    _validate_development_report(
        evaluation,
        schema=BILINGUAL_EVALUATION_REPORT_SCHEMA,
        label="selected fusion evaluation",
    )
    _require(
        evaluation.get("prediction_generation_provenance_verified") is True,
        "Selected fusion generation provenance is unverified",
    )

    dataset_sha256 = _object(cross.get("dataset"), "cross-family Dataset").get(
        "dataset_report_sha256"
    )
    _require(
        isinstance(dataset_sha256, str) and bool(_HEX_64.fullmatch(dataset_sha256)),
        "Invalid shared Dataset hash",
    )
    matrix_shared = _object(matrix.get("shared_provenance"), "matrix provenance")
    intervention_shared = _object(
        intervention.get("shared_provenance"), "intervention provenance"
    )
    evaluation_inputs = _object(evaluation.get("inputs"), "evaluation inputs")
    _require(
        dataset_sha256
        == matrix_shared.get("dataset_report_sha256")
        == intervention_shared.get("source_dataset_report_sha256")
        == evaluation_inputs.get("source_dataset_report_sha256"),
        "Development evidence does not share one Dataset",
    )
    selected_records_identity_sha256 = _record_identity_sha256(records)
    _require(
        selected_records_identity_sha256
        == matrix_shared.get("evaluation_record_identity_sha256"),
        "Selected evaluation-record identity does not match the fusion matrix",
    )
    shared_evidence = _validate_cross_family_bindings(
        cross,
        dataset_sha256=dataset_sha256,
        matrix_shared=matrix_shared,
        selected_evaluation_inputs=evaluation_inputs,
    )
    intervention_binding = _validate_intervention_binding(
        intervention,
        selected_records_identity_sha256=selected_records_identity_sha256,
        selected_evaluation_inputs=evaluation_inputs,
        matrix_shared=matrix_shared,
    )

    primary = _object(
        _object(matrix.get("matrix"), "fusion matrix").get("primary-seed17"),
        "primary seed-17 matrix row",
    )
    _require(
        primary.get("sensor_tokens") == 4 and primary.get("training_seed") == 17,
        "Selected matrix row is not the four-token seed-17 candidate",
    )
    hashes = _object(primary.get("sha256"), "primary seed-17 hashes")
    _require(
        hashes.get("evaluation_report") == sha256_file(evaluation_path),
        "Selected evaluation report is not the matrix seed-17 report",
    )
    _require(
        hashes.get("evaluation_manifest") == evaluation_manifest,
        "Selected evaluation manifest is not the matrix seed-17 manifest",
    )
    _require(
        hashes.get("evaluation_records") == sha256_file(records_path),
        "Selected evaluation records are not the matrix seed-17 records",
    )
    counts = _object(evaluation.get("counts"), "evaluation counts")
    _require(counts.get("predictions") == len(records), "Evaluation record count mismatch")
    cross_dataset = _object(cross.get("dataset"), "cross-family Dataset")
    _require(
        counts.get("independent_validation_assets")
        == cross_dataset.get("validation_assets"),
        "Validation asset-count drift",
    )
    _require(
        counts.get("validation_source_groups")
        == cross_dataset.get("validation_source_groups"),
        "Validation source-group drift",
    )

    intervention_training_sha256 = intervention_shared.get(
        "fusion_training_report_sha256"
    )
    selected_training_sha256 = hashes.get("training_report")
    intervention_manifest_sha256 = intervention_shared.get(
        "fusion_manifest_sha256"
    )
    selected_training_manifest_sha256 = hashes.get("training_manifest")
    _require(
        isinstance(intervention_training_sha256, str)
        and bool(_HEX_64.fullmatch(intervention_training_sha256)),
        "Invalid intervention training-report hash",
    )
    _require(
        isinstance(selected_training_sha256, str)
        and bool(_HEX_64.fullmatch(selected_training_sha256)),
        "Invalid selected training-report hash",
    )
    _require(
        isinstance(intervention_manifest_sha256, str)
        and bool(_HEX_64.fullmatch(intervention_manifest_sha256)),
        "Invalid intervention training-manifest hash",
    )
    _require(
        isinstance(selected_training_manifest_sha256, str)
        and bool(_HEX_64.fullmatch(selected_training_manifest_sha256)),
        "Invalid selected training-manifest hash",
    )
    intervention_matches_selected = (
        intervention_training_sha256 == selected_training_sha256
        and intervention_manifest_sha256 == selected_training_manifest_sha256
    )
    if intervention_matches_selected:
        intervention_note = (
            "The intervention analysis uses the exact selected matrix seed-17 checkpoint."
        )
    else:
        intervention_note = (
            "The intervention analysis uses an independently trained canonical seed-17 "
            "checkpoint, not the selected matrix checkpoint. It supports protocol-level "
            "modality reliance, but an exact-checkpoint intervention is still required "
            "before the selected candidate can enter sealed testing."
        )

    specialist = _object(cross.get("specialist_context"), "specialist context")
    qwen = _object(cross.get("qwen"), "cross-family Qwen rows")
    main_table = []
    for model_key, model_label in (
        ("zero_shot", "Qwen3-VL zero-shot"),
        ("lora", "Qwen3-VL image-only LoRA"),
    ):
        model = _object(qwen.get(model_key), model_label)
        for language in LANGUAGES:
            main_table.append(
                _qwen_row(
                    model, language, model_key=model_key, model_label=model_label
                )
            )
    for language in LANGUAGES:
        main_table.append(_fusion_row(matrix, language))
    main_table.extend(
        [
            _specialist_row(
                specialist, "sequence", "SequencePeakNet sequence-only"
            ),
            _specialist_row(
                specialist,
                "sequence_metadata",
                "SequencePeakNet sequence + metadata",
            ),
            _specialist_row(
                specialist, "chrompeakformer", "ChromPeakFormer image detector"
            ),
        ]
    )

    failure_summary, failure_cases = _failure_summary(_pair_records(records))
    failure_records_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in failure_cases
    )
    failure_records_sha256 = hashlib.sha256(
        failure_records_text.encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "leakage_safe_validation_development_dossier",
        "main_table": main_table,
        "fusion_reproducibility": matrix.get("primary_seed_aggregate"),
        "fusion_output_quality": matrix.get("primary_output_quality"),
        "fusion_language_gap": matrix.get("primary_en_minus_zh_cn"),
        "fusion_cross_language_consistency": matrix.get(
            "primary_cross_language_consistency"
        ),
        "sensor_token_ablation": matrix.get("token_ablation"),
        "sensor_gate_audit": matrix.get("gate_audit"),
        "causal_xic_intervention_evidence": {
            "selected_checkpoint_match": intervention_matches_selected,
            "selected_training_report_sha256": selected_training_sha256,
            "selected_training_manifest_sha256": selected_training_manifest_sha256,
            "intervention_training_report_sha256": intervention_training_sha256,
            "intervention_training_manifest_sha256": intervention_manifest_sha256,
            "note": intervention_note,
            "observed_metrics": intervention.get("observed_metrics"),
            "paired_group_bootstrap": intervention.get("paired_group_bootstrap"),
        },
        "failure_summary": failure_summary,
        "failure_cases": {
            "records": len(failure_cases),
            "ranking": "severity_descending_then_pair_id",
            "artifact": "fusion_failure_cases.jsonl",
            "sha256": failure_records_sha256,
        },
        "sources": {
            "cross_family": {
                "path": str(cross_family_report_path),
                "report_sha256": sha256_file(cross_family_report_path),
                "manifest_sha256": cross_manifest,
            },
            "fusion_matrix": {
                "path": str(fusion_matrix_report_path),
                "report_sha256": sha256_file(fusion_matrix_report_path),
                "manifest_sha256": matrix_manifest,
            },
            "xic_intervention": {
                "path": str(xic_intervention_report_path),
                "report_sha256": sha256_file(xic_intervention_report_path),
                "manifest_sha256": intervention_manifest,
            },
            "selected_fusion_evaluation": {
                "root": str(selected_fusion_evaluation_root),
                "report_sha256": sha256_file(evaluation_path),
                "records_sha256": sha256_file(records_path),
                "manifest_sha256": evaluation_manifest,
            },
            "shared_evidence": shared_evidence,
            "aligned_intervention_evaluation": intervention_binding,
        },
        "contracts": {
            "same_leakage_safe_validation_dataset": True,
            "same_hash_bound_prompt_answer_and_model_inputs": True,
            "evaluation_record_identities_match": True,
            "fusion_primary_has_three_training_seeds": True,
            "language_variants_are_paired_views": True,
            "failure_cases_come_from_hash_bound_evaluation_records": True,
            "selected_failure_candidate_is_four_token_seed17": True,
            "intervention_checkpoint_match_recorded": True,
            "internal_test_accessed": False,
        },
        "pre_internal_test_readiness": {
            "five_cell_matrix_complete": True,
            "three_seed_statistics_complete": True,
            "hash_bound_source_inputs_complete": True,
            "bilingual_failure_analysis_complete": True,
            "localization_failure_analysis_complete": True,
            "selected_checkpoint_intervention_complete": intervention_matches_selected,
            "ready": intervention_matches_selected,
        },
        "development_comparison_eligible": True,
        "final_benchmark_eligible": False,
        "internal_test_accessed": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    _require(not staging.exists(), f"Dossier staging exists: {staging}")
    staging.mkdir()
    try:
        report_path = staging / "development_dossier.json"
        main_table_path = staging / "development_main_table.md"
        failure_analysis_path = staging / "fusion_failure_analysis.md"
        failure_records_path = staging / "fusion_failure_cases.jsonl"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        main_table_path.write_text(_main_table_markdown(report), encoding="utf-8")
        failure_analysis_path.write_text(
            _failure_markdown(report, failure_cases), encoding="utf-8"
        )
        failure_records_path.write_text(failure_records_text, encoding="utf-8")
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in (
                    report_path,
                    main_table_path,
                    failure_analysis_path,
                    failure_records_path,
                )
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = output_dir / "development_dossier.json"
    return DevelopmentDossierResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        main_table_path=output_dir / "development_main_table.md",
        failure_analysis_path=output_dir / "fusion_failure_analysis.md",
        failure_records_path=output_dir / "fusion_failure_cases.jsonl",
        manifest_path=output_dir / "artifact_manifest.sha256",
    )
