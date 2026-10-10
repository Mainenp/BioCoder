"""Development inference with frozen SequencePeakNet predictions in the Qwen prompt."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.fusion_inference import _load_validation_inputs
from multimodal_science.qwen3vl.inference import (
    AdapterSpec,
    BatchGenerator,
    GenerationSettings,
    PromptRequest,
    QwenInferenceResult,
    _TransformersGenerator,
    _require,
    _verify_adapter,
    run_qwen_inference,
)
from multimodal_science.qwen3vl.sequence_prompt_data import (
    SEQUENCE_PROMPT_BUNDLE_SCHEMA,
    SEQUENCE_PROMPT_PREDICTION_SCHEMA,
)


SEQUENCE_PROMPT_BACKEND = "transformers-qwen3vl-image-sequence-prompt"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_EXPOSED_FIELDS = (
    "presence_probability",
    "start_normalized",
    "end_normalized",
)


@dataclass(frozen=True)
class _VerifiedSequenceExpert:
    predictions_by_asset: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label}: {path}") from error


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"Blank {label} line: {line_number}")
            try:
                rows.append(_object(json.loads(line), f"{label} line {line_number}"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {label} JSON at line {line_number}") from error
    _require(rows, f"{label} is empty")
    return rows


def _safe_artifact(root: Path, relative: str, label: str) -> Path:
    _require(bool(relative) and "\\" not in relative, f"Bad {label} path")
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute() and all(part not in {"", ".", ".."} for part in posix.parts),
        f"Unsafe {label} path",
    )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    _require(path.is_file(), f"Missing {label}: {path}")
    return path


def _verify_manifest(root: Path, expected_sha256: str) -> dict[str, str]:
    _require(bool(_HEX_64.fullmatch(expected_sha256)), "Bad sequence manifest hash")
    manifest_path = root / "artifact_manifest.sha256"
    _require(manifest_path.is_file(), f"Missing sequence manifest: {manifest_path}")
    _require(sha256_file(manifest_path) == expected_sha256, "Sequence manifest hash mismatch")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        _require(len(parts) == 2, f"Malformed sequence manifest line: {line_number}")
        digest, relative = parts
        _require(bool(_HEX_64.fullmatch(digest)), f"Bad sequence digest: {line_number}")
        _require(relative not in entries, f"Duplicate sequence artifact: {relative}")
        path = _safe_artifact(root, relative, "sequence artifact")
        _require(sha256_file(path) == digest, f"Sequence artifact hash mismatch: {relative}")
        entries[relative] = digest
    return entries


def _finite_unit(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{label} is outside [0, 1]")
    return result


def _verify_sequence_expert(
    *,
    root: Path,
    report_sha256: str,
    manifest_sha256: str,
    validation_links: dict[str, dict[str, Any]],
) -> _VerifiedSequenceExpert:
    root = root.resolve()
    _require(root.is_dir(), f"Sequence expert root not found: {root}")
    _require(bool(_HEX_64.fullmatch(report_sha256)), "Bad sequence report hash")
    manifest = _verify_manifest(root, manifest_sha256)
    report_path = root / "sequence_prompt_bundle_report.json"
    _require(report_path.is_file(), "Missing sequence prompt bundle report")
    _require(sha256_file(report_path) == report_sha256, "Sequence prompt report mismatch")
    _require(
        manifest.get("sequence_prompt_bundle_report.json") == report_sha256,
        "Sequence prompt report is not manifest-bound",
    )
    report = _read_json(report_path, "sequence prompt bundle report")
    _require(
        report.get("schema_version") == SEQUENCE_PROMPT_BUNDLE_SCHEMA,
        "Unsupported sequence prompt schema",
    )
    _require(report.get("development_comparison_eligible") is True, "Sequence run is ineligible")
    _require(
        report.get("final_benchmark_eligible") is False,
        "Sequence run is not development-only",
    )
    _require(report.get("internal_test_accessed") is False, "Sequence run accessed test data")
    contracts = _object(report.get("contracts"), "sequence prompt contracts")
    for name, expected in (
        ("frozen_sequence_predictions_only", True),
        ("target_fields_excluded", True),
        ("instruction_answers_opened", False),
        ("development_only", True),
        ("internal_test_accessed", False),
    ):
        _require(contracts.get(name) is expected, f"Sequence prompt contract failed: {name}")
    artifact = _object(
        _object(report.get("artifacts"), "sequence prompt artifacts").get("predictions"),
        "sequence prompt predictions",
    )
    relative = artifact.get("path")
    digest = artifact.get("sha256")
    _require(isinstance(relative, str), "Missing sequence prediction path")
    _require(isinstance(digest, str) and bool(_HEX_64.fullmatch(digest)), "Bad prediction hash")
    predictions_path = _safe_artifact(root, relative, "sequence predictions")
    _require(sha256_file(predictions_path) == digest, "Sequence prediction hash mismatch")
    _require(manifest.get(relative) == digest, "Sequence predictions are not manifest-bound")

    links_by_asset: dict[str, dict[str, Any]] = {}
    for link in validation_links.values():
        asset_id = str(link.get("asset_id") or "")
        group_id = str(link.get("group_id") or "")
        _require(asset_id and group_id, "Validation link lacks asset identity")
        previous = links_by_asset.setdefault(asset_id, link)
        _require(previous.get("group_id") == group_id, "Asset maps to multiple source groups")

    predictions_by_asset: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(predictions_path, "sequence predictions"):
        _require(
            row.get("schema_version") == SEQUENCE_PROMPT_PREDICTION_SCHEMA,
            "Bad prediction schema",
        )
        _require(
            set(row)
            == {"schema_version", "asset_id", "group_id", *_EXPOSED_FIELDS},
            "Sequence prompt row contains a non-allowlisted field",
        )
        asset_id = str(row.get("asset_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(asset_id and asset_id not in predictions_by_asset, "Duplicate sequence asset")
        _require(asset_id in links_by_asset, f"Unknown sequence validation asset: {asset_id}")
        _require(group_id == links_by_asset[asset_id].get("group_id"), "Sequence group mismatch")
        probability = _finite_unit(row.get("presence_probability"), "presence probability")
        start = _finite_unit(row.get("start_normalized"), "interval start")
        end = _finite_unit(row.get("end_normalized"), "interval end")
        _require(start <= end, "Sequence interval is reversed")
        predictions_by_asset[asset_id] = {
            "presence_probability": probability,
            "start_normalized": start,
            "end_normalized": end,
        }
    _require(
        set(predictions_by_asset) == set(links_by_asset),
        "Sequence predictions do not exactly cover validation assets",
    )
    records = artifact.get("records")
    _require(records == len(predictions_by_asset), "Sequence prediction count mismatch")
    metadata = {
        "kind": "frozen_sequence_peak_net_prompt_evidence",
        "report_sha256": report_sha256,
        "manifest_sha256": manifest_sha256,
        "predictions_sha256": digest,
        "prediction_records": len(predictions_by_asset),
        "bundle_code_revision": report.get("code_revision"),
        "source_sequence": _object(report.get("sources"), "sequence prompt sources"),
        "exposed_fields": list(_EXPOSED_FIELDS),
        "target_fields_exposed": False,
        "answer_key_opened_by_prompt_builder": False,
        "internal_test_accessed": False,
    }
    return _VerifiedSequenceExpert(predictions_by_asset, metadata)


def sequence_evidence_prompt(
    prompt: str,
    language: str | None,
    prediction: dict[str, Any],
) -> str:
    """Append the frozen expert output using an explicit prediction-only allowlist."""

    evidence = {field: prediction[field] for field in _EXPOSED_FIELDS}
    serialized = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if language == "zh-CN":
        note = (
            "外部冻结 SequencePeakNet 的预测（不是标准答案，请作为辅助证据独立判断）："
            f"{serialized}"
        )
    else:
        note = (
            "External frozen SequencePeakNet prediction (not ground truth; use only as "
            f"auxiliary evidence): {serialized}"
        )
    return f"{prompt.rstrip()}\n\n{note}"


class _SequencePromptAdapterVerifier:
    def __init__(self, expert: _VerifiedSequenceExpert) -> None:
        self._expert = expert

    def __call__(
        self,
        specification: AdapterSpec,
        *,
        model_name_or_path: str,
        model_revision: str,
        model_artifact_sha256: str | None,
    ) -> tuple[Path, dict[str, Any]]:
        adapter_dir, metadata = _verify_adapter(
            specification,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            model_artifact_sha256=model_artifact_sha256,
        )
        metadata = dict(metadata)
        metadata["kind"] = "image_lora_with_sequence_prompt"
        metadata["sequence_expert"] = dict(self._expert.metadata)
        return adapter_dir, metadata


class _SequencePromptGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        model_revision: str,
        settings: GenerationSettings,
        adapter_dir: Path | None,
        expert: _VerifiedSequenceExpert,
        validation_links: dict[str, dict[str, Any]],
    ) -> None:
        self._base = _TransformersGenerator(
            model_name_or_path,
            model_revision,
            settings,
            adapter_dir,
        )
        self._expert = expert
        self._links = validation_links
        self._verified_images: dict[str, str] = {}
        self._metadata = {
            **self._base.metadata,
            "sequence_prompt_execution_backend": SEQUENCE_PROMPT_BACKEND,
            "model_visible_prompt_augmented": True,
            "original_prompt_hash_preserved_in_generation_record": True,
            "sequence_expert": dict(expert.metadata),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def generate(
        self,
        requests: Sequence[PromptRequest],
        settings: GenerationSettings,
    ) -> list[str]:
        augmented: list[PromptRequest] = []
        for request in requests:
            link = _object(self._links.get(request.instruction_id), "validation fusion link")
            _require(request.task == link.get("task"), "Sequence prompt task mismatch")
            _require(request.language == link.get("language"), "Sequence prompt language mismatch")
            image_key = str(request.image_path)
            image_digest = self._verified_images.get(image_key)
            if image_digest is None:
                image_digest = sha256_file(request.image_path)
                self._verified_images[image_key] = image_digest
            _require(image_digest == link.get("image_sha256"), "Sequence prompt image mismatch")
            asset_id = str(link.get("asset_id") or "")
            prediction = self._expert.predictions_by_asset.get(asset_id)
            _require(prediction is not None, f"Missing sequence prediction: {asset_id}")
            augmented.append(
                replace(
                    request,
                    prompt=sequence_evidence_prompt(
                        request.prompt,
                        request.language,
                        prediction,
                    ),
                )
            )
        return self._base.generate(augmented, settings)


def run_sequence_prompt_inference(
    *,
    inference_bundle_root: Path,
    inference_bundle_report_sha256: str,
    fusion_bundle_root: Path,
    fusion_bundle_report_sha256: str,
    dataset_root: Path,
    dataset_report_sha256: str,
    assets_root: Path,
    image_adapter: AdapterSpec,
    sequence_prompt_bundle_root: Path,
    sequence_prompt_bundle_report_sha256: str,
    sequence_prompt_bundle_manifest_sha256: str,
    output_dir: Path,
    model_name_or_path: str,
    model_revision: str,
    model_artifact_sha256: str,
    settings: GenerationSettings = GenerationSettings(),
    max_records: int | None = None,
    resume: bool = False,
) -> QwenInferenceResult:
    """Run the development-only image-LoRA plus frozen-sequence-prompt baseline."""

    _require(not settings.do_sample, "Sequence prompt baseline requires greedy decoding")
    validation = _load_validation_inputs(
        fusion_bundle_root=fusion_bundle_root,
        fusion_bundle_report_sha256=fusion_bundle_report_sha256,
        dataset_root=dataset_root,
        dataset_report_sha256=dataset_report_sha256,
        inference_bundle_root=inference_bundle_root,
        inference_bundle_report_sha256=inference_bundle_report_sha256,
    )
    expert = _verify_sequence_expert(
        root=sequence_prompt_bundle_root,
        report_sha256=sequence_prompt_bundle_report_sha256,
        manifest_sha256=sequence_prompt_bundle_manifest_sha256,
        validation_links=validation.links_by_instruction_id,
    )

    def generator_factory(
        runtime_model_name_or_path: str,
        runtime_model_revision: str,
        runtime_settings: GenerationSettings,
        adapter_dir: Path | None,
    ) -> BatchGenerator:
        return _SequencePromptGenerator(
            runtime_model_name_or_path,
            runtime_model_revision,
            runtime_settings,
            adapter_dir,
            expert,
            validation.links_by_instruction_id,
        )

    return run_qwen_inference(
        inference_bundle_root,
        assets_root,
        output_dir,
        expected_bundle_report_sha256=inference_bundle_report_sha256,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        settings=settings,
        model_artifact_sha256=model_artifact_sha256,
        adapter=image_adapter,
        max_records=max_records,
        resume=resume,
        generator_factory=generator_factory,
        adapter_verifier=_SequencePromptAdapterVerifier(expert),
    )
