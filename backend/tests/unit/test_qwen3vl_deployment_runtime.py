from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from multimodal_science.qwen3vl.deployment_runtime import (
    DEPLOYMENT_SCHEMA,
    ChromatogramDeploymentRuntime,
    build_task_prompt,
    validate_task_output,
    verify_deployment_manifest,
)
from multimodal_science.qwen3vl.freeze_deployment_cli import main as freeze_deployment


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _file_manifest(root: Path, path: Path, relatives: list[str]) -> str:
    _write(
        path,
        "".join(f"{_digest(root / relative)}  {relative}\n" for relative in relatives),
    )
    return _digest(path)


def _deployment(tmp_path: Path, *, modality: str = "xic_only") -> Path:
    model = tmp_path / "model"
    _write(model / "config.json", "{}\n")
    model_manifest = tmp_path / "model.files.sha256"
    model_manifest_sha = _file_manifest(model, model_manifest, ["config.json"])

    adapter = tmp_path / "candidate" / "adapter"
    _write(adapter / "adapter_config.json", "{}\n")
    _write(adapter / "adapter_model.safetensors", "adapter")
    projector = tmp_path / "candidate" / "sensor_projector.safetensors"
    _write(projector, "projector")
    processor = tmp_path / "candidate" / "processor"
    _write(processor / "tokenizer_config.json", "{}\n")
    processor_manifest = tmp_path / "processor.files.sha256"
    processor_manifest_sha = _file_manifest(
        processor, processor_manifest, ["tokenizer_config.json"]
    )
    report = tmp_path / "candidate" / "fusion_training_report.json"
    _write(
        report,
        json.dumps(
            {
                "contracts": {
                    "internal_test_accessed": False,
                    "validation_answers_opened": False,
                    "validation_prompts_opened": False,
                }
            }
        ),
    )
    manifest = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "deployment_id": "test-candidate",
        "model": {
            "root": str(model),
            "revision": "frozen-revision",
            "manifest_path": str(model_manifest),
            "manifest_sha256": model_manifest_sha,
        },
        "artifacts": {
            "adapter_dir": str(adapter),
            "adapter_config_sha256": _digest(adapter / "adapter_config.json"),
            "adapter_weights_sha256": _digest(adapter / "adapter_model.safetensors"),
            "projector_path": str(projector),
            "projector_sha256": _digest(projector),
            "processor_dir": str(processor),
            "processor_manifest_path": str(processor_manifest),
            "processor_manifest_sha256": processor_manifest_sha,
        },
        "runtime": {
            "input_modality": modality,
            "attention_implementation": "sdpa",
            "sensor_projector": {
                "input_points": 160,
                "hidden_size": 2560,
                "sensor_tokens": 4,
                "base_channels": 32,
                "dropout": 0.1,
            },
            "min_pixels": 784,
            "max_pixels": 1_003_520,
            "max_new_tokens": 64,
        },
        "provenance": {
            "report_path": str(report),
            "report_sha256": _digest(report),
            "internal_test_accessed": False,
            "validation_answers_opened": False,
            "validation_prompts_opened": False,
        },
        "contracts": {
            "user_samples_only": True,
            "internal_test_accessed": False,
            "validation_answers_opened": False,
            "validation_prompts_opened": False,
        },
    }
    path = tmp_path / "deployment.json"
    _write(path, json.dumps(manifest))
    return path


def test_deployment_verifies_every_bound_artifact(tmp_path: Path) -> None:
    path = _deployment(tmp_path)

    verified = verify_deployment_manifest(path)

    assert verified.deployment_id == "test-candidate"
    assert verified.input_modality == "xic_only"
    assert verified.projector_spec.input_points == 160


def test_deployment_rejects_tampered_weights(tmp_path: Path) -> None:
    path = _deployment(tmp_path)
    weights = tmp_path / "candidate" / "adapter" / "adapter_model.safetensors"
    weights.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter weights hash mismatch"):
        verify_deployment_manifest(path)


def test_deployment_rejects_protected_research_path(tmp_path: Path) -> None:
    path = _deployment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["root"] = str(tmp_path / "validation" / "model")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Protected research artifact path"):
        verify_deployment_manifest(path)


def test_deployment_rejects_protected_path_marker_inside_component(
    tmp_path: Path,
) -> None:
    path = _deployment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["root"] = str(tmp_path / "candidate-validation-assets")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Protected research artifact path"):
        verify_deployment_manifest(path)


def test_deployment_rejects_provenance_that_opened_answers(tmp_path: Path) -> None:
    path = _deployment(tmp_path)
    report = tmp_path / "candidate" / "fusion_training_report.json"
    report.write_text(
        json.dumps({"contracts": {"validation_answers_opened": True}}),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["report_sha256"] = _digest(report)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="data isolation"):
        verify_deployment_manifest(path)


def test_deployment_rejects_training_report_without_explicit_isolation(
    tmp_path: Path,
) -> None:
    path = _deployment(tmp_path)
    report = tmp_path / "candidate" / "fusion_training_report.json"
    report.write_text(
        json.dumps(
            {
                "contracts": {
                    "internal_test_accessed": False,
                    "validation_answers_opened": False,
                }
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["report_sha256"] = _digest(report)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not prove validation_prompts_opened=false"):
        verify_deployment_manifest(path)


def test_deployment_rejects_protected_file_inside_bound_directory(
    tmp_path: Path,
) -> None:
    path = _deployment(tmp_path)
    model = tmp_path / "model"
    _write(model / "validation_answers.json", "{}\n")
    model_manifest = tmp_path / "model.files.sha256"
    manifest_sha = _file_manifest(
        model,
        model_manifest,
        ["config.json", "validation_answers.json"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["manifest_sha256"] = manifest_sha
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Protected research artifact in model manifest"):
        verify_deployment_manifest(path)


def test_deployment_rejects_unbound_file_inside_model_directory(tmp_path: Path) -> None:
    path = _deployment(tmp_path)
    _write(tmp_path / "model" / "unbound.bin", "untrusted")

    with pytest.raises(ValueError, match="does not bind the exact directory contents"):
        verify_deployment_manifest(path)


def test_runtime_is_lazy_and_validates_model_json(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    class Backend:
        def __init__(self, deployment) -> None:
            self.metadata = {"backend": "fake"}
            calls.append((deployment.deployment_id, 0))

        def generate(self, prompt, signal, available, image_path) -> str:
            calls.append((prompt, len(signal)))
            assert available is True
            assert image_path is None
            return '{"bbox_2d":[10,0,40,80]}'

    runtime = ChromatogramDeploymentRuntime(_deployment(tmp_path), Backend)
    assert calls == []

    result = runtime.analyze(
        task="peak_grounding",
        language="en",
        xic_signal=[0.5] * 160,
        xic_available=True,
        image_path=None,
        image_width=100,
        image_height=80,
    )

    assert result["result"] == {"bbox_2d": [10, 0, 40, 80]}
    assert result["runtime"] == {"backend": "fake"}
    assert len(calls) == 2


def test_runtime_reverifies_artifacts_before_first_model_load(tmp_path: Path) -> None:
    path = _deployment(tmp_path)
    runtime = ChromatogramDeploymentRuntime(path, lambda deployment: object())
    weights = tmp_path / "candidate" / "adapter" / "adapter_model.safetensors"
    weights.write_text("tampered-after-construction", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter weights hash mismatch"):
        runtime.analyze(
            task="peak_presence",
            language="en",
            xic_signal=[0.5] * 160,
            xic_available=True,
            image_path=None,
            image_width=100,
            image_height=80,
        )


def test_prompts_and_output_schema_match_fixed_tasks() -> None:
    prompt = build_task_prompt("peak_grounding", "zh-CN", width=256, height=128)
    assert "256×128" in prompt
    assert "bbox_2d" in prompt
    assert validate_task_output("peak_presence", '{"peak_present":true}', 256, 128) == {
        "peak_present": True
    }
    with pytest.raises(ValueError, match="grounding result"):
        validate_task_output("peak_grounding", '{"bbox_2d":[20,0,10,128]}', 256, 128)


def test_freeze_cli_writes_a_self_verifying_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "base-model"
    _write(model / "config.json", "{}\n")
    candidate = tmp_path / "candidate"
    adapter = candidate / "adapter"
    _write(adapter / "adapter_config.json", "{}\n")
    _write(adapter / "adapter_model.safetensors", "adapter")
    projector = candidate / "sensor_projector.safetensors"
    _write(projector, "projector")
    processor = candidate / "processor"
    _write(processor / "tokenizer_config.json", "{}\n")
    report = candidate / "fusion_training_report.json"
    _write(
        report,
        json.dumps(
            {
                "contracts": {
                    "internal_test_accessed": False,
                    "validation_answers_opened": False,
                    "validation_prompts_opened": False,
                }
            }
        ),
    )
    output = tmp_path / "frozen-deployment"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze-deployment",
            "--deployment-id",
            "test-freeze",
            "--output-dir",
            str(output),
            "--model-root",
            str(model),
            "--model-revision",
            "frozen-revision",
            "--adapter-dir",
            str(adapter),
            "--projector-path",
            str(projector),
            "--processor-dir",
            str(processor),
            "--training-report-path",
            str(report),
            "--input-modality",
            "xic_only",
        ],
    )

    freeze_deployment()

    result = json.loads(capsys.readouterr().out)
    manifest = output / "deployment_manifest.json"
    assert result["manifest_sha256"] == _digest(manifest)
    assert result["validation_prompts_opened"] is False
    assert verify_deployment_manifest(manifest).deployment_id == "test-freeze"
