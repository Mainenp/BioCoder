from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.services.attachments import AttachmentStore, build_user_content
from app.tools.chromatogram import (
    authorized_chromatogram_attachments,
    build_chromatogram_tool,
    register_chromatogram_tool,
)
from biocoder.security.permissions import ToolPermission
from biocoder.tools.registry import ToolRegistry

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8"
    b"\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeRuntime:
    def __init__(self, modality: str = "image_xic") -> None:
        self.calls: list[dict] = []
        self.modality = modality
        self.deployment = SimpleNamespace(input_modality=modality)

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "deployment_id": "frozen-test",
            "deployment_manifest_sha256": "a" * 64,
            "result": {"peak_present": True},
        }


def _settings(tmp_path: Path, *, enabled: bool) -> Settings:
    return Settings(
        attachments_dir=tmp_path / "attachments",
        vision_input_enabled=True,
        chromatogram_tool_enabled=enabled,
        chromatogram_deployment_manifest_path=(tmp_path / "deployment.json"),
    )


def test_tool_is_disabled_by_default(tmp_path: Path) -> None:
    assert (
        build_chromatogram_tool(Settings(attachments_dir=tmp_path), runtime=FakeRuntime()) is None
    )


def test_enabled_tool_is_registered_as_read_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    registry = ToolRegistry()

    register_chromatogram_tool(
        registry,
        settings,
        runtime=FakeRuntime(),
        attachment_store=AttachmentStore(settings),
    )

    assert [candidate.name for candidate in registry.tools()] == ["analyze_chromatogram"]
    metadata = registry.metadata("analyze_chromatogram")
    assert metadata.permission == ToolPermission.READ_ONLY
    assert metadata.side_effect is False
    assert metadata.retry_policy.max_attempts == 1


def test_attachment_header_exposes_id_for_authorized_tool_call(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    store = AttachmentStore(settings)
    descriptor = store.save("roi.png", PNG_1X1)

    content = build_user_content("analyze", store.prepare([descriptor.id]), 1000)

    assert descriptor.id in content[0]["text"]


def test_tool_rejects_attachment_outside_current_turn(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    store = AttachmentStore(settings)
    descriptor = store.save("roi.png", PNG_1X1)
    runtime = FakeRuntime()
    candidate = build_chromatogram_tool(settings, runtime=runtime, attachment_store=store)
    assert candidate is not None

    payload = json.loads(
        candidate.invoke(
            {
                "task": "peak_presence",
                "language": "en",
                "xic_signal": [0.5] * 160,
                "image_attachment_id": descriptor.id,
                "image_width": 1,
                "image_height": 1,
            }
        )
    )

    assert "not authorized" in payload["error"]
    assert runtime.calls == []


def test_tool_reads_only_authorized_image_and_cleans_temporary_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    store = AttachmentStore(settings)
    descriptor = store.save("roi.png", PNG_1X1)
    runtime = FakeRuntime()
    candidate = build_chromatogram_tool(settings, runtime=runtime, attachment_store=store)
    assert candidate is not None

    with authorized_chromatogram_attachments([descriptor.id]):
        payload = json.loads(
            candidate.invoke(
                {
                    "task": "peak_presence",
                    "language": "zh-CN",
                    "xic_signal": [0.5] * 160,
                    "image_attachment_id": descriptor.id,
                    "image_width": 1,
                    "image_height": 1,
                }
            )
        )

    assert payload["result"] == {"peak_present": True}
    assert len(runtime.calls) == 1
    temporary = runtime.calls[0]["image_path"]
    assert temporary is not None
    assert not temporary.exists()


def test_tool_rejects_declared_dimension_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    store = AttachmentStore(settings)
    descriptor = store.save("roi.png", PNG_1X1)
    runtime = FakeRuntime()
    candidate = build_chromatogram_tool(settings, runtime=runtime, attachment_store=store)
    assert candidate is not None

    with authorized_chromatogram_attachments([descriptor.id]):
        payload = json.loads(
            candidate.invoke(
                {
                    "task": "peak_presence",
                    "language": "en",
                    "xic_signal": [0.5] * 160,
                    "image_attachment_id": descriptor.id,
                    "image_width": 2,
                    "image_height": 1,
                }
            )
        )

    assert "dimensions" in payload["error"]
    assert runtime.calls == []


def test_xic_only_tool_ignores_text_attachment_id_and_preserves_signal(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    store = AttachmentStore(settings)
    descriptor = store.save("chromatogram.json", b'{"xic_signal": [0.5]}')
    runtime = FakeRuntime(modality="xic_only")
    candidate = build_chromatogram_tool(settings, runtime=runtime, attachment_store=store)
    assert candidate is not None
    signal = [index / 159 for index in range(160)]

    with authorized_chromatogram_attachments([descriptor.id]):
        payload = json.loads(
            candidate.invoke(
                {
                    "task": "peak_grounding",
                    "language": "zh-CN",
                    "xic_signal": signal,
                    "image_attachment_id": descriptor.id,
                    "image_width": 400,
                    "image_height": 300,
                }
            )
        )

    assert payload["result"] == {"peak_present": True}
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["xic_signal"] == signal
    assert runtime.calls[0]["image_path"] is None
