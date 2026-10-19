"""BioCoder Agent tool for the frozen Qwen/XIC deployment."""

from __future__ import annotations

import json
import struct
import tempfile
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.config import Settings
from app.services.attachments import AttachmentStore
from biocoder.security.permissions import ToolPermission
from biocoder.tools.registry import ToolRegistry
from biocoder.tools.schema import RetryPolicy, ToolMetadata

if TYPE_CHECKING:
    from multimodal_science.qwen3vl.deployment_runtime import ChromatogramDeploymentRuntime

_AUTHORIZED_ATTACHMENTS: ContextVar[frozenset[str]] = ContextVar(
    "chromatogram_authorized_attachments", default=frozenset()
)


@contextmanager
def authorized_chromatogram_attachments(attachment_ids: list[str]) -> Iterator[None]:
    """Bind tool file access to attachments authorized by the current chat request."""

    token = _AUTHORIZED_ATTACHMENTS.set(frozenset(attachment_ids))
    try:
        yield
    finally:
        _AUTHORIZED_ATTACHMENTS.reset(token)


class ChromatogramAnalysisInput(BaseModel):
    task: Literal[
        "peak_presence",
        "peak_presence_metadata",
        "peak_grounding",
        "scientific_qc",
    ]
    language: Literal["en", "zh-CN"] = "zh-CN"
    xic_signal: list[float] | None = Field(
        default=None,
        min_length=32,
        max_length=4096,
        description="Normalized [0,1] XIC values; length must match the frozen deployment.",
    )
    xic_available: bool = True
    image_attachment_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{32}$",
        description="Image attachment ID shown in the current user message.",
    )
    image_width: int = Field(
        gt=0,
        le=100_000,
        description=(
            "Source ROI coordinate-canvas width. Standard BioCoder ROIs use 400; "
            "this is independent of the 160 XIC samples."
        ),
    )
    image_height: int = Field(
        gt=0,
        le=100_000,
        description="Source ROI coordinate-canvas height. Standard BioCoder ROIs use 300.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=("For metadata/QC tasks: component, channel, q1, q3, expected_rt_minutes."),
    )


@lru_cache(maxsize=3)
def get_chromatogram_runtime(manifest_path: Path) -> ChromatogramDeploymentRuntime:
    """Return one lazy runtime per immutable deployment manifest.

    The HTTP endpoint and LangChain tool share this cache so enabling both
    surfaces cannot load two copies of the frozen model onto the GPU.
    """

    from multimodal_science.qwen3vl.deployment_runtime import (
        ChromatogramDeploymentRuntime,
    )

    return ChromatogramDeploymentRuntime(manifest_path.expanduser().resolve())


def configured_chromatogram_manifests(settings: Settings) -> dict[str, Path]:
    """Resolve explicit modality manifests with the v2 XIC-only setting as fallback."""

    manifests = {
        modality: path
        for modality, path in {
            "xic_only": settings.chromatogram_xic_only_manifest_path,
            "image_xic": settings.chromatogram_image_xic_manifest_path,
            "image_only": settings.chromatogram_image_only_manifest_path,
        }.items()
        if path is not None
    }
    if "xic_only" not in manifests and settings.chromatogram_deployment_manifest_path is not None:
        manifests["xic_only"] = settings.chromatogram_deployment_manifest_path
    return manifests


T = TypeVar("T")


class ChromatogramRuntimeRouter:
    """Select one immutable modality and keep at most one CUDA backend resident."""

    def __init__(
        self,
        manifests: dict[str, Path],
        *,
        runtime_factory: Callable[[Path], ChromatogramDeploymentRuntime] = get_chromatogram_runtime,
    ) -> None:
        if not manifests:
            raise ValueError("No chromatography deployments are configured")
        self._manifests = {
            modality: path.expanduser().resolve() for modality, path in manifests.items()
        }
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, ChromatogramDeploymentRuntime] = {}
        self._active_modality: str | None = None
        import threading

        self._lock = threading.RLock()

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))

    @staticmethod
    def modality_for(payload: ChromatogramAnalysisInput) -> str:
        has_xic = payload.xic_signal is not None
        has_image = payload.image_attachment_id is not None
        if has_xic and has_image:
            return "image_xic"
        if has_xic:
            return "xic_only"
        if has_image:
            return "image_only"
        raise ValueError("Chromatography analysis requires XIC values, an image, or both")

    def run(self, payload: ChromatogramAnalysisInput, operation: Callable[[Any], T]) -> T:
        modality = self.modality_for(payload)
        if modality not in self._manifests:
            raise ValueError(f"Chromatography modality is not deployed: {modality}")
        with self._lock:
            if self._active_modality is not None and self._active_modality != modality:
                active = self._runtimes.get(self._active_modality)
                if active is not None:
                    active.unload()
            runtime = self._runtimes.get(modality)
            if runtime is None:
                runtime = self._runtime_factory(self._manifests[modality])
                if runtime.deployment.input_modality != modality:
                    raise ValueError(
                        f"Configured {modality} manifest declares "
                        f"{runtime.deployment.input_modality}"
                    )
                self._runtimes[modality] = runtime
            self._active_modality = modality
            return operation(runtime)


@lru_cache(maxsize=4)
def get_chromatogram_router(
    configured: tuple[tuple[str, str], ...],
) -> ChromatogramRuntimeRouter:
    return ChromatogramRuntimeRouter(
        {modality: Path(path) for modality, path in configured}
    )


def build_chromatogram_router(settings: Settings) -> ChromatogramRuntimeRouter:
    manifests = configured_chromatogram_manifests(settings)
    configured = tuple(
        sorted(
            (modality, str(path.expanduser().resolve()))
            for modality, path in manifests.items()
        )
    )
    return get_chromatogram_router(configured)


def _image_dimensions(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 <= len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9}:
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset : offset + 2], "big")
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                if length >= 7 and offset + 7 <= len(data):
                    height = int.from_bytes(data[offset + 3 : offset + 5], "big")
                    width = int.from_bytes(data[offset + 5 : offset + 7], "big")
                    return width, height
                break
            if length < 2:
                break
            offset += length
    raise ValueError("The image dimensions could not be verified")


def normalize_chromatogram_attachment(
    payload: ChromatogramAnalysisInput,
    *,
    attachment_store: AttachmentStore,
    authorized_attachment_ids: Collection[str],
) -> ChromatogramAnalysisInput:
    """Drop non-image attachment IDs from XIC requests before modality routing."""

    attachment_id = payload.image_attachment_id
    if attachment_id is None:
        return payload
    if attachment_id not in authorized_attachment_ids:
        raise ValueError("Image attachment is not authorized for this chat turn")
    descriptor, _, _ = attachment_store.load(attachment_id)
    if descriptor.kind == "image":
        return payload
    if payload.xic_signal is None:
        raise ValueError("Chromatography image attachment must be an image")
    return payload.model_copy(update={"image_attachment_id": None})


def execute_chromatogram_analysis(
    payload: ChromatogramAnalysisInput,
    *,
    runtime: ChromatogramDeploymentRuntime,
    attachment_store: AttachmentStore,
    authorized_attachment_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Execute an already-validated request without an LLM serialization hop."""

    temporary_path: Path | None = None
    try:
        image_path: Path | None = None
        image_attachment_id = payload.image_attachment_id
        if runtime.deployment.input_modality == "xic_only":
            # Text/JSON attachments are data sources, never image pixels. Some
            # tool-calling models eagerly copy the current attachment ID into
            # every optional attachment field; ignoring it is safe because no
            # attachment bytes are read on the XIC-only path.
            image_attachment_id = None
        if image_attachment_id is not None:
            if image_attachment_id not in authorized_attachment_ids:
                raise ValueError("Image attachment is not authorized for this chat turn")
            descriptor, image_bytes, _ = attachment_store.load(image_attachment_id)
            if descriptor.kind != "image":
                raise ValueError("Chromatography image attachment must be an image")
            actual_width, actual_height = _image_dimensions(image_bytes)
            if (actual_width, actual_height) != (payload.image_width, payload.image_height):
                raise ValueError("Declared image dimensions do not match the attachment")
            suffix = str(descriptor.metadata.get("suffix", ".png"))
            with tempfile.NamedTemporaryFile(
                prefix="biocoder-chromatogram-", suffix=suffix, delete=False
            ) as stream:
                stream.write(image_bytes)
                temporary_path = Path(stream.name)
            image_path = temporary_path
        return runtime.analyze(
            task=payload.task,
            language=payload.language,
            xic_signal=payload.xic_signal,
            xic_available=payload.xic_available,
            image_path=image_path,
            image_width=payload.image_width,
            image_height=payload.image_height,
            metadata=payload.metadata,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_chromatogram_tool(
    settings: Settings,
    *,
    runtime: ChromatogramDeploymentRuntime | None = None,
    router: ChromatogramRuntimeRouter | None = None,
    attachment_store: AttachmentStore | None = None,
) -> BaseTool | None:
    """Return the optional tool, validating its deployment before registration."""

    if not settings.chromatogram_tool_enabled:
        return None
    if runtime is not None and router is not None:
        raise ValueError("Provide either a chromatography runtime or router, not both")
    resolved_router = (
        None if runtime is not None else (router or build_chromatogram_router(settings))
    )
    store = attachment_store or AttachmentStore(settings)

    @tool(args_schema=ChromatogramAnalysisInput)
    def analyze_chromatogram(
        task: str,
        language: str,
        image_width: int,
        image_height: int,
        xic_signal: list[float] | None = None,
        xic_available: bool = True,
        image_attachment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Analyze chromatography with the frozen model selected by supplied XIC/image inputs."""

        try:
            payload = ChromatogramAnalysisInput(
                task=task,
                language=language,
                xic_signal=xic_signal,
                xic_available=xic_available,
                image_attachment_id=image_attachment_id,
                image_width=image_width,
                image_height=image_height,
                metadata=metadata,
            )
            payload = normalize_chromatogram_attachment(
                payload,
                attachment_store=store,
                authorized_attachment_ids=_AUTHORIZED_ATTACHMENTS.get(),
            )
            execute = lambda selected: execute_chromatogram_analysis(
                payload,
                runtime=selected,
                attachment_store=store,
                authorized_attachment_ids=_AUTHORIZED_ATTACHMENTS.get(),
            )
            if runtime is not None:
                result = execute(runtime)
            else:
                assert resolved_router is not None
                result = resolved_router.run(payload, execute)
            return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return json.dumps(
                {"error": f"Chromatogram analysis failed: {exc}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

    return analyze_chromatogram


def register_chromatogram_tool(
    registry: ToolRegistry,
    settings: Settings,
    *,
    runtime: ChromatogramDeploymentRuntime | None = None,
    router: ChromatogramRuntimeRouter | None = None,
    attachment_store: AttachmentStore | None = None,
) -> None:
    candidate = build_chromatogram_tool(
        settings,
        runtime=runtime,
        router=router,
        attachment_store=attachment_store,
    )
    if candidate is None:
        return
    registry.register(
        candidate,
        ToolMetadata(
            name=candidate.name,
            description=candidate.description,
            parameters=(
                candidate.args_schema.model_json_schema()
                if candidate.args_schema is not None
                else candidate.args
            ),
            permission=ToolPermission.READ_ONLY,
            timeout_seconds=settings.chromatogram_tool_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=1),
            side_effect=False,
        ),
    )


__all__ = [
    "ChromatogramAnalysisInput",
    "ChromatogramRuntimeRouter",
    "authorized_chromatogram_attachments",
    "build_chromatogram_router",
    "build_chromatogram_tool",
    "configured_chromatogram_manifests",
    "execute_chromatogram_analysis",
    "get_chromatogram_router",
    "get_chromatogram_runtime",
    "normalize_chromatogram_attachment",
    "register_chromatogram_tool",
]
