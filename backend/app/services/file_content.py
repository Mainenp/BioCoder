from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pypdf import PdfReader

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
TEXT_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
}
DOCUMENT_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SUPPORTED_ATTACHMENT_SUFFIXES = frozenset(
    {*IMAGE_MEDIA_TYPES, *TEXT_MEDIA_TYPES, *DOCUMENT_MEDIA_TYPES}
)
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


class FileContentError(ValueError):
    """Raised when an attachment is unsupported, malformed, or unsafe to parse."""


@dataclass(frozen=True)
class ExtractedFile:
    name: str
    suffix: str
    kind: str
    media_type: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def safe_filename(value: str | None) -> str:
    candidate = (value or "attachment").replace("\\", "/").split("/")[-1]
    candidate = re.sub(r"[\x00-\x1f\x7f]", "_", candidate).strip(" .")
    if not candidate:
        candidate = "attachment"
    path = Path(candidate)
    stem = path.stem[:120] or "attachment"
    return f"{stem}{path.suffix.lower()}"[:160]


def _image_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_docx_archive(archive: zipfile.ZipFile) -> None:
    names = set(archive.namelist())
    if "[Content_Types].xml" not in names or "word/document.xml" not in names:
        raise FileContentError("文件不是有效的 .docx Word 文档。")
    total_size = sum(item.file_size for item in archive.infolist())
    if total_size > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise FileContentError("Word 文档解压后过大，已拒绝处理。")


def _docx_part_text(payload: bytes) -> str:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise FileContentError("Word 文档 XML 已损坏。") from exc
    paragraphs: list[str] = []
    paragraph_tag = f"{{{WORD_NAMESPACE}}}p"
    text_tag = f"{{{WORD_NAMESPACE}}}t"
    tab_tag = f"{{{WORD_NAMESPACE}}}tab"
    break_tag = f"{{{WORD_NAMESPACE}}}br"
    for paragraph in root.iter(paragraph_tag):
        fragments: list[str] = []
        for node in paragraph.iter():
            if node.tag == text_tag and node.text:
                fragments.append(node.text)
            elif node.tag == tab_tag:
                fragments.append("\t")
            elif node.tag == break_tag:
                fragments.append("\n")
        value = "".join(fragments).strip()
        if value:
            paragraphs.append(value)
    return "\n".join(paragraphs)


def extract_docx_text(data: bytes) -> tuple[str, dict[str, Any]]:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            _validate_docx_archive(archive)
            names = set(archive.namelist())
            part_names = ["word/document.xml"]
            part_names.extend(
                sorted(
                    name
                    for name in names
                    if re.fullmatch(
                        r"word/(?:header\d+|footer\d+|footnotes|endnotes|comments)\.xml",
                        name,
                    )
                )
            )
            parts = [_docx_part_text(archive.read(name)) for name in part_names]
            media_count = sum(1 for name in names if name.startswith("word/media/"))
    except (zipfile.BadZipFile, KeyError) as exc:
        raise FileContentError("文件不是有效的 .docx Word 文档。") from exc
    text = "\n\n".join(part for part in parts if part).strip()
    return text, {"embedded_images": media_count}


def extract_docx_images(data: bytes, *, limit: int) -> list[tuple[str, bytes]]:
    if limit <= 0:
        return []
    images: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            _validate_docx_archive(archive)
            for name in sorted(archive.namelist()):
                if not name.startswith("word/media/") or name.endswith("/"):
                    continue
                payload = archive.read(name)
                media_type = _image_media_type(payload)
                if media_type:
                    images.append((media_type, payload))
                if len(images) >= limit:
                    break
    except (zipfile.BadZipFile, KeyError) as exc:
        raise FileContentError("文件不是有效的 .docx Word 文档。") from exc
    return images


def extract_pdf_text(data: bytes) -> tuple[str, dict[str, Any]]:
    if b"%PDF-" not in data[:1024]:
        raise FileContentError("文件不是有效的 PDF。")
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise FileContentError("暂不支持有密码的 PDF。")
        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            value = (page.extract_text() or "").strip()
            if value:
                pages.append(f"[第 {index + 1} 页]\n{value}")
    except FileContentError:
        raise
    except Exception as exc:
        raise FileContentError("PDF 已损坏或无法解析。") from exc
    return "\n\n".join(pages), {"pages": len(reader.pages)}


def render_pdf_pages(data: bytes, *, limit: int) -> list[tuple[str, bytes]]:
    if limit <= 0:
        return []
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency is part of the default install
        raise FileContentError("缺少 PDF 视觉渲染依赖 PyMuPDF。") from exc
    images: list[tuple[str, bytes]] = []
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            for page in document[:limit]:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(1.5, 1.5),
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                images.append(("image/jpeg", pixmap.tobytes("jpeg")))
    except Exception as exc:
        raise FileContentError("PDF 页面无法转换为视觉输入。") from exc
    return images


def extract_file(name: str | None, data: bytes) -> ExtractedFile:
    if not data:
        raise FileContentError("附件不能为空。")
    filename = safe_filename(name)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_ATTACHMENT_SUFFIXES:
        raise FileContentError(
            "支持的附件格式：PNG、JPEG、WebP、GIF、PDF、DOCX、TXT、Markdown、JSON。"
        )

    if suffix in IMAGE_MEDIA_TYPES:
        detected = _image_media_type(data)
        if detected is None or detected != IMAGE_MEDIA_TYPES[suffix]:
            raise FileContentError("图片内容与文件扩展名不匹配或文件已损坏。")
        return ExtractedFile(filename, suffix, "image", detected)

    if suffix == ".pdf":
        text, metadata = extract_pdf_text(data)
        return ExtractedFile(filename, suffix, "pdf", DOCUMENT_MEDIA_TYPES[suffix], text, metadata)

    if suffix == ".docx":
        text, metadata = extract_docx_text(data)
        return ExtractedFile(filename, suffix, "word", DOCUMENT_MEDIA_TYPES[suffix], text, metadata)

    if b"\x00" in data:
        raise FileContentError("文本附件包含二进制内容。")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FileContentError("文本附件必须使用 UTF-8 编码。") from exc
    if suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FileContentError("JSON 附件格式无效。") from exc
        text = json.dumps(value, ensure_ascii=False, indent=2)
    return ExtractedFile(filename, suffix, "text", TEXT_MEDIA_TYPES[suffix], text.strip())
