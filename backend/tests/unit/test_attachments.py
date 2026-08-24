import io
import zipfile

import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.services.attachments import AttachmentStore, FileContentError, build_user_content
from app.services.file_content import extract_file

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8"
    b"\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _docx_bytes(text: str, *, include_image: bool = False) -> bytes:
    output = io.BytesIO()
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", document)
        if include_image:
            archive.writestr("word/media/image1.png", PNG_1X1)
    return output.getvalue()


def _blank_pdf_bytes() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(output)
    return output.getvalue()


def test_extracts_docx_text_and_embedded_image_metadata() -> None:
    extracted = extract_file("研究报告.docx", _docx_bytes("EGFR C797S", include_image=True))

    assert extracted.kind == "word"
    assert extracted.text == "EGFR C797S"
    assert extracted.metadata["embedded_images"] == 1


def test_rejects_an_image_whose_content_does_not_match_its_suffix() -> None:
    with pytest.raises(FileContentError, match="不匹配"):
        extract_file("fake.png", b"not an image")


def test_extracts_pdf_page_metadata() -> None:
    extracted = extract_file("paper.pdf", _blank_pdf_bytes())

    assert extracted.kind == "pdf"
    assert extracted.metadata["pages"] == 1


def test_attachment_store_round_trip_builds_multimodal_message(tmp_path) -> None:
    settings = Settings(
        attachments_dir=tmp_path / "attachments",
        vision_input_enabled=True,
        attachment_max_vision_images=4,
    )
    store = AttachmentStore(settings)
    image = store.save("cell.png", PNG_1X1)
    word = store.save("report.docx", _docx_bytes("肿瘤细胞实验结果", include_image=True))

    prepared = store.prepare([image.id, word.id])
    content = build_user_content("分析附件", prepared, max_characters=10_000)

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "肿瘤细胞实验结果" in content[0]["text"]
    assert [block["type"] for block in content].count("image_url") == 2


def test_text_documents_work_when_vision_input_is_disabled(tmp_path) -> None:
    settings = Settings(
        attachments_dir=tmp_path / "attachments",
        vision_input_enabled=False,
    )
    store = AttachmentStore(settings)
    document = store.save("notes.txt", "PARP 抑制剂".encode())

    prepared = store.prepare([document.id])

    assert prepared[0].text == "PARP 抑制剂"
    assert build_user_content("总结", prepared, 1000).startswith("用户请求")


def test_json_payload_and_metadata_are_stored_separately(tmp_path) -> None:
    settings = Settings(attachments_dir=tmp_path / "attachments")
    store = AttachmentStore(settings)
    document = store.save("data.json", b'{"target":"EGFR"}')

    descriptor, payload, text = store.load(document.id)

    assert descriptor.name == "data.json"
    assert payload == b'{"target":"EGFR"}'
    assert '"target": "EGFR"' in text


def test_images_require_a_vision_capable_model(tmp_path) -> None:
    settings = Settings(
        attachments_dir=tmp_path / "attachments",
        vision_input_enabled=False,
    )
    store = AttachmentStore(settings)
    image = store.save("cell.png", PNG_1X1)

    with pytest.raises(FileContentError, match="未启用视觉输入"):
        store.prepare([image.id])
