from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from multimodal_science.data.contracts import MatchStrategy, decide_eligibility
from multimodal_science.data.manifest import build_manifest, mzml_chromatogram_count
from multimodal_science.data.xlsx import read_first_flat_worksheet


def column_name(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def cell_xml(reference: str, value: object) -> str:
    if isinstance(value, str):
        return f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    return f'<c r="{reference}"><v>{value}</v></c>'


def write_flat_xlsx(path: Path, headers: list[str], rows: list[list[object | None]]) -> None:
    sheet_rows = []
    for row_index, row in enumerate([headers, *rows], start=1):
        cells = []
        for column_index, value in enumerate(row):
            if value is None:
                continue
            reference = f"{column_name(column_index)}{row_index}"
            cells.append(cell_xml(reference, value))
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Labels" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>"""
            + "".join(sheet_rows)
            + "</sheetData></worksheet>",
        )


def label_headers() -> list[str]:
    return [
        "roi_id",
        "comonent",
        "channel",
        "rt",
        "peak_label",
        "peak_count",
        "peak_start1",
        "peak_end1",
        "area1",
        "snr",
        "instrument",
        "raw_file",
        "sample_id",
        "product_id",
    ]


class MultimodalManifestTests(unittest.TestCase):
    def test_xlsx_reader_preserves_sparse_columns_and_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook_path = Path(directory) / "labels.xlsx"
            write_flat_xlsx(
                workbook_path,
                ["roi_id", "sample_id", "note"],
                [[1, "样本一.mzML", None], [2, None, "empty sample id"]],
            )

            worksheet = read_first_flat_worksheet(workbook_path)

        self.assertEqual(worksheet.name, "Labels")
        self.assertEqual(worksheet.headers, ("roi_id", "sample_id", "note"))
        self.assertEqual(
            worksheet.rows[0],
            {"roi_id": 1, "sample_id": "样本一.mzML", "note": None},
        )
        self.assertIsNone(worksheet.rows[1]["sample_id"])

    def test_exact_positive_is_eligible(self) -> None:
        decision = decide_eligibility(
            {
                "sample_id": "sample.mzML",
                "peak_label": 1,
                "peak_count": 1,
                "peak_start1": 1.0,
                "peak_end1": 1.2,
            },
            matched_source_count=1,
            alternate_label_variant=False,
        )

        self.assertEqual(decision.match_strategy, MatchStrategy.EXACT_SAMPLE_ID)
        self.assertEqual(str(decision.match_strategy), "exact_sample_id")
        self.assertEqual(decision.fallback_order, 0)
        self.assertTrue(decision.train_eligible)
        self.assertFalse(decision.audit_bucket)

    def test_positive_without_valid_boundary_enters_audit_bucket(self) -> None:
        decision = decide_eligibility(
            {"sample_id": "sample.mzML", "peak_label": 1, "peak_count": 1},
            matched_source_count=1,
            alternate_label_variant=False,
        )

        self.assertFalse(decision.train_eligible)
        self.assertTrue(decision.audit_bucket)
        self.assertIn("invalid_positive_boundary", decision.reasons)

    def test_alternate_label_variant_cannot_train(self) -> None:
        decision = decide_eligibility(
            {"sample_id": "sample.mzML", "peak_label": 0, "peak_count": 0},
            matched_source_count=1,
            alternate_label_variant=True,
        )

        self.assertFalse(decision.train_eligible)
        self.assertEqual(decision.fallback_order, 0)
        self.assertEqual(decision.reasons, ("alternate_label_variant",))

    def test_invalid_source_signal_enters_audit_bucket(self) -> None:
        decision = decide_eligibility(
            {"sample_id": "sample.mzML", "peak_label": 0, "peak_count": 0},
            matched_source_count=1,
            alternate_label_variant=False,
            source_signal_valid=False,
        )

        self.assertFalse(decision.train_eligible)
        self.assertIn("invalid_source_signal", decision.reasons)

    def test_mzml_probe_recovers_invalid_utf8_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-encoding.mzML"
            path.write_bytes(
                b"<mzML><userParam value='\xb2\xe2\xca\xd4'/><chromatogram id='one'/></mzML>"
            )

            count, warning = mzml_chromatogram_count(path)

        self.assertEqual(count, 1)
        self.assertIsNotNone(warning)
        self.assertIn("recovered_invalid_utf8", warning)

    def test_build_manifest_matches_exact_sources_and_isolates_audit_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_root = tmp_path / "data"
            label_dir = data_root / "label"
            mzml_dir = data_root / "mzml" / "train"
            label_dir.mkdir(parents=True)
            mzml_dir.mkdir(parents=True)
            (mzml_dir / "sample-a.mzML").write_text(
                "<mzML><chromatogram id='sample-a'/></mzML>", encoding="utf-8"
            )
            (mzml_dir / "unlabelled.mzML").write_text(
                "<mzML><chromatogram id='unlabelled'/></mzML>", encoding="utf-8"
            )
            rows = [
                [
                    1,
                    "compound-a",
                    "quantifier",
                    1.1,
                    1,
                    1,
                    1.0,
                    1.2,
                    10.0,
                    5.0,
                    "test",
                    "raw",
                    "sample-a.mzML",
                    "p1",
                ],
                [
                    2,
                    "compound-b",
                    "quantifier",
                    2.1,
                    0,
                    0,
                    None,
                    None,
                    0.0,
                    2.0,
                    "test",
                    "raw",
                    "missing.mzML",
                    "p2",
                ],
            ]
            write_flat_xlsx(label_dir / "train.xlsx", label_headers(), rows)
            write_flat_xlsx(label_dir / "train_traditional.xlsx", label_headers(), rows[:1])

            result = build_manifest(data_root, tmp_path / "output", source_archive_sha256="a" * 64)
            records = [
                json.loads(line)
                for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            self.assertEqual(result.dataset_version, "raw-aaaaaaaa")
            self.assertEqual(result.record_count, 3)
            self.assertEqual(result.train_eligible_count, 1)
            self.assertEqual(result.audit_count, 2)
            self.assertEqual(records[0]["component"], "compound-a")
            self.assertEqual(records[0]["source_mzml"], "mzml/train/sample-a.mzML")
            self.assertEqual(records[0]["fallback_order"], 0)
            self.assertEqual(records[0]["chromatogram_count"], 1)
            self.assertEqual(records[0]["source_signal_status"], "strict")
            self.assertIsNone(records[1]["source_mzml"])
            self.assertEqual(records[2]["label_variant"], "alternate")
            self.assertTrue(
                all(
                    not str(value).startswith(str(tmp_path))
                    for record in records
                    for value in record.values()
                )
            )
            self.assertEqual(report["counts"]["unlabeled_mzml"], 1)
            self.assertEqual(report["duplicate_mzml_content"], [])
            self.assertEqual(report["label_summary"]["positive"], 1)
            self.assertEqual(report["mzml_signal_summary"][0]["invalid_files"], 0)
            self.assertEqual(report["mzml_signal_summary"][0]["recovered_files"], 0)


if __name__ == "__main__":
    unittest.main()
