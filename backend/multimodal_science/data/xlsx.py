from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF = re.compile(r"^([A-Z]+)([0-9]+)$")


@dataclass(frozen=True)
class FlatWorksheet:
    name: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


def _column_index(reference: str) -> int:
    match = CELL_REF.match(reference)
    if not match:
        raise ValueError(f"Unsupported XLSX cell reference: {reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    return ["".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")) for item in root]


def _first_sheet(archive: ZipFile) -> tuple[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(f"{{{MAIN_NS}}}sheets")
    if sheets is None or len(sheets) == 0:
        raise ValueError("XLSX workbook contains no worksheets")
    first = sheets[0]
    relationship_id = first.attrib.get(f"{{{REL_NS}}}id")
    if not relationship_id:
        raise ValueError("First XLSX worksheet has no relationship id")

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relationship in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target")
            break
    if not target:
        raise ValueError(f"Missing XLSX worksheet relationship: {relationship_id}")
    if target.startswith("/"):
        sheet_path = target.lstrip("/")
    else:
        sheet_path = posixpath.normpath(posixpath.join("xl", target))
    return first.attrib.get("name", "Sheet1"), sheet_path


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))

    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        index = int(raw)
        if index >= len(shared_strings):
            raise ValueError(f"Shared-string index out of range: {index}")
        return shared_strings[index]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def read_first_flat_worksheet(path: Path) -> FlatWorksheet:
    """Read a flat first worksheet without adding an Excel runtime dependency."""

    try:
        archive = ZipFile(path)
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"Cannot open XLSX workbook: {path}") from exc

    with archive:
        shared_strings = _shared_strings(archive)
        sheet_name, sheet_path = _first_sheet(archive)
        materialized_rows: list[list[Any]] = []
        with archive.open(sheet_path) as stream:
            for _, row_element in ET.iterparse(stream, events=("end",)):
                if row_element.tag != f"{{{MAIN_NS}}}row":
                    continue
                row: list[Any] = []
                for cell in row_element.findall(f"{{{MAIN_NS}}}c"):
                    reference = cell.attrib.get("r")
                    if not reference:
                        raise ValueError(f"XLSX cell without a reference in {path}")
                    index = _column_index(reference)
                    if index >= len(row):
                        row.extend([None] * (index + 1 - len(row)))
                    row[index] = _cell_value(cell, shared_strings)
                if any(value is not None and value != "" for value in row):
                    materialized_rows.append(row)
                row_element.clear()

    if not materialized_rows:
        return FlatWorksheet(name=sheet_name, headers=(), rows=())
    headers = tuple(
        str(value).strip() if value is not None else "" for value in materialized_rows[0]
    )
    if any(not header for header in headers):
        raise ValueError(f"Blank header in first worksheet of {path}")
    if len(set(headers)) != len(headers):
        raise ValueError(f"Duplicate header in first worksheet of {path}")

    records = []
    for row in materialized_rows[1:]:
        padded = row + [None] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded[: len(headers)], strict=True)))
    return FlatWorksheet(name=sheet_name, headers=headers, rows=tuple(records))
