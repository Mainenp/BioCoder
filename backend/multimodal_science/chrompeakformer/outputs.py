from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = frozenset({"Compound Name", "native_id", "mz", "q3", "RT"})
WINDOW_COLUMNS = frozenset({"image", "rt_lo", "rt_hi"})
NUMPY_MAGIC = b"\x93NUMPY"
DTYPE_PATTERN = re.compile(r"^[<>=|]?([?bBiufcSUV])(\d+)$")


@dataclass(frozen=True)
class NpyMetadata:
    dtype: str
    fortran_order: bool
    shape: tuple[int, ...]
    data_offset: int
    byte_count: int


@dataclass(frozen=True)
class OutputSummary:
    status: str
    feature_count: int
    window_count: int
    image_count: int
    sequence_count: int
    sequence_points: int
    output_sha256: str
    files: tuple[str, ...]


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output path escapes job directory: {relative}") from exc
    return candidate


def _read_csv(path: Path, required_columns: frozenset[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required output is missing: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        return list(reader)


def _dtype_item_size(descriptor: str) -> int:
    match = DTYPE_PATTERN.fullmatch(descriptor)
    if not match:
        raise ValueError(f"Unsupported NumPy dtype descriptor: {descriptor}")
    kind, size_text = match.groups()
    size = int(size_text)
    if size <= 0:
        raise ValueError(f"Invalid NumPy dtype item size: {descriptor}")
    return size * 4 if kind == "U" else size


def read_npy_metadata(path: Path) -> NpyMetadata:
    if not path.is_file():
        raise FileNotFoundError(f"Required output is missing: {path.name}")
    with path.open("rb") as stream:
        if stream.read(len(NUMPY_MAGIC)) != NUMPY_MAGIC:
            raise ValueError(f"Invalid NumPy magic bytes: {path.name}")
        version_bytes = stream.read(2)
        if len(version_bytes) != 2:
            raise ValueError(f"Truncated NumPy version header: {path.name}")
        major, minor = version_bytes
        if major == 1:
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"Truncated NumPy header length: {path.name}")
            header_length = struct.unpack("<H", length_bytes)[0]
            encoding = "latin1"
        elif major in {2, 3}:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError(f"Truncated NumPy header length: {path.name}")
            header_length = struct.unpack("<I", length_bytes)[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"Unsupported NumPy format version: {major}.{minor}")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"Truncated NumPy header: {path.name}")
        try:
            header = ast.literal_eval(header_bytes.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid NumPy header: {path.name}") from exc
        data_offset = stream.tell()

    if not isinstance(header, dict):
        raise ValueError(f"NumPy header must be a dictionary: {path.name}")
    descriptor = header.get("descr")
    shape = header.get("shape")
    fortran_order = header.get("fortran_order")
    if not isinstance(descriptor, str):
        raise ValueError(f"Structured NumPy dtypes are not supported: {path.name}")
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"NumPy shape must be a non-empty tuple: {path.name}")
    if any(not isinstance(value, int) or value < 0 for value in shape):
        raise ValueError(f"NumPy shape contains invalid dimensions: {shape}")
    if not isinstance(fortran_order, bool):
        raise ValueError(f"NumPy fortran_order must be boolean: {path.name}")

    item_size = _dtype_item_size(descriptor)
    value_count = math.prod(shape)
    byte_count = value_count * item_size
    actual_size = path.stat().st_size
    if actual_size != data_offset + byte_count:
        raise ValueError(
            f"NumPy payload size mismatch for {path.name}: "
            f"expected {data_offset + byte_count}, got {actual_size}"
        )
    return NpyMetadata(
        dtype=descriptor,
        fortran_order=fortran_order,
        shape=shape,
        data_offset=data_offset,
        byte_count=byte_count,
    )


def _validate_windows(root: Path, rows: list[dict[str, str]]) -> list[str]:
    image_paths = []
    seen = set()
    for index, row in enumerate(rows, start=2):
        image = str(row.get("image") or "").strip().replace("\\", "/")
        if not image:
            raise ValueError(f"roi_windows.csv:{index} has an empty image path")
        if image in seen:
            raise ValueError(f"roi_windows.csv contains duplicate image: {image}")
        seen.add(image)
        image_path = _safe_child(root, image)
        if image_path.suffix.casefold() not in {".jpeg", ".jpg"}:
            raise ValueError(f"ROI image must be JPEG: {image}")
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"ROI image is missing or empty: {image}")
        image_bytes = image_path.read_bytes()
        if not image_bytes.startswith(b"\xff\xd8") or not image_bytes.endswith(b"\xff\xd9"):
            raise ValueError(f"ROI image does not contain a complete JPEG stream: {image}")
        try:
            rt_lo = float(row["rt_lo"])
            rt_hi = float(row["rt_hi"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"roi_windows.csv:{index} has invalid RT bounds") from exc
        if not math.isfinite(rt_lo) or not math.isfinite(rt_hi) or rt_hi <= rt_lo:
            raise ValueError(f"roi_windows.csv:{index} has invalid RT interval")
        image_paths.append(image)
    return image_paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_signature(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_file_sha256(_safe_child(root, relative)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_outputs(output_dir: Path) -> OutputSummary:
    root = output_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Job output directory not found: {root}")

    feature_rows = _read_csv(root / "feature.csv", FEATURE_COLUMNS)
    window_rows = _read_csv(root / "roi_windows.csv", WINDOW_COLUMNS)
    if not feature_rows:
        raise ValueError("feature.csv contains no extracted channels")
    for index, row in enumerate(feature_rows, start=2):
        if not str(row.get("native_id") or "").strip():
            raise ValueError(f"feature.csv:{index} has an empty native_id")
        try:
            rt = float(row["RT"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"feature.csv:{index} has an invalid RT") from exc
        if not math.isfinite(rt):
            raise ValueError(f"feature.csv:{index} has a non-finite RT")
    if len(feature_rows) != len(window_rows):
        raise ValueError(
            f"Feature/window count mismatch: {len(feature_rows)} != {len(window_rows)}"
        )
    image_paths = _validate_windows(root, window_rows)
    matrix = read_npy_metadata(root / "xic_matrix.npy")
    if len(matrix.shape) != 2:
        raise ValueError(f"xic_matrix.npy must be two-dimensional, got {matrix.shape}")
    sequence_count = matrix.shape[0] - 1
    sequence_points = matrix.shape[1]
    if sequence_count != len(feature_rows):
        raise ValueError(
            f"Feature/sequence count mismatch: {len(feature_rows)} != {sequence_count}"
        )
    if sequence_points < 2:
        raise ValueError("xic_matrix.npy must contain at least two RT points")

    required_files = ["feature.csv", "roi_windows.csv", "xic_matrix.npy", *image_paths]
    return OutputSummary(
        status="ok",
        feature_count=len(feature_rows),
        window_count=len(window_rows),
        image_count=len(image_paths),
        sequence_count=sequence_count,
        sequence_points=sequence_points,
        output_sha256=_output_signature(root, required_files),
        files=tuple(sorted(required_files)),
    )


def summary_payload(summary: OutputSummary) -> dict[str, Any]:
    return {
        "status": summary.status,
        "feature_count": summary.feature_count,
        "window_count": summary.window_count,
        "image_count": summary.image_count,
        "sequence_count": summary.sequence_count,
        "sequence_points": summary.sequence_points,
        "output_sha256": summary.output_sha256,
        "files": list(summary.files),
    }


def write_summary(path: Path, summary: OutputSummary) -> None:
    path.write_text(
        json.dumps(summary_payload(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
