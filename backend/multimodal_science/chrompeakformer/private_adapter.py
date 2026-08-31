"""Environment-driven adapter for an authorized private ChromPeakFormer checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

SOURCE_ROOT_ENV = "CHROMPEAKFORMER_SOURCE_ROOT"
SMOOTH_SIGMA_ENV = "CHROMPEAKFORMER_SMOOTH_SIGMA"
ADAPTER_VERSION = "chrompeak-private-adapter-v1"
SOURCE_API = "extract_xic_with_pyopenms"

PrivateExtractor = Callable[..., Mapping[str, Any] | None]


def _private_source() -> tuple[Path, Path]:
    configured = os.environ.get(SOURCE_ROOT_ENV, "").strip()
    if not configured:
        raise RuntimeError(f"Set {SOURCE_ROOT_ENV} to the authorized private source root")

    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Configured {SOURCE_ROOT_ENV} is not a directory")

    candidates = (
        (root / "model" / "preprocessing" / "xic_extraction.py", root / "model"),
        (root / "preprocessing" / "xic_extraction.py", root),
    )
    for module_path, model_root in candidates:
        if module_path.is_file():
            return module_path, model_root
    raise FileNotFoundError(
        f"Configured {SOURCE_ROOT_ENV} does not contain "
        "model/preprocessing/xic_extraction.py"
    )


@lru_cache(maxsize=4)
def _load_private_extractor(module_path_text: str, model_root_text: str) -> PrivateExtractor:
    module_path = Path(module_path_text)
    model_root = Path(model_root_text)
    model_root_text = str(model_root)
    if model_root_text not in sys.path:
        sys.path.insert(0, model_root_text)

    os.environ.setdefault("MPLBACKEND", "Agg")
    module_token = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_biocoder_chrompeak_private_{module_token}"
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise ImportError("Could not create a module specification for private source")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    extractor = getattr(module, SOURCE_API, None)
    if not callable(extractor):
        raise AttributeError(f"Private source does not provide callable {SOURCE_API}")
    return extractor


def _smooth_sigma() -> float:
    value = os.environ.get(SMOOTH_SIGMA_ENV, "1.0").strip()
    try:
        sigma = float(value)
    except ValueError as exc:
        raise ValueError(f"{SMOOTH_SIGMA_ENV} must be a number") from exc
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"{SMOOTH_SIGMA_ENV} must be finite and non-negative")
    return sigma


def _source_fingerprint(module_path: Path, model_root: Path) -> str:
    candidates = (
        module_path,
        model_root / "utils" / "mzml_load.py",
        model_root / "utils" / "mzml_chromatogram_ids.py",
    )
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in candidates if candidate.is_file()):
        relative = path.relative_to(model_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _required_text(label: Mapping[str, Any], field: str, index: int) -> str:
    value = str(label.get(field) or "").strip()
    if not value:
        raise ValueError(f"labels[{index}].{field} must be non-empty")
    return value


def _label_rt(label: Mapping[str, Any], index: int) -> str:
    raw_value = label.get("rt")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"labels[{index}].rt must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"labels[{index}].rt must be finite and non-negative")
    return format(value, ".15g")


def _private_labels(job: Mapping[str, Any]) -> list[dict[str, str]] | None:
    mode = job.get("derivation_mode")
    if mode == "channel_driven_inference":
        return None
    if mode != "label_driven":
        raise ValueError(f"Unsupported derivation_mode: {mode}")

    labels = job.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("label_driven jobs require a non-empty labels list")
    converted = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise TypeError(f"labels[{index}] must be an object")
        converted.append(
            {
                "compound": _required_text(label, "component", index),
                "channel": _required_text(label, "channel", index),
                "rt": _label_rt(label, index),
                "ert": "",
            }
        )
    return converted


def extract_job(
    job: dict[str, Any], source_path: Path, output_dir: Path
) -> Mapping[str, Any]:
    """Run one BioCoder derivation job through the authorized private extractor."""

    sigma = _smooth_sigma()
    labels = _private_labels(job)
    module_path, model_root = _private_source()
    source_sha256 = _source_fingerprint(module_path, model_root)
    extractor = _load_private_extractor(str(module_path), str(model_root))
    result = extractor(
        str(source_path),
        str(output_dir),
        smooth_sigma=sigma,
        labels=labels,
    )
    if result is not None and not isinstance(result, Mapping):
        raise TypeError("Private extractor result must be a mapping or None")
    return {
        **dict(result or {}),
        "adapter_version": ADAPTER_VERSION,
        "source_api": SOURCE_API,
        "private_code_sha256": source_sha256,
        "smooth_sigma": sigma,
    }
