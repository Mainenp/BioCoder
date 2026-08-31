from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multimodal_science.chrompeakformer.private_adapter import extract_job


FAKE_PRIVATE_MODULE = """
import json
from pathlib import Path

def extract_xic_with_pyopenms(mzml_path, output_dir, smooth_sigma=1.0, labels=None):
    payload = {
        "mzml_path": mzml_path,
        "output_dir": output_dir,
        "smooth_sigma": smooth_sigma,
        "labels": labels,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "call.json").write_text(json.dumps(payload), encoding="utf-8")
    return {"n_roi_images": 2}
"""


def make_private_source(root: Path) -> Path:
    source_root = root / "private-source"
    module_path = source_root / "model" / "preprocessing" / "xic_extraction.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(FAKE_PRIVATE_MODULE, encoding="utf-8")
    return source_root


class ChromPeakFormerPrivateAdapterTests(unittest.TestCase):
    def test_label_driven_job_is_converted_to_private_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_root = make_private_source(root)
            source_path = root / "sample.mzML"
            output_dir = root / "output"
            source_path.write_text("mzML", encoding="utf-8")
            job = {
                "derivation_mode": "label_driven",
                "labels": [
                    {"component": "compound-a", "channel": "quant", "rt": 12.25}
                ],
            }
            environment = {
                "CHROMPEAKFORMER_SOURCE_ROOT": str(private_root),
                "CHROMPEAKFORMER_SMOOTH_SIGMA": "1.5",
            }
            with patch.dict(os.environ, environment, clear=False):
                result = extract_job(job, source_path, output_dir)
            call = json.loads((output_dir / "call.json").read_text(encoding="utf-8"))

        self.assertEqual(call["labels"], [
            {
                "compound": "compound-a",
                "channel": "quant",
                "rt": "12.25",
                "ert": "",
            }
        ])
        self.assertEqual(call["smooth_sigma"], 1.5)
        self.assertEqual(result["adapter_version"], "chrompeak-private-adapter-v1")
        self.assertEqual(result["source_api"], "extract_xic_with_pyopenms")
        self.assertRegex(result["private_code_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(str(private_root), json.dumps(result))

    def test_channel_driven_job_passes_no_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_root = make_private_source(root)
            output_dir = root / "output"
            with patch.dict(
                os.environ,
                {"CHROMPEAKFORMER_SOURCE_ROOT": str(private_root)},
                clear=False,
            ):
                extract_job(
                    {"derivation_mode": "channel_driven_inference", "labels": []},
                    root / "sample.mzML",
                    output_dir,
                )
            call = json.loads((output_dir / "call.json").read_text(encoding="utf-8"))

        self.assertIsNone(call["labels"])

    def test_missing_source_configuration_fails_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CHROMPEAKFORMER_SOURCE_ROOT"):
                extract_job(
                    {"derivation_mode": "channel_driven_inference", "labels": []},
                    Path("sample.mzML"),
                    Path("output"),
                )

    def test_invalid_label_rt_is_rejected_before_private_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_root = make_private_source(root)
            job = {
                "derivation_mode": "label_driven",
                "labels": [
                    {"component": "compound-a", "channel": "quant", "rt": "unknown"}
                ],
            }
            with patch.dict(
                os.environ,
                {"CHROMPEAKFORMER_SOURCE_ROOT": str(private_root)},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, r"labels\[0\]\.rt"):
                    extract_job(job, root / "sample.mzML", root / "output")

    def test_unknown_derivation_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_root = make_private_source(root)
            with patch.dict(
                os.environ,
                {"CHROMPEAKFORMER_SOURCE_ROOT": str(private_root)},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported derivation_mode"):
                    extract_job(
                        {"derivation_mode": "unexpected", "labels": []},
                        root / "sample.mzML",
                        root / "output",
                    )

    def test_invalid_smoothing_configuration_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"CHROMPEAKFORMER_SMOOTH_SIGMA": "nan"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                extract_job(
                    {"derivation_mode": "channel_driven_inference", "labels": []},
                    Path("sample.mzML"),
                    Path("output"),
                )
