"""Build the preregistered C0/C1/R1/R2/R3 selection checkpoint matrix."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths
from multimodal_science.qwen3vl.rl_selection import (
    _hex,
    _manifest_entries,
    _object,
    _read_json,
    verify_candidate_descriptor,
)


CANDIDATE_MATRIX_SCHEMA = "chrompeak-qwen3vl-rl-selection-candidate-matrix-v1"
EXPECTED_CHECKPOINT_STEPS = (100, 200, 300)
EXPECTED_RUN_LABELS = ("R1", "R2", "R3", "C1-original", "C1-1e-6")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CandidateMatrixResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    candidate_list_path: Path
    candidates: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _report_path(root: Path, family: str) -> Path:
    name = (
        "sft_continue_training_report.json"
        if family == "C1"
        else "fusion_training_report.json"
    )
    return root / name


def build_candidate_matrix(
    *,
    code_revision: str,
    rl_pool_report_sha256: str,
    c0_root: Path,
    c0_report_sha256: str,
    c0_manifest_sha256: str,
    run_spec: Mapping[str, Any],
    output_dir: Path,
) -> CandidateMatrixResult:
    """Emit exactly 16 immutable candidate descriptors from completed formal runs."""

    _require(bool(_HEX_40.fullmatch(code_revision)), "Code revision must be a full Git SHA")
    _hex(rl_pool_report_sha256, "RL pool report SHA-256")
    _hex(c0_report_sha256, "C0 report SHA-256")
    _hex(c0_manifest_sha256, "C0 manifest SHA-256")
    c0_root = c0_root.resolve()
    output_dir = output_dir.resolve()
    runs = _object(run_spec.get("runs"), "candidate run specification")
    _require(tuple(sorted(runs)) == tuple(sorted(EXPECTED_RUN_LABELS)), "Candidate run labels drifted")
    roots = [c0_root]
    for label in EXPECTED_RUN_LABELS:
        entry = _object(runs[label], f"candidate run {label}")
        roots.append(Path(str(entry.get("root") or "")).resolve())
    assert_no_protected_input_paths([*roots, output_dir])
    _require(not output_dir.exists(), f"Candidate matrix output exists: {output_dir}")

    c0_report_path = c0_root / "fusion_training_report.json"
    c0_manifest_path = c0_root / "artifact_manifest.sha256"
    _require(sha256_file(c0_report_path) == c0_report_sha256, "C0 report drift")
    _require(sha256_file(c0_manifest_path) == c0_manifest_sha256, "C0 manifest drift")
    _, c0_entries = _manifest_entries(c0_root)
    for relative in (
        "fusion_training_report.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "sensor_projector.safetensors",
    ):
        _require(relative in c0_entries, f"C0 artifact is not manifest-bound: {relative}")
    c0_report = _read_json(c0_report_path, "C0 training report")
    c0_contracts = _object(c0_report.get("contracts"), "C0 contracts")
    _require(
        c0_contracts.get("internal_test_accessed") is False
        and c0_contracts.get("train_split_only") is True,
        "C0 is not an eligible train-only starting point",
    )

    descriptors: list[tuple[str, dict[str, Any]]] = []
    c0 = {
        "family": "C0",
        "config": "xic-only-sft-seed17",
        "checkpoint": {
            "step": 0,
            "path": str(c0_root / "sensor_projector.safetensors"),
            "sha256": c0_entries["sensor_projector.safetensors"],
        },
        "training_report": {
            "path": str(c0_report_path),
            "sha256": c0_report_sha256,
        },
        "sources": {
            "rl_pool_report_sha256": rl_pool_report_sha256,
            "starting_adapter_report_sha256": c0_report_sha256,
            "starting_adapter_manifest_sha256": c0_manifest_sha256,
        },
        "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0},
    }
    descriptors.append(("c0-step-00000000.json", c0))

    for label in EXPECTED_RUN_LABELS:
        entry = _object(runs[label], f"candidate run {label}")
        root = Path(str(entry["root"])).resolve()
        family = "C1" if label.startswith("C1-") else label
        config = entry.get("config")
        _require(isinstance(config, str) and bool(config), f"Missing config for {label}")
        report_path = _report_path(root, family)
        _require(report_path.is_file(), f"Missing completed report for {label}")
        _, root_entries = _manifest_entries(root)
        relative_report = report_path.relative_to(root).as_posix()
        _require(relative_report in root_entries, f"Training report is not manifest-bound: {label}")
        report = _read_json(report_path, f"training report {label}")
        sources = _object(report.get("sources"), f"training sources {label}")
        _require(sources.get("rl_pool_report_sha256") == rl_pool_report_sha256, f"RL pool provenance drift: {label}")
        _require(
            sources.get("starting_adapter_report_sha256") == c0_report_sha256
            and sources.get("starting_adapter_manifest_sha256") == c0_manifest_sha256,
            f"C0 lineage drift: {label}",
        )
        contracts = _object(report.get("contracts"), f"training contracts {label}")
        for key in (
            "selection_groups_trained",
            "validation_prompts_opened",
            "validation_answers_opened",
            "internal_test_accessed",
        ):
            _require(contracts.get(key) is False, f"Protected-data contract failed for {label}: {key}")
        _require(report.get("development_training_complete") is True, f"Incomplete formal run: {label}")
        for step in EXPECTED_CHECKPOINT_STEPS:
            checkpoint_root = root / "checkpoints" / f"step-{step:08d}"
            _, checkpoint_entries = _manifest_entries(checkpoint_root)
            checkpoint_path = checkpoint_root / "checkpoint.json"
            _require(
                checkpoint_entries.get("checkpoint.json") == sha256_file(checkpoint_path),
                f"Checkpoint is not manifest-bound: {label}/{step}",
            )
            state = _read_json(checkpoint_path, f"checkpoint {label}/{step}")
            _require(state.get("global_step") == step, f"Checkpoint step drift: {label}/{step}")
            descriptor = {
                "family": family,
                "config": config,
                "checkpoint": {
                    "step": step,
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_entries["checkpoint.json"],
                },
                "training_report": {
                    "path": str(report_path),
                    "sha256": root_entries[relative_report],
                },
                "sources": {
                    key: value
                    for key, value in sources.items()
                    if key.endswith("_sha256")
                },
                "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0},
            }
            verify_candidate_descriptor(
                descriptor, expected_pool_report_sha256=rl_pool_report_sha256
            )
            safe_label = label.lower().replace("-", "_")
            descriptors.append((f"{safe_label}-step-{step:08d}.json", descriptor))
    _require(len(descriptors) == 16, "Preregistered selection matrix must contain 16 candidates")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temporary:
        staging = Path(temporary)
        descriptor_rows = []
        descriptor_paths: list[Path] = []
        for name, descriptor in descriptors:
            path = staging / name
            _write_json(path, descriptor)
            descriptor_paths.append(path)
            bound = verify_candidate_descriptor(
                descriptor, expected_pool_report_sha256=rl_pool_report_sha256
            )
            descriptor_rows.append(
                {
                    "candidate_id": bound["candidate_id"],
                    "family": bound["family"],
                    "config": bound["config"],
                    "checkpoint_step": bound["checkpoint"]["step"],
                    "path": name,
                    "sha256": sha256_file(path),
                }
            )
        candidate_list_path = staging / "candidate_list.jsonl"
        candidate_list_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in descriptor_rows),
            encoding="utf-8",
        )
        spec_path = staging / "candidate_matrix_spec.json"
        _write_json(spec_path, run_spec)
        report = {
            "schema_version": CANDIDATE_MATRIX_SCHEMA,
            "code_revision": code_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "rl_pool_report_sha256": rl_pool_report_sha256,
                "c0_report_sha256": c0_report_sha256,
                "c0_manifest_sha256": c0_manifest_sha256,
                "candidate_matrix_spec_sha256": sha256_file(spec_path),
            },
            "counts": {
                "candidates": len(descriptor_rows),
                "families": {family: sum(row["family"] == family for row in descriptor_rows) for family in ("C0", "C1", "R1", "R2", "R3")},
                "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
            },
            "contracts": {
                "greedy_decode_only": True,
                "selection_answers_opened": False,
                "validation_prompts_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            "artifacts": {
                "candidate_list": {
                    "path": candidate_list_path.name,
                    "sha256": sha256_file(candidate_list_path),
                    "records": len(descriptor_rows),
                },
                "candidate_matrix_spec": {
                    "path": spec_path.name,
                    "sha256": sha256_file(spec_path),
                },
            },
        }
        report_path = staging / "candidate_matrix_report.json"
        _write_json(report_path, report)
        manifest_path = staging / "artifact_manifest.sha256"
        manifest_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in (*descriptor_paths, candidate_list_path, spec_path, report_path)
            ),
            encoding="utf-8",
        )
        staging.replace(output_dir)
    final_report = output_dir / "candidate_matrix_report.json"
    return CandidateMatrixResult(
        output_dir=output_dir,
        report_path=final_report,
        report_sha256=sha256_file(final_report),
        candidate_list_path=output_dir / "candidate_list.jsonl",
        candidates=len(descriptors),
    )
