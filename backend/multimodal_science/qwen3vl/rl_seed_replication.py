"""Selection-only seed replication for the selected v2 R2 and C1 recipes.

This is a post-selection robustness supplement.  It does not participate in the
preregistered model choice and deliberately has no validation or internal-test input.
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_data import assert_no_protected_input_paths
from multimodal_science.qwen3vl.rl_selection import (
    _evaluation_descriptor,
    _hex,
    _manifest_entries,
    _object,
    _read_json,
    verify_candidate_descriptor,
)


REPLICATION_CANDIDATE_SCHEMA = "chrompeak-qwen3vl-rl-seed-replication-candidates-v1"
REPLICATION_ANALYSIS_SCHEMA = "chrompeak-qwen3vl-rl-seed-replication-analysis-v1"
REPLICATION_SEEDS = (17, 29)
REPLICATION_STEP = 300


@dataclass(frozen=True)
class SeedReplicationResult:
    output_dir: Path
    report_path: Path
    report_sha256: str
    manifest_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, paths: list[Path]) -> Path:
    manifest = root / "artifact_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in sorted(paths)
        ),
        encoding="utf-8",
    )
    return manifest


def _verified_training_run(
    *,
    root: Path,
    family: str,
    seed: int,
    pool_sha: str,
    c0_report_sha: str,
    c0_manifest_sha: str,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    root = root.resolve()
    assert_no_protected_input_paths([root])
    report_name = (
        "fusion_training_report.json"
        if family == "R2"
        else "sft_continue_training_report.json"
    )
    report_path = root / report_name
    config_path = root / "training_config.json"
    _, entries = _manifest_entries(root)
    for relative in (report_name, "training_config.json"):
        path = root / relative
        _require(
            entries.get(relative) == sha256_file(path),
            f"Seed-{seed} {family} artifact is not manifest-bound: {relative}",
        )
    report = _read_json(report_path, f"seed-{seed} {family} training report")
    config = _read_json(config_path, f"seed-{seed} {family} training config")
    settings = _object(config.get("settings"), f"seed-{seed} {family} settings")
    _require(settings.get("seed") == seed, f"Seed drift for {family}")
    _require(settings.get("run_mode") == "formal", f"Non-formal seed-{seed} {family} run")
    if family == "R2":
        _require(settings.get("experiment") == "R2", "R2 experiment drift")
    else:
        _require(
            settings.get("learning_rate_choice") == "1e-6",
            "C1 learning-rate choice drift",
        )
    _require(report.get("development_training_complete") is True, f"Incomplete {family} run")
    sources = _object(report.get("sources"), f"seed-{seed} {family} sources")
    _require(sources.get("rl_pool_report_sha256") == pool_sha, f"{family} RL pool drift")
    _require(
        sources.get("starting_adapter_report_sha256") == c0_report_sha
        and sources.get("starting_adapter_manifest_sha256") == c0_manifest_sha,
        f"{family} C0 lineage drift",
    )
    contracts = _object(report.get("contracts"), f"seed-{seed} {family} contracts")
    if family == "R2":
        training = _object(report.get("training"), f"seed-{seed} R2 training")
        _require(training.get("experiment") == "R2", "R2 report experiment drift")
        _require(contracts.get("projector_trainable") is False, "R2 projector must be frozen")
        _require(
            math.isclose(float(training.get("lora_learning_rate")), 5e-6),
            "R2 learning-rate drift",
        )
    for key in (
        "selection_groups_trained",
        "validation_prompts_opened",
        "validation_answers_opened",
        "internal_test_accessed",
    ):
        _require(contracts.get(key) is False, f"Protected-data contract failed: {family}/{key}")
    return report_path, report, entries


def build_seed_replication_candidates(
    *,
    code_revision: str,
    rl_pool_report_sha256: str,
    c0_report_sha256: str,
    c0_manifest_sha256: str,
    r2_root: Path,
    c1_root: Path,
    output_dir: Path,
    seed: int = 29,
) -> SeedReplicationResult:
    """Build two immutable final-checkpoint descriptors for the seed replication."""

    _require(len(code_revision) == 40 and all(c in "0123456789abcdef" for c in code_revision), "Code revision must be a full Git SHA")
    pool_sha = _hex(rl_pool_report_sha256, "RL pool report SHA-256")
    c0_report_sha = _hex(c0_report_sha256, "C0 report SHA-256")
    c0_manifest_sha = _hex(c0_manifest_sha256, "C0 manifest SHA-256")
    _require(seed == 29, "This supplement is frozen to seed 29")
    output_dir = output_dir.resolve()
    assert_no_protected_input_paths([r2_root, c1_root, output_dir])
    _require(not output_dir.exists(), f"Seed replication output exists: {output_dir}")

    specs = (
        ("R2", r2_root, "lora-lr-5e-6-projector-frozen-seed29"),
        ("C1", c1_root, "lora-lr-1e-6-seed29"),
    )
    descriptors: list[tuple[str, dict[str, Any]]] = []
    for family, root_value, config_name in specs:
        root = root_value.resolve()
        report_path, report, root_entries = _verified_training_run(
            root=root,
            family=family,
            seed=seed,
            pool_sha=pool_sha,
            c0_report_sha=c0_report_sha,
            c0_manifest_sha=c0_manifest_sha,
        )
        checkpoint_root = root / "checkpoints" / f"step-{REPLICATION_STEP:08d}"
        _, checkpoint_entries = _manifest_entries(checkpoint_root)
        checkpoint_path = checkpoint_root / "checkpoint.json"
        _require(
            checkpoint_entries.get("checkpoint.json") == sha256_file(checkpoint_path),
            f"Seed-{seed} {family} checkpoint is not manifest-bound",
        )
        checkpoint = _read_json(checkpoint_path, f"seed-{seed} {family} checkpoint")
        _require(checkpoint.get("global_step") == REPLICATION_STEP, f"{family} checkpoint step drift")
        sources = _object(report.get("sources"), f"seed-{seed} {family} sources")
        descriptor = {
            "family": family,
            "config": config_name,
            "checkpoint": {
                "step": REPLICATION_STEP,
                "path": str(checkpoint_path),
                "sha256": checkpoint_entries["checkpoint.json"],
            },
            "training_report": {
                "path": str(report_path),
                "sha256": root_entries[report_path.name],
            },
            "sources": {
                key: value for key, value in sources.items() if key.endswith("_sha256")
            },
            "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0},
        }
        verify_candidate_descriptor(descriptor, expected_pool_report_sha256=pool_sha)
        descriptors.append((f"{family.lower()}-seed29-step-{REPLICATION_STEP:08d}.json", descriptor))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary)
        artifacts: list[Path] = []
        rows: list[dict[str, Any]] = []
        for name, descriptor in descriptors:
            path = staging / name
            _write_json(path, descriptor)
            artifacts.append(path)
            bound = verify_candidate_descriptor(descriptor, expected_pool_report_sha256=pool_sha)
            rows.append(
                {
                    "candidate_id": bound["candidate_id"],
                    "family": bound["family"],
                    "seed": seed,
                    "checkpoint_step": REPLICATION_STEP,
                    "path": name,
                    "sha256": sha256_file(path),
                }
            )
        candidate_list = staging / "candidate_list.jsonl"
        candidate_list.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        artifacts.append(candidate_list)
        report = {
            "schema_version": REPLICATION_CANDIDATE_SCHEMA,
            "code_revision": code_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "post_selection_seed_robustness_supplement",
            "sources": {
                "rl_pool_report_sha256": pool_sha,
                "c0_report_sha256": c0_report_sha,
                "c0_manifest_sha256": c0_manifest_sha,
            },
            "counts": {"candidates": 2, "seed": seed, "checkpoint_step": REPLICATION_STEP},
            "contracts": {
                "changes_preregistered_selection": False,
                "selection_groups_trained": False,
                "selection_set_evaluation_only": True,
                "validation_prompts_opened": False,
                "validation_answers_opened": False,
                "internal_test_accessed": False,
            },
            "artifacts": {
                "candidate_list": {
                    "path": candidate_list.name,
                    "sha256": sha256_file(candidate_list),
                    "records": 2,
                },
                "candidates": [
                    {"path": row["path"], "sha256": row["sha256"]} for row in rows
                ],
            },
        }
        report_path = staging / "seed_replication_candidate_report.json"
        _write_json(report_path, report)
        artifacts.append(report_path)
        _write_manifest(staging, artifacts)
        staging.replace(output_dir)
    report_path = output_dir / "seed_replication_candidate_report.json"
    return SeedReplicationResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        manifest_path=output_dir / "artifact_manifest.sha256",
    )


def _load_selection_evaluation(
    path: Path, *, expected_seed: int, expected_family: str
) -> dict[str, Any]:
    descriptor = _evaluation_descriptor(path.resolve())
    report = _read_json(path.resolve(), "seed replication selection evaluation")
    contracts = _object(report.get("contracts"), "selection evaluation contracts")
    _require(contracts.get("selection_groups_only") is True, "Evaluation is not selection-only")
    _require(contracts.get("validation_answers_opened") is False, "Validation answers were opened")
    _require(contracts.get("internal_test_accessed") is False, "Internal test was accessed")
    candidate = _object(descriptor.get("candidate"), "selection candidate")
    _require(candidate.get("family") == expected_family, "Candidate family drift")
    config = candidate.get("config")
    expected_config = (
        "lora-lr-5e-6-projector-frozen"
        if expected_family == "R2"
        else "lora-lr-1e-6"
    )
    if expected_seed == 29:
        expected_config += "-seed29"
    _require(config == expected_config, "Candidate seed/config drift")
    _require(
        _object(candidate.get("checkpoint"), "candidate checkpoint").get("step")
        == REPLICATION_STEP,
        "Candidate checkpoint drift",
    )
    return descriptor


def analyze_seed_replication(
    *,
    seed17_r2_evaluation: Path,
    seed17_c1_evaluation: Path,
    seed29_r2_evaluation: Path,
    seed29_c1_evaluation: Path,
    output_dir: Path,
) -> SeedReplicationResult:
    """Summarize paired R2/C1 selection-set results for seeds 17 and 29."""

    inputs = {
        17: {
            "R2": seed17_r2_evaluation.resolve(),
            "C1": seed17_c1_evaluation.resolve(),
        },
        29: {
            "R2": seed29_r2_evaluation.resolve(),
            "C1": seed29_c1_evaluation.resolve(),
        },
    }
    output_dir = output_dir.resolve()
    assert_no_protected_input_paths([*(path for rows in inputs.values() for path in rows.values()), output_dir])
    _require(not output_dir.exists(), f"Seed replication analysis exists: {output_dir}")
    loaded: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, rows in inputs.items():
        loaded[seed] = {
            family: _load_selection_evaluation(path, expected_seed=seed, expected_family=family)
            for family, path in rows.items()
        }
    bundle_hashes = {
        item["selection_bundle_report_sha256"]
        for rows in loaded.values()
        for item in rows.values()
    }
    _require(len(bundle_hashes) == 1, "Seed replication evaluations use different selection sets")

    metric_names = ("grounding_mean_iou", "json_valid_rate", "peak_presence_macro_f1")
    per_seed: dict[str, Any] = {}
    for seed in REPLICATION_SEEDS:
        per_seed[str(seed)] = {
            family.lower(): loaded[seed][family]["metrics"] for family in ("R2", "C1")
        }
        per_seed[str(seed)]["r2_minus_c1"] = {
            metric: loaded[seed]["R2"]["metrics"][metric]
            - loaded[seed]["C1"]["metrics"][metric]
            for metric in metric_names
        }

    across: dict[str, Any] = {}
    for label, extractor in (
        ("r2", lambda seed, metric: loaded[seed]["R2"]["metrics"][metric]),
        ("c1", lambda seed, metric: loaded[seed]["C1"]["metrics"][metric]),
        (
            "r2_minus_c1",
            lambda seed, metric: loaded[seed]["R2"]["metrics"][metric]
            - loaded[seed]["C1"]["metrics"][metric],
        ),
    ):
        across[label] = {}
        for metric in metric_names:
            values = [float(extractor(seed, metric)) for seed in REPLICATION_SEEDS]
            across[label][metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "seeds": list(REPLICATION_SEEDS),
            }

    report = {
        "schema_version": REPLICATION_ANALYSIS_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post_selection_seed_robustness_supplement",
        "sources": {
            "selection_bundle_report_sha256": next(iter(bundle_hashes)),
            "evaluations": {
                str(seed): {
                    family.lower(): {
                        "path": str(inputs[seed][family]),
                        "sha256": loaded[seed][family]["sha256"],
                    }
                    for family in ("R2", "C1")
                }
                for seed in REPLICATION_SEEDS
            },
        },
        "per_seed": per_seed,
        "across_seed_summary": across,
        "interpretation": {
            "seeds": 2,
            "inferential_claim_allowed": False,
            "reason": "Two seeds provide a robustness check, not a stable variance estimate.",
        },
        "contracts": {
            "changes_preregistered_selection": False,
            "selection_set_evaluation_only": True,
            "validation_prompts_opened": False,
            "validation_answers_opened": False,
            "internal_test_accessed": False,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary)
        report_path = staging / "seed_replication_analysis.json"
        _write_json(report_path, report)
        markdown_path = staging / "seed_replication_analysis.md"
        markdown_path.write_text(
            "# V2 RL seed replication on the frozen selection set\n\n"
            "This post-selection supplement compares the selected R2 and C1 recipes at seeds 17 and 29. "
            "It does not alter preregistered selection and does not use validation or internal-test data.\n\n"
            "| Seed | R2 mean IoU | C1 mean IoU | R2 - C1 |\n"
            "| ---: | ---: | ---: | ---: |\n"
            + "".join(
                f"| {seed} | {per_seed[str(seed)]['r2']['grounding_mean_iou']:.6f} | "
                f"{per_seed[str(seed)]['c1']['grounding_mean_iou']:.6f} | "
                f"{per_seed[str(seed)]['r2_minus_c1']['grounding_mean_iou']:+.6f} |\n"
                for seed in REPLICATION_SEEDS
            )
            + "\nWith only two seeds, the reported sample standard deviations are descriptive and not inferential.\n",
            encoding="utf-8",
        )
        manifest_path = _write_manifest(staging, [report_path, markdown_path])
        staging.replace(output_dir)
    report_path = output_dir / "seed_replication_analysis.json"
    return SeedReplicationResult(
        output_dir=output_dir,
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        manifest_path=output_dir / manifest_path.name,
    )
