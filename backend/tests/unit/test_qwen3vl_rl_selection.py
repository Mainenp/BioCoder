import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_selection import (
    JSON_VALIDITY_THRESHOLD,
    begin_one_time_validation_access,
    build_selection_bundle,
    choose_preregistered_candidate,
    complete_one_time_validation_access,
    evaluate_selection_predictions,
    seal_selection_result,
)


SHA = "a" * 64
REVISION = "b" * 40


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _bind_manifest(root: Path, *paths: Path) -> None:
    (root / "artifact_manifest.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )


def _row(index: int, task: str, answer: str, group: str) -> dict:
    return {
        "instruction_id": f"selection-{index}",
        "task": task,
        "language": "en",
        "group_id": group,
        "source_path": f"mzml/train/{group}.mzML",
        "source_stratum": "sample",
        "target_peak_present": answer == '{"peak_present": true}',
        "prompt": "Locate the peak for a 100x10 image" if task == "peak_grounding" else "Decide peak presence",
        "expected_response": answer,
    }


def _pool(root: Path) -> tuple[Path, str]:
    pool = root / "pool"; pool.mkdir()
    selection = [
        _row(0, "peak_grounding", '{"bbox_2d": [10, 0, 40, 10]}', "s0"),
        _row(1, "peak_grounding", '{"bbox_2d": [50, 0, 80, 10]}', "s1"),
        _row(2, "peak_presence", '{"peak_present": true}', "s2"),
        _row(3, "peak_presence", '{"peak_present": false}', "s3"),
        _row(4, "peak_presence_metadata", '{"peak_present": true}', "s4"),
        _row(5, "scientific_qc", '{"qc_state": "pass", "reason": "ok"}', "s5"),
        _row(6, "peak_grounding", '{"bbox_2d": [20, 0, 60, 10]}', "s6"),
        _row(7, "peak_presence", '{"peak_present": true}', "s7"),
    ]
    train = [_row(99, "peak_grounding", '{"bbox_2d": [10, 0, 40, 10]}', "training")]
    selection_path, training_path = pool / "rl_selection_set.jsonl", pool / "rl_training_pool.jsonl"
    _jsonl(selection_path, selection); _jsonl(training_path, train)
    groups_path = pool / "rl_selection_groups.jsonl"
    _jsonl(groups_path, [{"group_id": f"s{index}"} for index in range(8)])
    report = {
        "schema_version": "chrompeak-qwen3vl-rl-pool-v1",
        "selection": {"group_ids": [f"s{index}" for index in range(8)]},
        "artifacts": {
            "selection_set": {"path": selection_path.name, "sha256": sha256_file(selection_path), "records": len(selection)},
            "training_pool": {"path": training_path.name, "sha256": sha256_file(training_path), "records": len(train)},
            "selection_groups": {"path": groups_path.name, "sha256": sha256_file(groups_path), "records": 8},
        },
    }
    report_path = pool / "rl_pool_report.json"; _json(report_path, report)
    (pool / "artifact_manifest.sha256").write_text(f"{sha256_file(selection_path)}  {selection_path.name}\n{sha256_file(training_path)}  {training_path.name}\n{sha256_file(groups_path)}  {groups_path.name}\n{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8")
    return pool, sha256_file(report_path)


def _candidate(root: Path, family: str, config: str, step: int, pool_sha: str) -> dict:
    candidate_root = root / f"candidate-{family}-{config}-{step}"
    checkpoint_root = candidate_root / "checkpoint"
    training_root = candidate_root / "training"
    checkpoint_root.mkdir(parents=True)
    training_root.mkdir()
    checkpoint = checkpoint_root / "checkpoint.json"; checkpoint.write_bytes(f"{family}:{config}:{step}".encode())
    _bind_manifest(checkpoint_root, checkpoint)
    training = training_root / f"{family}-{config}.json"
    report_sources = {} if family == "C0" else {"rl_pool_report_sha256": pool_sha}
    _json(training, {"sources": report_sources})
    _bind_manifest(training_root, training)
    return {"family": family, "config": config, "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "step": step}, "training_report": {"path": str(training), "sha256": sha256_file(training)}, "sources": {"rl_pool_report_sha256": pool_sha}, "decode": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0}}


def _predictions(bundle: Path, correct: bool = True) -> Path:
    rows = []
    for prompt in (bundle / "selection_prompts.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(prompt); task = row["task"]
        if task == "peak_grounding": response = '{"bbox_2d": [10, 0, 40, 10]}' if correct else "not json"
        elif task == "peak_presence": response = '{"peak_present": true}' if correct else "not json"
        elif task == "peak_presence_metadata": response = '{"peak_present": true}'
        else: response = '{"qc_state": "pass", "reason": "ok"}'
        rows.append({"instruction_id": row["instruction_id"], "response": response})
    output = bundle.parent / f"predictions-{correct}.jsonl"; _jsonl(output, rows); return output


class SelectionTests(unittest.TestCase):
    def test_bundle_evaluation_and_sealed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool, pool_sha = _pool(root)
            bundle = build_selection_bundle(pool_root=pool, expected_pool_report_sha256=pool_sha, output_dir=root / "bundle", code_revision=REVISION)
            self.assertTrue(bundle.answers_path.is_file())
            self.assertNotIn("expected_response", bundle.prompts_path.read_text(encoding="utf-8"))
            c0 = _candidate(root, "C0", "start", 0, pool_sha)
            r1 = _candidate(root, "R1", "lr1e-6", 100, pool_sha)
            c1 = _candidate(root, "C1", "lr1e-6", 100, pool_sha)
            c0_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir, True), candidate=c0, output_dir=root / "c0")
            r1_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir, True), candidate=r1, output_dir=root / "r1")
            c1_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir, True), candidate=c1, output_dir=root / "c1")
            result = seal_selection_result(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, c0_evaluation=c0_eval.report_path, rl_evaluations=[r1_eval.report_path], c1_evaluations=[c1_eval.report_path], output_dir=root / "sealed", code_revision=REVISION)
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.report_sha256)

    def test_ranking_ties_and_format_only_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool, pool_sha = _pool(root); bundle = build_selection_bundle(pool_root=pool, expected_pool_report_sha256=pool_sha, output_dir=root / "bundle", code_revision=REVISION)
            c0 = _candidate(root, "C0", "start", 0, pool_sha); r1 = _candidate(root, "R1", "a", 100, pool_sha); r2 = _candidate(root, "R2", "z", 100, pool_sha)
            c0_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=c0, output_dir=root / "c0")
            r1_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=r1, output_dir=root / "r1")
            r2_eval = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=r2, output_dir=root / "r2")
            choice = choose_preregistered_candidate(baseline_evaluation=c0_eval.report_path, candidate_evaluations=[r2_eval.report_path, r1_eval.report_path], family_prefixes=("R1", "R2"))
            self.assertEqual(choice["selected"]["candidate"]["candidate_id"], "R1:a:step-00000100")
            report = json.loads(r1_eval.report_path.read_text(encoding="utf-8")); report["metrics"]["peak_presence_macro_f1"] = 0.0; report["metrics"]["grounding_mean_iou"] = 0.9; _json(r1_eval.report_path, report); _bind_manifest(r1_eval.output_dir, r1_eval.records_path, r1_eval.report_path)
            fallback = choose_preregistered_candidate(baseline_evaluation=c0_eval.report_path, candidate_evaluations=[r1_eval.report_path], family_prefixes=("R1",))
            self.assertTrue(fallback["fallback_format_only"])
            self.assertEqual(fallback["capability_thresholds"]["json_valid_rate_at_least"], JSON_VALIDITY_THRESHOLD)

    def test_provenance_protected_paths_and_one_time_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool, pool_sha = _pool(root); bundle = build_selection_bundle(pool_root=pool, expected_pool_report_sha256=pool_sha, output_dir=root / "bundle", code_revision=REVISION)
            bad = _candidate(root, "R1", "bad", 100, "c" * 64)
            with self.assertRaisesRegex(ValueError, "provenance"):
                evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=bad, output_dir=root / "bad")
            drift = _candidate(root, "R1", "drift", 100, pool_sha)
            Path(drift["checkpoint"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint drift"):
                evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=drift, output_dir=root / "drift")
            (pool / "rl_selection_set.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drift"):
                build_selection_bundle(pool_root=pool, expected_pool_report_sha256=pool_sha, output_dir=root / "tampered", code_revision=REVISION)
            c0 = _candidate(root, "C0", "start", 0, pool_sha); r1 = _candidate(root, "R1", "ok", 100, pool_sha); c1 = _candidate(root, "C1", "ok", 100, pool_sha)
            c0e = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=c0, output_dir=root / "c0")
            r1e = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=r1, output_dir=root / "r1")
            c1e = evaluate_selection_predictions(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, predictions_path=_predictions(bundle.output_dir), candidate=c1, output_dir=root / "c1")
            sealed = seal_selection_result(bundle_root=bundle.output_dir, expected_bundle_report_sha256=bundle.report_sha256, c0_evaluation=c0e.report_path, rl_evaluations=[r1e.report_path], c1_evaluations=[c1e.report_path], output_dir=root / "sealed", code_revision=REVISION)
            first = begin_one_time_validation_access(selection_root=sealed.output_dir, expected_selection_result_sha256=sealed.report_sha256, ledger_root=root / "ledger")
            resumed = begin_one_time_validation_access(selection_root=sealed.output_dir, expected_selection_result_sha256=sealed.report_sha256, ledger_root=root / "ledger")
            self.assertEqual(resumed.mode, "exact_selection_crash_resume")
            complete_one_time_validation_access(access=first, selection_result_sha256=sealed.report_sha256)
            with self.assertRaisesRegex(ValueError, "second run"):
                begin_one_time_validation_access(selection_root=sealed.output_dir, expected_selection_result_sha256=sealed.report_sha256, ledger_root=root / "ledger")
            with self.assertRaisesRegex(ValueError, "Protected"):
                build_selection_bundle(pool_root=Path("validation_answers.jsonl"), expected_pool_report_sha256=pool_sha, output_dir=root / "x", code_revision=REVISION)

    def test_training_report_must_bind_the_same_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool, pool_sha = _pool(root)
            bundle = build_selection_bundle(
                pool_root=pool,
                expected_pool_report_sha256=pool_sha,
                output_dir=root / "bundle",
                code_revision=REVISION,
            )
            candidate = _candidate(root, "R1", "wrong-report-source", 100, pool_sha)
            training_report = Path(candidate["training_report"]["path"])
            _json(training_report, {"sources": {"rl_pool_report_sha256": "d" * 64}})
            candidate["training_report"]["sha256"] = sha256_file(training_report)
            _bind_manifest(training_report.parent, training_report)

            with self.assertRaisesRegex(ValueError, "training report RL pool provenance"):
                evaluate_selection_predictions(
                    bundle_root=bundle.output_dir,
                    expected_bundle_report_sha256=bundle.report_sha256,
                    predictions_path=_predictions(bundle.output_dir),
                    candidate=candidate,
                    output_dir=root / "evaluation",
                )


if __name__ == "__main__":
    unittest.main()
