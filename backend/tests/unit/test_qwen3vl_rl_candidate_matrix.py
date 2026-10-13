import json
import tempfile
import unittest
from pathlib import Path

from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.rl_candidate_matrix import build_candidate_matrix


REVISION = "a" * 40
POOL_SHA = "b" * 64


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(root: Path, paths: list[Path]) -> Path:
    path = root / "artifact_manifest.sha256"
    path.write_text(
        "".join(
            f"{sha256_file(artifact)}  {artifact.relative_to(root).as_posix()}\n"
            for artifact in paths
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / "checkpoints" / f"step-{step:08d}"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    projector = checkpoint / "sensor_projector.safetensors"
    state = checkpoint / "checkpoint.json"
    _json(config, {"r": 8})
    weights.write_bytes(f"weights-{step}".encode())
    projector.write_bytes(f"projector-{step}".encode())
    _json(state, {"global_step": step})
    _manifest(checkpoint, [config, weights, projector, state])
    return checkpoint


class CandidateMatrixTests(unittest.TestCase):
    def test_builds_exact_preregistered_sixteen_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c0 = root / "c0"
            (c0 / "adapter").mkdir(parents=True)
            c0_report = c0 / "fusion_training_report.json"
            c0_config = c0 / "adapter" / "adapter_config.json"
            c0_weights = c0 / "adapter" / "adapter_model.safetensors"
            c0_projector = c0 / "sensor_projector.safetensors"
            _json(
                c0_report,
                {
                    "contracts": {
                        "internal_test_accessed": False,
                        "train_split_only": True,
                    }
                },
            )
            _json(c0_config, {"r": 8})
            c0_weights.write_bytes(b"c0")
            c0_projector.write_bytes(b"projector")
            c0_manifest = _manifest(
                c0, [c0_report, c0_config, c0_weights, c0_projector]
            )
            c0_report_sha = sha256_file(c0_report)
            c0_manifest_sha = sha256_file(c0_manifest)

            runs = {}
            for label in ("R1", "R2", "R3", "C1-original", "C1-1e-6"):
                run = root / label
                run.mkdir()
                family = "C1" if label.startswith("C1-") else label
                report = run / (
                    "sft_continue_training_report.json"
                    if family == "C1"
                    else "fusion_training_report.json"
                )
                _json(
                    report,
                    {
                        "sources": {
                            "rl_pool_report_sha256": POOL_SHA,
                            "starting_adapter_report_sha256": c0_report_sha,
                            "starting_adapter_manifest_sha256": c0_manifest_sha,
                        },
                        "contracts": {
                            "selection_groups_trained": False,
                            "validation_prompts_opened": False,
                            "validation_answers_opened": False,
                            "internal_test_accessed": False,
                        },
                        "development_training_complete": True,
                    },
                )
                _manifest(run, [report])
                for step in (100, 200, 300):
                    _checkpoint(run, step)
                runs[label] = {"root": str(run), "config": label.lower()}

            result = build_candidate_matrix(
                code_revision=REVISION,
                rl_pool_report_sha256=POOL_SHA,
                c0_root=c0,
                c0_report_sha256=c0_report_sha,
                c0_manifest_sha256=c0_manifest_sha,
                run_spec={"runs": runs},
                output_dir=root / "matrix",
            )
            self.assertEqual(result.candidates, 16)
            rows = [
                json.loads(line)
                for line in result.candidate_list_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                {family: sum(row["family"] == family for row in rows) for family in ("C0", "C1", "R1", "R2", "R3")},
                {"C0": 1, "C1": 6, "R1": 3, "R2": 3, "R3": 3},
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["contracts"]["validation_answers_opened"])
            self.assertEqual(report["counts"]["checkpoint_steps"], [100, 200, 300])


if __name__ == "__main__":
    unittest.main()
