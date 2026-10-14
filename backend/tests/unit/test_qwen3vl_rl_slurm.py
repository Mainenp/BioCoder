import unittest
from pathlib import Path

from multimodal_science.qwen3vl.select_rl_candidate_cli import parser


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "multimodal_science" / "qwen3vl" / "slurm"


class RlSlurmContractTests(unittest.TestCase):
    def _script(self, name: str) -> str:
        path = SLURM / name
        self.assertTrue(path.is_file(), name)
        text = path.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn('git -C "$repo', text)
        self.assertIn("archive", text)
        return text

    def test_prepare_and_difficulty_are_hash_bound(self) -> None:
        prepare = self._script("coder_rl_prepare.sbatch")
        difficulty = self._script("coder_rl_difficulty.sbatch")
        self.assertIn("RL_POOL_REPORT_SHA256", prepare)
        self.assertIn("STAGED_TRAIN_ONLY_INPUTS=OK", difficulty)
        self.assertIn('"rl_selection_set.jsonl" not in lowered', difficulty)
        self.assertIn("VALIDATION_ANSWERS_OPENED=false", difficulty)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", difficulty)

    def test_train_script_stages_no_selection_or_validation_samples(self) -> None:
        train = self._script("coder_rl_train.sbatch")
        self.assertIn("BIOCODER_RL_TRAINER", train)
        self.assertIn("SELECTION_SAMPLES_STAGED=false", train)
        self.assertIn('"rl_selection_set.jsonl" not in names', train)
        self.assertIn("--experiment", train)
        self.assertIn("--learning-rate-choice", train)
        self.assertIn("--resume", train)
        self.assertIn("VALIDATION_ANSWERS_OPENED=false", train)

    def test_train_script_preserves_verified_runtime_contract_in_clean_environment(self) -> None:
        train = self._script("coder_rl_train.sbatch")
        self.assertIn('actual_revision="$(git -C "$repo_root" rev-parse "$revision^{commit}")"', train)
        self.assertIn('[[ "$actual_revision" == "$revision" ]]', train)
        self.assertIn('BIOCODER_VERIFIED_CODE_REVISION="$actual_revision"', train)
        self.assertIn("BIOCODER_TRAIN_INPUT_SCOPE=staged_train_artifacts_only", train)
        self.assertIn(
            "BIOCODER_GPU_ALLOCATION_MODE=manual_physical_index_guard_no_slurm_gres",
            train,
        )
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", train)
        self.assertIn('env -i "${environment[@]}"', train)

    def test_selection_gate_and_analysis_require_a_completed_ledger(self) -> None:
        selection = self._script("coder_rl_selection.sbatch")
        localization = self._script("coder_rl_localization.sbatch")
        self.assertIn("begin-validation|complete-validation", selection)
        self.assertIn("access_completed.json", localization)
        self.assertIn("chrompeak-qwen3vl-rl-selection-completion-v1", localization)
        self.assertIn("ONE_TIME_VALIDATION_LEDGER=OK", localization)

    def test_localization_metadata_job_is_answer_free_and_hash_bound(self) -> None:
        metadata = self._script("coder_rl_localization_metadata.sbatch")
        self.assertIn("BIOCODER_DATASET_REPORT_SHA256", metadata)
        self.assertIn("BIOCODER_FUSION_BUNDLE_REPORT_SHA256", metadata)
        self.assertIn("build_localization_metadata_cli", metadata)
        self.assertIn("VALIDATION_ANSWERS_OPENED=false", metadata)
        self.assertIn("INTERNAL_TEST_ACCESSED=false", metadata)
        self.assertNotIn("ANSWER_PATH", metadata)
        self.assertNotIn("BIOCODER_INTERNAL_TEST", metadata)
        self.assertIn("QWEN3VL_RL_LOCALIZATION_METADATA=OK", metadata)

    def test_selection_inference_stages_prompt_only_view_and_verified_runtime(self) -> None:
        inference = self._script("coder_rl_selection_inference.sbatch")
        self.assertIn(
            '("selection_prompts", "selection_manifest", "selection_prompt_manifest")',
            inference,
        )
        self.assertIn('not (bundle_dst / "selection_answers.jsonl").exists()', inference)
        self.assertIn('not (bundle_dst / "artifact_manifest.sha256").exists()', inference)
        self.assertIn(
            'BIOCODER_SELECTION_INPUT_SCOPE="staged_selection_prompts_and_train_signals_only"',
            inference,
        )
        self.assertIn("--candidate-json-sha256", inference)
        self.assertIn("BIOCODER_RL_CANDIDATE_MATRIX_ROOT", inference)
        self.assertIn("BIOCODER_RL_CANDIDATE_MATRIX_REPORT_SHA256", inference)
        self.assertIn("BIOCODER_RL_SELECTION_OUTPUT_ROOT", inference)
        self.assertIn("SLURM_ARRAY_TASK_ID", inference)
        self.assertIn('report["counts"]["candidates"] != 16', inference)
        self.assertIn('output="${BIOCODER_RL_SELECTION_OUTPUT_ROOT%/}/$candidate_id"', inference)
        self.assertIn("QWEN3VL_RL_SELECTION_INFERENCE=OK", inference)
        selection = self._script("coder_rl_selection.sbatch")
        self.assertIn("build-candidates", selection)
        self.assertIn("build_rl_candidate_matrix_cli", selection)

    def test_selection_cli_exposes_validation_lifecycle(self) -> None:
        begin = parser().parse_args(
            [
                "begin-validation",
                "--selection-root", "selection",
                "--selection-result-sha256", "a" * 64,
                "--ledger-root", "ledger",
            ]
        )
        complete = parser().parse_args(
            [
                "complete-validation",
                "--selection-root", "selection",
                "--selection-result-sha256", "a" * 64,
                "--ledger-root", "ledger",
            ]
        )
        self.assertEqual(begin.command, "begin-validation")
        self.assertEqual(complete.command, "complete-validation")


if __name__ == "__main__":
    unittest.main()
