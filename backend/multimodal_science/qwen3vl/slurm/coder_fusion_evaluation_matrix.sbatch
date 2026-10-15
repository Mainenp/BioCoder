#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2

set -euo pipefail
umask 077

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
run_root="${BIOCODER_RUN_ROOT:?Set BIOCODER_RUN_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
training_root="${BIOCODER_MATRIX_TRAINING_ROOT:?Set BIOCODER_MATRIX_TRAINING_ROOT}"
training_revision="${BIOCODER_MATRIX_TRAINING_REVISION:?Set BIOCODER_MATRIX_TRAINING_REVISION}"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this script as its declared Slurm array}"

evaluation_script="$repo_root/backend/multimodal_science/qwen3vl/slurm/coder_fusion_evaluate.sbatch"
[[ -f "$evaluation_script" ]] || {
  echo "Missing fusion evaluation script: $evaluation_script" >&2
  exit 2
}
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 2
}
[[ "$training_revision" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Invalid matrix training revision: $training_revision" >&2
  exit 2
}

case "$task_id" in
  0) sensor_tokens=4; training_seed=17; experiment="primary-seed17" ;;
  1) sensor_tokens=4; training_seed=29; experiment="primary-seed29" ;;
  2) sensor_tokens=4; training_seed=43; experiment="primary-seed43" ;;
  3) sensor_tokens=1; training_seed=17; experiment="tokens1-seed17" ;;
  4) sensor_tokens=8; training_seed=17; experiment="tokens8-seed17" ;;
  *) echo "Unsupported matrix task: $task_id" >&2; exit 2 ;;
esac

training_short="${training_revision:0:7}"
adapter_root="$training_root/image-xic-full-tokens${sensor_tokens}-epoch1-b1-ga16-seed${training_seed}-${training_short}"
training_report="$adapter_root/fusion_training_report.json"
adapter_manifest="$adapter_root/artifact_manifest.sha256"

[[ -f "$training_report" ]] || {
  echo "Missing matrix training report: $training_report" >&2
  exit 2
}
[[ -f "$adapter_manifest" ]] || {
  echo "Missing matrix adapter manifest: $adapter_manifest" >&2
  exit 2
}
(
  cd "$adapter_root"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$training_report" "$training_revision" "$sensor_tokens" "$training_seed" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_revision = sys.argv[2]
expected_tokens = int(sys.argv[3])
expected_seed = int(sys.argv[4])

assert report["schema_version"] == "chrompeak-qwen3vl-xic-fusion-training-v1"
assert report["code_revision"] == expected_revision
assert report["training"]["seed"] == expected_seed
assert report["model"]["sensor_projector"]["sensor_tokens"] == expected_tokens
assert report["training"]["optimizer_updates"] == 3396
assert report["development_training_complete"] is True
assert report["development_comparison_eligible"] is False
assert report["final_benchmark_eligible"] is False
assert report["internal_test_accessed"] is False
assert report["contracts"]["train_split_only"] is True
assert report["contracts"]["validation_prompts_opened"] is False
assert report["contracts"]["validation_answers_opened"] is False
assert report["contracts"]["internal_test_accessed"] is False
print("MATRIX_TRAINING_INPUT_CONTRACT=OK")
PY

export BIOCODER_FUSION_ADAPTER_ROOT="$adapter_root"
export BIOCODER_FUSION_TRAINING_REPORT_SHA256="$(sha256sum "$training_report" | awk '{print $1}')"
export BIOCODER_FUSION_ADAPTER_MANIFEST_SHA256="$(sha256sum "$adapter_manifest" | awk '{print $1}')"
export BIOCODER_GPU_INDEX=auto
export BIOCODER_SEED=17
export BIOCODER_MAX_NEW_TOKENS=64
export BIOCODER_BOOTSTRAP_ITERATIONS=1000
export BIOCODER_XIC_INTERVENTION=aligned
export BIOCODER_XIC_INTERVENTION_SEED=17

echo "=== FUSION VALIDATION MATRIX ENTRY ==="
echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unscheduled}"
echo "ARRAY_TASK_ID=$task_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "EVALUATOR_CODE_REVISION=$actual_revision"
echo "TRAINING_CODE_REVISION=$training_revision"
echo "EXPERIMENT=$experiment"
echo "SENSOR_TOKENS=$sensor_tokens"
echo "TRAINING_SEED=$training_seed"
echo "XIC_INTERVENTION=aligned"
echo "INTERNAL_TEST_ACCESSED=false"

exec bash "$evaluation_script"
