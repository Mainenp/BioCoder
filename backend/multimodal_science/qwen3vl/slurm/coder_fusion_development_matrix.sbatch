#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%2

set -euo pipefail
umask 077

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
run_root="${BIOCODER_RUN_ROOT:?Set BIOCODER_RUN_ROOT}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this script as its declared Slurm array}"

training_script="$repo_root/backend/multimodal_science/qwen3vl/slurm/coder_fusion_train.sbatch"
[[ -f "$training_script" ]] || {
  echo "Missing fusion training script: $training_script" >&2
  exit 2
}
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 2
}

# The primary four-token architecture receives three independent training seeds.
# One- and eight-token runs change only the pooling/token count at the primary seed.
case "$task_id" in
  0) sensor_tokens=4; training_seed=17; experiment="primary-seed17" ;;
  1) sensor_tokens=4; training_seed=29; experiment="primary-seed29" ;;
  2) sensor_tokens=4; training_seed=43; experiment="primary-seed43" ;;
  3) sensor_tokens=1; training_seed=17; experiment="tokens1-seed17" ;;
  4) sensor_tokens=8; training_seed=17; experiment="tokens8-seed17" ;;
  *) echo "Unsupported matrix task: $task_id" >&2; exit 2 ;;
esac

# Auxiliary projector pretraining remains a separate initialization ablation.
unset BIOCODER_PRETRAINED_PROJECTOR_ROOT
unset BIOCODER_PRETRAINED_PROJECTOR_REPORT_SHA256
unset BIOCODER_PRETRAINED_PROJECTOR_MANIFEST_SHA256

export BIOCODER_GPU_INDEX=auto
export BIOCODER_EPOCHS=1
export BIOCODER_BATCH_SIZE=1
export BIOCODER_GRADIENT_ACCUMULATION_STEPS=16
export BIOCODER_LORA_LEARNING_RATE=2e-7
export BIOCODER_PROJECTOR_LEARNING_RATE=1e-4
export BIOCODER_SAVE_STEPS=250
export BIOCODER_LOG_STEPS=10
export BIOCODER_MAX_STEPS=""
export BIOCODER_SENSOR_TOKENS="$sensor_tokens"
export BIOCODER_SEED="$training_seed"
export BIOCODER_FUSION_OUTPUT_ROOT="${BIOCODER_MATRIX_OUTPUT_ROOT:-$run_root/qwen3vl/fusion/runs/random-init-matrix-${code_revision:0:7}}"

echo "=== FUSION DEVELOPMENT MATRIX ENTRY ==="
echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unscheduled}"
echo "ARRAY_TASK_ID=$task_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "EXPERIMENT=$experiment"
echo "INITIALIZATION=random"
echo "SENSOR_TOKENS=$sensor_tokens"
echo "TRAINING_SEED=$training_seed"
echo "MAX_STEPS=none"
echo "INTERNAL_TEST_ACCESSED=false"

exec bash "$training_script"
