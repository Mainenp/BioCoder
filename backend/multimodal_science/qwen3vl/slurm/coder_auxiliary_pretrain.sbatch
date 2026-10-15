#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
umask 077

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }
}

for name in \
  BIOCODER_REPO_ROOT \
  BIOCODER_RUN_ROOT \
  BIOCODER_PYTHON \
  BIOCODER_CODE_REVISION \
  BIOCODER_AUXILIARY_INDEX_ROOT \
  BIOCODER_AUXILIARY_INDEX_REPORT_SHA256 \
  BIOCODER_AUXILIARY_INDEX_MANIFEST_SHA256 \
  BIOCODER_AUXILIARY_ASSETS_ROOT
do
  require_variable "$name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"
index_root="$BIOCODER_AUXILIARY_INDEX_ROOT"
index_report_sha256="$BIOCODER_AUXILIARY_INDEX_REPORT_SHA256"
index_manifest_sha256="$BIOCODER_AUXILIARY_INDEX_MANIFEST_SHA256"
assets_root="$BIOCODER_AUXILIARY_ASSETS_ROOT"
gpu_index="${BIOCODER_GPU_INDEX:-auto}"
epochs="${BIOCODER_AUXILIARY_EPOCHS:-20}"
batch_size="${BIOCODER_AUXILIARY_BATCH_SIZE:-64}"
max_steps="${BIOCODER_AUXILIARY_MAX_STEPS:-}"
seed="${BIOCODER_SEED:-17}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${code_revision:0:7}"

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
for digest in "$index_report_sha256" "$index_manifest_sha256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "Bad SHA-256: $digest" >&2; exit 2; }
done
for value in "$epochs" "$batch_size" "$seed" "$max_gpu_used_mib" "$max_gpu_utilization"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected integer: $value" >&2; exit 2; }
done
(( epochs >= 1 && batch_size >= 2 ))
[[ -z "$max_steps" || "$max_steps" =~ ^[1-9][0-9]*$ ]]
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$index_root"
test -d "$assets_root"
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || { echo "Code revision mismatch" >&2; exit 3; }

echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$code_revision"
echo "INTERNAL_TEST_ACCESSED=false"

dataset_name="auxiliary-signal-v1-${index_report_sha256:0:12}"
dataset_output="$run_root/qwen3vl/auxiliary/datasets/$dataset_name"
mkdir -p "$(dirname "$dataset_output")"
if [[ ! -d "$dataset_output" ]]; then
  echo "=== MATERIALIZE ZERO-LABEL AUXILIARY SIGNALS ==="
  (
    cd "$repo_root/backend"
    PYTHONPATH="$repo_root/backend" "$python_bin" \
      -m multimodal_science.qwen3vl.build_auxiliary_pretraining_data_cli \
      --auxiliary-index-root "$index_root" \
      --auxiliary-index-report-sha256 "$index_report_sha256" \
      --auxiliary-index-manifest-sha256 "$index_manifest_sha256" \
      --assets-root "$assets_root" \
      --output-dir "$dataset_output" \
      --target-points 160
  )
fi
(
  cd "$dataset_output"
  sha256sum -c artifact_manifest.sha256
)
dataset_report_sha256="$(sha256sum "$dataset_output/auxiliary_pretraining_dataset_report.json" | awk '{print $1}')"
dataset_manifest_sha256="$(sha256sum "$dataset_output/artifact_manifest.sha256" | awk '{print $1}')"
echo "AUXILIARY_DATASET_REPORT_SHA256=$dataset_report_sha256"
echo "AUXILIARY_DATASET_MANIFEST_SHA256=$dataset_manifest_sha256"

run_mode="full"
if [[ -n "$max_steps" ]]; then run_mode="calibration-steps${max_steps}"; fi
run_name="xic-auxiliary-${run_mode}-epoch${epochs}-b${batch_size}-seed${seed}-${code_short}"
output_root="$run_root/qwen3vl/auxiliary/runs"
work_output="$output_root/.${run_name}.incomplete"
final_output="$output_root/$run_name"
mkdir -p "$output_root"
exec 7>"$output_root/.${run_name}.lock"
flock -n 7 || { echo "This exact auxiliary run is already active" >&2; exit 7; }
if [[ -d "$final_output" ]]; then
  (cd "$final_output" && sha256sum -c artifact_manifest.sha256)
  echo "PERSISTENT_OUTPUT=$final_output"
  echo "QWEN3VL_AUXILIARY_PRETRAINING=ALREADY_COMPLETE"
  exit 0
fi
test ! -e "$work_output"

scratch="$(mktemp -d "/tmp/biocoder-${USER}-auxiliary-pretrain-${job_id}.XXXXXX")"
cleanup() { case "${scratch:-}" in "/tmp/biocoder-${USER}-auxiliary-pretrain-${job_id}."*) rm -rf -- "$scratch" ;; esac; }
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/dataset" "$scratch/tmp"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$scratch/code"
cp -a "$dataset_output/." "$scratch/dataset/"
(cd "$scratch/dataset" && sha256sum -c artifact_manifest.sha256)

gpu_passes_guard() {
  local candidate="$1" sample used util
  for sample in 1 2 3; do
    IFS=',' read -r used util < <(
      nvidia-smi -i "$candidate" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    echo "GPU=$candidate SAMPLE=$sample USED_MIB=$used UTIL=$util" >&2
    (( used <= max_gpu_used_mib && util <= max_gpu_utilization )) || return 1
    sleep 2
  done
}

select_gpu() {
  local candidate lock_path
  if [[ "$gpu_index" == "auto" ]]; then candidates=(7 6 5 4 3 2 1 0); else candidates=("$gpu_index"); fi
  for candidate in "${candidates[@]}"; do
    lock_path="/tmp/biocoder-${USER}-gpu-${candidate}.lock"
    exec 9>"$lock_path"
    if flock -n 9 && gpu_passes_guard "$candidate"; then
      selected_gpu="$candidate"
      return 0
    fi
    exec 9>&-
  done
  return 1
}

selected_gpu=""
select_gpu || { echo "No guarded RTX 5090 is available" >&2; exit 5; }
echo "SELECTED_PHYSICAL_GPU=$selected_gpu"
limit_arguments=()
if [[ -n "$max_steps" ]]; then limit_arguments+=(--max-steps "$max_steps"); fi

echo "=== TRAIN UNLABELED XIC MORPHOLOGY PROJECTOR ==="
(
  cd "$scratch/code/backend"
  env -i \
    HOME="$HOME" USER="$USER" \
    PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    LANG="${LANG:-C.UTF-8}" \
    TMPDIR="$scratch/tmp" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$selected_gpu" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$scratch/code/backend" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
    SLURM_JOB_ID="$job_id" \
    "$python_bin" -m multimodal_science.qwen3vl.train_auxiliary_pretraining_cli \
      --dataset-root "$scratch/dataset" \
      --dataset-report-sha256 "$dataset_report_sha256" \
      --dataset-manifest-sha256 "$dataset_manifest_sha256" \
      --output-dir "$work_output" \
      --code-revision "$code_revision" \
      --epochs "$epochs" \
      --batch-size "$batch_size" \
      --seed "$seed" \
      "${limit_arguments[@]}"
)

(
  cd "$work_output"
  sha256sum -c artifact_manifest.sha256
)
"$python_bin" - "$work_output" "$run_mode" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
report = json.loads((root / "auxiliary_pretraining_report.json").read_text(encoding="utf-8"))
assert report["schema_version"] == "chrompeak-xic-auxiliary-pretraining-v1"
assert report["development_training_complete"] is (mode == "full")
assert report["development_comparison_eligible"] is False
assert report["final_benchmark_eligible"] is False
assert report["internal_test_accessed"] is False
assert report["contracts"]["unlabeled_auxiliary_only"] is True
assert report["contracts"]["validation_opened"] is False
assert report["contracts"]["metrics_allowed"] is False
print("AUXILIARY_PRETRAINING_CONTRACT=OK")
PY
mv "$work_output" "$final_output"
echo "AUXILIARY_PRETRAINING_REPORT_SHA256=$(sha256sum "$final_output/auxiliary_pretraining_report.json" | awk '{print $1}')"
echo "AUXILIARY_PRETRAINING_MANIFEST_SHA256=$(sha256sum "$final_output/artifact_manifest.sha256" | awk '{print $1}')"
echo "PERSISTENT_OUTPUT=$final_output"
echo "QWEN3VL_AUXILIARY_PRETRAINING=OK"
