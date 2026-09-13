#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

require_variable() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

for required_name in \
  BIOCODER_RUN_ROOT \
  BIOCODER_PYTHON \
  BIOCODER_MODEL_DIR \
  BIOCODER_MODEL_ARTIFACT_SHA256 \
  BIOCODER_MODEL_REVISION \
  BIOCODER_INSTRUCTION_REPORT_SHA256 \
  BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256 \
  BIOCODER_CODE_REVISION \
  BIOCODER_GPU_INDEX
do
  require_variable "$required_name"
done

repo_root="${BIOCODER_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
model_dir="$BIOCODER_MODEL_DIR"
assets_root="${BIOCODER_ASSETS_ROOT:-$run_root/assets}"
instruction_root="${BIOCODER_INSTRUCTION_ROOT:-$run_root/qwen3vl/instructions-v2-bilingual}"
inference_bundle_root="${BIOCODER_INFERENCE_BUNDLE_ROOT:-$run_root/qwen3vl/inference-bundle-v2-bilingual-2b8d246}"
gpu_index="$BIOCODER_GPU_INDEX"
seed="${BIOCODER_SEED:-17}"
max_records="${BIOCODER_MAX_RECORDS:-128}"
max_steps="${BIOCODER_MAX_STEPS:-2}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-1024}"
inference_records="${BIOCODER_INFERENCE_RECORDS:-16}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${BIOCODER_CODE_REVISION:0:7}"

for numeric_value in \
  "$gpu_index" "$seed" "$max_records" "$max_steps" \
  "$max_gpu_used_mib" "$inference_records"
do
  [[ "$numeric_value" =~ ^[0-9]+$ ]] || {
    echo "Expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  }
done
[[ "$BIOCODER_MODEL_ARTIFACT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$BIOCODER_INSTRUCTION_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$BIOCODER_CODE_REVISION" =~ ^[0-9a-f]{40}$ ]]

test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$model_dir"
test -d "$assets_root"
test -d "$instruction_root"
test -d "$inference_bundle_root"
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$BIOCODER_CODE_REVISION" ]] || {
  echo "Code revision mismatch: expected $BIOCODER_CODE_REVISION, got $actual_revision" >&2
  exit 3
}

echo "=== JOB CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-manual}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "PHYSICAL_GPU=$gpu_index"
echo "MAX_RECORDS=$max_records"
echo "MAX_STEPS=$max_steps"

lock_path="/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
exec 9>"$lock_path"
flock -n 9 || {
  echo "BioCoder GPU lock is already held: $lock_path" >&2
  exit 4
}

echo "=== GPU STARTUP GUARD ==="
for sample in 1 2 3
do
  gpu_state="$(
    nvidia-smi -i "$gpu_index" \
      --query-gpu=memory.used,memory.free,utilization.gpu,name \
      --format=csv,noheader,nounits
  )"
  echo "sample=$sample $gpu_state"
  used_mib="${gpu_state%%,*}"
  used_mib="${used_mib// /}"
  (( used_mib <= max_gpu_used_mib )) || {
    echo "GPU $gpu_index is already using ${used_mib} MiB" >&2
    exit 5
  }
  sleep 2
done

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$repo_root/backend"
export TOKENIZERS_PARALLELISM=false

"$python_bin" - <<'PY'
import torch
import transformers
import peft
import safetensors

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
print("python_dependencies=OK")
print("visible_gpu=" + torch.cuda.get_device_name(0))
print("torch=" + torch.__version__)
print("transformers=" + transformers.__version__)
print("peft=" + peft.__version__)
print("safetensors=" + safetensors.__version__)
PY

bundle_dir="$run_root/qwen3vl/lora/bundles/smoke-${max_records}-seed${seed}-${code_short}"
if [[ -d "$bundle_dir" ]]; then
  echo "=== VERIFY EXISTING TRAIN-ONLY BUNDLE ==="
  (
    cd "$bundle_dir"
    sha256sum -c artifact_manifest.sha256
  )
else
  echo "=== BUILD TRAIN-ONLY BUNDLE ==="
  nice -n 10 ionice -c 3 "$python_bin" \
    -m multimodal_science.qwen3vl.build_lora_bundle_cli \
    --instruction-root "$instruction_root" \
    --instruction-report-sha256 "$BIOCODER_INSTRUCTION_REPORT_SHA256" \
    --assets-root "$assets_root" \
    --output-dir "$bundle_dir" \
    --max-records "$max_records" \
    --seed "$seed"
fi
bundle_report_sha256="$(sha256sum "$bundle_dir/lora_bundle_report.json" | awk '{print $1}')"
echo "LORA_BUNDLE_REPORT_SHA256=$bundle_report_sha256"

scratch="/tmp/biocoder-${USER}-qwen3vl-lora-${job_id}"
test ! -e "$scratch"
mkdir -p "$scratch"
training_output="$scratch/training"

echo "=== TRAIN TWO-STEP LORA SMOKE ==="
nice -n 10 ionice -c 3 "$python_bin" \
  -m multimodal_science.qwen3vl.train_lora_cli \
  --bundle-root "$bundle_dir" \
  --bundle-report-sha256 "$bundle_report_sha256" \
  --assets-root "$assets_root" \
  --output-dir "$training_output" \
  --model-name-or-path "$model_dir" \
  --model-revision "$BIOCODER_MODEL_REVISION" \
  --model-artifact-sha256 "$BIOCODER_MODEL_ARTIFACT_SHA256" \
  --code-revision "$BIOCODER_CODE_REVISION" \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-steps "$max_steps" \
  --save-steps 1 \
  --log-steps 1 \
  --seed "$seed"

(
  cd "$training_output"
  sha256sum -c artifact_manifest.sha256
)
training_report_sha256="$(
  sha256sum "$training_output/lora_training_report.json" | awk '{print $1}'
)"
adapter_manifest_sha256="$(
  sha256sum "$training_output/artifact_manifest.sha256" | awk '{print $1}'
)"

echo "=== RELOAD ADAPTER AND GENERATE ==="
inference_output="$scratch/adapter-inference"
nice -n 10 ionice -c 3 "$python_bin" \
  -m multimodal_science.qwen3vl.run_inference_cli \
  --bundle-root "$inference_bundle_root" \
  --bundle-report-sha256 "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" \
  --assets-root "$assets_root" \
  --output-dir "$inference_output" \
  --model-name-or-path "$model_dir" \
  --model-revision "$BIOCODER_MODEL_REVISION" \
  --model-artifact-sha256 "$BIOCODER_MODEL_ARTIFACT_SHA256" \
  --adapter-root "$training_output" \
  --adapter-training-report-sha256 "$training_report_sha256" \
  --adapter-manifest-sha256 "$adapter_manifest_sha256" \
  --batch-size 2 \
  --max-new-tokens 64 \
  --seed "$seed" \
  --dtype bfloat16 \
  --device-map auto \
  --attention-implementation sdpa \
  --max-records "$inference_records"

"$python_bin" - "$training_output" "$inference_output" <<'PY'
import json
import pathlib
import sys

training_root = pathlib.Path(sys.argv[1])
inference_root = pathlib.Path(sys.argv[2])
training = json.loads((training_root / "lora_training_report.json").read_text())
generation = json.loads((inference_root / "generation_report.json").read_text())
assert training["training"]["smoke_test"] is True
assert training["development_training_complete"] is False
assert generation["runtime"]["adapter_loaded"] is True
assert generation["model"]["adapter"]["development_training_complete"] is False
assert generation["development_comparison_candidate"] is False
assert generation["internal_test_accessed"] is False
print("LORA_CUDA_SMOKE_CONTRACT=OK")
PY

persistent_output="${BIOCODER_LORA_OUTPUT_ROOT:-$run_root/qwen3vl/lora/runs}/qwen3vl-4b-lora-smoke-${max_records}-steps${max_steps}-seed${seed}-${code_short}-job${job_id}"
test ! -e "$persistent_output"
mkdir -p "$(dirname "$persistent_output")"
cp -a "$scratch" "$persistent_output"
(
  cd "$persistent_output/training"
  sha256sum -c artifact_manifest.sha256
)

echo "TRAINING_REPORT_SHA256=$training_report_sha256"
echo "ADAPTER_MANIFEST_SHA256=$adapter_manifest_sha256"
echo "PERSISTENT_OUTPUT=$persistent_output"
echo "SCRATCH_RETAINED=$scratch"
echo "QWEN3VL_LORA_SMOKE=OK"
