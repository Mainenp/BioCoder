#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

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
gpu_index="$BIOCODER_GPU_INDEX"
seed="${BIOCODER_SEED:-17}"
batch_size="${BIOCODER_BATCH_SIZE:-4}"
gradient_accumulation_steps="${BIOCODER_GRADIENT_ACCUMULATION_STEPS:-4}"
epochs="${BIOCODER_EPOCHS:-1}"
learning_rate="${BIOCODER_LEARNING_RATE:-1e-6}"
save_steps="${BIOCODER_SAVE_STEPS:-250}"
log_steps="${BIOCODER_LOG_STEPS:-10}"
max_records="${BIOCODER_MAX_RECORDS:-}"
max_steps="${BIOCODER_MAX_STEPS:-}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-1024}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
min_local_free_gib="${BIOCODER_MIN_LOCAL_FREE_GIB:-24}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${BIOCODER_CODE_REVISION:0:7}"
model_short="${BIOCODER_MODEL_ARTIFACT_SHA256:0:16}"

for numeric_value in \
  "$seed" "$batch_size" "$gradient_accumulation_steps" "$epochs" \
  "$save_steps" "$log_steps" "$max_gpu_used_mib" "$max_gpu_utilization" \
  "$min_local_free_gib"
do
  [[ "$numeric_value" =~ ^[0-9]+$ ]] || {
    echo "Expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  }
done
for positive_value in \
  "$batch_size" "$gradient_accumulation_steps" "$epochs" "$save_steps" "$log_steps"
do
  (( positive_value >= 1 )) || {
    echo "Expected a positive integer, got: $positive_value" >&2
    exit 2
  }
done
for optional_value in "$max_records" "$max_steps"
do
  [[ -z "$optional_value" || "$optional_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "Optional record/step limits must be positive integers" >&2
    exit 2
  }
done
[[ "$learning_rate" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]] || {
  echo "BIOCODER_LEARNING_RATE must be a positive decimal" >&2
  exit 2
}
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]] || {
  echo "BIOCODER_GPU_INDEX must be a physical index or auto" >&2
  exit 2
}
[[ "$BIOCODER_MODEL_ARTIFACT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$BIOCODER_INSTRUCTION_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$BIOCODER_CODE_REVISION" =~ ^[0-9a-f]{40}$ ]]

test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$model_dir"
test -d "$assets_root"
test -d "$instruction_root"
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$BIOCODER_CODE_REVISION" ]] || {
  echo "Code revision mismatch: expected $BIOCODER_CODE_REVISION, got $actual_revision" >&2
  exit 3
}
export PYTHONPATH="$repo_root/backend"

run_mode="full"
bundle_label="full"
if [[ -n "$max_records" || -n "$max_steps" ]]; then
  run_mode="calibration"
  bundle_label="cal-${max_records:-all}"
fi
run_name="qwen3vl-4b-lora-${run_mode}-epoch${epochs}-b${batch_size}-ga${gradient_accumulation_steps}-seed${seed}-${code_short}"
output_root="${BIOCODER_LORA_OUTPUT_ROOT:-$run_root/qwen3vl/lora/runs}"
work_output="$output_root/.${run_name}.incomplete"
final_output="$output_root/$run_name"
bundle_dir="$run_root/qwen3vl/lora/bundles/${bundle_label}-seed${seed}-${code_short}"

verify_training_output() {
  local output="$1"
  (
    cd "$output"
    sha256sum -c artifact_manifest.sha256
  )
  "$python_bin" - "$output" "$run_mode" "$BIOCODER_CODE_REVISION" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
revision = sys.argv[3]
report = json.loads((root / "lora_training_report.json").read_text(encoding="utf-8"))
assert report["code_revision"] == revision
assert report["contracts"]["train_split_only"] is True
assert report["contracts"]["validation_prompts_opened"] is False
assert report["contracts"]["validation_answers_opened"] is False
assert report["contracts"]["internal_test_accessed"] is False
assert report["contracts"]["bitwise_determinism_claimed"] is False
assert report["development_training_complete"] is (mode == "full")
assert report["training"]["smoke_test"] is (mode != "full")
print("LORA_TRAINING_REPORT_CONTRACT=OK")
PY
}

echo "=== FORMAL TRAINING CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-manual}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "RUN_MODE=$run_mode"
echo "BATCH_SIZE=$batch_size"
echo "GRADIENT_ACCUMULATION_STEPS=$gradient_accumulation_steps"
echo "EFFECTIVE_BATCH_SIZE=$((batch_size * gradient_accumulation_steps))"
echo "EPOCHS=$epochs"
echo "MAX_RECORDS=${max_records:-none}"
echo "MAX_STEPS=${max_steps:-none}"
echo "SAVE_STEPS=$save_steps"

if [[ -d "$final_output" ]]; then
  echo "=== VERIFY ALREADY-COMPLETED RUN ==="
  verify_training_output "$final_output"
  echo "PERSISTENT_OUTPUT=$final_output"
  echo "QWEN3VL_LORA_TRAINING=ALREADY_COMPLETE"
  exit 0
fi

mkdir -p "$(dirname "$bundle_dir")" "$output_root"
if [[ -d "$bundle_dir" ]]; then
  echo "=== VERIFY EXISTING TRAIN-ONLY BUNDLE ==="
  (
    cd "$bundle_dir"
    sha256sum -c artifact_manifest.sha256
  )
else
  echo "=== BUILD TRAIN-ONLY BUNDLE ==="
  bundle_arguments=(
    --instruction-root "$instruction_root"
    --instruction-report-sha256 "$BIOCODER_INSTRUCTION_REPORT_SHA256"
    --assets-root "$assets_root"
    --output-dir "$bundle_dir"
    --seed "$seed"
  )
  if [[ -n "$max_records" ]]; then
    bundle_arguments+=(--max-records "$max_records")
  fi
  nice -n 10 ionice -c 3 "$python_bin" \
    -m multimodal_science.qwen3vl.build_lora_bundle_cli \
    "${bundle_arguments[@]}"
fi
bundle_report_sha256="$(sha256sum "$bundle_dir/lora_bundle_report.json" | awk '{print $1}')"
echo "LORA_BUNDLE_REPORT_SHA256=$bundle_report_sha256"

scratch="/tmp/biocoder-${USER}-qwen3vl-lora-${job_id}"
[[ "$scratch" == /tmp/biocoder-"$USER"-qwen3vl-lora-* ]]
test ! -e "$scratch"
mkdir -p "$scratch"
cleanup() {
  if [[ "$scratch" == /tmp/biocoder-"$USER"-qwen3vl-lora-* && -d "$scratch" ]]; then
    rm -rf -- "$scratch"
  fi
}
trap cleanup EXIT

available_kib="$(df -Pk /tmp | awk 'NR==2 {print $4}')"
required_kib="$((min_local_free_gib * 1024 * 1024))"
(( available_kib >= required_kib )) || {
  echo "Insufficient /tmp capacity: need at least ${min_local_free_gib} GiB free" >&2
  exit 6
}

echo "=== STAGE IMMUTABLE CODE ==="
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$BIOCODER_CODE_REVISION" | tar -xf - -C "$scratch/code"
local_bundle="$scratch/bundle"
cp -a "$bundle_dir" "$local_bundle"
(
  cd "$local_bundle"
  sha256sum -c artifact_manifest.sha256
)

echo "=== STAGE HASH-VERIFIED TRAIN IMAGES ==="
local_assets="$scratch/assets"
"$python_bin" - "$local_bundle/selection_manifest.jsonl" "$assets_root" "$local_assets" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

selection_path = pathlib.Path(sys.argv[1]).resolve()
source_root = pathlib.Path(sys.argv[2]).resolve()
target_root = pathlib.Path(sys.argv[3]).resolve()
seen = {}

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

for line_number, line in enumerate(selection_path.read_text(encoding="utf-8").splitlines(), 1):
    row = json.loads(line)
    relative = pathlib.PurePosixPath(row["image"])
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe image path at selection row {line_number}")
    expected = row.get("image_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing image hash at selection row {line_number}")
    key = relative.as_posix()
    if key in seen:
        if seen[key] != expected:
            raise ValueError(f"Conflicting image hash at selection row {line_number}")
        continue
    source = (source_root / pathlib.Path(*relative.parts)).resolve()
    target = (target_root / pathlib.Path(*relative.parts)).resolve()
    source.relative_to(source_root)
    target.relative_to(target_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != expected:
        raise ValueError(f"Staged image hash mismatch: {key}")
    seen[key] = expected

print(f"LOCAL_TRAIN_IMAGES={len(seen)}")
print("LOCAL_TRAIN_IMAGE_HASHES=OK")
PY

echo "=== STAGE MODEL CACHE ==="
model_cache_root="/tmp/biocoder-${USER}-models"
local_model="$model_cache_root/qwen3vl-${model_short}"
mkdir -p "$model_cache_root"
exec 8>"$model_cache_root/qwen3vl-${model_short}.lock"
flock 8
if [[ ! -d "$local_model" ]]; then
  model_staging="$scratch/model-staging"
  mkdir -p "$model_staging"
  cp -a "$model_dir/." "$model_staging/"
  mv "$model_staging" "$local_model"
fi
"$python_bin" - "$model_dir" "$local_model" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2]).resolve()

def inventory(root):
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }

source_inventory = inventory(source)
target_inventory = inventory(target)
assert source_inventory
assert target_inventory == source_inventory
print(f"LOCAL_MODEL_FILES={len(target_inventory)}")
print(f"LOCAL_MODEL_BYTES={sum(target_inventory.values())}")
print("LOCAL_MODEL_INVENTORY=OK")
PY
flock -u 8

gpu_passes_guard() {
  local candidate="$1"
  local sample gpu_state used_mib remainder utilization
  for sample in 1 2 3
  do
    gpu_state="$(
      nvidia-smi -i "$candidate" \
        --query-gpu=memory.used,memory.free,utilization.gpu,name \
        --format=csv,noheader,nounits
    )" || return 1
    echo "candidate=$candidate sample=$sample $gpu_state" >&2
    used_mib="${gpu_state%%,*}"
    remainder="${gpu_state#*,}"
    remainder="${remainder#*,}"
    utilization="${remainder%%,*}"
    used_mib="${used_mib// /}"
    utilization="${utilization// /}"
    [[ "$used_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
    (( used_mib <= max_gpu_used_mib )) || return 1
    (( utilization <= max_gpu_utilization )) || return 1
    sleep 2
  done
}

echo "=== GPU STARTUP GUARD ==="
if [[ "$gpu_index" == "auto" ]]; then
  gpu_index=""
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  for (( candidate=0; candidate<gpu_count; candidate++ ))
  do
    lock_path="/tmp/biocoder-${USER}-gpu-${candidate}.lock"
    exec 9>"$lock_path"
    if flock -n 9 && gpu_passes_guard "$candidate"
    then
      gpu_index="$candidate"
      break
    fi
  done
  [[ -n "$gpu_index" ]] || {
    echo "No unlocked GPU passed the memory/utilization guard" >&2
    exit 5
  }
else
  lock_path="/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
  exec 9>"$lock_path"
  flock -n 9 || {
    echo "BioCoder GPU lock is already held: $lock_path" >&2
    exit 4
  }
  gpu_passes_guard "$gpu_index" || {
    echo "GPU $gpu_index failed the memory/utilization guard" >&2
    exit 5
  }
fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$scratch/code/backend"
export TOKENIZERS_PARALLELISM=false
"$python_bin" - <<'PY'
import peft
import safetensors
import torch
import transformers

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

if [[ -f "$work_output/lora_training_report.json" ]]; then
  echo "=== RECOVER COMPLETED BUT UNPUBLISHED RUN ==="
  verify_training_output "$work_output"
else
  resume_arguments=()
  if [[ -d "$work_output" ]]; then
    resume_arguments+=(--resume)
    echo "RESUME_FROM_PERSISTENT_CHECKPOINT=YES"
  else
    echo "RESUME_FROM_PERSISTENT_CHECKPOINT=NO"
  fi
  limit_arguments=()
  if [[ -n "$max_steps" ]]; then
    limit_arguments+=(--max-steps "$max_steps")
  fi
  echo "=== TRAIN QWEN3-VL LORA ==="
  nice -n 5 ionice -c 2 -n 7 "$python_bin" \
    -m multimodal_science.qwen3vl.train_lora_cli \
    --bundle-root "$local_bundle" \
    --bundle-report-sha256 "$bundle_report_sha256" \
    --assets-root "$local_assets" \
    --output-dir "$work_output" \
    --model-name-or-path "$local_model" \
    --model-revision "$BIOCODER_MODEL_REVISION" \
    --model-artifact-sha256 "$BIOCODER_MODEL_ARTIFACT_SHA256" \
    --code-revision "$BIOCODER_CODE_REVISION" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$gradient_accumulation_steps" \
    --learning-rate "$learning_rate" \
    --save-steps "$save_steps" \
    --log-steps "$log_steps" \
    --seed "$seed" \
    "${limit_arguments[@]}" \
    "${resume_arguments[@]}"
  verify_training_output "$work_output"
fi

test ! -e "$final_output"
mv "$work_output" "$final_output"
training_report_sha256="$(sha256sum "$final_output/lora_training_report.json" | awk '{print $1}')"
adapter_manifest_sha256="$(sha256sum "$final_output/artifact_manifest.sha256" | awk '{print $1}')"

"$python_bin" - "$final_output/lora_training_report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print("WALL_TIME_SECONDS=", round(report["wall_time_seconds"], 3))
print("PEAK_CUDA_GIB=", round(report["runtime"]["peak_cuda_memory_bytes"] / 2**30, 3))
print("TRAINING_RECORDS=", report["training"]["training_records"])
print("OPTIMIZER_UPDATES=", report["training"]["optimizer_updates"])
print("DEVELOPMENT_TRAINING_COMPLETE=", report["development_training_complete"])
PY

echo "TRAINING_REPORT_SHA256=$training_report_sha256"
echo "ADAPTER_MANIFEST_SHA256=$adapter_manifest_sha256"
echo "PERSISTENT_OUTPUT=$final_output"
echo "QWEN3VL_LORA_TRAINING=OK"
