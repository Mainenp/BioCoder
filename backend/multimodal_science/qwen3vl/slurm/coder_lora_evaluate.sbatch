#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

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
  BIOCODER_ADAPTER_ROOT \
  BIOCODER_ADAPTER_TRAINING_REPORT_SHA256 \
  BIOCODER_ADAPTER_MANIFEST_SHA256 \
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
bundle_root="${BIOCODER_INFERENCE_BUNDLE_ROOT:-$run_root/qwen3vl/inference-bundle-v2-bilingual-2b8d246}"
adapter_root="$BIOCODER_ADAPTER_ROOT"
gpu_index="$BIOCODER_GPU_INDEX"
seed="${BIOCODER_SEED:-17}"
batch_size="${BIOCODER_INFERENCE_BATCH_SIZE:-2}"
bootstrap_iterations="${BIOCODER_BOOTSTRAP_ITERATIONS:-1000}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-1024}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
min_local_free_gib="${BIOCODER_MIN_LOCAL_FREE_GIB:-24}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${BIOCODER_CODE_REVISION:0:7}"
model_short="${BIOCODER_MODEL_ARTIFACT_SHA256:0:16}"
adapter_short="${BIOCODER_ADAPTER_TRAINING_REPORT_SHA256:0:12}"

ensure_artifact_manifest() {
  local root="$1"
  shift
  "$python_bin" - "$root" "$@" <<'PY'
import hashlib
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
required = sys.argv[2:]
if not root.is_dir():
    raise FileNotFoundError(root)
for relative in required:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)

manifest = root / "artifact_manifest.sha256"
if not manifest.exists():
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and not path.name.startswith(".artifact_manifest.sha256.")
    )
    if not files:
        raise ValueError(f"No artifacts found under {root}")

    def digest(path):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    temporary = root / f".artifact_manifest.sha256.{os.getpid()}.tmp"
    temporary.write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    print(f"ARTIFACT_MANIFEST_CREATED={manifest}")
else:
    print(f"ARTIFACT_MANIFEST_REUSED={manifest}")
PY
  (
    cd "$root"
    sha256sum -c artifact_manifest.sha256
  )
}

for numeric_value in \
  "$seed" "$batch_size" "$bootstrap_iterations" "$max_gpu_used_mib" \
  "$max_gpu_utilization" "$min_local_free_gib"
do
  [[ "$numeric_value" =~ ^[0-9]+$ ]] || {
    echo "Expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  }
done
(( batch_size >= 1 && bootstrap_iterations >= 1 ))
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]] || {
  echo "BIOCODER_GPU_INDEX must be a physical index or auto" >&2
  exit 2
}
for digest in \
  "$BIOCODER_MODEL_ARTIFACT_SHA256" \
  "$BIOCODER_INSTRUCTION_REPORT_SHA256" \
  "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" \
  "$BIOCODER_ADAPTER_TRAINING_REPORT_SHA256" \
  "$BIOCODER_ADAPTER_MANIFEST_SHA256"
do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
done
[[ "$BIOCODER_CODE_REVISION" =~ ^[0-9a-f]{40}$ ]]

test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$model_dir"
test -d "$assets_root"
test -d "$instruction_root"
test -d "$bundle_root"
test -d "$adapter_root"
actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$BIOCODER_CODE_REVISION" ]] || {
  echo "Code revision mismatch: expected $BIOCODER_CODE_REVISION, got $actual_revision" >&2
  exit 3
}

run_name="qwen3vl-4b-lora-validation-${adapter_short}-${code_short}"
generation_final="$run_root/qwen3vl/runs/$run_name"
generation_resume="$run_root/qwen3vl/runs/.${run_name}.work"
evaluation_final="$run_root/qwen3vl/evaluations/${run_name}-evaluation"

echo "=== LORA EVALUATION CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-manual}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "ADAPTER_TRAINING_REPORT_SHA256=$BIOCODER_ADAPTER_TRAINING_REPORT_SHA256"
echo "ADAPTER_MANIFEST_SHA256=$BIOCODER_ADAPTER_MANIFEST_SHA256"
echo "INFERENCE_BATCH_SIZE=$batch_size"
echo "BOOTSTRAP_ITERATIONS=$bootstrap_iterations"

scratch="/tmp/biocoder-${USER}-qwen3vl-lora-eval-${job_id}"
[[ "$scratch" == /tmp/biocoder-"$USER"-qwen3vl-lora-eval-* ]]
test ! -e "$scratch"
mkdir -p "$scratch"
cleanup() {
  if [[ "$scratch" == /tmp/biocoder-"$USER"-qwen3vl-lora-eval-* && -d "$scratch" ]]; then
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

echo "=== STAGE IMMUTABLE CODE AND PROMPTS ==="
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$BIOCODER_CODE_REVISION" | tar -xf - -C "$scratch/code"
export PYTHONPATH="$scratch/code/backend"
local_bundle="$scratch/inference-bundle"
cp -a "$bundle_root" "$local_bundle"
"$python_bin" - "$local_bundle/inference_bundle_report.json" \
  "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print("LOCAL_INFERENCE_BUNDLE_REPORT=OK")
PY

echo "=== STAGE VALIDATION IMAGES WITH COPY VERIFICATION ==="
local_assets="$scratch/assets"
"$python_bin" - "$local_bundle/inference_prompts.jsonl" "$assets_root" "$local_assets" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

prompt_path = pathlib.Path(sys.argv[1]).resolve()
source_root = pathlib.Path(sys.argv[2]).resolve()
target_root = pathlib.Path(sys.argv[3]).resolve()
seen = {}

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

for line_number, line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines(), 1):
    row = json.loads(line)
    relative = pathlib.PurePosixPath(row["image"])
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe validation image path at row {line_number}")
    key = relative.as_posix()
    if key in seen:
        continue
    source = (source_root / pathlib.Path(*relative.parts)).resolve()
    target = (target_root / pathlib.Path(*relative.parts)).resolve()
    source.relative_to(source_root)
    target.relative_to(target_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    shutil.copy2(source, target)
    if digest(target) != source_hash:
        raise ValueError(f"Validation image copy mismatch: {key}")
    seen[key] = source_hash

print(f"LOCAL_VALIDATION_IMAGES={len(seen)}")
print("LOCAL_VALIDATION_IMAGE_COPIES=OK")
PY

echo "=== STAGE OR REUSE MODEL CACHE ==="
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
export TOKENIZERS_PARALLELISM=false
"$python_bin" - <<'PY'
import peft
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
PY

mkdir -p "$(dirname "$generation_final")" "$(dirname "$evaluation_final")"
if [[ -d "$generation_final" ]]; then
  echo "=== VERIFY EXISTING COMPLETE GENERATION ==="
  ensure_artifact_manifest \
    "$generation_final" \
    generation_report.json predictions.jsonl generation_records.jsonl \
    run_config.json runtime_metadata.json
else
  resume_arguments=()
  if [[ -d "$generation_resume" ]]; then
    resume_arguments+=(--resume)
    echo "RESUME_GENERATION=YES"
  else
    echo "RESUME_GENERATION=NO"
  fi
  echo "=== RUN COMPLETE ADAPTER INFERENCE ==="
  nice -n 5 ionice -c 2 -n 7 "$python_bin" \
    -m multimodal_science.qwen3vl.run_inference_cli \
    --bundle-root "$local_bundle" \
    --bundle-report-sha256 "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" \
    --assets-root "$local_assets" \
    --output-dir "$generation_final" \
    --model-name-or-path "$local_model" \
    --model-revision "$BIOCODER_MODEL_REVISION" \
    --model-artifact-sha256 "$BIOCODER_MODEL_ARTIFACT_SHA256" \
    --adapter-root "$adapter_root" \
    --adapter-training-report-sha256 "$BIOCODER_ADAPTER_TRAINING_REPORT_SHA256" \
    --adapter-manifest-sha256 "$BIOCODER_ADAPTER_MANIFEST_SHA256" \
    --batch-size "$batch_size" \
    --max-new-tokens 64 \
    --seed "$seed" \
    --dtype bfloat16 \
    --device-map auto \
    --attention-implementation sdpa \
    "${resume_arguments[@]}"
  ensure_artifact_manifest \
    "$generation_final" \
    generation_report.json predictions.jsonl generation_records.jsonl \
    run_config.json runtime_metadata.json
fi

generation_report_sha256="$(sha256sum "$generation_final/generation_report.json" | awk '{print $1}')"
echo "GENERATION_REPORT_SHA256=$generation_report_sha256"

if [[ -d "$evaluation_final" ]]; then
  echo "=== VERIFY EXISTING COMPLETE EVALUATION ==="
  ensure_artifact_manifest \
    "$evaluation_final" qwen_evaluation_report.json evaluation_records.jsonl
else
  evaluation_staging="$scratch/evaluation"
  echo "=== RUN ANSWER-SEPARATED EVALUATION ==="
  nice -n 10 ionice -c 3 "$python_bin" \
    -m multimodal_science.qwen3vl.evaluate_predictions_cli \
    --instruction-root "$instruction_root" \
    --predictions "$generation_final/predictions.jsonl" \
    --output-dir "$evaluation_staging" \
    --instruction-report-sha256 "$BIOCODER_INSTRUCTION_REPORT_SHA256" \
    --generation-report "$generation_final/generation_report.json" \
    --generation-report-sha256 "$generation_report_sha256" \
    --bootstrap-iterations "$bootstrap_iterations" \
    --seed "$seed"
  ensure_artifact_manifest \
    "$evaluation_staging" qwen_evaluation_report.json evaluation_records.jsonl
  mv "$evaluation_staging" "$evaluation_final"
fi

evaluation_report_sha256="$(sha256sum "$evaluation_final/qwen_evaluation_report.json" | awk '{print $1}')"
"$python_bin" - "$generation_final/generation_report.json" \
  "$evaluation_final/qwen_evaluation_report.json" \
  "$BIOCODER_ADAPTER_TRAINING_REPORT_SHA256" \
  "$BIOCODER_ADAPTER_MANIFEST_SHA256" \
  "$BIOCODER_MODEL_ARTIFACT_SHA256" \
  "$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256" \
  "$BIOCODER_INSTRUCTION_REPORT_SHA256" \
  "$seed" "$bootstrap_iterations" <<'PY'
import hashlib
import json
import pathlib
import sys

generation = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_training_report_sha256 = sys.argv[3]
expected_adapter_manifest_sha256 = sys.argv[4]
expected_model_artifact_sha256 = sys.argv[5]
expected_bundle_report_sha256 = sys.argv[6]
expected_instruction_report_sha256 = sys.argv[7]
expected_seed = int(sys.argv[8])
expected_bootstrap_iterations = int(sys.argv[9])
assert generation["counts"]["predictions"] == 13708
assert generation["scope"]["complete_prompt_coverage"] is True
assert generation["development_comparison_candidate"] is True
assert generation["runtime"]["adapter_loaded"] is True
assert generation["model"]["adapter"]["training_report_sha256"] == expected_training_report_sha256
assert generation["model"]["adapter"]["manifest_sha256"] == expected_adapter_manifest_sha256
assert generation["model"]["artifact_sha256"] == expected_model_artifact_sha256
assert generation["source"]["bundle_report_sha256"] == expected_bundle_report_sha256
assert generation["source"]["instruction_report_sha256"] == expected_instruction_report_sha256
assert generation["generation"]["do_sample"] is False
assert generation["generation"]["max_new_tokens"] == 64
assert generation["generation"]["seed"] == expected_seed
assert generation["generation"]["attention_implementation"] == "sdpa"
assert generation["internal_test_accessed"] is False
assert evaluation["counts"]["predictions"] == 13708
assert evaluation["inputs"]["instruction_report_sha256"] == expected_instruction_report_sha256
assert evaluation["inputs"]["generation_report_sha256"] == hashlib.sha256(
    pathlib.Path(sys.argv[1]).read_bytes()
).hexdigest()
assert evaluation["evaluation"]["seed"] == expected_seed
assert evaluation["evaluation"]["bootstrap_iterations"] == expected_bootstrap_iterations
assert evaluation["prediction_generation_provenance_verified"] is True
assert evaluation["development_comparison_eligible"] is True
assert evaluation["final_benchmark_eligible"] is False
assert evaluation["internal_test_accessed"] is False
print("LORA_FULL_GENERATION_CONTRACT=OK")
print("LORA_FULL_EVALUATION_CONTRACT=OK")
print("SCHEMA_VALID_RECORDS=", evaluation["counts"]["schema_valid"])
PY

echo "EVALUATION_REPORT_SHA256=$evaluation_report_sha256"
echo "PERSISTENT_GENERATION=$generation_final"
echo "PERSISTENT_EVALUATION=$evaluation_final"
echo "QWEN3VL_LORA_EVALUATION=OK"
