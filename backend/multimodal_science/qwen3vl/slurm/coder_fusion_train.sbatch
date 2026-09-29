#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
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
  BIOCODER_REPO_ROOT \
  BIOCODER_RUN_ROOT \
  BIOCODER_PYTHON \
  BIOCODER_CODE_REVISION \
  BIOCODER_MODEL_ROOT \
  BIOCODER_MODEL_REVISION \
  BIOCODER_MODEL_ARTIFACT_SHA256 \
  BIOCODER_MODEL_MANIFEST_PATH \
  BIOCODER_MODEL_MANIFEST_SHA256 \
  BIOCODER_FUSION_BUNDLE_ROOT \
  BIOCODER_FUSION_BUNDLE_REPORT_SHA256 \
  BIOCODER_LORA_BUNDLE_ROOT \
  BIOCODER_LORA_BUNDLE_REPORT_SHA256 \
  BIOCODER_DATASET_ROOT \
  BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_ASSETS_ROOT \
  BIOCODER_INITIAL_ADAPTER_ROOT \
  BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256
do
  require_variable "$required_name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
model_source="$BIOCODER_MODEL_ROOT"
model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"
fusion_source="$BIOCODER_FUSION_BUNDLE_ROOT"
lora_source="$BIOCODER_LORA_BUNDLE_ROOT"
dataset_source="$BIOCODER_DATASET_ROOT"
assets_source="$BIOCODER_ASSETS_ROOT"
adapter_source="$BIOCODER_INITIAL_ADAPTER_ROOT"
code_revision="$BIOCODER_CODE_REVISION"
model_revision="$BIOCODER_MODEL_REVISION"
model_artifact_sha256="$BIOCODER_MODEL_ARTIFACT_SHA256"
model_manifest_sha256="$BIOCODER_MODEL_MANIFEST_SHA256"
fusion_report_sha256="$BIOCODER_FUSION_BUNDLE_REPORT_SHA256"
lora_report_sha256="$BIOCODER_LORA_BUNDLE_REPORT_SHA256"
dataset_report_sha256="$BIOCODER_DATASET_REPORT_SHA256"
adapter_report_sha256="$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256"
adapter_manifest_sha256="$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"
pretrained_projector_source="${BIOCODER_PRETRAINED_PROJECTOR_ROOT:-}"
pretrained_projector_report_sha256="${BIOCODER_PRETRAINED_PROJECTOR_REPORT_SHA256:-}"
pretrained_projector_manifest_sha256="${BIOCODER_PRETRAINED_PROJECTOR_MANIFEST_SHA256:-}"
gpu_index="${BIOCODER_GPU_INDEX:-7}"
epochs="${BIOCODER_EPOCHS:-1}"
batch_size="${BIOCODER_BATCH_SIZE:-1}"
gradient_accumulation_steps="${BIOCODER_GRADIENT_ACCUMULATION_STEPS:-16}"
lora_learning_rate="${BIOCODER_LORA_LEARNING_RATE:-2e-7}"
projector_learning_rate="${BIOCODER_PROJECTOR_LEARNING_RATE:-1e-4}"
save_steps="${BIOCODER_SAVE_STEPS:-250}"
log_steps="${BIOCODER_LOG_STEPS:-10}"
seed="${BIOCODER_SEED:-17}"
max_steps="${BIOCODER_MAX_STEPS:-}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
min_local_free_gib="${BIOCODER_MIN_LOCAL_FREE_GIB:-32}"
job_id="${SLURM_JOB_ID:-manual}"
job_name="${SLURM_JOB_NAME:-coder}"
code_short="${code_revision:0:7}"
model_short="${model_artifact_sha256:0:16}"

for value in \
  "$epochs" "$batch_size" "$gradient_accumulation_steps" "$save_steps" \
  "$log_steps" "$seed" "$max_gpu_used_mib" "$max_gpu_utilization" \
  "$min_local_free_gib"
do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "Expected a nonnegative integer, got: $value" >&2
    exit 2
  }
done
for value in \
  "$epochs" "$batch_size" "$gradient_accumulation_steps" "$save_steps" "$log_steps"
do
  (( value >= 1 )) || {
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  }
done
[[ -z "$max_steps" || "$max_steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "BIOCODER_MAX_STEPS must be empty or a positive integer" >&2
  exit 2
}
[[ "$batch_size" == "1" ]] || {
  echo "Fusion training requires BIOCODER_BATCH_SIZE=1" >&2
  exit 2
}
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]] || {
  echo "BIOCODER_GPU_INDEX must be a physical index or auto" >&2
  exit 2
}
for digest in \
  "$model_artifact_sha256" \
  "$model_manifest_sha256" \
  "$fusion_report_sha256" \
  "$lora_report_sha256" \
  "$dataset_report_sha256" \
  "$adapter_report_sha256" \
  "$adapter_manifest_sha256"
do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Expected a lowercase SHA-256 digest: $digest" >&2
    exit 2
  }
done
if [[ -n "$pretrained_projector_source" ]]; then
  for digest in \
    "$pretrained_projector_report_sha256" \
    "$pretrained_projector_manifest_sha256"
  do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
      echo "Expected a lowercase auxiliary-pretraining SHA-256 digest: $digest" >&2
      exit 2
    }
  done
  test -d "$pretrained_projector_source"
else
  [[ -z "$pretrained_projector_report_sha256" ]]
  [[ -z "$pretrained_projector_manifest_sha256" ]]
fi
[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$model_source"
test -f "$model_manifest"
test -d "$fusion_source"
test -d "$lora_source"
test -d "$dataset_source"
test -d "$assets_source"
test -d "$adapter_source"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 3
}

run_mode="full"
if [[ -n "$max_steps" ]]; then
  run_mode="calibration-steps${max_steps}"
fi
run_name="image-xic-${run_mode}-epoch${epochs}-b1-ga${gradient_accumulation_steps}-seed${seed}-${code_short}"
output_root="${BIOCODER_FUSION_OUTPUT_ROOT:-$run_root/qwen3vl/fusion/runs}"
work_output="$output_root/.${run_name}.incomplete"
final_output="$output_root/$run_name"
mkdir -p "$output_root"

run_lock="$output_root/.${run_name}.lock"
exec 7>"$run_lock"
flock -n 7 || {
  echo "Another job already owns this exact fusion run: $run_name" >&2
  exit 7
}
echo "RUN_LOCK_ACQUIRED=$run_lock"

verify_training_output() {
  local output="$1"
  (
    cd "$output"
    sha256sum -c artifact_manifest.sha256
  )
  "$python_bin" - "$output" "$run_mode" "$code_revision" \
    "${pretrained_projector_source:+true}" <<'PY'
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


root = Path(sys.argv[1])
mode = sys.argv[2]
revision = sys.argv[3]
uses_auxiliary_pretraining = sys.argv[4] == "true"
report = json.loads(
    (root / "fusion_training_report.json").read_text(encoding="utf-8")
)
require(
    report["schema_version"] == "chrompeak-qwen3vl-xic-fusion-training-v1",
    "Unexpected fusion training schema",
)
require(report["code_revision"] == revision, "Code revision mismatch")
contracts = report["contracts"]
for name in (
    "base_weights_frozen",
    "vision_tower_frozen",
    "vision_merger_frozen",
    "assistant_tokens_only_supervision",
    "image_and_xic_forward",
    "native_multimodal_rope_positions",
    "lora_and_projector_backward",
    "parameter_updates_verified",
    "train_split_only",
    "staged_train_only_input_roots",
    "fusion_lora_training_artifacts_bound",
    "initial_adapter_training_rows_bound",
):
    require(contracts[name] is True, f"Contract failed: {name}")
require(contracts["validation_prompts_opened"] is False, "Validation prompt leak")
require(contracts["validation_answers_opened"] is False, "Validation answer leak")
require(contracts["internal_test_accessed"] is False, "Internal-test leak")
require(
    contracts["auxiliary_pretraining_bound"] is uses_auxiliary_pretraining,
    "Auxiliary-pretraining binding mismatch",
)
require(contracts["bitwise_determinism_claimed"] is False, "Bad determinism claim")
require(
    report["development_training_complete"] is (mode == "full"),
    "Development completion flag mismatch",
)
require(
    report["training"]["calibration_run"] is (mode != "full"),
    "Calibration flag mismatch",
)
require(report["development_comparison_eligible"] is False, "Unevaluated run eligible")
require(report["final_benchmark_eligible"] is False, "Development run is final")
state = report["training"]["parameter_state_sha256"]
require(state["lora_initial"] != state["lora_after"], "LoRA did not update")
require(
    state["projector_initial"] != state["projector_after"],
    "Projector did not update",
)
print("QWEN3VL_XIC_FUSION_TRAINING_CONTRACT=OK")
PY
}

echo "=== FORMAL FUSION TRAINING CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=$job_name"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "RUN_MODE=$run_mode"
echo "EPOCHS=$epochs"
echo "BATCH_SIZE=$batch_size"
echo "GRADIENT_ACCUMULATION_STEPS=$gradient_accumulation_steps"
echo "EFFECTIVE_BATCH_SIZE=$((batch_size * gradient_accumulation_steps))"
echo "MAX_STEPS=${max_steps:-none}"
echo "SAVE_STEPS=$save_steps"

if [[ -d "$final_output" ]]; then
  echo "=== VERIFY ALREADY-COMPLETED RUN ==="
  verify_training_output "$final_output"
  echo "PERSISTENT_OUTPUT=$final_output"
  echo "QWEN3VL_XIC_FUSION_TRAINING=ALREADY_COMPLETE"
  exit 0
fi

available_kib="$(df -Pk /tmp | awk 'NR==2 {print $4}')"
required_kib="$((min_local_free_gib * 1024 * 1024))"
(( available_kib >= required_kib )) || {
  echo "Insufficient /tmp capacity: need ${min_local_free_gib} GiB free" >&2
  exit 6
}
scratch="$(mktemp -d "/tmp/biocoder-${USER}-qwen3vl-fusion-train-${job_id}.XXXXXX")"
cleanup() {
  case "${scratch:-}" in
    "/tmp/biocoder-${USER}-qwen3vl-fusion-train-${job_id}."*) rm -rf -- "$scratch" ;;
  esac
}
trap cleanup EXIT

echo "=== STAGE IMMUTABLE CODE ==="
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$scratch/code"

echo "=== STAGE TRAIN-ONLY INPUT ROOTS ==="
train_view="$scratch/train-only"
fusion_root="$train_view/fusion"
lora_root="$train_view/lora"
dataset_root="$train_view/dataset"
assets_root="$train_view/assets"
adapter_root="$train_view/initial-adapter"
pretrained_projector_root="$train_view/auxiliary-pretraining"
"$python_bin" - \
  "$fusion_source" "$fusion_root" \
  "$lora_source" "$lora_root" \
  "$dataset_source" "$dataset_root" \
  "$assets_source" "$assets_root" \
  "$adapter_source" "$adapter_root" \
  "$adapter_manifest_sha256" \
  "$pretrained_projector_source" "$pretrained_projector_root" \
  "$pretrained_projector_manifest_sha256" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_child(root: Path, relative: object) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative, "Bad path")
    pure = PurePosixPath(relative)
    require(
        not pure.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"Unsafe path: {relative}",
    )
    candidate = (root / Path(*pure.parts)).resolve()
    candidate.relative_to(root)
    return candidate


def copy_file(source_root: Path, target_root: Path, relative: object) -> Path:
    source = safe_child(source_root, relative)
    require(source.is_file(), f"Missing source artifact: {source}")
    target = safe_child(target_root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def stage_report(
    source_root: Path,
    target_root: Path,
    report_name: str,
    artifact_keys: tuple[str, ...],
) -> dict:
    source_report = source_root / report_name
    report = json.loads(source_report.read_text(encoding="utf-8"))
    target_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_report, target_root / report_name)
    artifacts = report.get("artifacts")
    require(isinstance(artifacts, dict), f"Missing artifacts: {source_report}")
    for key in artifact_keys:
        item = artifacts.get(key)
        require(isinstance(item, dict), f"Missing artifact: {key}")
        target = copy_file(source_root, target_root, item.get("path"))
        require(digest(target) == item.get("sha256"), f"Artifact hash mismatch: {key}")
    return report


fusion_source, fusion_target = map(Path, sys.argv[1:3])
lora_source, lora_target = map(Path, sys.argv[3:5])
dataset_source, dataset_target = map(Path, sys.argv[5:7])
assets_source, assets_target = map(Path, sys.argv[7:9])
adapter_source, adapter_target = map(Path, sys.argv[9:11])
expected_adapter_manifest_sha256 = sys.argv[11]
pretraining_source_text = sys.argv[12]
pretraining_target = Path(sys.argv[13])
expected_pretraining_manifest_sha256 = sys.argv[14]
for root in (fusion_source, lora_source, dataset_source, assets_source, adapter_source):
    require(root.resolve().is_dir(), f"Missing source root: {root}")
fusion_source = fusion_source.resolve()
lora_source = lora_source.resolve()
dataset_source = dataset_source.resolve()
assets_source = assets_source.resolve()
adapter_source = adapter_source.resolve()

stage_report(
    fusion_source,
    fusion_target,
    "fusion_bundle_report.json",
    ("train_xic_links",),
)
lora_report = stage_report(
    lora_source,
    lora_target,
    "lora_bundle_report.json",
    ("train_qwen", "selection_manifest"),
)
stage_report(
    dataset_source,
    dataset_target,
    "dataset_report.json",
    ("train_signals", "train_examples"),
)

selection_item = lora_report["artifacts"]["selection_manifest"]
selection_path = safe_child(lora_target, selection_item["path"])
assets_target.mkdir(parents=True, exist_ok=False)
seen: dict[str, str] = {}
with selection_path.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        row = json.loads(line)
        relative = row.get("image")
        expected = row.get("image_sha256")
        require(
            isinstance(expected, str) and len(expected) == 64,
            f"Missing image hash at row {line_number}",
        )
        key = PurePosixPath(str(relative)).as_posix()
        if key in seen:
            require(seen[key] == expected, f"Conflicting image hash: {key}")
            continue
        target = copy_file(assets_source, assets_target, relative)
        require(digest(target) == expected, f"Staged image hash mismatch: {key}")
        seen[key] = expected

adapter_target.mkdir(parents=True, exist_ok=False)
manifest_source = adapter_source / "artifact_manifest.sha256"
require(manifest_source.is_file(), "Missing initial-adapter manifest")
require(
    digest(manifest_source) == expected_adapter_manifest_sha256,
    "Initial-adapter manifest hash mismatch",
)
shutil.copy2(manifest_source, adapter_target / manifest_source.name)
for line_number, line in enumerate(
    manifest_source.read_text(encoding="utf-8").splitlines(), 1
):
    parts = line.split("  ", maxsplit=1)
    require(len(parts) == 2, f"Malformed adapter manifest line {line_number}")
    expected, relative = parts
    target = copy_file(adapter_source, adapter_target, relative)
    require(digest(target) == expected, f"Initial-adapter hash mismatch: {relative}")

if pretraining_source_text:
    pretraining_source = Path(pretraining_source_text).resolve()
    require(pretraining_source.is_dir(), "Missing auxiliary-pretraining source")
    pretraining_target.mkdir(parents=True, exist_ok=False)
    pretraining_manifest = pretraining_source / "artifact_manifest.sha256"
    require(pretraining_manifest.is_file(), "Missing auxiliary-pretraining manifest")
    require(
        digest(pretraining_manifest) == expected_pretraining_manifest_sha256,
        "Auxiliary-pretraining manifest hash mismatch",
    )
    shutil.copy2(pretraining_manifest, pretraining_target / pretraining_manifest.name)
    for line_number, line in enumerate(
        pretraining_manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.split("  ", maxsplit=1)
        require(len(parts) == 2, f"Malformed pretraining manifest line {line_number}")
        expected, relative = parts
        target = copy_file(pretraining_source, pretraining_target, relative)
        require(digest(target) == expected, f"Auxiliary-pretraining drift: {relative}")

for forbidden in (
    fusion_target / "validation_xic_links.jsonl",
    dataset_target / "validation",
    assets_target / "jobs" / "validation",
):
    require(not forbidden.exists(), f"Validation artifact entered train view: {forbidden}")
print(f"LOCAL_TRAIN_IMAGES={len(seen)}")
print("STAGED_TRAIN_ONLY_INPUT_ROOTS=OK")
print("STAGED_INITIAL_ADAPTER=OK")
print(f"STAGED_AUXILIARY_PRETRAINING={'YES' if pretraining_source_text else 'NO'}")
PY

echo "=== STAGE HASH-VERIFIED MODEL CACHE ==="
model_cache_root="/tmp/biocoder-${USER}-models"
local_model="$model_cache_root/qwen3vl-${model_short}"
mkdir -p "$model_cache_root"
exec 8>"$model_cache_root/qwen3vl-${model_short}.lock"
flock 8
verify_model_cache() {
  (
    cd "$scratch/code/backend"
    PYTHONNOUSERSITE=1 PYTHONPATH="$scratch/code/backend" "$python_bin" - \
      "$local_model" "$model_manifest" "$model_manifest_sha256" <<'PY'
import sys
from pathlib import Path

from multimodal_science.qwen3vl.fusion_smoke import _verify_model_manifest

inventory = _verify_model_manifest(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
print(f"LOCAL_MODEL_FILES={inventory['files']}")
print(f"LOCAL_MODEL_BYTES={inventory['bytes']}")
PY
  )
}
if [[ -d "$local_model" ]] && ! verify_model_cache; then
  [[ "$local_model" == "$model_cache_root"/qwen3vl-* ]] || exit 6
  rm -rf -- "$local_model"
fi
if [[ ! -d "$local_model" ]]; then
  model_staging="$scratch/model-staging"
  mkdir -p "$model_staging"
  cp -a "$model_source/." "$model_staging/"
  mv "$model_staging" "$local_model"
fi
verify_model_cache
flock -u 8

gpu_passes_guard() {
  local candidate="$1"
  local sample line used remainder utilization
  for sample in 1 2 3
  do
    line="$(nvidia-smi -i "$candidate" \
      --query-gpu=memory.used,memory.free,utilization.gpu,name \
      --format=csv,noheader,nounits)" || return 1
    echo "candidate=$candidate sample=$sample $line" >&2
    used="${line%%,*}"
    remainder="${line#*,}"
    remainder="${remainder#*,}"
    utilization="${remainder%%,*}"
    used="${used// /}"
    utilization="${utilization// /}"
    [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
    (( used <= max_gpu_used_mib && utilization <= max_gpu_utilization )) || return 1
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
    if flock -n 9 && gpu_passes_guard "$candidate"; then
      gpu_index="$candidate"
      break
    fi
  done
  [[ -n "$gpu_index" ]] || {
    echo "No unlocked GPU passed the startup guard" >&2
    exit 5
  }
else
  lock_path="/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
  exec 9>"$lock_path"
  flock -n 9 || { echo "GPU lock already held: $lock_path" >&2; exit 4; }
  gpu_passes_guard "$gpu_index" || {
    echo "GPU $gpu_index failed the startup guard" >&2
    exit 5
  }
fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

if [[ -f "$work_output/fusion_training_report.json" ]]; then
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
  pretrained_projector_arguments=()
  if [[ -n "$pretrained_projector_source" ]]; then
    pretrained_projector_arguments+=(
      --pretrained-projector-root "$pretrained_projector_root"
      --pretrained-projector-report-sha256 "$pretrained_projector_report_sha256"
      --pretrained-projector-manifest-sha256 "$pretrained_projector_manifest_sha256"
    )
  fi
  mkdir -p "$scratch/tmp" "$scratch/cache"
  for variable in "${!BIOCODER_@}"
  do
    unset "$variable"
  done
  echo "=== TRAIN QWEN3-VL IMAGE-XIC FUSION ==="
  (
    cd "$scratch/code/backend"
    env -i \
      HOME="$HOME" \
      USER="$USER" \
      PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      LANG="${LANG:-C.UTF-8}" \
      TMPDIR="$scratch/tmp" \
      XDG_CACHE_HOME="$scratch/cache" \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu_index" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONPATH="$scratch/code/backend" \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      TRANSFORMERS_OFFLINE=1 \
      HF_HUB_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" \
      SLURM_JOB_ID="$job_id" \
      SLURM_JOB_NAME="$job_name" \
      BIOCODER_GPU_ALLOCATION_MODE=manual_physical_index_guard_no_slurm_gres \
      BIOCODER_TRAIN_INPUT_SCOPE=staged_train_artifacts_only \
      BIOCODER_VERIFIED_CODE_REVISION="$actual_revision" \
      "$python_bin" -m multimodal_science.qwen3vl.train_fusion_cli \
        --fusion-bundle-root "$fusion_root" \
        --fusion-bundle-report-sha256 "$fusion_report_sha256" \
        --lora-bundle-root "$lora_root" \
        --lora-bundle-report-sha256 "$lora_report_sha256" \
        --dataset-root "$dataset_root" \
        --dataset-report-sha256 "$dataset_report_sha256" \
        --assets-root "$assets_root" \
        --initial-adapter-root "$adapter_root" \
        --initial-adapter-report-sha256 "$adapter_report_sha256" \
        --initial-adapter-manifest-sha256 "$adapter_manifest_sha256" \
        "${pretrained_projector_arguments[@]}" \
        --output-dir "$work_output" \
        --model-name-or-path "$local_model" \
        --model-revision "$model_revision" \
        --model-artifact-sha256 "$model_artifact_sha256" \
        --model-manifest-path "$model_manifest" \
        --model-manifest-sha256 "$model_manifest_sha256" \
        --code-revision "$code_revision" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --gradient-accumulation-steps "$gradient_accumulation_steps" \
        --lora-learning-rate "$lora_learning_rate" \
        --projector-learning-rate "$projector_learning_rate" \
        --save-steps "$save_steps" \
        --log-steps "$log_steps" \
        --seed "$seed" \
        "${limit_arguments[@]}" \
        "${resume_arguments[@]}"
  )
  verify_training_output "$work_output"
fi

test ! -e "$final_output"
mv "$work_output" "$final_output"
report_sha256="$(sha256sum "$final_output/fusion_training_report.json" | awk '{print $1}')"
manifest_sha256="$(sha256sum "$final_output/artifact_manifest.sha256" | awk '{print $1}')"
"$python_bin" - "$final_output/fusion_training_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("WALL_TIME_SECONDS=", round(report["wall_time_seconds"], 3))
print("PEAK_CUDA_GIB=", round(report["runtime"]["peak_cuda_memory_bytes"] / 2**30, 3))
print("TRAINING_RECORDS=", report["training"]["training_records"])
print("OPTIMIZER_UPDATES=", report["training"]["optimizer_updates"])
print("DEVELOPMENT_TRAINING_COMPLETE=", report["development_training_complete"])
PY
echo "FUSION_TRAINING_REPORT_SHA256=$report_sha256"
echo "ARTIFACT_MANIFEST_SHA256=$manifest_sha256"
echo "PERSISTENT_OUTPUT=$final_output"
echo "QWEN3VL_XIC_FUSION_TRAINING=OK"
