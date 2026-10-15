#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

set -euo pipefail
umask 077

require_variable() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_RL_POOL_ROOT BIOCODER_RL_POOL_REPORT_SHA256 \
  BIOCODER_RL_DIFFICULTY_ROOT BIOCODER_RL_DIFFICULTY_REPORT_SHA256 \
  BIOCODER_DATASET_ROOT BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_ROOT BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256 BIOCODER_MODEL_ROOT \
  BIOCODER_MODEL_REVISION BIOCODER_MODEL_ARTIFACT_SHA256 \
  BIOCODER_MODEL_MANIFEST_PATH BIOCODER_MODEL_MANIFEST_SHA256 \
  BIOCODER_RL_TRAINER BIOCODER_RL_RUN_MODE
do require_variable "$name"; done

repo_root="$BIOCODER_REPO_ROOT"; run_root="$BIOCODER_RUN_ROOT"; python_bin="$BIOCODER_PYTHON"
revision="$BIOCODER_CODE_REVISION"; trainer="$BIOCODER_RL_TRAINER"; run_mode="$BIOCODER_RL_RUN_MODE"
pool_source="$BIOCODER_RL_POOL_ROOT"; pool_sha="$BIOCODER_RL_POOL_REPORT_SHA256"
difficulty_source="$BIOCODER_RL_DIFFICULTY_ROOT"; difficulty_sha="$BIOCODER_RL_DIFFICULTY_REPORT_SHA256"
dataset_source="$BIOCODER_DATASET_ROOT"; dataset_sha="$BIOCODER_DATASET_REPORT_SHA256"
adapter_source="$BIOCODER_INITIAL_ADAPTER_ROOT"; adapter_report_sha="$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256"
adapter_manifest_sha="$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"; model_source="$BIOCODER_MODEL_ROOT"
model_revision="$BIOCODER_MODEL_REVISION"; model_artifact_sha="$BIOCODER_MODEL_ARTIFACT_SHA256"
model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"; model_manifest_sha="$BIOCODER_MODEL_MANIFEST_SHA256"
experiment="${BIOCODER_RL_EXPERIMENT:-R1}"; learning_rate_choice="${BIOCODER_C1_LEARNING_RATE_CHOICE:-original}"
seed="${BIOCODER_SEED:-17}"; gpu_index="${BIOCODER_GPU_INDEX:-auto}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"; max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
job_id="${SLURM_JOB_ID:-manual}"; short="${revision:0:7}"; model_short="${model_artifact_sha:0:16}"

[[ "$trainer" == grpo || "$trainer" == c1 ]]
[[ "$run_mode" == smoke || "$run_mode" == calibration || "$run_mode" == formal || "$run_mode" == reduced-prompts || "$run_mode" == reduced-steps ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ && "$seed" =~ ^[0-9]+$ ]]
for digest in "$pool_sha" "$difficulty_sha" "$dataset_sha" "$adapter_report_sha" "$adapter_manifest_sha" "$model_artifact_sha" "$model_manifest_sha"; do [[ "$digest" =~ ^[0-9a-f]{64}$ ]]; done
test -x "$python_bin"; test -d "$repo_root/.git"; test -d "$pool_source"; test -d "$difficulty_source"
test -d "$dataset_source"; test -d "$adapter_source"; test -d "$model_source"; test -f "$model_manifest"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$revision" ]]

if [[ "$trainer" == grpo ]]; then
  run_name="v2-grpo-${experiment}-${run_mode}-seed${seed}-${short}"
else
  run_name="v2-c1-${learning_rate_choice}-${run_mode}-seed${seed}-${short}"
fi
output_root="${BIOCODER_RL_TRAIN_OUTPUT_ROOT:-$run_root/qwen3vl/rl/runs}"
output="$output_root/$run_name"; mkdir -p "$output_root"
exec 7>"$output_root/.${run_name}.lock"; flock -n 7 || { echo "Another job owns $run_name" >&2; exit 7; }

echo "=== V2 RL TRAINING CONTRACT ==="
echo "JOB_ID=$job_id"; echo "NODE=$(hostname -s)"; echo "CODE_REVISION=$revision"
echo "TRAINER=$trainer"; echo "RUN_MODE=$run_mode"; echo "TRAIN_INPUT_SCOPE=staged_train_artifacts_only"
echo "SELECTION_SAMPLES_STAGED=false"; echo "VALIDATION_ANSWERS_OPENED=false"; echo "INTERNAL_TEST_ACCESSED=false"

if [[ -f "$output/fusion_training_report.json" || -f "$output/sft_continue_training_report.json" ]]; then
  (cd "$output" && sha256sum -c artifact_manifest.sha256)
  echo "PERSISTENT_OUTPUT=$output"; echo "QWEN3VL_RL_TRAINING=ALREADY_COMPLETE"; exit 0
fi

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-train-${job_id}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-train-* ]] && rm -rf -- "$scratch"; }
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/inputs" "$scratch/tmp" "$scratch/cache"
git -C "$repo_root" archive "$revision" | tar -xf - -C "$scratch/code"
actual_revision="$(git -C "$repo_root" rev-parse "$revision^{commit}")"
[[ "$actual_revision" == "$revision" ]]

echo "=== STAGE HASH-BOUND TRAIN-ONLY INPUT VIEW ==="
PYTHONPATH="$scratch/code/backend" "$python_bin" - \
  "$pool_source" "$pool_sha" "$scratch/inputs/pool" \
  "$difficulty_source" "$difficulty_sha" "$scratch/inputs/difficulty" \
  "$dataset_source" "$dataset_sha" "$scratch/inputs/dataset" \
  "$adapter_source" "$adapter_report_sha" "$adapter_manifest_sha" "$scratch/inputs/adapter" <<'PY'
import hashlib, json, pathlib, shutil, sys
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def copy_bound(source_root, target_root, relative, expected):
    source_root, target_root = source_root.resolve(), target_root.resolve()
    source, target = (source_root / relative).resolve(), (target_root / relative).resolve()
    require(source.is_relative_to(source_root) and target.is_relative_to(target_root), "Unsafe artifact path")
    require(source.is_file() and digest(source) == expected, f"Artifact drift: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
pool, pool_sha, pool_out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
difficulty, difficulty_sha, difficulty_out = pathlib.Path(sys.argv[4]), sys.argv[5], pathlib.Path(sys.argv[6])
dataset, dataset_sha, dataset_out = pathlib.Path(sys.argv[7]), sys.argv[8], pathlib.Path(sys.argv[9])
adapter, adapter_report_sha, adapter_manifest_sha, adapter_out = pathlib.Path(sys.argv[10]), sys.argv[11], sys.argv[12], pathlib.Path(sys.argv[13])
for target in (pool_out, difficulty_out, dataset_out, adapter_out): target.mkdir(parents=True, exist_ok=False)
pool_report = pool / "rl_pool_report.json"; require(digest(pool_report) == pool_sha, "RL pool report drift")
pool_body = json.loads(pool_report.read_text(encoding="utf-8")); shutil.copy2(pool_report, pool_out / pool_report.name)
item = pool_body["artifacts"]["training_pool"]; copy_bound(pool, pool_out, item["path"], item["sha256"])
difficulty_report = difficulty / "rl_difficulty_report.json"; require(digest(difficulty_report) == difficulty_sha, "Difficulty report drift")
difficulty_body = json.loads(difficulty_report.read_text(encoding="utf-8")); shutil.copy2(difficulty_report, difficulty_out / difficulty_report.name)
item = difficulty_body["artifacts"]["difficulty_training_pool"]; copy_bound(difficulty, difficulty_out, item["path"], item["sha256"])
dataset_report = dataset / "dataset_report.json"; require(digest(dataset_report) == dataset_sha, "Dataset report drift")
dataset_body = json.loads(dataset_report.read_text(encoding="utf-8")); shutil.copy2(dataset_report, dataset_out / dataset_report.name)
item = dataset_body["artifacts"]["train_signals"]; copy_bound(dataset, dataset_out, item["path"], item["sha256"])
adapter_report = adapter / "fusion_training_report.json"; require(digest(adapter_report) == adapter_report_sha, "Adapter report drift")
manifest = adapter / "artifact_manifest.sha256"; require(digest(manifest) == adapter_manifest_sha, "Adapter manifest drift")
shutil.copy2(manifest, adapter_out / manifest.name)
for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    parts = line.split("  ", 1); require(len(parts) == 2, f"Malformed adapter manifest line {number}")
    copy_bound(adapter, adapter_out, parts[1], parts[0])
for target in (pool_out, difficulty_out, dataset_out, adapter_out):
    names = {part.lower().replace("-", "_") for path in target.rglob("*") for part in path.relative_to(target).parts}
    require("validation" not in names and "internal_test" not in names and "rl_selection_set.jsonl" not in names, "Protected input staged")
print("STAGED_TRAIN_ONLY_INPUTS=OK")
PY

echo "=== STAGE HASH-VERIFIED MODEL CACHE ==="
model_cache="/tmp/biocoder-${USER}-models/qwen3vl-${model_short}"; mkdir -p "$(dirname "$model_cache")"
exec 8>"${model_cache}.lock"; flock 8
if [[ ! -d "$model_cache" ]]; then mkdir -p "$scratch/model"; cp -a "$model_source/." "$scratch/model/"; mv "$scratch/model" "$model_cache"; fi
(cd "$scratch/code/backend" && PYTHONPATH="$scratch/code/backend" "$python_bin" - "$model_cache" "$model_manifest" "$model_manifest_sha" <<'PY'
import sys
from pathlib import Path
from multimodal_science.qwen3vl.fusion_smoke import _verify_model_manifest
inventory = _verify_model_manifest(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
print(f"LOCAL_MODEL_FILES={inventory['files']}"); print(f"LOCAL_MODEL_BYTES={inventory['bytes']}")
PY
)
flock -u 8

gpu_passes_guard() { local c="$1" s line used rest util; for s in 1 2 3; do line="$(nvidia-smi -i "$c" --query-gpu=memory.used,memory.free,utilization.gpu,name --format=csv,noheader,nounits)" || return 1; echo "candidate=$c sample=$s $line" >&2; used="${line%%,*}"; rest="${line#*,}"; rest="${rest#*,}"; util="${rest%%,*}"; used="${used// /}"; util="${util// /}"; (( used <= max_gpu_used_mib && util <= max_gpu_utilization )) || return 1; sleep 2; done; }
if [[ "$gpu_index" == auto ]]; then gpu_index=""; count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"; for ((c=0;c<count;c++)); do exec 9>"/tmp/biocoder-${USER}-gpu-${c}.lock"; if flock -n 9 && gpu_passes_guard "$c"; then gpu_index="$c"; break; fi; done; [[ -n "$gpu_index" ]]; else exec 9>"/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"; flock -n 9; gpu_passes_guard "$gpu_index"; fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

resume=(); [[ -d "$output" ]] && resume=(--resume)
common=(--code-revision "$revision" --rl-pool-root "$scratch/inputs/pool" --rl-pool-report-sha256 "$pool_sha" --difficulty-root "$scratch/inputs/difficulty" --difficulty-report-sha256 "$difficulty_sha" --dataset-root "$scratch/inputs/dataset" --dataset-report-sha256 "$dataset_sha" --starting-adapter-root "$scratch/inputs/adapter" --starting-adapter-report-sha256 "$adapter_report_sha" --starting-adapter-manifest-sha256 "$adapter_manifest_sha" --model-name-or-path "$model_cache" --model-revision "$model_revision" --model-artifact-sha256 "$model_artifact_sha" --model-manifest-path "$model_manifest" --model-manifest-sha256 "$model_manifest_sha" --output-dir "$output" --run-mode "$run_mode" --seed "$seed")
environment=(HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" LANG="${LANG:-C.UTF-8}" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 BIOCODER_VERIFIED_CODE_REVISION="$actual_revision" BIOCODER_TRAIN_INPUT_SCOPE=staged_train_artifacts_only BIOCODER_GPU_ALLOCATION_MODE=manual_physical_index_guard_no_slurm_gres CUBLAS_WORKSPACE_CONFIG=:4096:8)
if [[ "$trainer" == grpo ]]; then
  env -i "${environment[@]}" "$python_bin" -m multimodal_science.qwen3vl.train_rl_cli "${common[@]}" --experiment "$experiment" "${resume[@]}"
else
  env -i "${environment[@]}" "$python_bin" -m multimodal_science.qwen3vl.train_sft_continue_cli "${common[@]}" --learning-rate-choice "$learning_rate_choice" "${resume[@]}"
fi
(cd "$output" && sha256sum -c artifact_manifest.sha256)
echo "PERSISTENT_OUTPUT=$output"; echo "QWEN3VL_RL_TRAINING=OK"
