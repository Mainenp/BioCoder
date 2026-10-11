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

require_variable() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_RL_POOL_ROOT BIOCODER_RL_POOL_REPORT_SHA256 \
  BIOCODER_DATASET_ROOT BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_ROOT BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256 BIOCODER_MODEL_ROOT \
  BIOCODER_MODEL_REVISION BIOCODER_MODEL_ARTIFACT_SHA256 \
  BIOCODER_MODEL_MANIFEST_PATH BIOCODER_MODEL_MANIFEST_SHA256
do require_variable "$name"; done

repo_root="$BIOCODER_REPO_ROOT"; run_root="$BIOCODER_RUN_ROOT"; python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"; pool_source="$BIOCODER_RL_POOL_ROOT"
pool_sha="$BIOCODER_RL_POOL_REPORT_SHA256"; dataset_source="$BIOCODER_DATASET_ROOT"
dataset_sha="$BIOCODER_DATASET_REPORT_SHA256"; adapter_source="$BIOCODER_INITIAL_ADAPTER_ROOT"
adapter_report_sha="$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256"
adapter_manifest_sha="$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"
model_source="$BIOCODER_MODEL_ROOT"; model_revision="$BIOCODER_MODEL_REVISION"
model_artifact_sha="$BIOCODER_MODEL_ARTIFACT_SHA256"; model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"
model_manifest_sha="$BIOCODER_MODEL_MANIFEST_SHA256"; gpu_index="${BIOCODER_GPU_INDEX:-auto}"
max_records="${BIOCODER_MAX_RECORDS:-}"; max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"; job_id="${SLURM_JOB_ID:-manual}"
code_short="${code_revision:0:7}"; model_short="${model_artifact_sha:0:16}"

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
for digest in "$pool_sha" "$dataset_sha" "$adapter_report_sha" "$adapter_manifest_sha" "$model_artifact_sha" "$model_manifest_sha"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid SHA-256: $digest" >&2; exit 2; }
done
[[ -z "$max_records" || "$max_records" =~ ^[1-9][0-9]*$ ]]
[[ "$gpu_index" == auto || "$gpu_index" =~ ^[0-9]+$ ]]
test -x "$python_bin"; test -d "$repo_root/.git"; test -d "$pool_source"
test -d "$dataset_source"; test -d "$adapter_source"; test -d "$model_source"; test -f "$model_manifest"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$code_revision" ]]

mode="formal"; [[ -n "$max_records" ]] && mode="calibration-${max_records}"
run_name="v2-difficulty-${mode}-${code_short}"
output_root="${BIOCODER_RL_DIFFICULTY_OUTPUT_ROOT:-$run_root/qwen3vl/rl/difficulty}"
work_output="$output_root/.${run_name}.incomplete"; final_output="$output_root/$run_name"
mkdir -p "$output_root"; exec 7>"$output_root/.${run_name}.lock"
flock -n 7 || { echo "Another job owns this difficulty run" >&2; exit 7; }

verify_output() {
  local root="$1"
  (cd "$root" && sha256sum -c artifact_manifest.sha256)
  "$python_bin" - "$root" "$code_revision" "$mode" <<'PY'
import json, pathlib, sys
report = json.loads((pathlib.Path(sys.argv[1]) / "rl_difficulty_report.json").read_text(encoding="utf-8"))
assert report["code_revision"] == sys.argv[2]
contracts = report["contracts"]
assert contracts["train_pool_only"] is True
assert contracts["selection_groups_trained"] is False
assert contracts["validation_answers_opened"] is False
assert contracts["internal_test_accessed"] is False
assert contracts["visual_tower_unused"] is True
assert report["development_screening_complete"] is (sys.argv[3] == "formal")
print("RL_DIFFICULTY_REPORT_CONTRACT=OK")
PY
}

echo "=== V2 RL DIFFICULTY CONTRACT ==="
echo "JOB_ID=$job_id"; echo "NODE=$(hostname -s)"; echo "CODE_REVISION=$code_revision"
echo "RUN_MODE=$mode"; echo "TRAIN_INPUT_SCOPE=staged_train_artifacts_only"
echo "VALIDATION_ANSWERS_OPENED=false"; echo "INTERNAL_TEST_ACCESSED=false"
if [[ -d "$final_output" ]]; then
  verify_output "$final_output"; echo "RL_DIFFICULTY_OUTPUT=$final_output"; echo "QWEN3VL_RL_DIFFICULTY=ALREADY_COMPLETE"; exit 0
fi
test ! -e "$work_output"

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-difficulty-${job_id}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-difficulty-* ]] && rm -rf -- "$scratch"; }
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/inputs" "$scratch/tmp" "$scratch/cache"
git -C "$repo_root" archive "$code_revision" | tar -xf - -C "$scratch/code"

echo "=== STAGE HASH-BOUND TRAIN-ONLY INPUT VIEW ==="
PYTHONPATH="$scratch/code/backend" "$python_bin" - \
  "$pool_source" "$pool_sha" "$scratch/inputs/pool" \
  "$dataset_source" "$dataset_sha" "$scratch/inputs/dataset" \
  "$adapter_source" "$adapter_report_sha" "$adapter_manifest_sha" "$scratch/inputs/adapter" <<'PY'
import hashlib, json, pathlib, shutil, sys
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def safe_copy(source_root, target_root, relative, expected):
    source_root = source_root.resolve(); target_root = target_root.resolve()
    source = (source_root / relative).resolve(); target = (target_root / relative).resolve()
    require(source.is_relative_to(source_root) and target.is_relative_to(target_root), "Unsafe artifact path")
    require(source.is_file() and digest(source) == expected, f"Artifact drift: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
pool_src, pool_sha, pool_dst = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
data_src, data_sha, data_dst = pathlib.Path(sys.argv[4]), sys.argv[5], pathlib.Path(sys.argv[6])
adapter_src, adapter_report_sha, adapter_manifest_sha, adapter_dst = pathlib.Path(sys.argv[7]), sys.argv[8], sys.argv[9], pathlib.Path(sys.argv[10])
for root in (pool_dst, data_dst, adapter_dst): root.mkdir(parents=True, exist_ok=False)
pool_report_path = pool_src / "rl_pool_report.json"; require(digest(pool_report_path) == pool_sha, "RL pool report drift")
pool_report = json.loads(pool_report_path.read_text(encoding="utf-8")); shutil.copy2(pool_report_path, pool_dst / pool_report_path.name)
artifact = pool_report["artifacts"]["training_pool"]; safe_copy(pool_src, pool_dst, artifact["path"], artifact["sha256"])
data_report_path = data_src / "dataset_report.json"; require(digest(data_report_path) == data_sha, "Dataset report drift")
data_report = json.loads(data_report_path.read_text(encoding="utf-8")); shutil.copy2(data_report_path, data_dst / data_report_path.name)
artifact = data_report["artifacts"]["train_signals"]; safe_copy(data_src, data_dst, artifact["path"], artifact["sha256"])
adapter_report = adapter_src / "fusion_training_report.json"; require(digest(adapter_report) == adapter_report_sha, "Adapter report drift")
manifest = adapter_src / "artifact_manifest.sha256"; require(digest(manifest) == adapter_manifest_sha, "Adapter manifest drift")
shutil.copy2(manifest, adapter_dst / manifest.name)
for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    parts = line.split("  ", 1); require(len(parts) == 2, f"Malformed adapter manifest line {number}")
    safe_copy(adapter_src, adapter_dst, parts[1], parts[0])
for root in (pool_dst, data_dst, adapter_dst):
    lowered = {part.lower().replace("-", "_") for path in root.rglob("*") for part in path.relative_to(root).parts}
    require("validation" not in lowered and "internal_test" not in lowered and "rl_selection_set.jsonl" not in lowered, "Protected split entered train view")
print("STAGED_TRAIN_ONLY_INPUTS=OK")
PY

echo "=== STAGE HASH-VERIFIED MODEL CACHE ==="
model_cache_root="/tmp/biocoder-${USER}-models"; local_model="$model_cache_root/qwen3vl-${model_short}"
mkdir -p "$model_cache_root"; exec 8>"$model_cache_root/qwen3vl-${model_short}.lock"; flock 8
verify_model() { (cd "$scratch/code/backend" && PYTHONPATH="$scratch/code/backend" "$python_bin" - "$local_model" "$model_manifest" "$model_manifest_sha" <<'PY'
import sys
from pathlib import Path
from multimodal_science.qwen3vl.fusion_smoke import _verify_model_manifest
inventory = _verify_model_manifest(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
print(f"LOCAL_MODEL_FILES={inventory['files']}"); print(f"LOCAL_MODEL_BYTES={inventory['bytes']}")
PY
); }
if [[ -d "$local_model" ]] && ! verify_model; then rm -rf -- "$local_model"; fi
if [[ ! -d "$local_model" ]]; then mkdir -p "$scratch/model"; cp -a "$model_source/." "$scratch/model/"; mv "$scratch/model" "$local_model"; fi
verify_model; flock -u 8

gpu_passes_guard() { local c="$1" s line used rest util; for s in 1 2 3; do line="$(nvidia-smi -i "$c" --query-gpu=memory.used,memory.free,utilization.gpu,name --format=csv,noheader,nounits)" || return 1; echo "candidate=$c sample=$s $line" >&2; used="${line%%,*}"; rest="${line#*,}"; rest="${rest#*,}"; util="${rest%%,*}"; used="${used// /}"; util="${util// /}"; (( used <= max_gpu_used_mib && util <= max_gpu_utilization )) || return 1; sleep 2; done; }
echo "=== GPU STARTUP GUARD ==="
if [[ "$gpu_index" == auto ]]; then
  gpu_index=""; gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  for ((c=0; c<gpu_count; c++)); do exec 9>"/tmp/biocoder-${USER}-gpu-${c}.lock"; if flock -n 9 && gpu_passes_guard "$c"; then gpu_index="$c"; break; fi; done
  [[ -n "$gpu_index" ]] || { echo "No unlocked GPU passed guard" >&2; exit 5; }
else exec 9>"/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"; flock -n 9; gpu_passes_guard "$gpu_index"; fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

limit=(); [[ -n "$max_records" ]] && limit=(--max-records "$max_records")
env -i HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" LANG="${LANG:-C.UTF-8}" \
  TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" CUDA_VISIBLE_DEVICES="$gpu_index" \
  PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "$python_bin" -m multimodal_science.qwen3vl.rl_difficulty_cli \
    --code-revision "$code_revision" --rl-pool-root "$scratch/inputs/pool" --rl-pool-report-sha256 "$pool_sha" \
    --dataset-root "$scratch/inputs/dataset" --dataset-report-sha256 "$dataset_sha" \
    --starting-adapter-root "$scratch/inputs/adapter" --starting-adapter-report-sha256 "$adapter_report_sha" \
    --starting-adapter-manifest-sha256 "$adapter_manifest_sha" --model-name-or-path "$local_model" \
    --model-revision "$model_revision" --model-artifact-sha256 "$model_artifact_sha" \
    --model-manifest-path "$model_manifest" --model-manifest-sha256 "$model_manifest_sha" \
    --output-dir "$work_output" "${limit[@]}"

verify_output "$work_output"; test ! -e "$final_output"; mv "$work_output" "$final_output"
rm -rf -- "$output_root/.${run_name}.incomplete.work"
echo "RL_DIFFICULTY_REPORT_SHA256=$(sha256sum "$final_output/rl_difficulty_report.json" | awk '{print $1}')"
echo "RL_DIFFICULTY_OUTPUT=$final_output"; echo "QWEN3VL_RL_DIFFICULTY=OK"
