#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail
umask 077

require_variable() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_RL_SELECTION_BUNDLE_ROOT BIOCODER_RL_SELECTION_BUNDLE_SHA256 \
  BIOCODER_DATASET_ROOT BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_ROOT BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 \
  BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256 BIOCODER_MODEL_ROOT BIOCODER_MODEL_REVISION \
  BIOCODER_MODEL_ARTIFACT_SHA256 BIOCODER_MODEL_MANIFEST_PATH \
  BIOCODER_MODEL_MANIFEST_SHA256
do require_variable "$name"; done

repo_root="$BIOCODER_REPO_ROOT"; run_root="$BIOCODER_RUN_ROOT"; python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"; bundle_source="$BIOCODER_RL_SELECTION_BUNDLE_ROOT"
bundle_sha="$BIOCODER_RL_SELECTION_BUNDLE_SHA256"; dataset_source="$BIOCODER_DATASET_ROOT"
dataset_sha="$BIOCODER_DATASET_REPORT_SHA256"; adapter_source="$BIOCODER_INITIAL_ADAPTER_ROOT"
adapter_report_sha="$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256"
adapter_manifest_sha="$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"
model_source="$BIOCODER_MODEL_ROOT"; model_revision="$BIOCODER_MODEL_REVISION"
model_artifact_sha="$BIOCODER_MODEL_ARTIFACT_SHA256"; model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"
model_manifest_sha="$BIOCODER_MODEL_MANIFEST_SHA256"
gpu_index="${BIOCODER_GPU_INDEX:-auto}"; max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"; job_id="${SLURM_JOB_ID:-manual}"
model_short="${model_artifact_sha:0:16}"

candidate_json="${BIOCODER_RL_CANDIDATE_JSON:-}"
candidate_sha="${BIOCODER_RL_CANDIDATE_JSON_SHA256:-}"
output="${BIOCODER_RL_SELECTION_OUTPUT:-}"
if [[ -z "$candidate_json" || -z "$candidate_sha" ]]; then
  require_variable BIOCODER_RL_CANDIDATE_MATRIX_ROOT
  require_variable BIOCODER_RL_CANDIDATE_MATRIX_REPORT_SHA256
  require_variable BIOCODER_RL_SELECTION_OUTPUT_ROOT
  require_variable SLURM_ARRAY_TASK_ID
  matrix_root="$BIOCODER_RL_CANDIDATE_MATRIX_ROOT"
  matrix_sha="$BIOCODER_RL_CANDIDATE_MATRIX_REPORT_SHA256"
  array_task_id="$SLURM_ARRAY_TASK_ID"
  [[ "$array_task_id" =~ ^([0-9]|1[0-5])$ ]] || { echo "Candidate array task must be in [0, 15]" >&2; exit 2; }
  [[ "$matrix_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid candidate matrix SHA-256" >&2; exit 2; }
  test -d "$matrix_root"
  mapfile -t candidate_info < <(
    "$python_bin" - "$matrix_root" "$matrix_sha" "$array_task_id" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected_report_sha = sys.argv[2]
index = int(sys.argv[3])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report_path = root / "candidate_matrix_report.json"
if digest(report_path) != expected_report_sha:
    raise ValueError("Candidate matrix report drift")
report = json.loads(report_path.read_text(encoding="utf-8"))
if report["schema_version"] != "chrompeak-qwen3vl-rl-selection-candidate-matrix-v1":
    raise ValueError("Unexpected candidate matrix schema")
if report["counts"]["candidates"] != 16:
    raise ValueError("Candidate matrix must contain exactly 16 candidates")
artifact = report["artifacts"]["candidate_list"]
candidate_list = (root / artifact["path"]).resolve()
if not candidate_list.is_relative_to(root) or digest(candidate_list) != artifact["sha256"]:
    raise ValueError("Candidate list drift")
rows = [json.loads(line) for line in candidate_list.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 16 or artifact["records"] != 16:
    raise ValueError("Candidate list must contain exactly 16 rows")
row = rows[index]
candidate = (root / row["path"]).resolve()
if not candidate.is_relative_to(root) or digest(candidate) != row["sha256"]:
    raise ValueError("Candidate descriptor drift")
candidate_id = row.get("candidate_id")
if not isinstance(candidate_id, str) or not candidate_id or "/" in candidate_id or "\\" in candidate_id or candidate_id in {".", ".."}:
    raise ValueError("Unsafe candidate ID")
print(candidate)
print(row["sha256"])
print(candidate_id)
PY
  )
  [[ "${#candidate_info[@]}" -eq 3 ]] || { echo "Candidate matrix resolution failed" >&2; exit 2; }
  candidate_json="${candidate_info[0]}"
  candidate_sha="${candidate_info[1]}"
  candidate_id="${candidate_info[2]}"
  output="${BIOCODER_RL_SELECTION_OUTPUT_ROOT%/}/$candidate_id"
  echo "ARRAY_TASK_ID=$array_task_id"
  echo "CANDIDATE_ID=$candidate_id"
else
  [[ -n "$candidate_json" && -n "$candidate_sha" ]] || { echo "Candidate JSON and SHA-256 must be provided together" >&2; exit 2; }
  [[ -n "$output" ]] || { echo "Missing required environment variable: BIOCODER_RL_SELECTION_OUTPUT" >&2; exit 2; }
fi

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
for digest in "$bundle_sha" "$dataset_sha" "$adapter_report_sha" "$adapter_manifest_sha" "$candidate_sha" "$model_artifact_sha" "$model_manifest_sha"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid SHA-256: $digest" >&2; exit 2; }
done
[[ "$gpu_index" == auto || "$gpu_index" =~ ^[0-9]+$ ]]
test -x "$python_bin"; test -d "$repo_root/.git"; test -d "$bundle_source"
test -d "$dataset_source"; test -d "$adapter_source"; test -f "$candidate_json"
test -d "$model_source"; test -f "$model_manifest"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$code_revision" ]]
mkdir -p "$(dirname "$output")"
exec 7>"$(dirname "$output")/.$(basename "$output").lock"
flock -n 7 || { echo "Another job owns this selection candidate" >&2; exit 7; }

verify_output() {
  local root="$1"
  (cd "$root" && sha256sum -c artifact_manifest.sha256)
  "$python_bin" - "$root" "$code_revision" "$candidate_sha" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
report = json.loads((root / "selection_inference_report.json").read_text(encoding="utf-8"))
assert report["code_revision"] == sys.argv[2]
assert report["sources"]["candidate_descriptor_sha256"] == sys.argv[3]
assert report["decode"] == {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64}
contracts = report["contracts"]
assert contracts == {
    "selection_answers_opened": False,
    "validation_prompts_opened": False,
    "validation_answers_opened": False,
    "internal_test_accessed": False,
    "visual_tower_unused": True,
}
assert report["runtime"]["visual_forward_calls"] == 0
print("RL_SELECTION_INFERENCE_REPORT_CONTRACT=OK")
PY
}

echo "=== V2 RL ANSWER-FREE SELECTION INFERENCE CONTRACT ==="
echo "JOB_ID=$job_id"; echo "NODE=$(hostname -s)"; echo "CODE_REVISION=$code_revision"
echo "SELECTION_INPUT_SCOPE=staged_selection_prompts_and_train_signals_only"
echo "SELECTION_ANSWERS_OPENED=false"; echo "VALIDATION_ANSWERS_OPENED=false"; echo "INTERNAL_TEST_ACCESSED=false"
if [[ -d "$output" ]]; then
  verify_output "$output"; echo "RL_SELECTION_INFERENCE_OUTPUT=$output"; echo "QWEN3VL_RL_SELECTION_INFERENCE=ALREADY_COMPLETE"; exit 0
fi

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-selection-inference-${job_id}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-selection-inference-* ]] && rm -rf -- "$scratch"; }
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/inputs" "$scratch/tmp" "$scratch/cache"
git -C "$repo_root" archive "$code_revision" | tar -xf - -C "$scratch/code"

echo "=== STAGE ANSWER-FREE SELECTION VIEW ==="
PYTHONPATH="$scratch/code/backend" "$python_bin" - \
  "$bundle_source" "$bundle_sha" "$scratch/inputs/selection" \
  "$dataset_source" "$dataset_sha" "$scratch/inputs/dataset" \
  "$adapter_source" "$adapter_report_sha" "$adapter_manifest_sha" "$scratch/inputs/adapter" \
  "$candidate_json" "$candidate_sha" "$scratch/inputs/candidate.json" <<'PY'
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
bundle_src, bundle_sha, bundle_dst = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
data_src, data_sha, data_dst = pathlib.Path(sys.argv[4]), sys.argv[5], pathlib.Path(sys.argv[6])
adapter_src, adapter_report_sha, adapter_manifest_sha, adapter_dst = pathlib.Path(sys.argv[7]), sys.argv[8], sys.argv[9], pathlib.Path(sys.argv[10])
candidate_src, candidate_sha, candidate_dst = pathlib.Path(sys.argv[11]), sys.argv[12], pathlib.Path(sys.argv[13])
for root in (bundle_dst, data_dst, adapter_dst): root.mkdir(parents=True, exist_ok=False)
bundle_report_path = bundle_src / "selection_bundle_report.json"
require(digest(bundle_report_path) == bundle_sha, "Selection bundle report drift")
bundle_report = json.loads(bundle_report_path.read_text(encoding="utf-8"))
require(bundle_report["contracts"]["answer_free_generation_view_available"] is True, "No answer-free selection contract")
shutil.copy2(bundle_report_path, bundle_dst / bundle_report_path.name)
for name in ("selection_prompts", "selection_manifest", "selection_prompt_manifest"):
    artifact = bundle_report["artifacts"][name]
    safe_copy(bundle_src, bundle_dst, artifact["path"], artifact["sha256"])
require(not (bundle_dst / "selection_answers.jsonl").exists(), "Selection answers were staged")
require(not (bundle_dst / "artifact_manifest.sha256").exists(), "Full selection manifest was staged")
data_report_path = data_src / "dataset_report.json"
require(digest(data_report_path) == data_sha, "Dataset report drift")
data_report = json.loads(data_report_path.read_text(encoding="utf-8")); shutil.copy2(data_report_path, data_dst / data_report_path.name)
artifact = data_report["artifacts"]["train_signals"]; safe_copy(data_src, data_dst, artifact["path"], artifact["sha256"])
adapter_report = adapter_src / "fusion_training_report.json"; require(digest(adapter_report) == adapter_report_sha, "Adapter report drift")
manifest = adapter_src / "artifact_manifest.sha256"; require(digest(manifest) == adapter_manifest_sha, "Adapter manifest drift")
shutil.copy2(manifest, adapter_dst / manifest.name)
for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    parts = line.split("  ", 1); require(len(parts) == 2, f"Malformed adapter manifest line {number}")
    safe_copy(adapter_src, adapter_dst, parts[1], parts[0])
require(digest(candidate_src) == candidate_sha, "Candidate descriptor drift")
shutil.copy2(candidate_src, candidate_dst)
for root in (bundle_dst, data_dst, adapter_dst):
    lowered = {part.lower().replace("-", "_") for path in root.rglob("*") for part in path.relative_to(root).parts}
    require("validation" not in lowered and "internal_test" not in lowered, "Protected split entered selection inference")
print("STAGED_ANSWER_FREE_SELECTION_INPUTS=OK")
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

resume=(); [[ -d "$(dirname "$output")/.$(basename "$output").work" ]] && resume=(--resume)
env -i HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" LANG="${LANG:-C.UTF-8}" \
  TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" CUDA_VISIBLE_DEVICES="$gpu_index" \
  PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  BIOCODER_VERIFIED_CODE_REVISION="$code_revision" \
  BIOCODER_SELECTION_INPUT_SCOPE="staged_selection_prompts_and_train_signals_only" \
  BIOCODER_GPU_ALLOCATION_MODE="manual_physical_index_guard_no_slurm_gres" \
  "$python_bin" -m multimodal_science.qwen3vl.run_rl_selection_inference_cli \
    --code-revision "$code_revision" --bundle-root "$scratch/inputs/selection" \
    --bundle-report-sha256 "$bundle_sha" --dataset-root "$scratch/inputs/dataset" \
    --dataset-report-sha256 "$dataset_sha" --starting-adapter-root "$scratch/inputs/adapter" \
    --starting-adapter-report-sha256 "$adapter_report_sha" \
    --starting-adapter-manifest-sha256 "$adapter_manifest_sha" \
    --candidate-json "$scratch/inputs/candidate.json" --candidate-json-sha256 "$candidate_sha" \
    --model-name-or-path "$local_model" --model-revision "$model_revision" \
    --model-artifact-sha256 "$model_artifact_sha" --model-manifest-path "$model_manifest" \
    --model-manifest-sha256 "$model_manifest_sha" --output-dir "$output" "${resume[@]}"

verify_output "$output"
echo "RL_SELECTION_INFERENCE_REPORT_SHA256=$(sha256sum "$output/selection_inference_report.json" | awk '{print $1}')"
echo "RL_SELECTION_INFERENCE_OUTPUT=$output"
echo "QWEN3VL_RL_SELECTION_INFERENCE=OK"
