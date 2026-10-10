#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

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
  BIOCODER_INFERENCE_BUNDLE_ROOT \
  BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256 \
  BIOCODER_FUSION_BUNDLE_ROOT \
  BIOCODER_FUSION_BUNDLE_REPORT_SHA256 \
  BIOCODER_DATASET_ROOT \
  BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_ASSETS_ROOT \
  BIOCODER_FUSION_ADAPTER_ROOT \
  BIOCODER_FUSION_TRAINING_REPORT_SHA256 \
  BIOCODER_FUSION_ADAPTER_MANIFEST_SHA256 \
  BIOCODER_INSTRUCTION_ROOT \
  BIOCODER_INSTRUCTION_REPORT_SHA256
do
  require_variable "$required_name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"
model_source="$BIOCODER_MODEL_ROOT"
model_revision="$BIOCODER_MODEL_REVISION"
model_artifact_sha256="$BIOCODER_MODEL_ARTIFACT_SHA256"
model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"
model_manifest_sha256="$BIOCODER_MODEL_MANIFEST_SHA256"
inference_source="$BIOCODER_INFERENCE_BUNDLE_ROOT"
inference_report_sha256="$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256"
fusion_source="$BIOCODER_FUSION_BUNDLE_ROOT"
fusion_report_sha256="$BIOCODER_FUSION_BUNDLE_REPORT_SHA256"
dataset_source="$BIOCODER_DATASET_ROOT"
dataset_report_sha256="$BIOCODER_DATASET_REPORT_SHA256"
assets_source="$BIOCODER_ASSETS_ROOT"
adapter_source="$BIOCODER_FUSION_ADAPTER_ROOT"
adapter_report_sha256="$BIOCODER_FUSION_TRAINING_REPORT_SHA256"
adapter_manifest_sha256="$BIOCODER_FUSION_ADAPTER_MANIFEST_SHA256"
instruction_root="$BIOCODER_INSTRUCTION_ROOT"
instruction_report_sha256="$BIOCODER_INSTRUCTION_REPORT_SHA256"
gpu_index="${BIOCODER_GPU_INDEX:-auto}"
seed="${BIOCODER_SEED:-17}"
max_new_tokens="${BIOCODER_MAX_NEW_TOKENS:-64}"
bootstrap_iterations="${BIOCODER_BOOTSTRAP_ITERATIONS:-1000}"
xic_intervention="${BIOCODER_XIC_INTERVENTION:-aligned}"
xic_intervention_seed="${BIOCODER_XIC_INTERVENTION_SEED:-17}"
input_modality="${BIOCODER_INPUT_MODALITY:-image_xic}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
min_local_free_gib="${BIOCODER_MIN_LOCAL_FREE_GIB:-32}"
job_id="${SLURM_JOB_ID:-manual}"
job_name="${SLURM_JOB_NAME:-coder}"
code_short="${code_revision:0:7}"
adapter_short="${adapter_report_sha256:0:12}"
model_short="${model_artifact_sha256:0:16}"

for value in \
  "$seed" "$max_new_tokens" "$bootstrap_iterations" "$xic_intervention_seed" "$max_gpu_used_mib" \
  "$max_gpu_utilization" "$min_local_free_gib"
do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "Expected a nonnegative integer, got: $value" >&2
    exit 2
  }
done
(( max_new_tokens >= 1 && bootstrap_iterations >= 2 ))
case "$xic_intervention" in
  aligned|shuffled|zero|availability-off) ;;
  *) echo "Unsupported BIOCODER_XIC_INTERVENTION: $xic_intervention" >&2; exit 2 ;;
esac
[[ "$input_modality" == "image_xic" || "$input_modality" == "xic_only" ]] || {
  echo "Unsupported BIOCODER_INPUT_MODALITY: $input_modality" >&2
  exit 2
}
[[ "$seed" == "17" && "$max_new_tokens" == "64" && "$bootstrap_iterations" == "1000" ]] || {
  echo "Formal fusion evaluation is frozen at seed=17, max_new_tokens=64, bootstrap_iterations=1000" >&2
  exit 2
}
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]] || {
  echo "BIOCODER_GPU_INDEX must be a physical index or auto" >&2
  exit 2
}
for digest in \
  "$model_artifact_sha256" "$model_manifest_sha256" \
  "$inference_report_sha256" "$fusion_report_sha256" \
  "$dataset_report_sha256" "$adapter_report_sha256" \
  "$adapter_manifest_sha256" "$instruction_report_sha256"
do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Expected a lowercase SHA-256 digest: $digest" >&2
    exit 2
  }
done
[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$model_source"
test -f "$model_manifest"
test -d "$inference_source"
test -d "$fusion_source"
test -d "$dataset_source"
test -d "$assets_source"
test -d "$adapter_source"
test -d "$instruction_root"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 3
}

modality_slug="${input_modality//_/-}"
run_name="${modality_slug}-${xic_intervention}-iseed${xic_intervention_seed}-validation-${adapter_short}-${code_short}"
generation_root="$run_root/qwen3vl/fusion/generations"
evaluation_root="$run_root/qwen3vl/fusion/evaluations"
generation_final="$generation_root/$run_name"
evaluation_final="$evaluation_root/${run_name}-evaluation"
mkdir -p "$generation_root" "$evaluation_root"

run_lock="$evaluation_root/.${run_name}.lock"
exec 7>"$run_lock"
flock -n 7 || {
  echo "Another job already owns this exact fusion evaluation: $run_name" >&2
  exit 7
}

echo "=== FORMAL FUSION EVALUATION CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=$job_name"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "FUSION_TRAINING_REPORT_SHA256=$adapter_report_sha256"
echo "FUSION_ADAPTER_MANIFEST_SHA256=$adapter_manifest_sha256"
echo "GENERATION_RUN_NAME=$run_name"
echo "GENERATION_BATCH_SIZE=1"
echo "MAX_NEW_TOKENS=$max_new_tokens"
echo "BOOTSTRAP_ITERATIONS=$bootstrap_iterations"
echo "XIC_INTERVENTION=$xic_intervention"
echo "XIC_INTERVENTION_SEED=$xic_intervention_seed"
echo "INPUT_MODALITY=$input_modality"
echo "GENERATION_ANSWER_KEY_AVAILABLE=false"
echo "INTERNAL_TEST_ACCESSED=false"

available_kib="$(df -Pk /tmp | awk 'NR==2 {print $4}')"
required_kib="$((min_local_free_gib * 1024 * 1024))"
(( available_kib >= required_kib )) || {
  echo "Insufficient /tmp capacity: need ${min_local_free_gib} GiB free" >&2
  exit 6
}
scratch="$(mktemp -d "/tmp/biocoder-${USER}-qwen3vl-fusion-eval-${job_id}.XXXXXX")"
cleanup() {
  case "${scratch:-}" in
    "/tmp/biocoder-${USER}-qwen3vl-fusion-eval-${job_id}."*) rm -rf -- "$scratch" ;;
  esac
}
trap cleanup EXIT

write_exact_manifest() {
  local root="$1"
  shift
  "$python_bin" - "$root" "$@" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = tuple(sys.argv[2:])
for relative in required:
    if not (root / relative).is_file():
        raise FileNotFoundError(root / relative)

manifest = root / "artifact_manifest.sha256"
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path != manifest and not path.name.startswith(".artifact_manifest.")
)

def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

content = "".join(
    f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in files
)
if manifest.exists():
    if manifest.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Existing artifact manifest does not exactly bind {root}")
else:
    temporary = root / f".artifact_manifest.{os.getpid()}.tmp"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, manifest)
print(f"EXACT_ARTIFACTS={len(files)}")
PY
  (
    cd "$root"
    sha256sum -c artifact_manifest.sha256
  )
}

echo "=== STAGE IMMUTABLE CODE ==="
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$scratch/code"
export PYTHONPATH="$scratch/code/backend"

echo "=== STAGE PROMPT-ONLY VALIDATION INPUTS ==="
inference_root="$scratch/inference"
fusion_root="$scratch/fusion"
dataset_root="$scratch/dataset"
assets_root="$scratch/assets"
"$python_bin" - \
  "$inference_source" "$inference_root" \
  "$fusion_source" "$fusion_root" \
  "$dataset_source" "$dataset_root" \
  "$assets_source" "$assets_root" \
  "$inference_report_sha256" "$fusion_report_sha256" "$dataset_report_sha256" <<'PY'
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

def safe(root: Path, relative: object) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative, "Bad path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and all(p not in {"", ".", ".."} for p in pure.parts), "Unsafe path")
    path = (root / Path(*pure.parts)).resolve()
    path.relative_to(root)
    return path

def copy_artifact(source: Path, target: Path, record: dict) -> Path:
    src = safe(source, record["path"])
    dst = safe(target, record["path"])
    require(src.is_file(), f"Missing artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    require(digest(dst) == record["sha256"], f"Artifact hash mismatch: {src}")
    return dst

inference_source, inference_target = map(Path, sys.argv[1:3])
fusion_source, fusion_target = map(Path, sys.argv[3:5])
dataset_source, dataset_target = map(Path, sys.argv[5:7])
assets_source, assets_target = map(Path, sys.argv[7:9])
expected_inference, expected_fusion, expected_dataset = sys.argv[9:12]
for source in (inference_source, fusion_source, dataset_source, assets_source):
    require(source.resolve().is_dir(), f"Missing input root: {source}")
inference_source = inference_source.resolve()
fusion_source = fusion_source.resolve()
dataset_source = dataset_source.resolve()
assets_source = assets_source.resolve()

reports = []
for source, target, name, expected in (
    (inference_source, inference_target, "inference_bundle_report.json", expected_inference),
    (fusion_source, fusion_target, "fusion_bundle_report.json", expected_fusion),
    (dataset_source, dataset_target, "dataset_report.json", expected_dataset),
):
    path = source / name
    require(digest(path) == expected, f"Report hash mismatch: {name}")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, target / name)
    reports.append(json.loads(path.read_text(encoding="utf-8")))

inference, fusion, dataset = reports
prompts = copy_artifact(
    inference_source, inference_target, inference["artifacts"]["inference_prompts"]
)
validation_links = copy_artifact(
    fusion_source, fusion_target, fusion["artifacts"]["validation_xic_links"]
)
for key in ("validation_signals", "validation_examples"):
    copy_artifact(dataset_source, dataset_target, dataset["artifacts"][key])

assets_target.mkdir(parents=True, exist_ok=False)
seen = {}
prompt_rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines()]
link_rows = [json.loads(line) for line in validation_links.read_text(encoding="utf-8").splitlines()]
require(len(prompt_rows) == len(link_rows), "Prompt/link count mismatch")
for prompt, row in zip(prompt_rows, link_rows):
    require(prompt["instruction_id"] == row["instruction_id"], "Prompt/link order mismatch")
    require(prompt["image"] == row["image"], "Prompt/link image mismatch")
for line_number, row in enumerate(link_rows, 1):
    relative = row["image"]
    expected = row["image_sha256"]
    if relative in seen:
        require(seen[relative] == expected, f"Conflicting image hash: {relative}")
        continue
    src = safe(assets_source, relative)
    dst = safe(assets_target, relative)
    require(src.is_file(), f"Missing validation image at row {line_number}: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    require(digest(dst) == expected, f"Validation image hash mismatch: {relative}")
    seen[relative] = expected

require(not (inference_target / "validation_answers.jsonl").exists(), "Answer key staged")
require(not (fusion_target / "train_xic_links.jsonl").exists(), "Train fusion rows staged")
require(not (dataset_target / "train").exists(), "Train Dataset staged")
require(not (assets_target / "jobs" / "train").exists(), "Train images staged")
print(f"LOCAL_VALIDATION_IMAGES={len(seen)}")
print("STAGED_PROMPT_ONLY_VALIDATION_INPUTS=OK")
PY

echo "=== STAGE HASH-VERIFIED FUSION ADAPTER CACHE ==="
adapter_cache_root="/tmp/biocoder-${USER}-fusion-adapters"
adapter_root="$adapter_cache_root/$adapter_short"
mkdir -p "$adapter_cache_root"
exec 6>"$adapter_cache_root/${adapter_short}.lock"
flock 6
verify_adapter_cache() {
  [[ "$(sha256sum "$adapter_root/fusion_training_report.json" | awk '{print $1}')" == "$adapter_report_sha256" ]]
  [[ "$(sha256sum "$adapter_root/artifact_manifest.sha256" | awk '{print $1}')" == "$adapter_manifest_sha256" ]]
  (
    cd "$adapter_root"
    sha256sum -c artifact_manifest.sha256
  )
}
if [[ -d "$adapter_root" ]] && ! verify_adapter_cache; then
  [[ "$adapter_root" == "$adapter_cache_root"/* ]] || exit 6
  rm -rf -- "$adapter_root"
fi
if [[ ! -d "$adapter_root" ]]; then
  adapter_staging="$scratch/adapter-staging"
  mkdir -p "$adapter_staging"
  cp -a "$adapter_source/." "$adapter_staging/"
  mv "$adapter_staging" "$adapter_root"
fi
verify_adapter_cache
flock -u 6

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
    line="$(nvidia-smi -i "$candidate" --query-gpu=memory.used,memory.free,utilization.gpu,name --format=csv,noheader,nounits)" || return 1
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
  for (( candidate=0; candidate<gpu_count; candidate++ )); do
    lock_path="/tmp/biocoder-${USER}-gpu-${candidate}.lock"
    exec 9>"$lock_path"
    if flock -n 9 && gpu_passes_guard "$candidate"; then
      gpu_index="$candidate"
      break
    fi
  done
  [[ -n "$gpu_index" ]] || { echo "No unlocked GPU passed the startup guard" >&2; exit 5; }
else
  lock_path="/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
  exec 9>"$lock_path"
  flock -n 9 || { echo "GPU lock already held: $lock_path" >&2; exit 4; }
  gpu_passes_guard "$gpu_index" || { echo "GPU $gpu_index failed the startup guard" >&2; exit 5; }
fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

if [[ ! -d "$generation_final" ]]; then
  resume_arguments=()
  [[ -d "$generation_root/.${run_name}.work" ]] && resume_arguments+=(--resume)
  echo "=== RUN ANSWER-ISOLATED FUSED GENERATION ==="
  (
    cd "$scratch/code/backend"
    env -i \
      HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG="${LANG:-C.UTF-8}" \
      TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache" \
      CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH="$scratch/code/backend" \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 \
      HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      "$python_bin" -m multimodal_science.qwen3vl.run_fusion_inference_cli \
        --inference-bundle-root "$inference_root" \
        --inference-bundle-report-sha256 "$inference_report_sha256" \
        --fusion-bundle-root "$fusion_root" \
        --fusion-bundle-report-sha256 "$fusion_report_sha256" \
        --dataset-root "$dataset_root" \
        --dataset-report-sha256 "$dataset_report_sha256" \
        --assets-root "$assets_root" \
        --fusion-adapter-root "$adapter_root" \
        --fusion-training-report-sha256 "$adapter_report_sha256" \
        --fusion-adapter-manifest-sha256 "$adapter_manifest_sha256" \
        --output-dir "$generation_final" \
        --model-name-or-path "$local_model" \
        --model-revision "$model_revision" \
        --model-artifact-sha256 "$model_artifact_sha256" \
        --max-new-tokens "$max_new_tokens" \
        --seed "$seed" \
        --dtype bfloat16 \
        --device-map single_cuda \
        --attention-implementation sdpa \
        --xic-intervention "$xic_intervention" \
        --xic-intervention-seed "$xic_intervention_seed" \
        --input-modality "$input_modality" \
        "${resume_arguments[@]}"
  )
fi
write_exact_manifest "$generation_final" \
  generation_report.json predictions.jsonl generation_records.jsonl \
  run_config.json runtime_metadata.json
generation_report_sha256="$(sha256sum "$generation_final/generation_report.json" | awk '{print $1}')"
echo "GENERATION_REPORT_SHA256=$generation_report_sha256"

echo "=== OPEN ANSWERS ONLY AFTER GENERATION IS IMMUTABLE ==="
if [[ ! -d "$evaluation_final" ]]; then
  evaluation_staging="$scratch/evaluation"
  nice -n 10 ionice -c 3 "$python_bin" \
    -m multimodal_science.qwen3vl.evaluate_predictions_cli \
    --instruction-root "$instruction_root" \
    --instruction-report-sha256 "$instruction_report_sha256" \
    --predictions "$generation_final/predictions.jsonl" \
    --generation-report "$generation_final/generation_report.json" \
    --generation-report-sha256 "$generation_report_sha256" \
    --output-dir "$evaluation_staging" \
    --bootstrap-iterations "$bootstrap_iterations" \
    --seed "$seed"
  write_exact_manifest "$evaluation_staging" qwen_evaluation_report.json evaluation_records.jsonl
  mv "$evaluation_staging" "$evaluation_final"
fi
write_exact_manifest "$evaluation_final" qwen_evaluation_report.json evaluation_records.jsonl
evaluation_report_sha256="$(sha256sum "$evaluation_final/qwen_evaluation_report.json" | awk '{print $1}')"

"$python_bin" - \
  "$generation_final/generation_report.json" \
  "$evaluation_final/qwen_evaluation_report.json" \
  "$adapter_report_sha256" "$adapter_manifest_sha256" \
  "$model_artifact_sha256" "$inference_report_sha256" \
  "$instruction_report_sha256" "$fusion_report_sha256" \
  "$dataset_report_sha256" "$seed" "$max_new_tokens" "$bootstrap_iterations" \
  "$xic_intervention" "$xic_intervention_seed" "$input_modality" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

generation_path = Path(sys.argv[1])
evaluation_path = Path(sys.argv[2])
generation = json.loads(generation_path.read_text(encoding="utf-8"))
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
(
    adapter_report, adapter_manifest, model_artifact, inference_report,
    instruction_report, fusion_report, dataset_report,
) = sys.argv[3:10]
seed, max_tokens, bootstrap = map(int, sys.argv[10:13])
xic_intervention = sys.argv[13]
xic_intervention_seed = int(sys.argv[14])
input_modality = sys.argv[15]

assert generation["counts"]["predictions"] == 13708
assert generation["scope"]["complete_prompt_coverage"] is True
assert generation["development_comparison_candidate"] is True
assert generation["runtime"]["backend"] == "transformers"
expected_backend = (
    "transformers-qwen3vl-image-xic"
    if input_modality == "image_xic"
    else "transformers-qwen3vl-xic-only"
)
assert generation["runtime"]["fusion_execution_backend"] == expected_backend
assert generation["runtime"]["input_modality"] == input_modality
assert generation["runtime"]["image_pixels_forwarded"] is (input_modality == "image_xic")
assert generation["runtime"]["adapter_loaded"] is True
assert generation["runtime"]["manual_cached_greedy_decode"] is True
assert generation["runtime"]["explicit_fused_mrope_positions"] is True
assert generation["runtime"]["fusion_bundle_report_sha256"] == fusion_report
assert generation["runtime"]["dataset_report_sha256"] == dataset_report
intervention = generation["runtime"]["xic_intervention"]
assert intervention["mode"] == xic_intervention
assert intervention["seed"] == xic_intervention_seed
assert intervention["answer_key_used"] is False
assert intervention["language_variants_share_one_asset_intervention"] is True
adapter = generation["model"]["adapter"]
expected_kind = "image_xic_fusion" if input_modality == "image_xic" else "xic_only_qwen"
assert adapter["kind"] == expected_kind
assert adapter["input_modality"] == input_modality
assert adapter["training_report_sha256"] == adapter_report
assert adapter["manifest_sha256"] == adapter_manifest
assert adapter["development_training_complete"] is True
assert adapter["xic_intervention"] == intervention
assert generation["model"]["artifact_sha256"] == model_artifact
assert generation["source"]["bundle_report_sha256"] == inference_report
assert generation["source"]["instruction_report_sha256"] == instruction_report
assert generation["generation"]["batch_size"] == 1
assert generation["generation"]["do_sample"] is False
assert generation["generation"]["seed"] == seed
assert generation["generation"]["max_new_tokens"] == max_tokens
assert generation["contracts"]["answer_key_opened"] is False
assert generation["internal_test_accessed"] is False

assert evaluation["counts"]["predictions"] == 13708
assert evaluation["inputs"]["instruction_report_sha256"] == instruction_report
assert evaluation["inputs"]["generation_report_sha256"] == hashlib.sha256(generation_path.read_bytes()).hexdigest()
assert evaluation["evaluation"]["seed"] == seed
assert evaluation["evaluation"]["bootstrap_iterations"] == bootstrap
assert evaluation["prediction_generation_provenance_verified"] is True
assert evaluation["development_comparison_eligible"] is True
assert evaluation["final_benchmark_eligible"] is False
assert evaluation["internal_test_accessed"] is False
print("FUSION_FULL_GENERATION_CONTRACT=OK")
print("FUSION_FULL_EVALUATION_CONTRACT=OK")
print("SCHEMA_VALID_RECORDS=", evaluation["counts"]["schema_valid"])
PY

echo "EVALUATION_REPORT_SHA256=$evaluation_report_sha256"
echo "PERSISTENT_GENERATION=$generation_final"
echo "PERSISTENT_EVALUATION=$evaluation_final"
if [[ "$input_modality" == "xic_only" ]]; then
  echo "QWEN3VL_XIC_ONLY_EVALUATION=OK"
else
  echo "QWEN3VL_XIC_FUSION_EVALUATION=OK"
fi
