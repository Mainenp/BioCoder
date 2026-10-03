#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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
  BIOCODER_FINAL_PROTOCOL_ROOT \
  BIOCODER_FINAL_PROTOCOL_SHA256 \
  BIOCODER_ASSET_INDEX_PATH \
  BIOCODER_READINESS_REPORT_PATH \
  BIOCODER_SEQUENCE_PREFLIGHT_PATH \
  BIOCODER_ASSETS_ROOT \
  BIOCODER_PRIVATE_DETECTOR_SOURCE
do
  require_variable "$required_name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"
protocol_root="$BIOCODER_FINAL_PROTOCOL_ROOT"
protocol_sha256="$BIOCODER_FINAL_PROTOCOL_SHA256"
asset_index="$BIOCODER_ASSET_INDEX_PATH"
readiness_report="$BIOCODER_READINESS_REPORT_PATH"
sequence_preflight="$BIOCODER_SEQUENCE_PREFLIGHT_PATH"
assets_root="$BIOCODER_ASSETS_ROOT"
private_detector_source="$BIOCODER_PRIVATE_DETECTOR_SOURCE"
gpu_index="${BIOCODER_GPU_INDEX:-auto}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${code_revision:0:7}"
protocol_short="${protocol_sha256:0:12}"
final_root="$run_root/final-benchmark/runs/final-$protocol_short"
ledger_dir="$run_root/final-benchmark/access/access-$protocol_short"

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$protocol_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]]
[[ "$max_gpu_used_mib" =~ ^[0-9]+$ ]]
[[ "$max_gpu_utilization" =~ ^[0-9]+$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$protocol_root"
test -d "$assets_root"
test -d "$private_detector_source"
for path in "$asset_index" "$readiness_report" "$sequence_preflight"; do
  test -f "$path"
done

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 3
}

mkdir -p "$run_root/final-benchmark/runs" "$run_root/final-benchmark/access"
exec 7>"$run_root/final-benchmark/.run-$protocol_short.lock"
flock -n 7 || {
  echo "Another job owns this exact final benchmark protocol" >&2
  exit 7
}

scratch="$(mktemp -d "/tmp/biocoder-${USER}-final-benchmark-${job_id}.XXXXXX")"
cleanup() {
  case "${scratch:-}" in
    "/tmp/biocoder-${USER}-final-benchmark-${job_id}."*) rm -rf -- "$scratch" ;;
  esac
}
trap cleanup EXIT
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$scratch/code"
code_root="$scratch/code"
export PYTHONPATH="$code_root/backend"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$scratch"
export XDG_CACHE_HOME="$scratch/cache"

echo "=== ONE-TIME FINAL BENCHMARK PREFLIGHT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "PROTOCOL_SHA256=$protocol_sha256"
echo "FINAL_ROOT=$final_root"
echo "LEDGER_DIR=$ledger_dir"

mapfile -t locked < <(
  "$python_bin" - \
    "$protocol_root" "$protocol_sha256" \
    "$asset_index" "$readiness_report" "$sequence_preflight" \
    "$private_detector_source" <<'PY'
import json
import sys
from pathlib import Path

from multimodal_science.chrompeakformer.final_detector_evaluation import (
    _verify_candidate,
)
from multimodal_science.data.manifest import sha256_file
from multimodal_science.qwen3vl.final_benchmark_protocol import _verify_protocol
from multimodal_science.qwen3vl.fusion_smoke import _verify_model_manifest

protocol_root = Path(sys.argv[1]).resolve()
protocol_sha = sys.argv[2]
asset_index = Path(sys.argv[3]).resolve()
readiness = Path(sys.argv[4]).resolve()
preflight = Path(sys.argv[5]).resolve()
private_source = Path(sys.argv[6]).resolve()

protocol = _verify_protocol(protocol_root, protocol_sha)
descriptor = protocol["locks"]["candidate"]
candidate_path = protocol_root / descriptor["path"]
if sha256_file(candidate_path) != descriptor["sha256"]:
    raise RuntimeError("Candidate lock drift")
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
shared = candidate["shared"]

dataset_root = Path(shared["dataset_root"]).resolve()
dataset_report_path = dataset_root / "dataset_report.json"
if sha256_file(dataset_report_path) != shared["dataset_report_sha256"]:
    raise RuntimeError("Frozen development Dataset drift")
dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
for path, field, label in (
    (asset_index, "asset_index_sha256", "asset index"),
    (readiness, "readiness_report_sha256", "readiness report"),
    (preflight, "sequence_preflight_sha256", "sequence preflight"),
):
    if sha256_file(path) != dataset_report[field]:
        raise RuntimeError(f"Frozen {label} drift")

normalization = shared["scalar_normalization"]
normalization_path = Path(normalization["path"]).resolve()
if sha256_file(normalization_path) != normalization["sha256"]:
    raise RuntimeError("Frozen scalar normalization drift")

base_model = shared["base_model"]
model_root = Path(base_model["name_or_path"]).resolve()
manifest_path = Path(base_model["verification_manifest_path"]).resolve()
_verify_model_manifest(
    model_root,
    manifest_path,
    base_model["verification_manifest_sha256"],
)

models = candidate["models"]
_verify_candidate(models["chrompeakformer"], private_source)
policy = shared["inference_policy"]

values = (
    str(model_root),
    base_model["revision"],
    base_model["artifact_sha256"],
    str(manifest_path),
    base_model["verification_manifest_sha256"],
    str(normalization_path),
    normalization["sha256"],
    models["qwen3vl_image_lora"]["root"],
    models["qwen3vl_image_lora"]["training_report_sha256"],
    models["qwen3vl_image_lora"]["manifest_sha256"],
    models["qwen3vl_image_xic_fusion"]["root"],
    models["qwen3vl_image_xic_fusion"]["training_report_sha256"],
    models["qwen3vl_image_xic_fusion"]["manifest_sha256"],
    policy["fusion_attention_implementation"],
    str(dataset_root),
    shared["dataset_report_sha256"],
)
for value in values:
    if "\n" in str(value):
        raise RuntimeError("Locked path contains a newline")
    print(value)
PY
)
[[ "${#locked[@]}" -eq 16 ]] || {
  echo "Unexpected frozen-runtime field count: ${#locked[@]}" >&2
  exit 3
}
model_root="${locked[0]}"
model_revision="${locked[1]}"
model_artifact_sha256="${locked[2]}"
model_manifest="${locked[3]}"
model_manifest_sha256="${locked[4]}"
normalization_path="${locked[5]}"
normalization_sha256="${locked[6]}"
lora_root="${locked[7]}"
lora_report_sha256="${locked[8]}"
lora_manifest_sha256="${locked[9]}"
fusion_adapter_root="${locked[10]}"
fusion_report_sha256="${locked[11]}"
fusion_manifest_sha256="${locked[12]}"
fusion_attention="${locked[13]}"
development_dataset_root="${locked[14]}"
development_dataset_sha256="${locked[15]}"

echo "MODEL_ARTIFACT_SHA256=$model_artifact_sha256"
echo "MODEL_MANIFEST_SHA256=$model_manifest_sha256"
echo "DEVELOPMENT_DATASET_SHA256=$development_dataset_sha256"
echo "PRIVATE_DETECTOR_SOURCE_CONTRACT=OK"
echo "FROZEN_RUNTIME_INPUTS=OK"
echo "PRE_ACCESS_TEST_ROWS_PARSED=false"

verify_manifest() {
  local root="$1"
  test -f "$root/artifact_manifest.sha256"
  (
    cd "$root"
    sha256sum -c artifact_manifest.sha256
  )
}

verified_report_sha() {
  local root="$1"
  local report="$2"
  verify_manifest "$root" >&2
  test -f "$root/$report"
  sha256sum "$root/$report" | awk '{print $1}'
}

if [[ -f "$ledger_dir/access_completed.json" ]]; then
  "$python_bin" -m multimodal_science.qwen3vl.final_benchmark_access_cli complete \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --final-evidence-manifest "$final_root/report/artifact_manifest.sha256"
  verify_manifest "$final_root/report"
  verify_manifest "$ledger_dir"
  echo "FINAL_BENCHMARK_ALREADY_COMPLETED=OK"
  exit 0
fi

gpu_passes_guard() {
  local candidate="$1"
  local sample line used remainder utilization
  for sample in 1 2 3; do
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
  for ((candidate=0; candidate<gpu_count; candidate++)); do
    exec 9>"/tmp/biocoder-${USER}-gpu-${candidate}.lock"
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
  exec 9>"/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
  flock -n 9 || { echo "GPU lock is already held" >&2; exit 4; }
  gpu_passes_guard "$gpu_index" || {
    echo "GPU $gpu_index failed the startup guard" >&2
    exit 5
  }
fi
export CUDA_VISIBLE_DEVICES="$gpu_index"
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

echo "=== OPEN OR RESUME SOLE ACCESS EVENT ==="
if [[ -d "$ledger_dir" ]]; then
  "$python_bin" -m multimodal_science.qwen3vl.final_benchmark_access_cli verify \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir"
  echo "ACCESS_MODE=exact_protocol_crash_resume"
else
  "$python_bin" -m multimodal_science.qwen3vl.final_benchmark_access_cli open \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir"
  echo "ACCESS_MODE=first_and_only_open"
fi

mkdir -p "$final_root"
internal_dataset="$final_root/internal-test-dataset"
final_data="$final_root/separated-data"

echo "=== MATERIALIZE INTERNAL TEST WITH FROZEN TRAIN NORMALIZATION ==="
(
  cd "$code_root/backend"
  "$python_bin" -m multimodal_science.chrompeakformer.materialize_cli \
    --index "$asset_index" \
    --readiness-report "$readiness_report" \
    --sequence-preflight "$sequence_preflight" \
    --assets-root "$assets_root" \
    --output-dir "$internal_dataset" \
    --split internal_test \
    --target-points 160 \
    --scalar-normalization "$normalization_path" \
    --scalar-normalization-sha256 "$normalization_sha256"
)
internal_dataset_sha256="$(sha256sum "$internal_dataset/dataset_report.json" | awk '{print $1}')"
echo "INTERNAL_TEST_DATASET_SHA256=$internal_dataset_sha256"

echo "=== BUILD DISJOINT PROMPT AND ANSWER ROOTS ==="
(
  cd "$code_root/backend"
  "$python_bin" -m multimodal_science.qwen3vl.build_final_benchmark_data_cli \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --dataset-root "$internal_dataset" \
    --dataset-report-sha256 "$internal_dataset_sha256" \
    --assets-root "$assets_root" \
    --output-dir "$final_data"
)
verify_manifest "$final_data"
verify_manifest "$final_data/inference"
verify_manifest "$final_data/answers"
verify_manifest "$final_data/fusion"
verify_manifest "$final_data/detector"
inference_sha256="$(
  sha256sum "$final_data/inference/inference_bundle_report.json" | awk '{print $1}'
)"
answer_sha256="$(
  sha256sum "$final_data/answers/answer_key_report.json" | awk '{print $1}'
)"
fusion_bundle_sha256="$(
  sha256sum "$final_data/fusion/fusion_bundle_report.json" | awk '{print $1}'
)"
detector_data_sha256="$(
  sha256sum "$final_data/detector/detector_dataset_report.json" | awk '{print $1}'
)"

generation_root="$final_root/generations"
evaluation_root="$final_root/evaluations"
mkdir -p "$generation_root" "$evaluation_root"

run_image_candidate() {
  local candidate="$1"
  local output="$generation_root/$candidate"
  shift
  if [[ ! -d "$output" ]]; then
    "$python_bin" -m multimodal_science.qwen3vl.run_final_inference_cli \
      --candidate "$candidate" \
      --protocol-root "$protocol_root" \
      --protocol-sha256 "$protocol_sha256" \
      --ledger-dir "$ledger_dir" \
      --bundle-root "$final_data/inference" \
      --bundle-report-sha256 "$inference_sha256" \
      --assets-root "$assets_root" \
      --output-dir "$output" \
      --model-name-or-path "$model_root" \
      --model-revision "$model_revision" \
      --model-artifact-sha256 "$model_artifact_sha256" \
      --batch-size 1 \
      --max-new-tokens 64 \
      --seed 17 \
      --dtype bfloat16 \
      --device-map auto \
      --resume \
      "$@" >&2
  fi
  verified_report_sha "$output" generation_report.json
}

evaluate_qwen_candidate() {
  local candidate="$1"
  local generation="$generation_root/$candidate"
  local output="$evaluation_root/$candidate"
  local generation_sha256
  generation_sha256="$(verified_report_sha "$generation" generation_report.json)"
  if [[ ! -d "$output" ]]; then
    "$python_bin" -m multimodal_science.qwen3vl.evaluate_final_predictions_cli \
      --candidate "$candidate" \
      --protocol-root "$protocol_root" \
      --protocol-sha256 "$protocol_sha256" \
      --ledger-dir "$ledger_dir" \
      --inference-root "$final_data/inference" \
      --inference-report-sha256 "$inference_sha256" \
      --answer-root "$final_data/answers" \
      --answer-report-sha256 "$answer_sha256" \
      --predictions "$generation/predictions.jsonl" \
      --generation-report "$generation/generation_report.json" \
      --generation-report-sha256 "$generation_sha256" \
      --output-dir "$output" >&2
  fi
  verified_report_sha "$output" qwen_evaluation_report.json
}

echo "=== RUN FROZEN ZERO-SHOT CANDIDATE ==="
zero_generation_sha256="$(run_image_candidate qwen3vl_zero_shot)"
echo "ZERO_SHOT_GENERATION_SHA256=$zero_generation_sha256"
zero_evaluation_sha256="$(evaluate_qwen_candidate qwen3vl_zero_shot)"
echo "ZERO_SHOT_EVALUATION_SHA256=$zero_evaluation_sha256"

echo "=== RUN FROZEN IMAGE-LORA CANDIDATE ==="
lora_generation_sha256="$(
  run_image_candidate qwen3vl_image_lora \
    --adapter-root "$lora_root" \
    --adapter-training-report-sha256 "$lora_report_sha256" \
    --adapter-manifest-sha256 "$lora_manifest_sha256"
)"
echo "IMAGE_LORA_GENERATION_SHA256=$lora_generation_sha256"
lora_evaluation_sha256="$(evaluate_qwen_candidate qwen3vl_image_lora)"
echo "IMAGE_LORA_EVALUATION_SHA256=$lora_evaluation_sha256"

echo "=== RUN FROZEN ALIGNED IMAGE-XIC CANDIDATE ==="
fusion_generation="$generation_root/qwen3vl_image_xic_fusion"
if [[ ! -d "$fusion_generation" ]]; then
  "$python_bin" -m multimodal_science.qwen3vl.run_final_fusion_inference_cli \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --inference-bundle-root "$final_data/inference" \
    --inference-bundle-report-sha256 "$inference_sha256" \
    --fusion-bundle-root "$final_data/fusion" \
    --fusion-bundle-report-sha256 "$fusion_bundle_sha256" \
    --dataset-root "$internal_dataset" \
    --dataset-report-sha256 "$internal_dataset_sha256" \
    --assets-root "$assets_root" \
    --fusion-adapter-root "$fusion_adapter_root" \
    --fusion-training-report-sha256 "$fusion_report_sha256" \
    --fusion-adapter-manifest-sha256 "$fusion_manifest_sha256" \
    --output-dir "$fusion_generation" \
    --model-name-or-path "$model_root" \
    --model-revision "$model_revision" \
    --model-artifact-sha256 "$model_artifact_sha256" \
    --max-new-tokens 64 \
    --seed 17 \
    --attention-implementation "$fusion_attention" \
    --resume
fi
fusion_generation_sha256="$(verified_report_sha "$fusion_generation" generation_report.json)"
echo "FUSION_GENERATION_SHA256=$fusion_generation_sha256"
fusion_evaluation_sha256="$(evaluate_qwen_candidate qwen3vl_image_xic_fusion)"
echo "FUSION_EVALUATION_SHA256=$fusion_evaluation_sha256"

echo "=== RUN FROZEN SEQUENCEPEAKNET CANDIDATE ==="
sequence_evaluation="$evaluation_root/sequence_peak_net"
if [[ ! -d "$sequence_evaluation" ]]; then
  "$python_bin" -m multimodal_science.baselines.evaluate_final_sequence_cli \
    --candidate sequence_peak_net \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --dataset-root "$internal_dataset" \
    --dataset-report-sha256 "$internal_dataset_sha256" \
    --output-dir "$sequence_evaluation" \
    --device cuda \
    --batch-size 256
fi
sequence_evaluation_sha256="$(
  verified_report_sha "$sequence_evaluation" sequence_evaluation_report.json
)"
echo "SEQUENCE_EVALUATION_SHA256=$sequence_evaluation_sha256"

echo "=== RUN FROZEN CHROMPEAKFORMER CANDIDATE ==="
detector_evaluation="$evaluation_root/chrompeakformer"
if [[ ! -d "$detector_evaluation" ]]; then
  "$python_bin" -m multimodal_science.chrompeakformer.evaluate_final_detector_cli \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --source-root "$private_detector_source" \
    --detector-data-root "$final_data/detector" \
    --detector-report-sha256 "$detector_data_sha256" \
    --output-dir "$detector_evaluation" \
    --device cuda \
    --batch-size 16 \
    --num-workers 2
fi
detector_evaluation_sha256="$(
  verified_report_sha "$detector_evaluation" detector_evaluation_report.json
)"
echo "DETECTOR_EVALUATION_SHA256=$detector_evaluation_sha256"

echo "=== BUILD EXACT FIVE-CANDIDATE FINAL REPORT ==="
report_root="$final_root/report"
if [[ ! -d "$report_root" ]]; then
  "$python_bin" -m multimodal_science.qwen3vl.build_final_benchmark_report_cli \
    --protocol-root "$protocol_root" \
    --protocol-sha256 "$protocol_sha256" \
    --ledger-dir "$ledger_dir" \
    --zero-shot-root "$evaluation_root/qwen3vl_zero_shot" \
    --zero-shot-report-sha256 "$zero_evaluation_sha256" \
    --image-lora-root "$evaluation_root/qwen3vl_image_lora" \
    --image-lora-report-sha256 "$lora_evaluation_sha256" \
    --fusion-root "$evaluation_root/qwen3vl_image_xic_fusion" \
    --fusion-report-sha256 "$fusion_evaluation_sha256" \
    --sequence-root "$sequence_evaluation" \
    --sequence-report-sha256 "$sequence_evaluation_sha256" \
    --detector-root "$detector_evaluation" \
    --detector-report-sha256 "$detector_evaluation_sha256" \
    --output-dir "$report_root"
fi
final_report_sha256="$(verified_report_sha "$report_root" final_benchmark_report.json)"
echo "FINAL_BENCHMARK_REPORT_SHA256=$final_report_sha256"

echo "=== SEAL THE SOLE ACCESS EVENT ==="
"$python_bin" -m multimodal_science.qwen3vl.final_benchmark_access_cli complete \
  --protocol-root "$protocol_root" \
  --protocol-sha256 "$protocol_sha256" \
  --ledger-dir "$ledger_dir" \
  --final-evidence-manifest "$report_root/artifact_manifest.sha256"
verify_manifest "$report_root"
verify_manifest "$ledger_dir"

echo "FINAL_EVIDENCE_REGISTRY=$report_root/final_evidence_registry.json"
echo "FINAL_BENCHMARK_TABLE=$report_root/final_benchmark_table.md"
echo "INTERNAL_TEST_ACCESS_SEQUENCE=1"
echo "ADDITIONAL_TEST_ACCESS_AUTHORIZED=false"
echo "FINAL_BENCHMARK_ELIGIBLE=true"
echo "ONE_TIME_FINAL_BENCHMARK=OK"
