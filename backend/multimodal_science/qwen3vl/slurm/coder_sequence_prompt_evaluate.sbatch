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
  [[ -n "${!name:-}" ]] || { echo "Missing required variable: $name" >&2; exit 2; }
}

for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_MODEL_ROOT BIOCODER_MODEL_REVISION BIOCODER_MODEL_ARTIFACT_SHA256 \
  BIOCODER_MODEL_MANIFEST_PATH BIOCODER_MODEL_MANIFEST_SHA256 \
  BIOCODER_INFERENCE_BUNDLE_ROOT BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256 \
  BIOCODER_FUSION_BUNDLE_ROOT BIOCODER_FUSION_BUNDLE_REPORT_SHA256 \
  BIOCODER_DATASET_ROOT BIOCODER_DATASET_REPORT_SHA256 BIOCODER_ASSETS_ROOT \
  BIOCODER_IMAGE_ADAPTER_ROOT BIOCODER_IMAGE_ADAPTER_REPORT_SHA256 \
  BIOCODER_IMAGE_ADAPTER_MANIFEST_SHA256 BIOCODER_SEQUENCE_RUN_ROOT \
  BIOCODER_SEQUENCE_REPORT_SHA256 BIOCODER_SEQUENCE_MANIFEST_SHA256 \
  BIOCODER_INSTRUCTION_ROOT BIOCODER_INSTRUCTION_REPORT_SHA256
do
  require_variable "$name"
done

repo="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
revision="$BIOCODER_CODE_REVISION"
model_root="$BIOCODER_MODEL_ROOT"
model_revision="$BIOCODER_MODEL_REVISION"
model_sha="$BIOCODER_MODEL_ARTIFACT_SHA256"
model_manifest="$BIOCODER_MODEL_MANIFEST_PATH"
model_manifest_sha="$BIOCODER_MODEL_MANIFEST_SHA256"
inference_root="$BIOCODER_INFERENCE_BUNDLE_ROOT"
inference_sha="$BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256"
fusion_root="$BIOCODER_FUSION_BUNDLE_ROOT"
fusion_sha="$BIOCODER_FUSION_BUNDLE_REPORT_SHA256"
dataset_root="$BIOCODER_DATASET_ROOT"
dataset_sha="$BIOCODER_DATASET_REPORT_SHA256"
assets_root="$BIOCODER_ASSETS_ROOT"
adapter_root="$BIOCODER_IMAGE_ADAPTER_ROOT"
adapter_report_sha="$BIOCODER_IMAGE_ADAPTER_REPORT_SHA256"
adapter_manifest_sha="$BIOCODER_IMAGE_ADAPTER_MANIFEST_SHA256"
sequence_root="$BIOCODER_SEQUENCE_RUN_ROOT"
sequence_report_sha="$BIOCODER_SEQUENCE_REPORT_SHA256"
sequence_manifest_sha="$BIOCODER_SEQUENCE_MANIFEST_SHA256"
instruction_root="$BIOCODER_INSTRUCTION_ROOT"
instruction_sha="$BIOCODER_INSTRUCTION_REPORT_SHA256"
gpu_index="${BIOCODER_GPU_INDEX:-auto}"
max_gpu_used_mib="${BIOCODER_MAX_GPU_USED_MIB:-512}"
max_gpu_utilization="${BIOCODER_MAX_GPU_UTILIZATION:-10}"
job_id="${SLURM_JOB_ID:-manual}"
job_name="${SLURM_JOB_NAME:-coder}"

[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "$model_sha" "$model_manifest_sha" "$inference_sha" "$fusion_sha" \
  "$dataset_sha" "$adapter_report_sha" "$adapter_manifest_sha" \
  "$sequence_report_sha" "$sequence_manifest_sha" "$instruction_sha"
do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "Bad SHA-256: $digest" >&2; exit 2; }
done
[[ "$gpu_index" == "auto" || "$gpu_index" =~ ^[0-9]+$ ]]
test -x "$python_bin"
test -d "$repo/.git"
test -d "$model_root"
test -f "$model_manifest"
test -d "$inference_root"
test -d "$fusion_root"
test -d "$dataset_root"
test -d "$assets_root"
test -d "$adapter_root"
test -d "$sequence_root"
test -d "$instruction_root"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$revision" ]]
[[ "$(sha256sum "$model_manifest" | awk '{print $1}')" == "$model_manifest_sha" ]]
(cd "$model_root" && sha256sum -c "$model_manifest")

code_short="${revision:0:7}"
sequence_short="${sequence_report_sha:0:12}"
adapter_short="${adapter_report_sha:0:12}"
bundle_root="$run_root/qwen3vl/sequence-prompt/bundles/sequence-${sequence_short}-${code_short}"
run_name="image-lora-sequence-prompt-validation-${adapter_short}-${sequence_short}-${code_short}"
generation_root="$run_root/qwen3vl/sequence-prompt/generations"
evaluation_root="$run_root/qwen3vl/sequence-prompt/evaluations"
generation_final="$generation_root/$run_name"
evaluation_final="$evaluation_root/${run_name}-evaluation"
mkdir -p "$(dirname "$bundle_root")" "$generation_root" "$evaluation_root"
exec 7>"$evaluation_root/.${run_name}.lock"
flock -n 7 || { echo "Exact sequence-prompt evaluation is already owned" >&2; exit 7; }

scratch="$(mktemp -d "/tmp/biocoder-${USER}-sequence-prompt-${job_id}.XXXXXX")"
cleanup() {
  case "${scratch:-}" in
    "/tmp/biocoder-${USER}-sequence-prompt-${job_id}."*) rm -rf -- "$scratch" ;;
  esac
}
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/cache"
git -C "$repo" archive "$revision" | tar -x -C "$scratch/code"
export PYTHONPATH="$scratch/code/backend"

echo "=== SEQUENCE-PROMPT DEVELOPMENT CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=$job_name"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$revision"
echo "SEQUENCE_REPORT_SHA256=$sequence_report_sha"
echo "IMAGE_ADAPTER_REPORT_SHA256=$adapter_report_sha"
echo "INTERNAL_TEST_ACCESSED=false"

if [[ ! -d "$bundle_root" ]]; then
  "$python_bin" -m multimodal_science.qwen3vl.build_sequence_prompt_bundle_cli \
    --sequence-run-root "$sequence_root" \
    --sequence-report-sha256 "$sequence_report_sha" \
    --sequence-manifest-sha256 "$sequence_manifest_sha" \
    --output-dir "$bundle_root" \
    --code-revision "$revision"
fi
(cd "$bundle_root" && sha256sum -c artifact_manifest.sha256)
bundle_report_sha="$(sha256sum "$bundle_root/sequence_prompt_bundle_report.json" | awk '{print $1}')"
bundle_manifest_sha="$(sha256sum "$bundle_root/artifact_manifest.sha256" | awk '{print $1}')"
"$python_bin" - "$bundle_root/sequence_prompt_bundle_report.json" \
  "$revision" "$sequence_report_sha" "$sequence_manifest_sha" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["schema_version"] == "chrompeak-sequence-prompt-bundle-v1"
assert report["code_revision"] == sys.argv[2]
assert report["sources"]["sequence_report_sha256"] == sys.argv[3]
assert report["sources"]["sequence_manifest_sha256"] == sys.argv[4]
assert report["contracts"]["target_fields_excluded"] is True
assert report["contracts"]["instruction_answers_opened"] is False
assert report["internal_test_accessed"] is False
print("SEQUENCE_PROMPT_BUNDLE_CONTRACT=OK")
PY
echo "SEQUENCE_PROMPT_BUNDLE_REPORT_SHA256=$bundle_report_sha"

gpu_passes_guard() {
  local candidate="$1" line used utilization remainder
  for sample in 1 2 3; do
    line="$(nvidia-smi -i "$candidate" --query-gpu=memory.used,memory.free,utilization.gpu,name --format=csv,noheader,nounits)" || return 1
    echo "candidate=$candidate sample=$sample $line" >&2
    used="${line%%,*}"; remainder="${line#*,}"; remainder="${remainder#*,}"
    utilization="${remainder%%,*}"; used="${used// /}"; utilization="${utilization// /}"
    [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
    (( used <= max_gpu_used_mib && utilization <= max_gpu_utilization )) || return 1
    sleep 2
  done
}
if [[ "$gpu_index" == "auto" ]]; then
  gpu_index=""
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  for (( candidate=0; candidate<gpu_count; candidate++ )); do
    exec 9>"/tmp/biocoder-${USER}-gpu-${candidate}.lock"
    if flock -n 9 && gpu_passes_guard "$candidate"; then gpu_index="$candidate"; break; fi
  done
  [[ -n "$gpu_index" ]] || { echo "No GPU passed startup guard" >&2; exit 5; }
else
  exec 9>"/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
  flock -n 9 && gpu_passes_guard "$gpu_index"
fi
echo "SELECTED_PHYSICAL_GPU=$gpu_index"

write_manifest() {
  local root="$1"; shift
  (cd "$root" && : > artifact_manifest.sha256 && for path in "$@"; do sha256sum "$path" >> artifact_manifest.sha256; done)
}

if [[ ! -d "$generation_final" ]]; then
  resume_args=()
  [[ -d "$generation_root/.${run_name}.work" ]] && resume_args+=(--resume)
  env -i \
    HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG="${LANG:-C.UTF-8}" \
    TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache" CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$python_bin" -m multimodal_science.qwen3vl.run_sequence_prompt_inference_cli \
      --inference-bundle-root "$inference_root" \
      --inference-bundle-report-sha256 "$inference_sha" \
      --fusion-bundle-root "$fusion_root" \
      --fusion-bundle-report-sha256 "$fusion_sha" \
      --dataset-root "$dataset_root" \
      --dataset-report-sha256 "$dataset_sha" \
      --assets-root "$assets_root" \
      --image-adapter-root "$adapter_root" \
      --image-adapter-report-sha256 "$adapter_report_sha" \
      --image-adapter-manifest-sha256 "$adapter_manifest_sha" \
      --sequence-prompt-bundle-root "$bundle_root" \
      --sequence-prompt-bundle-report-sha256 "$bundle_report_sha" \
      --sequence-prompt-bundle-manifest-sha256 "$bundle_manifest_sha" \
      --output-dir "$generation_final" \
      --model-name-or-path "$model_root" \
      --model-revision "$model_revision" \
      --model-artifact-sha256 "$model_sha" \
      --batch-size 1 --max-new-tokens 64 --seed 17 --dtype bfloat16 \
      --device-map auto --attention-implementation sdpa \
      "${resume_args[@]}"
fi
write_manifest "$generation_final" generation_report.json predictions.jsonl generation_records.jsonl run_config.json runtime_metadata.json
generation_sha="$(sha256sum "$generation_final/generation_report.json" | awk '{print $1}')"
echo "GENERATION_REPORT_SHA256=$generation_sha"

if [[ ! -d "$evaluation_final" ]]; then
  evaluation_staging="$scratch/evaluation"
  "$python_bin" -m multimodal_science.qwen3vl.evaluate_predictions_cli \
    --instruction-root "$instruction_root" \
    --instruction-report-sha256 "$instruction_sha" \
    --predictions "$generation_final/predictions.jsonl" \
    --generation-report "$generation_final/generation_report.json" \
    --generation-report-sha256 "$generation_sha" \
    --output-dir "$evaluation_staging" \
    --bootstrap-iterations 1000 --seed 17
  write_manifest "$evaluation_staging" qwen_evaluation_report.json evaluation_records.jsonl
  mv "$evaluation_staging" "$evaluation_final"
fi
(cd "$generation_final" && sha256sum -c artifact_manifest.sha256)
(cd "$evaluation_final" && sha256sum -c artifact_manifest.sha256)
evaluation_sha="$(sha256sum "$evaluation_final/qwen_evaluation_report.json" | awk '{print $1}')"

"$python_bin" - "$generation_final/generation_report.json" "$evaluation_final/qwen_evaluation_report.json" "$bundle_report_sha" <<'PY'
import json
import sys
from pathlib import Path

generation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
bundle_sha = sys.argv[3]
assert generation["counts"]["predictions"] == 13708
assert generation["scope"]["complete_prompt_coverage"] is True
assert generation["development_comparison_candidate"] is True
assert generation["runtime"]["sequence_prompt_execution_backend"] == "transformers-qwen3vl-image-sequence-prompt"
assert generation["runtime"]["sequence_expert"]["report_sha256"] == bundle_sha
assert generation["runtime"]["sequence_expert"]["target_fields_exposed"] is False
assert generation["model"]["adapter"]["kind"] == "image_lora_with_sequence_prompt"
assert generation["contracts"]["answer_key_opened"] is False
assert generation["internal_test_accessed"] is False
assert evaluation["counts"]["predictions"] == 13708
assert evaluation["prediction_generation_provenance_verified"] is True
assert evaluation["development_comparison_eligible"] is True
assert evaluation["final_benchmark_eligible"] is False
assert evaluation["internal_test_accessed"] is False
print("SEQUENCE_PROMPT_GENERATION_CONTRACT=OK")
print("SEQUENCE_PROMPT_EVALUATION_CONTRACT=OK")
PY

echo "EVALUATION_REPORT_SHA256=$evaluation_sha"
echo "PERSISTENT_GENERATION=$generation_final"
echo "PERSISTENT_EVALUATION=$evaluation_final"
echo "QWEN3VL_SEQUENCE_PROMPT_EVALUATION=OK"
