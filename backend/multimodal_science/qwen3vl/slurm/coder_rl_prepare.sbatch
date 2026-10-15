#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }
}

for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_LORA_BUNDLE_ROOT BIOCODER_LORA_BUNDLE_REPORT_SHA256 \
  BIOCODER_FUSION_BUNDLE_ROOT BIOCODER_FUSION_BUNDLE_REPORT_SHA256 \
  BIOCODER_ASSETS_ROOT
do
  require_variable "$name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"
lora_root="$BIOCODER_LORA_BUNDLE_ROOT"
lora_sha="$BIOCODER_LORA_BUNDLE_REPORT_SHA256"
fusion_root="$BIOCODER_FUSION_BUNDLE_ROOT"
fusion_sha="$BIOCODER_FUSION_BUNDLE_REPORT_SHA256"
assets_root="$BIOCODER_ASSETS_ROOT"
seed="${BIOCODER_SEED:-17}"
job_id="${SLURM_JOB_ID:-manual}"
code_short="${code_revision:0:7}"

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$lora_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$fusion_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$seed" =~ ^[0-9]+$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$lora_root"
test -d "$fusion_root"
test -d "$assets_root"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$code_revision" ]]
[[ "$(sha256sum "$lora_root/lora_bundle_report.json" | awk '{print $1}')" == "$lora_sha" ]]
[[ "$(sha256sum "$fusion_root/fusion_bundle_report.json" | awk '{print $1}')" == "$fusion_sha" ]]

output_root="${BIOCODER_RL_POOL_OUTPUT_ROOT:-$run_root/qwen3vl/rl/pools}"
run_name="v2-train-selection-seed${seed}-${code_short}"
work_output="$output_root/.${run_name}.incomplete"
final_output="$output_root/$run_name"
mkdir -p "$output_root"
exec 7>"$output_root/.${run_name}.lock"
flock -n 7 || { echo "Another job owns this RL pool build" >&2; exit 7; }

verify_output() {
  local root="$1"
  (cd "$root" && sha256sum -c artifact_manifest.sha256)
  "$python_bin" - "$root" "$code_revision" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
report = json.loads((root / "rl_pool_report.json").read_text(encoding="utf-8"))
assert report["code_revision"] == sys.argv[2]
assert report["selection"]["source_groups"] == 8
assert report["counts"]["training_source_groups"] == 79
assert report["contracts"]["training_selection_source_group_overlap"] == 0
assert report["contracts"]["selection_has_blank_qc_sample_coverage"] is True
assert report["contracts"]["validation_answers_opened"] is False
assert report["contracts"]["internal_test_accessed"] is False
assert report["contracts"]["selection_groups_trained"] is False
print("RL_POOL_REPORT_CONTRACT=OK")
PY
}

echo "=== V2 RL POOL PREPARATION CONTRACT ==="
echo "JOB_ID=$job_id"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$code_revision"
echo "INTERNAL_TEST_ACCESSED=false"
echo "VALIDATION_ANSWERS_OPENED=false"

if [[ -d "$final_output" ]]; then
  verify_output "$final_output"
  echo "RL_POOL_OUTPUT=$final_output"
  echo "QWEN3VL_RL_POOL_PREPARATION=ALREADY_COMPLETE"
  exit 0
fi
test ! -e "$work_output"

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-prepare-${job_id}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-prepare-* ]] && rm -rf -- "$scratch"; }
trap cleanup EXIT

echo "=== STAGE IMMUTABLE CODE ==="
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$code_revision" | tar -xf - -C "$scratch/code"

echo "=== BUILD HASH-BOUND TRAIN/SELECTION POOL ==="
mkdir -p "$scratch/tmp" "$scratch/cache"
env -i \
  HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
  LANG="${LANG:-C.UTF-8}" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" \
  PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$python_bin" -m multimodal_science.qwen3vl.build_rl_pool_cli \
    --code-revision "$code_revision" \
    --lora-bundle-root "$lora_root" \
    --lora-bundle-report-sha256 "$lora_sha" \
    --fusion-bundle-root "$fusion_root" \
    --fusion-bundle-report-sha256 "$fusion_sha" \
    --assets-root "$assets_root" \
    --output-dir "$work_output" \
    --seed "$seed"

verify_output "$work_output"
test ! -e "$final_output"
mv "$work_output" "$final_output"
report_sha="$(sha256sum "$final_output/rl_pool_report.json" | awk '{print $1}')"
echo "RL_POOL_REPORT_SHA256=$report_sha"
echo "RL_POOL_OUTPUT=$final_output"
echo "QWEN3VL_RL_POOL_PREPARATION=OK"
