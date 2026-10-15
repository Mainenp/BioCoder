#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-4090
#SBATCH --nodelist=gnode01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

require_variable() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in BIOCODER_REPO_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION BIOCODER_RL_SEED_REPLICATION_ACTION BIOCODER_RL_SEED_REPLICATION_OUTPUT
do require_variable "$name"; done

repo_root="$BIOCODER_REPO_ROOT"
python_bin="$BIOCODER_PYTHON"
revision="$BIOCODER_CODE_REVISION"
action="$BIOCODER_RL_SEED_REPLICATION_ACTION"
output="$BIOCODER_RL_SEED_REPLICATION_OUTPUT"
job_id="${SLURM_JOB_ID:-manual}"

[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$action" == build-candidates || "$action" == analyze ]]
test -x "$python_bin"
test -d "$repo_root/.git"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$revision" ]]
[[ "$output" != *validation* && "$output" != *internal_test* && "$output" != *internal-test* ]]

echo "=== V2 RL SEED REPLICATION CONTRACT ==="
echo "JOB_ID=$job_id"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$revision"
echo "ACTION=$action"
echo "SELECTION_SET_EVALUATION_ONLY=true"
echo "VALIDATION_PROMPTS_OPENED=false"
echo "VALIDATION_ANSWERS_OPENED=false"
echo "INTERNAL_TEST_ACCESSED=false"

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-seed-replication-${job_id}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-seed-replication-* ]] && rm -rf -- "$scratch"; }
trap cleanup EXIT
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$revision" | tar -xf - -C "$scratch/code"

common=(--output-dir "$output")
if [[ "$action" == build-candidates ]]; then
  for name in BIOCODER_RL_POOL_REPORT_SHA256 BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256 BIOCODER_R2_SEED29_ROOT BIOCODER_C1_SEED29_ROOT
  do require_variable "$name"; done
  for digest in "$BIOCODER_RL_POOL_REPORT_SHA256" "$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256" "$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"
  do [[ "$digest" =~ ^[0-9a-f]{64}$ ]]; done
  test -d "$BIOCODER_R2_SEED29_ROOT"
  test -d "$BIOCODER_C1_SEED29_ROOT"
  args=(
    --code-revision "$revision"
    --rl-pool-report-sha256 "$BIOCODER_RL_POOL_REPORT_SHA256"
    --c0-report-sha256 "$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256"
    --c0-manifest-sha256 "$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256"
    --r2-root "$BIOCODER_R2_SEED29_ROOT"
    --c1-root "$BIOCODER_C1_SEED29_ROOT"
  )
else
  for name in BIOCODER_SEED17_R2_SELECTION_EVALUATION BIOCODER_SEED17_C1_SELECTION_EVALUATION BIOCODER_SEED29_R2_SELECTION_EVALUATION BIOCODER_SEED29_C1_SELECTION_EVALUATION
  do require_variable "$name"; test -f "${!name}/selection_evaluation_report.json"; done
  args=(
    --seed17-r2-evaluation "$BIOCODER_SEED17_R2_SELECTION_EVALUATION/selection_evaluation_report.json"
    --seed17-c1-evaluation "$BIOCODER_SEED17_C1_SELECTION_EVALUATION/selection_evaluation_report.json"
    --seed29-r2-evaluation "$BIOCODER_SEED29_R2_SELECTION_EVALUATION/selection_evaluation_report.json"
    --seed29-c1-evaluation "$BIOCODER_SEED29_C1_SELECTION_EVALUATION/selection_evaluation_report.json"
  )
fi

env -i HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" LANG="${LANG:-C.UTF-8}" \
  PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$python_bin" -m multimodal_science.qwen3vl.rl_seed_replication_cli \
    "$action" "${args[@]}" "${common[@]}"

(cd "$output" && sha256sum -c artifact_manifest.sha256)
echo "RL_SEED_REPLICATION_OUTPUT=$output"
echo "QWEN3VL_RL_SEED_REPLICATION=OK"
