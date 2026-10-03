#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

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
  BIOCODER_DEVELOPMENT_DOSSIER_ROOT \
  BIOCODER_SPLIT_MANIFEST_PATH \
  BIOCODER_SPLIT_REPORT_PATH \
  BIOCODER_DERIVATION_PLAN_PATH \
  BIOCODER_DERIVATION_REPORT_PATH \
  BIOCODER_DATASET_ROOT \
  BIOCODER_INSTRUCTION_ROOT \
  BIOCODER_MODEL_MANIFEST_PATH
do
  require_variable "$required_name"
done

repo_root="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
code_revision="$BIOCODER_CODE_REVISION"
code_short="${code_revision:0:7}"
expected_assets="${BIOCODER_EXPECTED_INTERNAL_TEST_ASSETS:-1815}"
expected_groups="${BIOCODER_EXPECTED_INTERNAL_TEST_SOURCE_GROUPS:-11}"
default_protocol_root="$run_root/final-benchmark/protocols/protocol-$code_short"
protocol_root="${BIOCODER_FINAL_PROTOCOL_ROOT:-$default_protocol_root}"
job_id="${SLURM_JOB_ID:-manual}"

[[ "$code_revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_assets" =~ ^[1-9][0-9]*$ ]]
[[ "$expected_groups" =~ ^[1-9][0-9]*$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test "$(git -C "$repo_root" rev-parse HEAD)" = "$code_revision"

for path in \
  "$BIOCODER_SPLIT_MANIFEST_PATH" \
  "$BIOCODER_SPLIT_REPORT_PATH" \
  "$BIOCODER_DERIVATION_PLAN_PATH" \
  "$BIOCODER_DERIVATION_REPORT_PATH" \
  "$BIOCODER_MODEL_MANIFEST_PATH"
do
  test -f "$path"
done
for path in \
  "$BIOCODER_DEVELOPMENT_DOSSIER_ROOT" \
  "$BIOCODER_DATASET_ROOT" \
  "$BIOCODER_INSTRUCTION_ROOT"
do
  test -d "$path"
done

protocol_parent="$(dirname "$protocol_root")"
mkdir -p "$protocol_parent"
exec 7>"$protocol_parent/.protocol-$code_short.lock"
flock -n 7 || {
  echo "Another process is freezing this exact final protocol" >&2
  exit 7
}

echo "=== FINAL BENCHMARK PRE-ACCESS CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$code_revision"
echo "EXPECTED_INTERNAL_TEST_ASSETS=$expected_assets"
echo "EXPECTED_INTERNAL_TEST_SOURCE_GROUPS=$expected_groups"
echo "INTERNAL_TEST_ACCESSED=false"

if [[ ! -d "$protocol_root" ]]; then
  (
    cd "$repo_root/backend"
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo_root/backend" \
      "$python_bin" -m multimodal_science.qwen3vl.freeze_final_benchmark_cli \
        --development-dossier-root "$BIOCODER_DEVELOPMENT_DOSSIER_ROOT" \
        --split-manifest "$BIOCODER_SPLIT_MANIFEST_PATH" \
        --split-report "$BIOCODER_SPLIT_REPORT_PATH" \
        --derivation-plan "$BIOCODER_DERIVATION_PLAN_PATH" \
        --derivation-report "$BIOCODER_DERIVATION_REPORT_PATH" \
        --dataset-root "$BIOCODER_DATASET_ROOT" \
        --instruction-root "$BIOCODER_INSTRUCTION_ROOT" \
        --base-model-manifest "$BIOCODER_MODEL_MANIFEST_PATH" \
        --output-dir "$protocol_root" \
        --expected-internal-test-assets "$expected_assets" \
        --expected-internal-test-source-groups "$expected_groups" \
        --bootstrap-iterations 10000 \
        --bootstrap-seed 17
  )
fi

(
  cd "$protocol_root"
  sha256sum -c artifact_manifest.sha256
)

protocol_sha256="$(sha256sum "$protocol_root/final_benchmark_protocol.json" | awk '{print $1}')"
PYTHONPATH="$repo_root/backend" "$python_bin" - \
  "$protocol_root" "$protocol_sha256" <<'PY'
import json
import sys
from pathlib import Path

from multimodal_science.qwen3vl.final_benchmark_protocol import _verify_protocol

root = Path(sys.argv[1])
report = _verify_protocol(root, sys.argv[2])
if report.get("pre_internal_test_ready") is not True:
    raise RuntimeError("Final protocol is not pre-access ready")
if report.get("internal_test_accessed") is not False:
    raise RuntimeError("Final protocol improperly records test access")
print("FINAL_PROTOCOL_CONTRACT=OK")
PY

pointer="$protocol_parent/final-benchmark-protocol.latest"
temporary="$pointer.$job_id.tmp"
printf '%s\n%s\n' "$protocol_root" "$protocol_sha256" > "$temporary"
mv "$temporary" "$pointer"

echo "FINAL_PROTOCOL_ROOT=$protocol_root"
echo "FINAL_PROTOCOL_SHA256=$protocol_sha256"
echo "FINAL_PROTOCOL_POINTER=$pointer"
echo "PRE_INTERNAL_TEST_READY=true"
echo "INTERNAL_TEST_ACCESSED=false"
echo "FINAL_BENCHMARK_PROTOCOL_FREEZE=OK"
