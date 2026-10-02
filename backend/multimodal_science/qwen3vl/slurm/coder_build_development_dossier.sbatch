#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
cross_family_report="${BIOCODER_CROSS_FAMILY_REPORT:?Set BIOCODER_CROSS_FAMILY_REPORT}"
fusion_matrix_report="${BIOCODER_FUSION_MATRIX_REPORT:?Set BIOCODER_FUSION_MATRIX_REPORT}"
xic_intervention_report="${BIOCODER_XIC_INTERVENTION_REPORT:?Set BIOCODER_XIC_INTERVENTION_REPORT}"
selected_evaluation_root="${BIOCODER_SELECTED_FUSION_EVALUATION_ROOT:?Set BIOCODER_SELECTED_FUSION_EVALUATION_ROOT}"
output_dir="${BIOCODER_DEVELOPMENT_DOSSIER_OUTPUT:?Set BIOCODER_DEVELOPMENT_DOSSIER_OUTPUT}"

require_file "$python_bin" "Python interpreter"
require_file "$cross_family_report" "cross-family report"
require_file "$fusion_matrix_report" "fusion-matrix report"
require_file "$xic_intervention_report" "XIC-intervention report"
require_file "$selected_evaluation_root/qwen_evaluation_report.json" "selected fusion evaluation report"
require_file "$selected_evaluation_root/evaluation_records.jsonl" "selected fusion evaluation records"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "$actual_revision" != "$code_revision" ]]; then
  echo "Code revision mismatch: expected=${code_revision} actual=${actual_revision}" >&2
  exit 2
fi

scratch_root="$(mktemp -d "/tmp/biocoder-${USER}-development-dossier-${SLURM_JOB_ID:-manual}.XXXXXX")"
cleanup() {
  rm -rf -- "$scratch_root"
}
trap cleanup EXIT
code_root="$scratch_root/code"
mkdir -p "$code_root"
git -C "$repo_root" cat-file -e "${code_revision}^{commit}"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$code_root"
test -f "$code_root/backend/multimodal_science/qwen3vl/development_dossier.py"

echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "IMMUTABLE_CODE_ARCHIVE=OK"
echo "INTERNAL_TEST_ACCESSED=false"

export PYTHONPATH="$code_root/backend"
"$python_bin" -m multimodal_science.qwen3vl.build_development_dossier_cli \
  --cross-family-report "$cross_family_report" \
  --fusion-matrix-report "$fusion_matrix_report" \
  --xic-intervention-report "$xic_intervention_report" \
  --selected-fusion-evaluation-root "$selected_evaluation_root" \
  --output-dir "$output_dir"

(
  cd "$output_dir"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$output_dir/development_dossier.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["development_comparison_eligible"] is True
assert report["final_benchmark_eligible"] is False
assert report["internal_test_accessed"] is False
assert report["contracts"]["same_leakage_safe_validation_dataset"] is True
assert report["contracts"]["same_hash_bound_prompt_answer_and_model_inputs"] is True
assert report["contracts"]["evaluation_record_identities_match"] is True
assert report["contracts"]["failure_cases_come_from_hash_bound_evaluation_records"] is True
assert len(report["main_table"]) == 9
print("DEVELOPMENT_DOSSIER_CONTRACT=OK")
print(
    "PRE_INTERNAL_TEST_READY="
    + str(report["pre_internal_test_readiness"]["ready"]).lower()
)
PY

echo "DEVELOPMENT_DOSSIER_OUTPUT=$output_dir"
echo "MULTIMODAL_DEVELOPMENT_DOSSIER=OK"
