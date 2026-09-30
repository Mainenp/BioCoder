#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
output_dir="${BIOCODER_XIC_ANALYSIS_OUTPUT:?Set BIOCODER_XIC_ANALYSIS_OUTPUT}"
iterations="${BIOCODER_BOOTSTRAP_ITERATIONS:-2000}"
seed="${BIOCODER_BOOTSTRAP_SEED:-17}"

# Required intervention roots:
# BIOCODER_ALIGNED_GENERATION_ROOT / BIOCODER_ALIGNED_EVALUATION_ROOT
# BIOCODER_SHUFFLED_GENERATION_ROOT / BIOCODER_SHUFFLED_EVALUATION_ROOT
# BIOCODER_ZERO_GENERATION_ROOT / BIOCODER_ZERO_EVALUATION_ROOT
# BIOCODER_AVAILABILITY_OFF_GENERATION_ROOT / BIOCODER_AVAILABILITY_OFF_EVALUATION_ROOT
declare -a arguments=()
for label in aligned shuffled zero availability-off
do
  variable_label="${label//-/_}"
  generation_variable="BIOCODER_${variable_label^^}_GENERATION_ROOT"
  evaluation_variable="BIOCODER_${variable_label^^}_EVALUATION_ROOT"
  generation_root="${!generation_variable:?Set ${generation_variable}}"
  evaluation_root="${!evaluation_variable:?Set ${evaluation_variable}}"
  [[ -f "$generation_root/generation_report.json" ]] || {
    echo "Missing ${label} generation report" >&2
    exit 2
  }
  [[ -f "$evaluation_root/qwen_evaluation_report.json" ]] || {
    echo "Missing ${label} evaluation report" >&2
    exit 2
  }
  arguments+=(
    "--${label}-generation-root" "$generation_root"
    "--${label}-evaluation-root" "$evaluation_root"
  )
done

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=${code_revision} actual=${actual_revision}" >&2
  exit 2
}
[[ ! -e "$output_dir" ]] || {
  echo "Analysis output already exists: $output_dir" >&2
  exit 2
}

echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "INTERNAL_TEST_ACCESSED=false"

export PYTHONPATH="$repo_root/backend"
"$python_bin" -m multimodal_science.qwen3vl.analyze_xic_interventions_cli \
  "${arguments[@]}" \
  --output-dir "$output_dir" \
  --bootstrap-iterations "$iterations" \
  --seed "$seed"

(
  cd "$output_dir"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$output_dir/xic_intervention_analysis.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["contracts"]["same_fusion_checkpoint"] is True
assert report["contracts"]["paired_source_group_bootstrap"] is True
assert report["paired_group_bootstrap"]["resampling_unit"] == "source_group"
assert report["internal_test_accessed"] is False
assert report["final_benchmark_eligible"] is False
print("XIC_INTERVENTION_ANALYSIS_CONTRACT=OK")
PY

echo "XIC_ANALYSIS_OUTPUT=$output_dir"
echo "QWEN3VL_XIC_INTERVENTION_ANALYSIS=OK"
