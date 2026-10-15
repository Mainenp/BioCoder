#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
run_root="${BIOCODER_RUN_ROOT:?Set BIOCODER_RUN_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
training_root="${BIOCODER_MATRIX_TRAINING_ROOT:?Set BIOCODER_MATRIX_TRAINING_ROOT}"
training_revision="${BIOCODER_MATRIX_TRAINING_REVISION:?Set BIOCODER_MATRIX_TRAINING_REVISION}"
evaluation_revision="${BIOCODER_MATRIX_EVALUATION_REVISION:?Set BIOCODER_MATRIX_EVALUATION_REVISION}"
output_dir="${BIOCODER_FUSION_MATRIX_ANALYSIS_OUTPUT:?Set BIOCODER_FUSION_MATRIX_ANALYSIS_OUTPUT}"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 2
}
for revision in "$training_revision" "$evaluation_revision"
do
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Invalid matrix revision: $revision" >&2
    exit 2
  }
done
[[ ! -e "$output_dir" ]] || {
  echo "Analysis output already exists: $output_dir" >&2
  exit 2
}

training_short="${training_revision:0:7}"
evaluation_short="${evaluation_revision:0:7}"
generation_parent="$run_root/qwen3vl/fusion/generations"
evaluation_parent="$run_root/qwen3vl/fusion/evaluations"
declare -a arguments=()

for specification in \
  "primary-seed17:4:17" \
  "primary-seed29:4:29" \
  "primary-seed43:4:43" \
  "tokens1-seed17:1:17" \
  "tokens8-seed17:8:17"
do
  IFS=: read -r label sensor_tokens training_seed <<<"$specification"
  adapter_root="$training_root/image-xic-full-tokens${sensor_tokens}-epoch1-b1-ga16-seed${training_seed}-${training_short}"
  training_report="$adapter_root/fusion_training_report.json"
  [[ -f "$training_report" ]] || {
    echo "Missing $label training report: $training_report" >&2
    exit 2
  }
  adapter_short="$(sha256sum "$training_report" | awk '{print substr($1,1,12)}')"
  run_name="image-xic-aligned-iseed17-validation-${adapter_short}-${evaluation_short}"
  generation_root="$generation_parent/$run_name"
  matrix_evaluation_root="$evaluation_parent/${run_name}-evaluation"
  [[ -f "$generation_root/generation_report.json" ]] || {
    echo "Missing $label generation report: $generation_root" >&2
    exit 2
  }
  [[ -f "$matrix_evaluation_root/qwen_evaluation_report.json" ]] || {
    echo "Missing $label evaluation report: $matrix_evaluation_root" >&2
    exit 2
  }
  arguments+=(
    "--${label}-training-root" "$adapter_root"
    "--${label}-generation-root" "$generation_root"
    "--${label}-evaluation-root" "$matrix_evaluation_root"
  )
done

echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "TRAINING_REVISION=$training_revision"
echo "EVALUATION_REVISION=$evaluation_revision"
echo "INTERNAL_TEST_ACCESSED=false"

export PYTHONPATH="$repo_root/backend"
"$python_bin" -m multimodal_science.qwen3vl.analyze_fusion_matrix_cli \
  "${arguments[@]}" \
  --output-dir "$output_dir"

(
  cd "$output_dir"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$output_dir/fusion_matrix_analysis.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert len(report["matrix"]) == 5
assert report["primary_seed_aggregate"]["overall"]
assert report["primary_output_quality"]
assert report["primary_cross_language_consistency"]
assert report["token_ablation"]["baseline_sensor_tokens"] == 4
assert report["contracts"]["evaluation_metrics_recomputed_from_records"] is True
assert report["contracts"]["evaluation_record_identities_match_across_runs"] is True
assert report["internal_test_accessed"] is False
assert report["final_benchmark_eligible"] is False
print("FUSION_MATRIX_ANALYSIS_CONTRACT=OK")
PY

echo "FUSION_MATRIX_ANALYSIS_OUTPUT=$output_dir"
echo "QWEN3VL_XIC_FUSION_MATRIX_ANALYSIS=OK"
