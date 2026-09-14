#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

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
specialist="${BIOCODER_SPECIALIST_COMPARISON:?Set BIOCODER_SPECIALIST_COMPARISON}"
zero_generation="${BIOCODER_ZERO_SHOT_GENERATION:?Set BIOCODER_ZERO_SHOT_GENERATION}"
zero_evaluation="${BIOCODER_ZERO_SHOT_EVALUATION:?Set BIOCODER_ZERO_SHOT_EVALUATION}"
lora_generation="${BIOCODER_LORA_GENERATION:?Set BIOCODER_LORA_GENERATION}"
lora_evaluation="${BIOCODER_LORA_EVALUATION:?Set BIOCODER_LORA_EVALUATION}"
output_dir="${BIOCODER_COMPARISON_OUTPUT:?Set BIOCODER_COMPARISON_OUTPUT}"

for item in \
  "$specialist:specialist comparison" \
  "$zero_generation:zero-shot generation" \
  "$zero_evaluation:zero-shot evaluation" \
  "$lora_generation:LoRA generation" \
  "$lora_evaluation:LoRA evaluation"
do
  require_file "${item%%:*}" "${item#*:}"
done
require_file "$python_bin" "Python interpreter"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "$actual_revision" != "$code_revision" ]]; then
  echo "Code revision mismatch: expected=${code_revision} actual=${actual_revision}" >&2
  exit 2
fi

echo "JOB_ID=${SLURM_JOB_ID:-unscheduled}"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "INTERNAL_TEST_ACCESSED=false"

export PYTHONPATH="$repo_root/backend"
"$python_bin" -m multimodal_science.qwen3vl.compare_development_cli \
  --specialist-comparison "$specialist" \
  --zero-shot-generation "$zero_generation" \
  --zero-shot-evaluation "$zero_evaluation" \
  --lora-generation "$lora_generation" \
  --lora-evaluation "$lora_evaluation" \
  --output-dir "$output_dir"

(
  cd "$output_dir"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$output_dir/cross_family_development_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["development_comparison_eligible"] is True
assert report["final_benchmark_eligible"] is False
assert report["internal_test_accessed"] is False
assert report["dataset"]["bilingual_prompts_are_paired_views"] is True
print("CROSS_FAMILY_DEVELOPMENT_CONTRACT=OK")
PY

echo "COMPARISON_OUTPUT=$output_dir"
echo "CROSS_FAMILY_DEVELOPMENT=OK"
