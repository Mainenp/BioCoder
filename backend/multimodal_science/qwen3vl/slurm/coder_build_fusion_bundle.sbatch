#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
dataset_root="${BIOCODER_DATASET_ROOT:?Set BIOCODER_DATASET_ROOT}"
lora_bundle_root="${BIOCODER_LORA_BUNDLE_ROOT:?Set BIOCODER_LORA_BUNDLE_ROOT}"
inference_bundle_root="${BIOCODER_INFERENCE_BUNDLE_ROOT:?Set BIOCODER_INFERENCE_BUNDLE_ROOT}"
output_dir="${BIOCODER_FUSION_BUNDLE_OUTPUT:?Set BIOCODER_FUSION_BUNDLE_OUTPUT}"

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
"$python_bin" -m multimodal_science.qwen3vl.build_fusion_bundle_cli \
  --dataset-root "$dataset_root" \
  --dataset-report-sha256 "${BIOCODER_DATASET_REPORT_SHA256:?Set BIOCODER_DATASET_REPORT_SHA256}" \
  --lora-bundle-root "$lora_bundle_root" \
  --lora-bundle-report-sha256 "${BIOCODER_LORA_BUNDLE_REPORT_SHA256:?Set BIOCODER_LORA_BUNDLE_REPORT_SHA256}" \
  --inference-bundle-root "$inference_bundle_root" \
  --inference-bundle-report-sha256 "${BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256:?Set BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256}" \
  --output-dir "$output_dir"

(
  cd "$output_dir"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$output_dir/fusion_bundle_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["counts"]["train_instruction_links"] == 54335
assert report["counts"]["train_independent_assets"] == 14355
assert report["counts"]["validation_prompt_links"] == 13708
assert report["counts"]["validation_independent_assets"] == 1815
assert report["contracts"]["validation_answer_key_opened"] is False
assert report["contracts"]["source_group_overlap"] == 0
assert report["internal_test_accessed"] is False
print("FUSION_BUNDLE_CONTRACT=OK")
PY

echo "FUSION_BUNDLE_OUTPUT=$output_dir"
echo "QWEN3VL_XIC_FUSION_BUNDLE=OK"
