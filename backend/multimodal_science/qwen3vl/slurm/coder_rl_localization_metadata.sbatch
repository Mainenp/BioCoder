#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }
}

for name in \
  BIOCODER_REPO_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_DATASET_ROOT BIOCODER_DATASET_REPORT_SHA256 \
  BIOCODER_FUSION_BUNDLE_ROOT BIOCODER_FUSION_BUNDLE_REPORT_SHA256 \
  BIOCODER_RL_LOCALIZATION_METADATA_OUTPUT
do
  require_variable "$name"
done

repo_root="$BIOCODER_REPO_ROOT"
python_bin="$BIOCODER_PYTHON"
revision="$BIOCODER_CODE_REVISION"
dataset_root="$BIOCODER_DATASET_ROOT"
dataset_sha="$BIOCODER_DATASET_REPORT_SHA256"
fusion_root="$BIOCODER_FUSION_BUNDLE_ROOT"
fusion_sha="$BIOCODER_FUSION_BUNDLE_REPORT_SHA256"
output="$BIOCODER_RL_LOCALIZATION_METADATA_OUTPUT"
job_id="${SLURM_JOB_ID:-manual}"

[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$dataset_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$fusion_sha" =~ ^[0-9a-f]{64}$ ]]
test -x "$python_bin"
test -d "$repo_root/.git"
test -d "$dataset_root"
test -d "$fusion_root"
test ! -e "$output"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$revision" ]]
[[ "$(sha256sum "$dataset_root/dataset_report.json" | awk '{print $1}')" == "$dataset_sha" ]]
[[ "$(sha256sum "$fusion_root/fusion_bundle_report.json" | awk '{print $1}')" == "$fusion_sha" ]]

scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-localization-metadata-${job_id}.XXXXXX")"
cleanup() {
  [[ "$scratch" == /tmp/biocoder-"$USER"-rl-localization-metadata-* ]] && rm -rf -- "$scratch"
}
trap cleanup EXIT
mkdir -p "$scratch/code" "$scratch/tmp" "$scratch/cache"
git -C "$repo_root" archive "$revision" | tar -xf - -C "$scratch/code"

echo "=== V2 ANSWER-FREE LOCALIZATION METADATA CONTRACT ==="
echo "JOB_ID=$job_id"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$revision"
echo "VALIDATION_ANSWERS_OPENED=false"
echo "INTERNAL_TEST_ACCESSED=false"

env -i \
  HOME="$HOME" USER="$USER" PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
  LANG="${LANG:-C.UTF-8}" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache" \
  PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$python_bin" -m multimodal_science.qwen3vl.build_localization_metadata_cli \
    --dataset-root "$dataset_root" \
    --dataset-report-sha256 "$dataset_sha" \
    --fusion-bundle-root "$fusion_root" \
    --fusion-bundle-report-sha256 "$fusion_sha" \
    --output-dir "$output"

(cd "$output" && sha256sum -c artifact_manifest.sha256)
"$python_bin" - "$output" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = json.loads(
    (root / "validation_localization_metadata_report.json").read_text(encoding="utf-8")
)
assert report["schema_version"] == "chrompeak-qwen3vl-v2-localization-metadata-report-v1"
assert report["counts"]["validation_source_groups"] == 11
assert report["counts"]["grounding_source_groups"] >= 2
assert report["contracts"]["answer_free"] is True
assert report["contracts"]["validation_answers_opened"] is False
assert report["contracts"]["internal_test_accessed"] is False
print("RL_LOCALIZATION_METADATA_REPORT_CONTRACT=OK")
PY

echo "RL_LOCALIZATION_METADATA_REPORT_SHA256=$(sha256sum "$output/validation_localization_metadata_report.json" | awk '{print $1}')"
echo "RL_LOCALIZATION_METADATA_SHA256=$(sha256sum "$output/validation_localization_metadata.jsonl" | awk '{print $1}')"
echo "RL_LOCALIZATION_METADATA_OUTPUT=$output"
echo "QWEN3VL_RL_LOCALIZATION_METADATA=OK"
