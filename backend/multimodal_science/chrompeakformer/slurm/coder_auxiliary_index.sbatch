#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/Gmengqi/biocoder-work/raw-072fee8e/auxiliary/slurm/coder-auxiliary-index-%j.out
#SBATCH --error=/home/Gmengqi/biocoder-work/raw-072fee8e/auxiliary/slurm/coder-auxiliary-index-%j.err

set -euo pipefail
umask 077

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: $name" >&2
    exit 2
  fi
}

require_env BIOCODER_REPO_ROOT
require_env BIOCODER_CODE_REVISION
require_env BIOCODER_AUXILIARY_ROOT
require_env BIOCODER_AUXILIARY_REPORT_SHA256
require_env BIOCODER_AUXILIARY_EXTRACTION_ROOT

python_bin="${BIOCODER_CHROMPEAK_PYTHON:-$HOME/miniconda3/envs/biocoder_chrompeak/bin/python}"
run_root="${BIOCODER_RUN_ROOT:-$HOME/biocoder-work/raw-072fee8e}"
persistent_root="${BIOCODER_AUXILIARY_INDEX_OUTPUT_ROOT:-$run_root/auxiliary/indexes/msdata-f589608b-v1}"
scratch="$(mktemp -d "/tmp/biocoder-${USER}-auxiliary-index-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  rm -rf -- "$scratch"
}
trap cleanup EXIT

echo "JOB_ID=$SLURM_JOB_ID"
echo "JOB_NAME=$SLURM_JOB_NAME"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$BIOCODER_CODE_REVISION"
echo "DATA_SCOPE=auxiliary_unlabeled_train"
echo "CUDA_VISIBLE_DEVICES=disabled"
echo "INTERNAL_TEST_ACCESSED=false"

test -x "$python_bin"
test -d "$BIOCODER_REPO_ROOT/.git"
test "$(git -C "$BIOCODER_REPO_ROOT" rev-parse "$BIOCODER_CODE_REVISION^{commit}")" = "$BIOCODER_CODE_REVISION"
test -f "$BIOCODER_AUXILIARY_ROOT/auxiliary_report.json"
test -f "$BIOCODER_AUXILIARY_ROOT/derivation_plan.jsonl"
test -f "$BIOCODER_AUXILIARY_ROOT/artifact_manifest.sha256"
test -d "$BIOCODER_AUXILIARY_EXTRACTION_ROOT/jobs/auxiliary_unlabeled_train"
test -f "$BIOCODER_AUXILIARY_EXTRACTION_ROOT/auxiliary_report.sha256"

actual_report_sha256="$(sha256sum "$BIOCODER_AUXILIARY_ROOT/auxiliary_report.json" | awk '{print $1}')"
test "$actual_report_sha256" = "$BIOCODER_AUXILIARY_REPORT_SHA256"
test "$(tr -d '[:space:]' < "$BIOCODER_AUXILIARY_EXTRACTION_ROOT/auxiliary_report.sha256")" = "$BIOCODER_AUXILIARY_REPORT_SHA256"
(
  cd "$BIOCODER_AUXILIARY_ROOT"
  sha256sum -c artifact_manifest.sha256
)
echo "AUXILIARY_IMPORT_MANIFEST=OK"

mkdir -p "$scratch/code" "$scratch/input" "$scratch/assets"
git -C "$BIOCODER_REPO_ROOT" archive "$BIOCODER_CODE_REVISION" | tar -x -C "$scratch/code"
cp "$BIOCODER_AUXILIARY_ROOT/derivation_plan.jsonl" "$scratch/input/derivation_plan.jsonl"
cp "$BIOCODER_AUXILIARY_ROOT/auxiliary_report.json" "$scratch/input/auxiliary_report.json"
cp -a "$BIOCODER_AUXILIARY_EXTRACTION_ROOT/." "$scratch/assets/"

export PYTHONPATH="$scratch/code/backend"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "=== BUILD COMPLETE AUXILIARY ASSET INDEX ==="
"$python_bin" -m multimodal_science.chrompeakformer.auxiliary_index_cli \
  --plan "$scratch/input/derivation_plan.jsonl" \
  --assets-root "$scratch/assets" \
  --output-dir "$scratch/index"

(
  cd "$scratch/index"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$scratch/input/auxiliary_report.json" "$scratch/index/auxiliary_asset_index_report.json" <<'PY'
import json
import pathlib
import sys

source = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
index = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
source_counts = source["counts"]
counts = index["counts"]
contracts = index["contracts"]
if counts["selected_jobs"] != source_counts["acquisition_frames"]:
    raise SystemExit("Acquisition-frame count drift")
if counts["extracted_trace_assets"] != source_counts["transition_traces"]:
    raise SystemExit("Transition-trace count drift")
if counts["independent_source_groups"] != source_counts["independent_source_groups"]:
    raise SystemExit("Independent-source-group count drift")
if counts["labels"] != 0 or counts["supervised_train_assets"] != 0:
    raise SystemExit("Auxiliary index unexpectedly contains supervision")
if contracts.get("metrics_allowed") is not False:
    raise SystemExit("Auxiliary metrics contract drift")
if contracts.get("internal_test_accessed") is not False:
    raise SystemExit("Internal-test contract drift")
print(f"AUXILIARY_INDEXED_TRACES={counts['extracted_trace_assets']}", flush=True)
print(f"AUXILIARY_SOURCE_GROUPS={counts['independent_source_groups']}", flush=True)
print("AUXILIARY_INDEX_CONTRACT=OK", flush=True)
PY

test ! -e "$persistent_root"
mkdir -p "$(dirname "$persistent_root")"
cp -a "$scratch/index" "$persistent_root"
printf '%s\n' "$BIOCODER_AUXILIARY_REPORT_SHA256" > "$persistent_root/auxiliary_report.sha256"
printf '%s\n' "$BIOCODER_CODE_REVISION" > "$persistent_root/code_revision.txt"

echo "PERSISTENT_OUTPUT=$persistent_root"
echo "AUXILIARY_LABELS=0"
echo "AUXILIARY_METRICS_ALLOWED=false"
echo "CHROMPEAK_AUXILIARY_INDEX=OK"
