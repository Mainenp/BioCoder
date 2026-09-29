#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/Gmengqi/biocoder-work/raw-072fee8e/auxiliary/slurm/coder-auxiliary-%j.out
#SBATCH --error=/home/Gmengqi/biocoder-work/raw-072fee8e/auxiliary/slurm/coder-auxiliary-%j.err

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
require_env BIOCODER_CHROMPEAKFORMER_SOURCE_ROOT

python_bin="${BIOCODER_CHROMPEAK_PYTHON:-$HOME/miniconda3/envs/biocoder_chrompeak/bin/python}"
run_root="${BIOCODER_RUN_ROOT:-$HOME/biocoder-work/raw-072fee8e}"
persistent_root="${BIOCODER_AUXILIARY_OUTPUT_ROOT:-$run_root/auxiliary/msdata-f589608b}"
slurm_root="$run_root/auxiliary/slurm"
scratch="$(mktemp -d "/tmp/biocoder-${USER}-auxiliary-${SLURM_JOB_ID}.XXXXXX")"

mkdir -p "$slurm_root"
cleanup() {
  rm -rf -- "$scratch"
}
trap cleanup EXIT

echo "JOB_ID=$SLURM_JOB_ID"
echo "JOB_NAME=$SLURM_JOB_NAME"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$BIOCODER_CODE_REVISION"
echo "DATA_SCOPE=auxiliary_unlabeled_train"
echo "INTERNAL_TEST_ACCESSED=false"

test -x "$python_bin"
test -d "$BIOCODER_REPO_ROOT/.git"
test "$(git -C "$BIOCODER_REPO_ROOT" rev-parse "$BIOCODER_CODE_REVISION^{commit}")" = "$BIOCODER_CODE_REVISION"
test -f "$BIOCODER_AUXILIARY_ROOT/auxiliary_report.json"
test -f "$BIOCODER_AUXILIARY_ROOT/derivation_plan.jsonl"
test -f "$BIOCODER_AUXILIARY_ROOT/artifact_manifest.sha256"
test -d "$BIOCODER_CHROMPEAKFORMER_SOURCE_ROOT"

actual_report_sha256="$(sha256sum "$BIOCODER_AUXILIARY_ROOT/auxiliary_report.json" | awk '{print $1}')"
test "$actual_report_sha256" = "$BIOCODER_AUXILIARY_REPORT_SHA256"
(
  cd "$BIOCODER_AUXILIARY_ROOT"
  sha256sum -c artifact_manifest.sha256
)
echo "AUXILIARY_INPUT_MANIFEST=OK"

mkdir -p "$scratch/code" "$scratch/input" "$scratch/private-source" "$scratch/assets"
git -C "$BIOCODER_REPO_ROOT" archive "$BIOCODER_CODE_REVISION" | tar -x -C "$scratch/code"
cp -a "$BIOCODER_AUXILIARY_ROOT/." "$scratch/input/"
cp -a "$BIOCODER_CHROMPEAKFORMER_SOURCE_ROOT/." "$scratch/private-source/"

export PYTHONPATH="$scratch/code/backend"
export CHROMPEAKFORMER_SOURCE_ROOT="$scratch/private-source"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$python_bin" - <<'PY'
import importlib.util

required = ("numpy", "pandas", "scipy", "matplotlib", "natsort", "pyopenms")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing extraction dependencies: {missing}")
print("EXTRACTION_DEPENDENCIES=OK", flush=True)
PY

echo "=== EXECUTE AUXILIARY EXTRACTION ==="
"$python_bin" -m multimodal_science.chrompeakformer.execute_cli \
  --plan "$scratch/input/derivation_plan.jsonl" \
  --data-root "$scratch/input" \
  --output-root "$scratch/assets" \
  --extractor "multimodal_science.chrompeakformer.private_adapter:extract_job" \
  --split auxiliary_unlabeled_train

"$python_bin" - "$scratch/assets" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
provenance = sorted(root.glob("jobs/auxiliary_unlabeled_train/*/derivation_provenance.json"))
failures = sorted(root.glob("failures/*.json"))
if len(provenance) != 77:
    raise SystemExit(f"Expected 77 completed frames, found {len(provenance)}")
if failures:
    raise SystemExit(f"Extraction failures were published: {len(failures)}")
for path in provenance:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok" or value.get("split") != "auxiliary_unlabeled_train":
        raise SystemExit(f"Invalid provenance: {path}")
print("AUXILIARY_EXTRACTION_CONTRACT=OK", flush=True)
PY

test ! -e "$persistent_root"
mkdir -p "$(dirname "$persistent_root")"
cp -a "$scratch/assets" "$persistent_root"
printf '%s\n' "$BIOCODER_AUXILIARY_REPORT_SHA256" > "$persistent_root/auxiliary_report.sha256"
printf '%s\n' "$BIOCODER_CODE_REVISION" > "$persistent_root/code_revision.txt"

echo "PERSISTENT_OUTPUT=$persistent_root"
echo "AUXILIARY_COMPLETED_FRAMES=77"
echo "AUXILIARY_LABELS=0"
echo "AUXILIARY_METRICS_ALLOWED=false"
echo "CHROMPEAK_AUXILIARY_EXTRACTION=OK"
