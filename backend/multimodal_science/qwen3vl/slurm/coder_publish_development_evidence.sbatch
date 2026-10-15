#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Missing required variable: $name" >&2; exit 2; }
}

for name in \
  BIOCODER_REPO_ROOT BIOCODER_RUN_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION \
  BIOCODER_DEVELOPMENT_DOSSIER_ROOT BIOCODER_LORA_TRAINING_ROOT \
  BIOCODER_FUSION_TRAINING_ROOT
do
  require_variable "$name"
done

repo="$BIOCODER_REPO_ROOT"
run_root="$BIOCODER_RUN_ROOT"
python_bin="$BIOCODER_PYTHON"
revision="$BIOCODER_CODE_REVISION"
dossier_root="$BIOCODER_DEVELOPMENT_DOSSIER_ROOT"
lora_root="$BIOCODER_LORA_TRAINING_ROOT"
fusion_root="$BIOCODER_FUSION_TRAINING_ROOT"
auxiliary_root="${BIOCODER_AUXILIARY_PRETRAINING_ROOT:-}"
random_evaluation_root="${BIOCODER_RANDOM_PROJECTOR_EVALUATION_ROOT:-}"
auxiliary_evaluation_root="${BIOCODER_AUXILIARY_PROJECTOR_EVALUATION_ROOT:-}"
xic_only_evaluation_root="${BIOCODER_XIC_ONLY_EVALUATION_ROOT:-}"
sequence_prompt_evaluation_root="${BIOCODER_SEQUENCE_PROMPT_EVALUATION_ROOT:-}"
code_short="${revision:0:7}"
output_root="${BIOCODER_PUBLIC_DEVELOPMENT_OUTPUT_ROOT:-}"
job_id="${SLURM_JOB_ID:-manual}"

[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
test -x "$python_bin"
test -d "$repo/.git"
test -d "$dossier_root"
test -d "$lora_root"
test -d "$fusion_root"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$revision" ]]

optional_count=0
for value in "$auxiliary_root" "$random_evaluation_root" "$auxiliary_evaluation_root"; do
  [[ -n "$value" ]] && optional_count=$((optional_count + 1))
done
[[ "$optional_count" == "0" || "$optional_count" == "3" ]] || {
  echo "All three auxiliary evidence roots must be supplied together" >&2
  exit 2
}
if [[ "$optional_count" == "3" ]]; then
  test -d "$auxiliary_root"
  test -d "$random_evaluation_root"
  test -d "$auxiliary_evaluation_root"
fi

control_count=0
for value in "$xic_only_evaluation_root" "$sequence_prompt_evaluation_root"; do
  [[ -n "$value" ]] && control_count=$((control_count + 1))
done
[[ "$control_count" == "0" || "$control_count" == "2" ]] || {
  echo "Both post-seal control evaluation roots must be supplied together" >&2
  exit 2
}
if [[ "$control_count" == "2" ]]; then
  test -d "$xic_only_evaluation_root"
  test -d "$sequence_prompt_evaluation_root"
fi
if [[ -z "$output_root" ]]; then
  completion_slug="core"
  [[ "$control_count" == "2" ]] && completion_slug="complete"
  output_root="$run_root/comparisons/public-development-evidence-${code_short}-${completion_slug}"
fi

echo "=== PUBLIC DEVELOPMENT EVIDENCE CONTRACT ==="
echo "JOB_ID=$job_id"
echo "JOB_NAME=${SLURM_JOB_NAME:-coder}"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$revision"
echo "SEALED_INTERNAL_TEST_REOPENED=false"

verify_output() {
  (
    cd "$output_root"
    sha256sum -c artifact_manifest.sha256
  )
  "$python_bin" - "$output_root" "$control_count" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_controls = int(sys.argv[2]) == 2
text = (root / "public_development_evidence.json").read_text(encoding="utf-8")
report = json.loads(text)
assert report["schema_version"] == "chrompeak-public-development-evidence-v1"
assert report["contracts"]["three_training_seeds"] == [17, 29, 43]
assert report["contracts"]["sample_standard_deviation_ddof"] == 1
assert set(report["sensor_token_ablation"]) == {"1", "4", "8"}
assert set(report["xic_interventions"]) == {
    "aligned", "shuffled", "zero", "availability-off"
}
assert report["contracts"]["sealed_internal_test_reopened"] is False
assert report["contracts"]["post_seal_controls_complete"] is expected_controls
assert report["internal_test_accessed"] is False
assert report["final_benchmark_eligible"] is False
assert "/home/" not in text and "C:\\" not in text
print("PUBLIC_DEVELOPMENT_REPORT_CONTRACT=OK")
PY
}

if [[ -d "$output_root" ]]; then
  verify_output
  echo "PUBLIC_DEVELOPMENT_EVIDENCE=ALREADY_COMPLETE"
else
  scratch="$(mktemp -d "/tmp/biocoder-${USER}-public-development-${job_id}.XXXXXX")"
  cleanup() {
    case "${scratch:-}" in
      "/tmp/biocoder-${USER}-public-development-${job_id}."*) rm -rf -- "$scratch" ;;
    esac
  }
  trap cleanup EXIT
  mkdir -p "$scratch/code"
  git -C "$repo" archive "$revision" | tar -x -C "$scratch/code"
  export PYTHONPATH="$scratch/code/backend"

  arguments=(
    --development-dossier-root "$dossier_root"
    --lora-training-root "$lora_root"
    --fusion-training-root "$fusion_root"
    --output-dir "$output_root"
  )
  if [[ "$optional_count" == "3" ]]; then
    arguments+=(
      --auxiliary-pretraining-root "$auxiliary_root"
      --random-projector-evaluation-root "$random_evaluation_root"
      --auxiliary-projector-evaluation-root "$auxiliary_evaluation_root"
    )
  fi
  if [[ "$control_count" == "2" ]]; then
    arguments+=(
      --xic-only-evaluation-root "$xic_only_evaluation_root"
      --sequence-prompt-evaluation-root "$sequence_prompt_evaluation_root"
    )
  fi
  "$python_bin" -m multimodal_science.qwen3vl.build_public_development_evidence_cli \
    "${arguments[@]}"
  verify_output
fi

report_sha="$(sha256sum "$output_root/public_development_evidence.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$output_root/artifact_manifest.sha256" | awk '{print $1}')"
echo "PUBLIC_DEVELOPMENT_REPORT_SHA256=$report_sha"
echo "PUBLIC_DEVELOPMENT_MANIFEST_SHA256=$manifest_sha"
echo "PUBLIC_DEVELOPMENT_OUTPUT=$output_root"
cat "$output_root/public_development_evidence.md"
echo "PUBLIC_DEVELOPMENT_EVIDENCE=OK"
