#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077
require() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in BIOCODER_REPO_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION BIOCODER_C0_EVALUATION_ROOT BIOCODER_C1_EVALUATION_ROOT BIOCODER_RL_EVALUATION_ROOT BIOCODER_VALIDATION_METADATA_PATH BIOCODER_VALIDATION_METADATA_SHA256 BIOCODER_RL_LOCALIZATION_OUTPUT BIOCODER_RL_VALIDATION_ACCESS_LEDGER; do require "$name"; done
repo="$BIOCODER_REPO_ROOT"; python_bin="$BIOCODER_PYTHON"; revision="$BIOCODER_CODE_REVISION"; ledger="$BIOCODER_RL_VALIDATION_ACCESS_LEDGER"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]; [[ "$(git -C "$repo" rev-parse HEAD)" == "$revision" ]]
"$python_bin" - "$ledger" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
if "internal_test" in {part.lower().replace("-", "_") for part in root.parts}: raise ValueError("Internal test is forbidden")
completion = root / "access_completed.json"
if not completion.is_file(): raise ValueError("Validation access is not completed and hash-bound")
body = json.loads(completion.read_text(encoding="utf-8"))
if body.get("schema_version") != "chrompeak-qwen3vl-rl-selection-completion-v1" or body.get("internal_test_accessed") is not False:
    raise ValueError("Validation ledger is incomplete")
print("ONE_TIME_VALIDATION_LEDGER=OK")
PY
scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-localization-${SLURM_JOB_ID:-manual}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-localization-* ]] && rm -rf -- "$scratch"; }; trap cleanup EXIT
mkdir -p "$scratch/code"; git -C "$repo" archive "$revision" | tar -xf - -C "$scratch/code"
echo "=== V2 RL LOCALIZATION ANALYSIS ==="; echo "CODE_REVISION=$revision"; echo "INTERNAL_TEST_ACCESSED=false"
env PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m multimodal_science.qwen3vl.analyze_localization_cli \
  --c0-evaluation-root "$BIOCODER_C0_EVALUATION_ROOT" --c1-evaluation-root "$BIOCODER_C1_EVALUATION_ROOT" --rl-evaluation-root "$BIOCODER_RL_EVALUATION_ROOT" \
  --validation-metadata-path "$BIOCODER_VALIDATION_METADATA_PATH" --expected-validation-metadata-sha256 "$BIOCODER_VALIDATION_METADATA_SHA256" --output-dir "$BIOCODER_RL_LOCALIZATION_OUTPUT"
(cd "$BIOCODER_RL_LOCALIZATION_OUTPUT" && sha256sum -c artifact_manifest.sha256)
echo "QWEN3VL_RL_LOCALIZATION_ANALYSIS=OK"
