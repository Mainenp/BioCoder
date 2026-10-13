#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077
require() { local name="$1"; [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 2; }; }
for name in BIOCODER_REPO_ROOT BIOCODER_PYTHON BIOCODER_CODE_REVISION BIOCODER_RL_SELECTION_ACTION; do require "$name"; done
repo="$BIOCODER_REPO_ROOT"; python_bin="$BIOCODER_PYTHON"; revision="$BIOCODER_CODE_REVISION"; action="$BIOCODER_RL_SELECTION_ACTION"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]; test -x "$python_bin"; test -d "$repo/.git"; [[ "$(git -C "$repo" rev-parse HEAD)" == "$revision" ]]
case "$action" in build-bundle|build-candidates|evaluate|seal|begin-validation|complete-validation) ;; *) echo "Invalid selection action: $action" >&2; exit 2;; esac
scratch="$(mktemp -d "/tmp/biocoder-${USER}-rl-selection-${SLURM_JOB_ID:-manual}.XXXXXX")"
cleanup() { [[ "$scratch" == /tmp/biocoder-"$USER"-rl-selection-* ]] && rm -rf -- "$scratch"; }; trap cleanup EXIT
mkdir -p "$scratch/code"; git -C "$repo" archive "$revision" | tar -xf - -C "$scratch/code"
args=("$action")
case "$action" in
  build-bundle)
    for name in BIOCODER_RL_POOL_ROOT BIOCODER_RL_POOL_REPORT_SHA256 BIOCODER_RL_SELECTION_OUTPUT; do require "$name"; done
    args+=(--pool-root "$BIOCODER_RL_POOL_ROOT" --pool-report-sha256 "$BIOCODER_RL_POOL_REPORT_SHA256" --output-dir "$BIOCODER_RL_SELECTION_OUTPUT" --code-revision "$revision") ;;
  build-candidates)
    for name in BIOCODER_RL_POOL_REPORT_SHA256 BIOCODER_INITIAL_ADAPTER_ROOT BIOCODER_INITIAL_ADAPTER_REPORT_SHA256 BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256 BIOCODER_RL_CANDIDATE_RUN_SPEC BIOCODER_RL_SELECTION_OUTPUT; do require "$name"; done
    echo "=== V2 RL CANDIDATE MATRIX CONTRACT ==="
    env PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m multimodal_science.qwen3vl.build_rl_candidate_matrix_cli \
      --code-revision "$revision" --rl-pool-report-sha256 "$BIOCODER_RL_POOL_REPORT_SHA256" \
      --c0-root "$BIOCODER_INITIAL_ADAPTER_ROOT" --c0-report-sha256 "$BIOCODER_INITIAL_ADAPTER_REPORT_SHA256" \
      --c0-manifest-sha256 "$BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256" --run-spec "$BIOCODER_RL_CANDIDATE_RUN_SPEC" \
      --output-dir "$BIOCODER_RL_SELECTION_OUTPUT"
    echo "QWEN3VL_RL_SELECTION_ACTION=OK"
    exit 0 ;;
  evaluate)
    for name in BIOCODER_RL_SELECTION_BUNDLE_ROOT BIOCODER_RL_SELECTION_BUNDLE_SHA256 BIOCODER_RL_PREDICTIONS BIOCODER_RL_CANDIDATE_JSON BIOCODER_RL_SELECTION_OUTPUT; do require "$name"; done
    args+=(--bundle-root "$BIOCODER_RL_SELECTION_BUNDLE_ROOT" --bundle-report-sha256 "$BIOCODER_RL_SELECTION_BUNDLE_SHA256" --predictions "$BIOCODER_RL_PREDICTIONS" --candidate-json "$BIOCODER_RL_CANDIDATE_JSON" --output-dir "$BIOCODER_RL_SELECTION_OUTPUT") ;;
  seal)
    for name in BIOCODER_RL_SELECTION_BUNDLE_ROOT BIOCODER_RL_SELECTION_BUNDLE_SHA256 BIOCODER_RL_C0_EVALUATION BIOCODER_RL_EVALUATION_LIST BIOCODER_C1_EVALUATION_LIST BIOCODER_RL_SELECTION_OUTPUT; do require "$name"; done
    args+=(--bundle-root "$BIOCODER_RL_SELECTION_BUNDLE_ROOT" --bundle-report-sha256 "$BIOCODER_RL_SELECTION_BUNDLE_SHA256" --c0-evaluation "$BIOCODER_RL_C0_EVALUATION" --output-dir "$BIOCODER_RL_SELECTION_OUTPUT" --code-revision "$revision")
    while IFS= read -r path; do [[ -n "$path" ]] && args+=(--rl-evaluation "$path"); done < "$BIOCODER_RL_EVALUATION_LIST"
    while IFS= read -r path; do [[ -n "$path" ]] && args+=(--c1-evaluation "$path"); done < "$BIOCODER_C1_EVALUATION_LIST" ;;
  begin-validation|complete-validation)
    for name in BIOCODER_RL_SELECTION_ROOT BIOCODER_RL_SELECTION_RESULT_SHA256 BIOCODER_RL_VALIDATION_LEDGER_ROOT; do require "$name"; done
    args+=(--selection-root "$BIOCODER_RL_SELECTION_ROOT" --selection-result-sha256 "$BIOCODER_RL_SELECTION_RESULT_SHA256" --ledger-root "$BIOCODER_RL_VALIDATION_LEDGER_ROOT") ;;
esac
echo "=== V2 RL SELECTION CONTRACT ==="; echo "ACTION=$action"; echo "CODE_REVISION=$revision"
echo "INTERNAL_TEST_ACCESSED=false"
if [[ "$action" != begin-validation && "$action" != complete-validation ]]; then echo "VALIDATION_ANSWERS_OPENED=false"; fi
env PYTHONPATH="$scratch/code/backend" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m multimodal_science.qwen3vl.select_rl_candidate_cli "${args[@]}"
echo "QWEN3VL_RL_SELECTION_ACTION=OK"
