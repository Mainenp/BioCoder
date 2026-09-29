#!/usr/bin/env bash
#SBATCH --job-name=coder
#SBATCH --partition=GPU-5090
#SBATCH --nodelist=gnode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

repo_root="${BIOCODER_REPO_ROOT:?Set BIOCODER_REPO_ROOT}"
python_bin="${BIOCODER_PYTHON:?Set BIOCODER_PYTHON}"
code_revision="${BIOCODER_CODE_REVISION:?Set BIOCODER_CODE_REVISION}"
run_root="${BIOCODER_RUN_ROOT:?Set BIOCODER_RUN_ROOT}"
model_source="${BIOCODER_MODEL_ROOT:?Set BIOCODER_MODEL_ROOT}"
model_manifest="${BIOCODER_MODEL_MANIFEST_PATH:?Set BIOCODER_MODEL_MANIFEST_PATH}"
model_manifest_sha256="${BIOCODER_MODEL_MANIFEST_SHA256:?Set BIOCODER_MODEL_MANIFEST_SHA256}"
fusion_source="${BIOCODER_FUSION_BUNDLE_ROOT:?Set BIOCODER_FUSION_BUNDLE_ROOT}"
fusion_report_sha256="${BIOCODER_FUSION_BUNDLE_REPORT_SHA256:?Set BIOCODER_FUSION_BUNDLE_REPORT_SHA256}"
lora_source="${BIOCODER_LORA_BUNDLE_ROOT:?Set BIOCODER_LORA_BUNDLE_ROOT}"
lora_report_sha256="${BIOCODER_LORA_BUNDLE_REPORT_SHA256:?Set BIOCODER_LORA_BUNDLE_REPORT_SHA256}"
dataset_source="${BIOCODER_DATASET_ROOT:?Set BIOCODER_DATASET_ROOT}"
dataset_report_sha256="${BIOCODER_DATASET_REPORT_SHA256:?Set BIOCODER_DATASET_REPORT_SHA256}"
assets_source="${BIOCODER_ASSETS_ROOT:?Set BIOCODER_ASSETS_ROOT}"
initial_adapter_root="${BIOCODER_INITIAL_ADAPTER_ROOT:?Set BIOCODER_INITIAL_ADAPTER_ROOT}"
initial_adapter_report_sha256="${BIOCODER_INITIAL_ADAPTER_REPORT_SHA256:?Set BIOCODER_INITIAL_ADAPTER_REPORT_SHA256}"
initial_adapter_manifest_sha256="${BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256:?Set BIOCODER_INITIAL_ADAPTER_MANIFEST_SHA256}"
model_revision="${BIOCODER_MODEL_REVISION:?Set BIOCODER_MODEL_REVISION}"
model_artifact_sha256="${BIOCODER_MODEL_ARTIFACT_SHA256:?Set BIOCODER_MODEL_ARTIFACT_SHA256}"
gpu_index="${BIOCODER_GPU_INDEX:-7}"
job_id="${SLURM_JOB_ID:-unscheduled}"
job_name="${SLURM_JOB_NAME:-coder}"
model_short="${model_artifact_sha256:0:16}"

actual_revision="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$actual_revision" == "$code_revision" ]] || {
  echo "Code revision mismatch: expected=$code_revision actual=$actual_revision" >&2
  exit 2
}
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  echo "Repository must be clean before an immutable fusion smoke" >&2
  exit 3
}
[[ "$model_manifest_sha256" =~ ^[0-9a-f]{64}$ ]]

scratch="$(mktemp -d "/tmp/biocoder-${USER}-qwen3vl-fusion-${job_id}.XXXXXX")"
publish_staging=""
cleanup() {
  case "${scratch:-}" in
    "/tmp/biocoder-${USER}-qwen3vl-fusion-${job_id}."*) rm -rf -- "$scratch" ;;
  esac
  if [[ -n "${publish_staging:-}" && -d "$publish_staging" ]]; then
    rm -rf -- "$publish_staging"
  fi
}
trap cleanup EXIT
mkdir -p "$scratch/code"
git -C "$repo_root" archive "$code_revision" | tar -x -C "$scratch/code"

echo "=== STAGE TRAIN-ONLY INPUT ROOTS ==="
train_view="$scratch/train-only"
fusion_root="$train_view/fusion"
lora_root="$train_view/lora"
dataset_root="$train_view/dataset"
assets_root="$train_view/assets"
"$python_bin" - \
  "$fusion_source" "$fusion_root" \
  "$lora_source" "$lora_root" \
  "$dataset_source" "$dataset_root" \
  "$assets_source" "$assets_root" <<'PY'
import json
import shutil
import sys
from pathlib import Path, PurePosixPath


def safe_child(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Unsafe staged path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe staged path: {relative}")
    candidate = (root / Path(*pure.parts)).resolve()
    candidate.relative_to(root)
    return candidate


def copy_file(source_root: Path, target_root: Path, relative: object) -> Path:
    source = safe_child(source_root, relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    target = safe_child(target_root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def stage_report(
    source_root: Path,
    target_root: Path,
    report_name: str,
    artifact_keys: tuple[str, ...],
) -> dict:
    source_report = source_root / report_name
    report = json.loads(source_report.read_text(encoding="utf-8"))
    target_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_report, target_root / report_name)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Missing artifacts in {source_report}")
    for key in artifact_keys:
        item = artifacts.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Missing artifact {key} in {source_report}")
        copy_file(source_root, target_root, item.get("path"))
    return report


fusion_source, fusion_target = map(Path, sys.argv[1:3])
lora_source, lora_target = map(Path, sys.argv[3:5])
dataset_source, dataset_target = map(Path, sys.argv[5:7])
assets_source, assets_target = map(Path, sys.argv[7:9])
fusion_source = fusion_source.resolve()
lora_source = lora_source.resolve()
dataset_source = dataset_source.resolve()
assets_source = assets_source.resolve()

stage_report(
    fusion_source,
    fusion_target,
    "fusion_bundle_report.json",
    ("train_xic_links",),
)
lora_report = stage_report(
    lora_source,
    lora_target,
    "lora_bundle_report.json",
    ("train_qwen", "selection_manifest"),
)
stage_report(
    dataset_source,
    dataset_target,
    "dataset_report.json",
    ("train_signals", "train_examples"),
)

train_item = lora_report["artifacts"]["train_qwen"]
train_path = safe_child(lora_target, train_item["path"])
assets_target.mkdir(parents=True, exist_ok=False)
with train_path.open(encoding="utf-8") as stream:
    for index, line in enumerate(stream):
        if index >= 8:
            break
        row = json.loads(line)
        copy_file(assets_source, assets_target, row.get("image"))

for forbidden in (
    fusion_target / "validation_xic_links.jsonl",
    dataset_target / "validation",
    assets_target / "jobs" / "validation",
):
    if forbidden.exists():
        raise RuntimeError(f"Validation artifact entered train-only view: {forbidden}")
print("STAGED_TRAIN_ONLY_INPUT_ROOTS=OK", flush=True)
PY

echo "=== STAGE HASH-VERIFIED MODEL CACHE ==="
model_cache_root="/tmp/biocoder-${USER}-models"
local_model="$model_cache_root/qwen3vl-${model_short}"
mkdir -p "$model_cache_root"
exec 8>"$model_cache_root/qwen3vl-${model_short}.lock"
flock 8
verify_model_cache() {
  (
    cd "$scratch/code/backend"
    PYTHONNOUSERSITE=1 PYTHONPATH="$scratch/code/backend" "$python_bin" - \
      "$local_model" "$model_manifest" "$model_manifest_sha256" <<'PY'
import sys
from pathlib import Path

from multimodal_science.qwen3vl.fusion_smoke import _verify_model_manifest

inventory = _verify_model_manifest(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
print(f"LOCAL_MODEL_FILES={inventory['files']}")
print(f"LOCAL_MODEL_BYTES={inventory['bytes']}")
PY
  )
}
if [[ -d "$local_model" ]] && ! verify_model_cache; then
  [[ "$local_model" == "$model_cache_root"/qwen3vl-* ]] || exit 6
  rm -rf -- "$local_model"
fi
if [[ ! -d "$local_model" ]]; then
  mkdir -p "$scratch/model-staging"
  cp -a "$model_source/." "$scratch/model-staging/"
  mv "$scratch/model-staging" "$local_model"
fi
verify_model_cache
flock -u 8

lock_path="/tmp/biocoder-${USER}-gpu-${gpu_index}.lock"
exec 9>"$lock_path"
flock -n 9 || { echo "GPU lock already held: $lock_path" >&2; exit 4; }
for sample in 1 2 3
do
  line="$(nvidia-smi -i "$gpu_index" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  used="${line%%,*}"
  util="${line##*,}"
  used="${used// /}"
  util="${util// /}"
  echo "GPU=$gpu_index SAMPLE=$sample USED_MIB=$used UTIL=$util"
  (( used <= 512 && util <= 10 )) || { echo "GPU failed startup guard" >&2; exit 5; }
  sleep 2
done

allocation_mode="manual_physical_index_guard_no_slurm_gres"
input_scope="staged_train_artifacts_only"
run_name="image-xic-smoke-${code_revision:0:7}-job${job_id}"
publish_parent="$run_root/qwen3vl/fusion/runs"
final_output="$publish_parent/$run_name"
smoke_output="$scratch/output"
mkdir -p "$scratch/tmp" "$scratch/cache"
for variable in "${!BIOCODER_@}"
do
  unset "$variable"
done

echo "JOB_ID=$job_id"
echo "JOB_NAME=$job_name"
echo "NODE=$(hostname -s)"
echo "CODE_REVISION=$actual_revision"
echo "SELECTED_PHYSICAL_GPU=$gpu_index"
echo "GPU_ALLOCATION_MODE=$allocation_mode"
echo "TRAIN_INPUT_SCOPE=$input_scope"
echo "INTERNAL_TEST_ACCESSED=false"

(
  cd "$scratch/code/backend"
  env -i \
    HOME="$HOME" \
    USER="$USER" \
    PATH="$(dirname "$python_bin"):/usr/bin:/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    LANG="${LANG:-C.UTF-8}" \
    TMPDIR="$scratch/tmp" \
    XDG_CACHE_HOME="$scratch/cache" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$scratch/code/backend" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=0 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" \
    SLURM_JOB_ID="$job_id" \
    SLURM_JOB_NAME="$job_name" \
    BIOCODER_GPU_ALLOCATION_MODE="$allocation_mode" \
    BIOCODER_TRAIN_INPUT_SCOPE="$input_scope" \
    BIOCODER_VERIFIED_CODE_REVISION="$actual_revision" \
    "$python_bin" -m multimodal_science.qwen3vl.run_fusion_smoke_cli \
      --fusion-bundle-root "$fusion_root" \
      --fusion-bundle-report-sha256 "$fusion_report_sha256" \
      --lora-bundle-root "$lora_root" \
      --lora-bundle-report-sha256 "$lora_report_sha256" \
      --dataset-root "$dataset_root" \
      --dataset-report-sha256 "$dataset_report_sha256" \
      --assets-root "$assets_root" \
      --initial-adapter-root "$initial_adapter_root" \
      --initial-adapter-report-sha256 "$initial_adapter_report_sha256" \
      --initial-adapter-manifest-sha256 "$initial_adapter_manifest_sha256" \
      --output-dir "$smoke_output" \
      --model-name-or-path "$local_model" \
      --model-revision "$model_revision" \
      --model-artifact-sha256 "$model_artifact_sha256" \
      --model-manifest-path "$model_manifest" \
      --model-manifest-sha256 "$model_manifest_sha256" \
      --code-revision "$code_revision" \
      --max-records 8 \
      --max-steps 2 \
      --seed 17
)

(
  cd "$smoke_output"
  sha256sum -c artifact_manifest.sha256
)

"$python_bin" - "$smoke_output/fusion_smoke_report.json" <<'PY'
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
require(
    report["schema_version"] == "chrompeak-qwen3vl-xic-fusion-smoke-v2",
    "Unexpected fusion smoke schema",
)
require(report["training"]["optimizer_updates"] == 2, "Bad optimizer update count")
require(report["training"]["processed_records"] == 2, "Bad processed-record count")
require(report["training"]["availability_counts"]["available"] >= 1, "No measured XIC")
state = report["training"]["parameter_state_sha256"]
require(state["lora_before"] != state["lora_after"], "LoRA did not update")
require(state["lora_cleared_before_reload"] != state["lora_after"], "Bad reload control")
require(state["lora_reloaded"] == state["lora_after"], "LoRA reload mismatch")
require(state["projector_before"] != state["projector_after"], "Projector did not update")
binding = report["sources"]["fusion_lora_content_binding"]
require(binding["mode"] == "training_artifact_sha256", "Bad LoRA binding mode")
require(len(binding["train_qwen_sha256"]) == 64, "Bad bound train-row hash")
require(len(binding["selection_manifest_sha256"]) == 64, "Bad bound selection hash")
require(
    binding["train_qwen_sha256"] == report["sources"]["adapter_train_rows_sha256"],
    "Adapter/fusion train-row binding mismatch",
)
for contract in (
    "image_and_xic_forward",
    "vision_tower_executed",
    "native_multimodal_rope_positions",
    "lora_and_projector_backward",
    "named_gradient_groups_verified",
    "parameter_updates_verified",
    "adapter_state_reloaded",
    "projector_reloaded",
    "reproducibility_seeded",
    "staged_train_only_input_roots",
    "fusion_lora_training_artifacts_bound",
    "initial_adapter_training_rows_bound",
):
    require(report["contracts"][contract] is True, f"Contract failed: {contract}")
require(
    report["contracts"]["bitwise_determinism_claimed"] is False,
    "Bitwise determinism must not be claimed",
)
require(report["contracts"]["validation_answers_opened"] is False, "Answers accessed")
require(report["internal_test_accessed"] is False, "Internal test accessed")
require(
    report["runtime"]["gpu_allocation_mode"]
    == "manual_physical_index_guard_no_slurm_gres",
    "GPU allocation provenance mismatch",
)
require(
    report["runtime"]["train_input_scope"] == "staged_train_artifacts_only",
    "Train input-scope provenance mismatch",
)
print("QWEN3VL_XIC_FORWARD_BACKWARD_CONTRACT=OK")
print(f"LORA_REPORT_SHA256_MATCH={binding['bundle_report_sha256_match']}")
PY

verify_exact_manifest() {
  "$python_bin" - "$1" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "artifact_manifest.sha256"
listed = {
    line.split("  ", maxsplit=1)[1]
    for line in manifest.read_text(encoding="utf-8").splitlines()
}
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != manifest
}
if listed != actual:
    raise RuntimeError(
        f"Manifest coverage mismatch: missing={sorted(actual - listed)} "
        f"extra={sorted(listed - actual)}"
    )
print("FUSION_OUTPUT_EXACT_MANIFEST=OK")
PY
}
verify_exact_manifest "$smoke_output"

mkdir -p "$publish_parent"
[[ ! -e "$final_output" ]] || { echo "Final output already exists: $final_output" >&2; exit 7; }
publish_staging="$(mktemp -d "$publish_parent/.${run_name}.incomplete.XXXXXX")"
cp -a "$smoke_output/." "$publish_staging/"
(
  cd "$publish_staging"
  sha256sum -c artifact_manifest.sha256
)
verify_exact_manifest "$publish_staging"
mv "$publish_staging" "$final_output"
publish_staging=""

echo "FUSION_SMOKE_OUTPUT=$final_output"
echo "ARTIFACT_MANIFEST_SHA256=$(sha256sum "$final_output/artifact_manifest.sha256" | awk '{print $1}')"
echo "MODEL_MANIFEST_SHA256=$model_manifest_sha256"
echo "QWEN3VL_XIC_FUSION_SMOKE=OK"
