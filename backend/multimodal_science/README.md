# Multimodal science substrate

This package is the HF/CUDA-independent data, training, and evaluation boundary for the
ChromPeakFormer roadmap. It does not call BioCoder's text/MLX SFT coordinator.

## Phase A data audit

The first implemented command builds a deterministic, eligibility-gated manifest from an
authorized raw-data directory:

```powershell
$env:PYTHONPATH = "backend"
python -m multimodal_science.data `
  --data-root "<extracted-data-root>" `
  --output-dir "work/chrompeak/audit/<dataset-version>" `
  --source-archive-sha256 "<64-character-sha256>"
```

The command writes:

- `manifest.jsonl`: one traceable record per label row.
- `audit_report.json`: eligibility counts, label distributions, signal probes, duplicate hashes,
  unlabelled sources, and audit reasons.

Generated data belongs under `work/`, which is ignored by Git. Raw data, labels, manifests, and
reports must not be committed unless their publication has been reviewed separately.

## Eligibility rules

A record enters the primary train and benchmark populations only when:

- `sample_id` matches exactly one mzML basename inside the corresponding dataset directory;
- the mzML contains at least one chromatogram signal;
- `peak_label` is `0` or `1`;
- positive labels contain the declared number of valid peak intervals; and
- the workbook is the primary label variant.

No row-order or sequence fallback is used. Weak, missing, ambiguous, alternate, or invalid records
remain in the manifest with `audit_bucket=true` and explicit `exclusion_reasons`.

Some older mzML exporters declare UTF-8 while embedding non-UTF-8 bytes in text metadata. The
probe first performs strict XML parsing, then uses replacement decoding only to verify the
chromatogram structure. These files are marked `source_signal_status=recovered`; raw bytes are
never rewritten.

## Phase A-plus split contract

The split builder consumes the eligible manifest and assigns entire source mzML groups instead of
individual ROI rows:

```powershell
$env:PYTHONPATH = "backend"
python -m multimodal_science.data.split_cli `
  --manifest "work/chrompeak/audit/<dataset-version>/manifest.jsonl" `
  --audit-report "work/chrompeak/audit/<dataset-version>/audit_report.json" `
  --output-dir "work/chrompeak/splits/<dataset-version>"
```

`traindata3` groups are deterministically stratified as `blank`, `qc`, or `sample` before the
80/10/10 train, validation, and internal-test allocation. Duplicate mzML content under different
group names is rejected. The report also proves zero protected split-group overlap, zero content
hash overlap, and zero audit-record contamination.

Other populations are deliberately not blended into the primary benchmark:

- `traindata1` and `traindata2` are `auxiliary_train` because they are small negative-only sets;
- `test1` is `legacy_external_non_pristine`, not a newly collected blind test set;
- alternate `test1` labels remain `audit_only`; and
- unlabelled `test2` files are inference-only and cannot produce supervised metrics.

## ChromPeakFormer derivation preflight

The derivation builder converts the split manifest into one hash-verified job per source mzML. A job
contains only traceable label inputs and relative output contracts for ROI images, `feature.csv`,
`roi_windows.csv`, and `xic_matrix.npy`:

```powershell
$env:PYTHONPATH = "backend"
python -m multimodal_science.data.derive_cli `
  --split-manifest "work/chrompeak/splits/<dataset-version>/split_manifest.jsonl" `
  --audit-report "work/chrompeak/audit/<dataset-version>/audit_report.json" `
  --data-root "<extracted-data-root>" `
  --output-dir "work/chrompeak/derivation/<dataset-version>"
```

### Auxiliary vendor-data import

New vendor containers without human peak labels never enter the supervised train, validation, or
benchmark populations. After an authorized converter has produced one mzML per acquisition frame,
bind the conversion to its redacted quarantine inventory with:

```bash
python -m multimodal_science.chrompeakformer.auxiliary_msdata_cli \
  --quarantine-manifest "<quarantine>/msdata_manifest.jsonl" \
  --quarantine-report "<quarantine>/msdata_quarantine_report.json" \
  --converted-root "<converted-msdata-root>" \
  --converter "<authorized-converter-binary>" \
  --output-dir "<external-run-root>/auxiliary-msdata"
```

The importer requires an exact source-group/frame inventory, checks every converted chromatogram
count against the quarantined transition table, hashes the converter and every mzML, and emits an
inference-only `derivation_plan.jsonl`. Vendor output containing recoverable non-UTF-8 metadata is
normalized into a hash-bound UTF-8 training copy and recorded explicitly. Every record is fixed to
`auxiliary_unlabeled_train`, has no metric eligibility, and cannot change validation, benchmark, or
internal-test membership. The resulting mzML and reports stay outside Git.

After transferring that external output to the Linux host, submit
`chrompeakformer/slurm/coder_auxiliary_extract.sbatch`. The job stages code, normalized mzML, and
the authorized private extractor to node-local storage; disables CUDA; verifies all 80 input
artifact hashes; and publishes results only when all 77 frame jobs finish without a failure record.
Before extraction it imports the private entrypoint and binds a caller-supplied fingerprint over
every Python source file beneath the private model root, preventing an undeclared package import
from escaping provenance or failing only after frame processing begins.
The persistent extraction output is still unlabeled and must pass a later auxiliary-asset index
before any self-supervised or weak-supervision objective can consume it.

After all extraction jobs pass, build that separate index with:

```bash
python -m multimodal_science.chrompeakformer.auxiliary_index_cli \
  --plan "<auxiliary-import-root>/derivation_plan.jsonl" \
  --assets-root "<auxiliary-extraction-root>" \
  --output-dir "<external-run-root>/auxiliary-index-v1"
```

The indexer rejects partial plans, labels, metric eligibility, validation membership, and benchmark
membership. It verifies every extraction provenance record and output signature, binds each JPEG
and XIC matrix by hash, and reports the actual number of extracted transition traces and independent
source groups. It deliberately emits no COCO file or target. The scheduled equivalent is
`chrompeakformer/slurm/coder_auxiliary_index.sbatch`; it additionally checks the frame and
source-group counts against the immutable vendor-import report, bounds the actual indexed trace
count by the transition-candidate count, and reports extractor exclusions or Q1/Q3 deduplication
instead of silently treating candidates as usable training assets.

The accepted September 2026 import contains 4 independent source groups, 77 acquisition frames,
and 1,610 usable traces from 1,610 candidates, with zero extraction failures or exclusions. These
are still **zero-label auxiliary inputs**, not an increase to the 14,355 supervised train assets.
Materialize them for signal-representation pretraining with:

```bash
python -m multimodal_science.qwen3vl.build_auxiliary_pretraining_data_cli \
  --auxiliary-index-root "<auxiliary-index-v1>" \
  --auxiliary-index-report-sha256 "<expected-report-sha256>" \
  --auxiliary-index-manifest-sha256 "<expected-manifest-sha256>" \
  --assets-root "<auxiliary-extraction-root>" \
  --output-dir "<external-run-root>/qwen3vl/auxiliary/datasets/signal-v1"
```

This step re-verifies every image and XIC hash, crops and resamples the raw RT-coordinate signal to
160 points, applies the same baseline/log/max normalization as the supervised Dataset, retains
constant traces with an explicit unavailable flag, and emits no target. The image and XIC are
recorded as derived views of the same trace; the report explicitly forbids an independence claim.
Vendor RT axes are normalized before interpolation by stable sorting and collapsing exact duplicate
timestamps with a maximum-intensity reducer. Reordered matrices, duplicate counts, and the largest
backward RT step are retained in the report and surfaced as a warning; an axis with fewer than two
unique timestamps remains a hard failure.

`qwen3vl/slurm/coder_auxiliary_pretrain.sbatch` then trains only the morphology portion of the XIC
sensor projector with deterministic paired augmentations and symmetric InfoNCE. The Qwen model,
vision tower, validation data, and answer keys are not loaded. Training loss is diagnostic only.
A complete projector can initialize formal fusion by setting all three variables together:

```bash
export BIOCODER_PRETRAINED_PROJECTOR_ROOT="<complete-auxiliary-run>"
export BIOCODER_PRETRAINED_PROJECTOR_REPORT_SHA256="<expected-report-sha256>"
export BIOCODER_PRETRAINED_PROJECTOR_MANIFEST_SHA256="<expected-manifest-sha256>"
```

The formal fusion launcher stages this artifact into its train-only view and refuses incomplete,
calibration, schema-drifted, spec-mismatched, or hash-drifted projector weights. The controlled
ablation must compare the existing random-projector fusion against this auxiliary-initialized
fusion under the same labeled training rows, validation prompts, seed, and evaluator.

The report records whether the runtime provides NumPy, Pandas, SciPy, Matplotlib, natsort, and
pyOpenMS. It does not install them. A blocked dependency gate means the plan is valid but
extraction has not run.

## Atomic extraction execution

The execution layer accepts a configured ChromPeakFormer callable with the signature
`extract_job(job, source_mzml, staging_dir)`. The built-in private adapter loads an authorized
ChromPeakFormer source tree from an environment variable; private code and absolute paths stay
outside this repository. The callable writes its outputs only to the supplied staging directory:

```powershell
$env:PYTHONPATH = "backend"
$env:CHROMPEAKFORMER_SOURCE_ROOT = "<authorized-private-source-root>"
$env:CHROMPEAKFORMER_SMOOTH_SIGMA = "1.0"
$env:MPLBACKEND = "Agg"
$env:CUDA_VISIBLE_DEVICES = ""
python -m multimodal_science.chrompeakformer.execute_cli `
  --plan "work/chrompeak/derivation/<dataset-version>/derivation_plan.jsonl" `
  --data-root "<extracted-data-root>" `
  --output-root "work/chrompeak/assets/<dataset-version>" `
  --extractor "multimodal_science.chrompeakformer.private_adapter:extract_job" `
  --split train `
  --max-jobs 1
```

Before calling the extractor, the runner verifies the source file hash and the complete scientific
dependency gate. After extraction it validates:

- required feature and RT-window CSV columns;
- one non-empty JPEG per RT window;
- a valid, non-truncated two-dimensional NumPy file without importing NumPy;
- `feature rows == RT windows == XIC matrix rows - 1`; and
- at least two RT points in every numerical sequence.

Only a fully valid staging directory is atomically promoted. Repeat runs verify provenance and
output hashes before returning a cache hit. Dependency, source, tool, and validation failures write
structured failure records under `failures/` and never publish partial assets.

The pinned CPU extraction environment is recorded in
`chrompeakformer/environment.yml`. Create it with `conda env create --file` on a fresh host, or
compare its exact versions with an existing environment before running extraction. The adapter
accepts a private root containing either `model/preprocessing/xic_extraction.py` or
`preprocessing/xic_extraction.py`. Label-driven jobs fail closed on missing component, channel, or
RT values; inference jobs deliberately call the source extractor without labels.

On a Linux extraction host, keep the private source and generated assets outside the repository:

```bash
conda env create --file backend/multimodal_science/chrompeakformer/environment.yml
conda activate biocoder_chrompeak
export PYTHONPATH="$PWD/backend"
export CHROMPEAKFORMER_SOURCE_ROOT="<authorized-private-source-root>"
export CHROMPEAKFORMER_SMOOTH_SIGMA="1.0"
export MPLBACKEND="Agg"
export CUDA_VISIBLE_DEVICES=""
python -m multimodal_science.chrompeakformer.execute_cli \
  --plan "<derivation-output>/derivation_plan.jsonl" \
  --data-root "<extracted-data-root>" \
  --output-root "<external-asset-root>" \
  --extractor "multimodal_science.chrompeakformer.private_adapter:extract_job" \
  --split train \
  --max-jobs 1
```

Successful provenance includes a SHA-256 fingerprint of the private extraction entry point and its
two mzML-loading helpers, but never stores the private source path.

## Verified ROI/XIC/COCO asset index

The asset-index builder joins each published ROI image and XIC signal row back to exactly one
derivation-plan `record_id`. It verifies job provenance, output hashes, 400x300 JPEG dimensions,
native-id label matching, RT windows, and positive peak visibility before writing a training index:

```bash
python -m multimodal_science.chrompeakformer.index_cli \
  --plan "<derivation-output>/derivation_plan.jsonl" \
  --assets-root "<external-asset-root>" \
  --output-dir "<external-index-root>" \
  --split train
```

The command writes `asset_index.jsonl`, `asset_index_report.json`, and one COCO JSON file per
selected split. Images are referenced relative to the external asset root and are not copied into
the repository. Negative ROIs remain COCO images without annotations. Positive RT intervals map
linearly into full-height bounding boxes on the 400x300 ROI. The builder never falls back to label
row order. Use `--allow-partial` only for an explicitly partial pilot; full builds fail when any
selected extraction job is missing. Long NFS-backed builds report one verified job at a time to
standard error while reserving standard output for the final machine-readable JSON result.

## Training-readiness QA

Before a baseline or Qwen3-VL run consumes the index, build a deterministic readiness report:

```bash
python -m multimodal_science.chrompeakformer.readiness_cli \
  --index "<external-index-root>/asset_index.jsonl" \
  --index-report "<external-index-root>/asset_index_report.json" \
  --output "<external-index-root>/training_readiness.json"
```

This pass streams the JSONL index and does not reopen the ROI images or XIC arrays. It fails closed
on a partial or tampered index, inconsistent declared counts, duplicate identities, non-metric
records, missing train/validation splits, or source artifacts crossing protected splits. The report
captures split and class balance, source-job and component coverage, XIC length, ROI width, positive
peak width, and COCO-box width distributions. It is descriptive data evidence, not a model metric.

The full-trace XIC length is not the sequence-model input length. Profile the actual ROI crops before
choosing a resampling size:

```bash
python -m multimodal_science.chrompeakformer.sequence_preflight_cli \
  --index "<external-index-root>/asset_index.jsonl" \
  --readiness-report "<external-index-root>/training_readiness.json" \
  --assets-root "<external-asset-root>" \
  --output "<external-index-root>/sequence_preflight.json"
```

The preflight memory-maps each referenced XIC matrix once, verifies matrix shape, unique signal-row
alignment, finite ROI values, strictly increasing RT axes, and multi-point ROI coverage. It reports
cropped point counts, sampling intervals, crop fractions, dynamic ranges, constant signals, and
negative values. Because scheduled acquisition can create clustered sub-cycle RT points, the report
separates raw adjacent-axis steps from the effective average step inside each ROI. Sequence
materialization must interpolate against the RT values themselves rather than resize by array index.
Progress is written to standard error; the final JSON remains on standard output.

## Unified multimodal Dataset

After the v2 sequence preflight passes, materialize the model-facing train and validation arrays:

```bash
python -m multimodal_science.chrompeakformer.materialize_cli \
  --index "<external-index-root>/asset_index.jsonl" \
  --readiness-report "<external-index-root>/training_readiness.json" \
  --sequence-preflight "<external-index-root>/sequence_preflight.json" \
  --assets-root "<external-asset-root>" \
  --output-dir "<external-dataset-root>/multimodal-v1" \
  --target-points 160
```

The builder interpolates every signal on 160 uniformly spaced RT coordinates inside its declared
ROI. It applies per-ROI fifth-percentile baseline correction, nonnegative clipping, `log1p`, and
shape normalization. Absolute scale is retained as `log1p` maximum and dynamic-range scalar
features. Q1, Q3, expected RT, ROI width, and signal availability complete the scalar vector; its
first six columns are standardized using train-only statistics.

Each split contains `signals.npy`, `scalar_features.npy`, `targets.npy`, and `examples.jsonl`.
Targets are `[peak_present, start_normalized, end_normalized]`; negative boundaries use `-1` only in
the array and remain `null` in JSON. Image and numerical boundaries must agree in the same `[0, 1]`
ROI coordinate system. Images are referenced by their verified relative paths and are not copied.
All files are staged and atomically published together, and repeat runs verify artifact hashes
before returning a cache hit.

## ChromPeakFormer specialist-detector baseline

The specialist detector is evaluated before the sequence-only ablation and any domain-adapted
Qwen3-VL run. Public BioCoder code owns the split and evidence boundary while the authorized model
source remains outside Git. First convert the verified index into the exact `train`/`val` layout
expected by the detector:

```bash
python -m multimodal_science.chrompeakformer.prepare_detector_cli \
  --asset-index "<external-index-root>/asset_index.jsonl" \
  --asset-index-report "<external-index-root>/asset_index_report.json" \
  --assets-root "<external-asset-root>" \
  --train-coco "<external-index-root>/train_coco.json" \
  --validation-coco "<external-index-root>/validation_coco.json" \
  --output-dir "<external-run-root>/detector-dataset" \
  --verify-image-hashes
```

The adapter rejects partial indices, image/annotation drift, duplicate identities, and any source
`job_id` shared by train and validation. It rewrites only the model-facing image locations; labels,
image IDs, boxes, and split membership must exactly match the immutable asset index.

Train through the audited external-source launcher. The source config must describe the three-layer
boundary-refinement variant; a resume checkpoint is accepted only with an explicit SHA-256:

```bash
python -m multimodal_science.chrompeakformer.train_detector_cli \
  --source-root "<authorized-detector-source>/model" \
  --source-config "<authorized-detector-config>" \
  --detector-dataset-root "<external-run-root>/detector-dataset/coco" \
  --detector-dataset-report "<external-run-root>/detector-dataset/detector_dataset_report.json" \
  --detector-dataset-report-sha256 "<dataset-report-sha256>" \
  --output-dir "<external-run-root>/chrompeakformer-seed17" \
  --device cuda --seed 17 --epochs 30 --batch-size 16 --num-workers 2
```

The training report distinguishes the best validation epoch observed in the log from the actual
final-epoch checkpoint; it never labels the final weights as best weights without matching evidence.
The peak-width-weighted localization loss is recalibrated from current training annotations only;
an older config statistic is never reused and validation boxes do not influence that value.
Use `--smoke-test --epochs 1` for the first scheduled CUDA contract run. Smoke reports are explicitly
ineligible for development comparisons; omit `--smoke-test` only for a complete declared run.
Produce standard COCO detections from that checkpoint, pinning both checkpoint and Dataset hashes:

```bash
python -m multimodal_science.chrompeakformer.run_detector_inference_cli \
  --source-root "<authorized-detector-source>/model" \
  --checkpoint "<external-run-root>/chrompeakformer-seed17/checkpoint.pth" \
  --checkpoint-sha256 "<checkpoint-sha256>" \
  --training-report "<external-run-root>/chrompeakformer-seed17/detector_training_report.json" \
  --training-report-sha256 "<training-report-sha256>" \
  --detector-dataset-root "<external-run-root>/detector-dataset/coco" \
  --detector-dataset-report "<external-run-root>/detector-dataset/detector_dataset_report.json" \
  --detector-dataset-report-sha256 "<dataset-report-sha256>" \
  --output-dir "<external-run-root>/chrompeakformer-seed17-predictions"
```

Finally, the unified evaluator reports official COCO AP@[.50:.95], AP50, AP75 and recall together
with image-level peak classification and best-box IoU. The fixed 0.5 threshold is always retained;
the validation-selected threshold is explicitly marked as development-only:

```bash
python -m multimodal_science.chrompeakformer.evaluate_detector_cli \
  --validation-coco "<external-run-root>/detector-dataset/coco/val/val_coco.json" \
  --predictions "<external-run-root>/chrompeakformer-seed17-predictions/coco_predictions.json" \
  --inference-report "<external-run-root>/chrompeakformer-seed17-predictions/detector_inference_report.json" \
  --inference-report-sha256 "<inference-report-sha256>" \
  --detector-dataset-report "<external-run-root>/detector-dataset/detector_dataset_report.json" \
  --detector-dataset-report-sha256 "<dataset-report-sha256>" \
  --output-dir "<external-run-root>/chrompeakformer-seed17-evaluation"
```

## Sequence-baseline evaluation contract

The sequence baseline loads only the materialized Dataset above. Its loader verifies the report
schema, every selected artifact hash, declared array shapes, JSON/NumPy target agreement, negative
boundary sentinels, and source-group identities before returning model inputs. This keeps training
code from silently bypassing the audited data boundary.

Detection reports use accuracy, balanced accuracy, positive and negative F1, Macro-F1, MCC,
AUROC, AUPRC (average-precision step integral), specificity, recall, and false-positive rate.
Boundary quality is evaluated only on human-labelled positive ROIs, using normalized start/end
MAE, physical-time MAE, width MAE, and 1D interval IoU. Confidence intervals resample complete
source mzML groups rather than treating the 16,170 correlated compound ROIs as independent
experiments. Validation threshold selection is deterministic; a selected threshold must be frozen
before the sealed internal-test split is opened.

Train the residual 1D detector first as a CPU smoke test. A sample cap is rejected unless the run
is explicitly marked as non-benchmark evidence:

```bash
CUDA_VISIBLE_DEVICES="" python -m multimodal_science.baselines.train_sequence_cli \
  --dataset-root "<external-dataset-root>/multimodal-v1" \
  --output-dir "<external-run-root>/sequence-smoke" \
  --modality sequence \
  --device cpu \
  --epochs 1 \
  --smoke-test \
  --max-train-samples 512 \
  --max-validation-samples 256 \
  --bootstrap-iterations 100
```

For a full development-comparison run, omit all smoke and sample-cap arguments and select a GPU
explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m multimodal_science.baselines.train_sequence_cli \
  --dataset-root "<external-dataset-root>/multimodal-v1" \
  --output-dir "<external-run-root>/sequence-seed17" \
  --modality sequence \
  --device cuda \
  --seed 17
```

The `sequence` and `sequence_metadata` modalities share the same residual 1D encoder, detection
head, and positive-only boundary head; only the latter receives the seven audited scalar features.
Equal-bin max pooling uses a non-overlapping deterministic implementation instead of PyTorch's
adaptive CUDA backward kernel, so strict deterministic training remains enabled on supported GPUs.
The runner selects its checkpoint by validation loss, reports both fixed-0.5 and
validation-selected detection metrics, freezes the selected threshold, and saves a source-grouped
bootstrap report. It refuses to overwrite an existing run directory and has no internal-test CLI
surface. A full run is eligible for validation-set ablations, not the final benchmark or model
promotion. Its evidence gate remains incomplete until sealed internal-test, blank-stratified,
quantification, and declared multimodal-ablation evidence exists. Model checkpoint, configuration,
epoch history, validation predictions, threshold, code revision, runtime versions, and dataset
hashes are published together.

Treat the report as a claim to be checked, not as self-validating evidence. After a run finishes,
the independent validator verifies every artifact hash and recomputes threshold selection,
classification metrics, physical and normalized boundary metrics, and source-grouped bootstrap
intervals from the saved per-asset predictions. It deliberately hashes but never deserializes the
PyTorch checkpoint:

```bash
python -m multimodal_science.baselines.validate_sequence_run_cli \
  --run-dir "<external-run-root>/sequence-seed17" \
  --dataset-root "<external-dataset-root>/multimodal-v1"
```

The command refuses to overwrite an existing verification report. Older runs whose prediction
records predate the required `roi_width_minutes` evidence must be rerun; physical-time metrics
cannot be reconstructed safely without it.

After the formal detector and both sequence modalities have independently passed their evidence
contracts, create a hash-bound development comparison. The builder verifies that all runs use the
same asset index, train/validation counts, and source-group boundary. It reports fixed-0.5 and
validation-selected classification results separately and refuses smoke, internal-test, or
unverified sequence reports:

```bash
python -m multimodal_science.baselines.compare_runs_cli \
  --detector-evaluation "<detector-run>/evaluation/detector_evaluation_report.json" \
  --detector-dataset-report "<detector-run>/detector-dataset/detector_dataset_report.json" \
  --sequence-report "<sequence-run>/scientific_report.json" \
  --sequence-verification "<sequence-run>/recovery_verification_report.json" \
  --sequence-metadata-report "<sequence-metadata-run>/scientific_report.json" \
  --sequence-metadata-verification "<sequence-metadata-run>/verification_report.json" \
  --output-dir "<external-run-root>/development-ablation"
```

The generated Markdown table uses the common fixed threshold for the primary classification
comparison. COCO AP remains detector-only, and the report explicitly preserves the distinction
between detector best-box IoU and sequence interval IoU. It is development evidence, not a sealed
test result.

## Qwen3-VL instruction and evaluation data

The instruction builder consumes only the hash-verified unified Dataset. It verifies every
declared Dataset artifact, rechecks train/validation source-group separation, and creates four
declared task families: image-only peak presence, image-plus-metadata peak presence, positive-only
peak grounding, and deterministic scientific QC. Run it without opening the sealed internal test:

```bash
python -m multimodal_science.qwen3vl.build_instruction_cli \
  --dataset-root "<external-dataset-root>/multimodal-v1" \
  --output-dir "<external-dataset-root>/qwen3vl-instructions-v1"
```

The versioned bilingual profile keeps the legacy English v1 unchanged while localizing the
instruction layer:

```bash
python -m multimodal_science.qwen3vl.build_instruction_cli \
  --dataset-root "<external-dataset-root>/multimodal-v1" \
  --output-dir "<external-dataset-root>/qwen3vl-instructions-v2-bilingual" \
  --language-profile bilingual \
  --chinese-train-ratio 0.6
```

The bilingual train file still contains one derived instruction per asset/task combination. Its
language is assigned reproducibly from the semantic instruction hash, so changing output paths or
rerunning the builder cannot reshuffle languages. Validation contains parallel `en` and `zh-CN`
prompts with distinct instruction IDs and a shared `pair_id`. JSON keys and controlled values stay
language-neutral English identifiers. The report records language counts and explicitly marks the
parallel prompts as correlated language views, not additional independent scientific samples.

`train_qwen.jsonl` follows the official Qwen3-VL single-image `image` plus `conversations`
contract, with exactly one `<image>` token in each human message and no visual tokens in model
answers. The format is grounded in the
[official Qwen3-VL fine-tuning documentation](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).
Validation is deliberately not emitted as another SFT file: `validation_prompts.jsonl` contains
model inputs with no answers, while `validation_answers.jsonl` is a separately hashed evaluation
key. `instruction_manifest.jsonl` maps every derived row to the original asset, source mzML group,
image hash, Dataset hash, task, modalities, and supervision source.

Before LoRA training, publish a train-only bundle. The builder verifies the instruction report,
official-format train rows, train-manifest prefix, response hashes, and selected image paths. It
does not open validation prompts or validation answers. A bounded smoke bundle is deterministically
stratified by task, language, and peak-presence label:

```bash
python -m multimodal_science.qwen3vl.build_lora_bundle_cli \
  --instruction-root "<external-dataset-root>/qwen3vl-instructions-v2-bilingual" \
  --instruction-report-sha256 "<expected-64-hex-digest>" \
  --assets-root "<external-assets-root>" \
  --output-dir "<external-dataset-root>/qwen3vl-lora-smoke-bundle" \
  --max-records 128 \
  --seed 17
```

The LoRA runner requires exactly one visible CUDA device and BF16 support. It freezes the base
language weights, vision tower, and visual merger; adapters target only language-attention
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. Loss is calculated only on assistant response tokens.
The default SDPA path avoids making FlashAttention or a local CUDA compiler a prerequisite:

```bash
CUDA_VISIBLE_DEVICES=0 python -m multimodal_science.qwen3vl.train_lora_cli \
  --bundle-root "<external-dataset-root>/qwen3vl-lora-smoke-bundle" \
  --bundle-report-sha256 "<expected-64-hex-digest>" \
  --assets-root "<external-assets-root>" \
  --output-dir "<external-run-root>/qwen3vl-lora-smoke" \
  --model-name-or-path "<Qwen3-VL-4B-checkpoint>" \
  --model-revision "<immutable-revision>" \
  --model-artifact-sha256 "<model-artifact-digest>" \
  --code-revision "<40-hex-git-revision>" \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-steps 2 \
  --save-steps 1
```

Install the pinned `peft` addition from `qwen3vl/requirements-lora.txt` into the existing Qwen
environment. Checkpoints contain adapter, optimizer, scheduler, trainer-state, and RNG state so a
matching `--resume` run can continue safely. A completed adapter is still not metric evidence:
it must subsequently run through the prompt-only inference bundle and answer-separated evaluator.
Sample-capped bundles and `--max-steps` runs are explicitly marked as smoke evidence and cannot
claim completed domain training.

Adapter inference is accepted only when all three adapter identity inputs are supplied. Before
loading PEFT, the runner verifies the training-report hash, the complete artifact-manifest hash,
every listed adapter byte, the frozen-base/train-only contracts, and the exact base-model artifact:

```bash
python -m multimodal_science.qwen3vl.run_inference_cli \
  --bundle-root "<external-dataset-root>/qwen3vl-inference-bundle-v2" \
  --bundle-report-sha256 "<expected-64-hex-digest>" \
  --assets-root "<external-assets-root>" \
  --output-dir "<external-run-root>/qwen3vl-lora-validation" \
  --model-name-or-path "<Qwen3-VL-4B-checkpoint>" \
  --model-revision "<immutable-revision>" \
  --model-artifact-sha256 "<model-artifact-digest>" \
  --adapter-root "<external-run-root>/qwen3vl-lora" \
  --adapter-training-report-sha256 "<training-report-digest>" \
  --adapter-manifest-sha256 "<artifact-manifest-digest>" \
  --batch-size 2 \
  --resume
```

A smoke-trained adapter may be loaded for a CUDA contract test, but its generation report remains
development-comparison ineligible. Only an uncapped, completed training report can qualify for the
same answer-separated development evaluation used by the zero-shot baseline.

For the first GPU contract test, use the checked-in `qwen3vl/slurm/coder_lora_smoke.sbatch`
instead of pasting the training command into an interactive shell. The script keeps the required
job name `coder`, selects an unlocked physical GPU at job start, refuses GPUs above either the
declared memory or utilization guard, verifies the repository revision, builds a 128-row train-only bundle, performs two
optimizer updates, reloads the saved adapter for bounded inference, and persists verified outputs.
Partition, node, log paths, and all site-specific paths stay submission-time settings:

```bash
export BIOCODER_RUN_ROOT="<external-run-root>"
export BIOCODER_REPO_ROOT="$PWD"
export BIOCODER_PYTHON="<qwen-python>"
export BIOCODER_MODEL_DIR="<Qwen3-VL-4B-checkpoint>"
export BIOCODER_MODEL_ARTIFACT_SHA256="<model-artifact-digest>"
export BIOCODER_MODEL_REVISION="<immutable-revision>"
export BIOCODER_INSTRUCTION_REPORT_SHA256="<instruction-report-digest>"
export BIOCODER_INFERENCE_BUNDLE_REPORT_SHA256="<inference-bundle-report-digest>"
export BIOCODER_CODE_REVISION="$(git rev-parse HEAD)"
export BIOCODER_GPU_INDEX="auto"

job_id="$(sbatch --parsable \
  --partition="<gpu-partition>" \
  --nodelist="<gpu-node>" \
  --output="<external-run-root>/qwen3vl/slurm/coder-lora-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-lora-%j.err" \
  backend/multimodal_science/qwen3vl/slurm/coder_lora_smoke.sbatch)"
echo "LORA_SMOKE_JOB_ID=$job_id"
```

The smoke profile is intentionally too small for throughput extrapolation at `batch_size=1`.
Before the uncapped run, use the formal runner for a short `batch_size=4` calibration. It stages
the exact Git revision, a stable node-local model cache, and only the bundle-selected images; each
staged image is verified against the train-manifest hash. Adapter checkpoints remain on persistent
storage so another matching job can resume without replaying already-consumed image batches:

```bash
export BIOCODER_BATCH_SIZE=4
export BIOCODER_GRADIENT_ACCUMULATION_STEPS=4
export BIOCODER_SAVE_STEPS=250
export BIOCODER_LOG_STEPS=10
export BIOCODER_MAX_RECORDS=128
export BIOCODER_MAX_STEPS=4

calibration_job_id="$(sbatch --parsable \
  --partition="<gpu-partition>" \
  --nodelist="<gpu-node>" \
  --output="<external-run-root>/qwen3vl/slurm/coder-lora-train-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-lora-train-%j.err" \
  backend/multimodal_science/qwen3vl/slurm/coder_lora_train.sbatch)"
echo "LORA_CALIBRATION_JOB_ID=$calibration_job_id"
```

After that calibration passes the memory guard, remove both explicit caps to request the complete
54,335-row, one-epoch development training run. The default effective batch size is 16. The
formal job still does not access validation answers, does not evaluate itself, and cannot claim a
final benchmark. A non-blocking persistent run lock also rejects duplicate jobs targeting the
same code revision and training configuration before they can share history or checkpoints:

```bash
unset BIOCODER_MAX_RECORDS BIOCODER_MAX_STEPS

formal_job_id="$(sbatch --parsable \
  --partition="<gpu-partition>" \
  --nodelist="<gpu-node>" \
  --output="<external-run-root>/qwen3vl/slurm/coder-lora-train-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-lora-train-%j.err" \
  backend/multimodal_science/qwen3vl/slurm/coder_lora_train.sbatch)"
echo "LORA_FORMAL_JOB_ID=$formal_job_id"
```

The formal runner records `bitwise_determinism_claimed=false`: deterministic data order, seeds,
and resumable RNG state do not justify a bit-exact claim for the current SDPA backward kernel.
Use the resulting hash-bound adapter in the existing full prompt-only inference and
answer-separated evaluation path before comparing it with zero-shot or specialist baselines.

Submit that full adapter validation through the dedicated `coder_lora_evaluate.sbatch` runner.
It stages immutable code, the prompt-only bundle, all 1,815 validation images, and a stable local
model cache; validation image copies are byte-verified before CUDA starts. The adapter is accepted
only with the exact training-report and artifact-manifest hashes printed by the formal training
job. Generation is greedy, covers all 13,708 bilingual prompts, and uses the inference runner's
persistent journal for safe resume. Only after generation is atomically published does the script
open the separate answer key through the evaluator. The runner materializes and verifies complete
directory manifests after each CLI atomically publishes its native files; if a scheduler retry
starts after generation, it validates and reuses those predictions instead of running the model
again:

For recovery of an already published generation produced by an older wrapper revision, set
`BIOCODER_LORA_EVALUATION_RUN_NAME` to that exact safe directory basename. The override changes
only artifact discovery; all model, adapter, bundle, and report hashes are still independently
verified.

```bash
export BIOCODER_ADAPTER_ROOT="<formal-training-output>"
export BIOCODER_ADAPTER_TRAINING_REPORT_SHA256="<training-report-digest>"
export BIOCODER_ADAPTER_MANIFEST_SHA256="<artifact-manifest-digest>"
export BIOCODER_INFERENCE_BATCH_SIZE=2
export BIOCODER_BOOTSTRAP_ITERATIONS=1000

evaluation_job_id="$(sbatch --parsable \
  --partition="<gpu-partition>" \
  --nodelist="<gpu-node>" \
  --output="<external-run-root>/qwen3vl/slurm/coder-lora-eval-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-lora-eval-%j.err" \
  backend/multimodal_science/qwen3vl/slurm/coder_lora_evaluate.sbatch)"
echo "LORA_EVALUATION_JOB_ID=$evaluation_job_id"
```

The runner rejects partial prompt coverage, an unloaded or hash-mismatched adapter, unverified
generation provenance, internal-test access, and any final-benchmark claim. Its output is valid
development evidence only; the sealed internal test remains untouched.

After both full Qwen runs and the specialist ablation are verified, build one cross-family report:

```bash
python -m multimodal_science.qwen3vl.compare_development_cli \
  --specialist-comparison "<comparison-root>/development_ablation_report.json" \
  --zero-shot-generation "<zero-shot-run>/generation_report.json" \
  --zero-shot-evaluation "<zero-shot-evaluation>/qwen_evaluation_report.json" \
  --lora-generation "<lora-run>/generation_report.json" \
  --lora-evaluation "<lora-evaluation>/qwen_evaluation_report.json" \
  --output-dir "<comparison-root>/cross-family-development"
```

The builder verifies every supplied report hash, the shared Dataset and prompt artifacts, the
immutable base model, the absence/presence of the LoRA adapter, and complete paired bilingual
coverage. Its primary Qwen rows are per-language: English and Chinese prompts are two views of
the same 1,815 validation assets, never 3,630 independent scientific samples. COCO AP remains
detector-only and Qwen grounding is reported separately as bbox IoU.
For shared servers, export those six report/output paths plus the repository, Python, and code
revision variables, then submit `qwen3vl/slurm/coder_compare_development.sbatch`; it uses the
required `coder` job name and requests only one CPU and 4 GiB RAM.

Before training image-plus-XIC fusion, materialize an answer-isolated link bundle. The builder
joins all 54,335 train instructions and 13,708 prompt-only validation rows to the normalized
160-point XIC arrays while preserving the true independent asset counts (14,355 train and 1,815
validation). It verifies every source report and array hash, rejects source-group overlap, and
never opens validation answers. Fusion bundle v2 also binds the complete `train_qwen.jsonl` and
`selection_manifest.jsonl` content hashes. This permits a metadata-identical LoRA bundle rebuilt
at a different time while rejecting any supervision or row-selection drift. The original LoRA
report hash remains recorded as build provenance. A missing/constant XIC remains a zero-valued
sensor input with an explicit availability flag; it is not silently dropped.

```bash
python -m multimodal_science.qwen3vl.build_fusion_bundle_cli \
  --code-revision "<40-character-clean-git-revision>" \
  --dataset-root "<multimodal-v1-160>" \
  --dataset-report-sha256 "<dataset-report-digest>" \
  --lora-bundle-root "<formal-lora-bundle>" \
  --lora-bundle-report-sha256 "<lora-bundle-report-digest>" \
  --inference-bundle-root "<prompt-only-inference-bundle>" \
  --inference-bundle-report-sha256 "<inference-bundle-report-digest>" \
  --output-dir "<external-run-root>/qwen3vl/fusion/bundles/image-xic-v2"
```

On Slurm, set the corresponding `BIOCODER_*` variables and submit
`qwen3vl/slurm/coder_build_fusion_bundle.sbatch`. This is a one-CPU metadata job and does not
reserve a GPU. It requires a clean repository at the exact declared revision and records that
revision in the v2 report. The accompanying trainable sensor projector converts each 160-point
XIC into four Qwen-width tokens behind a near-closed residual gate. A learned availability
embedding preserves the distinction between a measured zero trace and a missing XIC. Bundle and
projector completion alone do not constitute Qwen fusion: the next gate is injection into the
Qwen embedding stream followed by LoRA-plus-projector training and same-validation evaluation.

The bounded fusion smoke inserts four projected XIC embeddings immediately before the assistant
response without changing Qwen's tokenizer or output vocabulary. The inserted positions are
masked from the language-model loss. Shadow text IDs are used only to derive explicit native
Qwen3-VL multimodal RoPE positions; image placeholders are still replaced by the official vision
path, while the image-only LoRA initializes the language path.
Only LoRA weights and the sensor projector are trainable; the visual tower, merger, and base
language weights stay frozen. Submit `qwen3vl/slurm/coder_fusion_smoke.sbatch` after providing the
hash-verified fusion bundle, formal LoRA adapter, Dataset, assets, and immutable base model through
its `BIOCODER_*` variables. The job requires and verifies a full-file model manifest, stages an
immutable Git archive, local model cache, and train-only input roots containing only the selected
train rows/images, strips the original source-root variables from the child environment, rejects
dirty source trees, requires the initial image-only adapter's recorded training-row SHA-256 to
equal the exact fusion LoRA train file, requires the runtime LoRA train-row and selection hashes
to equal the complete hashes recorded by fusion bundle v2, and fails on drift from the verified
NumPy 1.26.4, Torch 2.11.0+cu128, Transformers 4.57.1, PEFT 0.17.1, and safetensors 0.6.2
runtime. This is an auditable input-scoping control, not a chroot/container security boundary. A
successful two-update smoke verifies native
M-RoPE prefix/suffix positions, observes the visual-tower forward hook, requires separate nonzero
gradients for every LoRA attention target and projector submodule, proves both parameter groups
changed by exact state digests, and reloads the serialized adapter/projector. It is evidence only
after its saved manifest and success markers pass, and remains ineligible for development
comparison.

The first externally verified execution of that contract is Slurm job `6626` at revision
`26d694619c1dddb9907f74604f6e2b1c3a0a8644`. Its fusion report SHA-256 is
`0f7ed28ca15e8e5feb511258302fde59bb4abf2b4404364b2ee9b160be2fb99f`, and its exact artifact
manifest SHA-256 is `f561296c048e4b1af169dc14ca7d8bb857a0205866f293074a5e0c7fa38f1a67`.
The two optimizer losses are retained only as runtime diagnostics. This run does not establish
validation quality or completed fusion training.

Formal fusion training uses `train_fusion_cli` through
`qwen3vl/slurm/coder_fusion_train.sbatch`. Its default contract is an uncapped one-epoch pass over
all 54,335 train instructions with `batch_size=1`, gradient accumulation 16, BF16, frozen base and
vision weights, trainable language-attention LoRA, and a trainable XIC projector. Batch size one is
intentional: it avoids padding-dependent M-RoPE ambiguity while continuous sensor tokens are
inserted into each example. The launcher stages and hash-verifies only train artifacts, all 14,355
unique train images, the complete initial adapter manifest, and the immutable base-model cache
before acquiring a GPU. It writes periodic joint adapter/projector/optimizer/scheduler/RNG
checkpoints into a persistent `.incomplete` run and resumes from the deterministic unseen suffix.
`BIOCODER_MAX_STEPS` is empty by default; setting it explicitly creates a calibration run that is
marked incomplete. Training output remains development-comparison ineligible until full
prompt-only fused generation and answer-separated validation evaluation succeed.

The default projector emits four sensor tokens. Formal token-count ablations set
`BIOCODER_SENSOR_TOKENS` to exactly `1`, `4`, or `8`; the value is included in the immutable run
name, training configuration, projector specification, and report. Adaptive average pooling keeps
the saved convolutional/projector weights shape-compatible across those token counts. When an
auxiliary-pretrained projector is reused with a different token count, the formal report records
both source and target counts and marks `token_pooling_remapped=true`. It is therefore an explicit
pooling ablation rather than an unreported architecture substitution. Every completed training
report also records the sensor gate's initial/final logits, probabilities, and probability change.

Formal fused validation uses `run_fusion_inference_cli`. It accepts only the prompt-only inference
bundle, validation XIC links/arrays, hash-verified images, the immutable base model, and a completed
formal fusion artifact. It rejects calibration adapters. Generation is greedy and batch-one; the
runner inserts the four projected XIC embeddings after the assistant-generation prefix and uses an
explicit cached decode loop so Qwen3-VL's three-axis multimodal RoPE positions are preserved rather
than silently rebuilt by a generic generation wrapper. Validation answers remain outside this
process and are opened only by the existing answer-separated evaluator after predictions and their
generation provenance have been persisted.

For modality-use controls, fused generation accepts exactly four audited XIC interventions:
`aligned`, `shuffled`, `zero`, and `availability-off`. `shuffled` applies a seeded Sattolo
single-cycle permutation over independent validation signal rows, so every asset receives another
asset's XIC with no fixed points and its English/Chinese prompts share the same donor. `zero` keeps
the original availability embedding while replacing signal values with zero; `availability-off`
also forces the availability flag false. Runtime provenance records the mode, seed, algorithm,
row-mapping hash, changed-row count, and the fact that no answer key was used. Set
`BIOCODER_XIC_INTERVENTION` and `BIOCODER_XIC_INTERVENTION_SEED` on
`coder_fusion_evaluate.sbatch`; each intervention receives a distinct locked run name.

Completed intervention runs are analyzed with `analyze_xic_interventions_cli` or
`qwen3vl/slurm/coder_analyze_xic_interventions.sbatch`. The analyzer refuses path-name-only
comparisons: every evaluation must bind its supplied generation report, all four generations must
share the exact base model, fusion checkpoint, prompt/answer artifacts, decoding settings, and
validation-record identities, and the reported metrics are independently recomputed from the
hash-bound evaluation records. Confidence intervals use a paired bootstrap over complete source
groups. They never resample individual prompts, and English/Chinese variants remain paired views
of the same scientific assets. The report retains raw `aligned - intervention` deltas, a
direction-adjusted benefit (so lower FPR is correctly treated as better), and the bootstrap
probability that aligned XIC is better.

`qwen3vl/slurm/coder_fusion_evaluate.sbatch` operationalizes that boundary on the RTX 5090 node.
It stages only prompt-side validation inputs for generation and launches the model under an
`env -i` child environment that has no instruction-Dataset or answer-key variable. It accepts no
sample cap, requires all 13,708 bilingual validation prompts, writes an exact artifact manifest,
and only then invokes the evaluator against the separately hash-bound instruction root. A result is
accepted for development comparison only when fused runtime provenance, model/adapter/Dataset
hashes, greedy batch-one decoding, complete prompt coverage, evaluator provenance, and the
no-internal-test contract all pass. The same validation set remains development-only and must not
be used as the reward source for later RL or preference optimization.

The canonical random-initialized fusion checkpoint and all four aligned/shuffled/zero/
availability-off validation runs completed without opening the internal test. Their
provenance-bound 10,000-resample source-group bootstrap also completed: aligned XIC was better
than every intervention on the primary classification, localization, and QC endpoints. This is
the causal modality-use check; the learned gate magnitude is not interpreted as a percentage of
XIC use.

`qwen3vl/slurm/coder_fusion_development_matrix.sbatch` fixes that remaining training matrix. Its
five array entries run the primary four-token configuration at seeds `17/29/43` and the one- and
eight-token pooling variants at seed `17`. The array is throttled to two concurrent entries,
forbids step caps, explicitly removes auxiliary-projector initialization, and delegates to the
same resumable, hash-bound formal trainer. Auxiliary initialization remains a separate ablation.
All five entries have now completed 3,396 optimizer updates and retained exactly one final
checkpoint each. Their final gate probabilities are approximately `0.01847`, `0.01856`, and
`0.01851` for the three four-token seeds, `0.01797` for one token, and `0.01829` for eight tokens.

Run `qwen3vl/slurm/coder_fusion_evaluation_matrix.sbatch` to evaluate those five immutable
training artifacts on the same 13,708 bilingual validation prompts. The array keeps generation
seed `17` fixed for every entry; seed `17/29/43` denotes training repetition and must not leak into
the inference protocol. After all entries pass, `coder_analyze_fusion_matrix.sbatch` verifies
every training/generation/evaluation manifest, recomputes selected metrics from the 13,708
hash-bound evaluation records, checks identical scientific-unit identities, and publishes:

- mean and sample standard deviation across the three four-token training seeds;
- English and Chinese metrics plus their per-seed gaps and cross-language consistency;
- the seed-17 one/four/eight-token ablation without inventing cross-seed uncertainty; and
- initial/final gate values, output-schema rates, and exact provenance hashes.

Both stages expose no internal-test input. Their results remain development evidence until the
protocol is frozen and the sealed internal test is opened once.

After the five-cell analysis succeeds, build the final pre-test development dossier:

```bash
python -m multimodal_science.qwen3vl.build_development_dossier_cli \
  --cross-family-report "<comparison-root>/cross_family_development_report.json" \
  --fusion-matrix-report "<matrix-analysis-root>/fusion_matrix_analysis.json" \
  --xic-intervention-report "<intervention-root>/xic_intervention_analysis.json" \
  --selected-fusion-evaluation-root "<four-token-seed17-evaluation-root>" \
  --output-dir "<comparison-root>/multimodal-development-dossier"
```

`qwen3vl/slurm/coder_build_development_dossier.sbatch` provides the scheduled form. It verifies
all four input manifests plus the hash-bound reports behind the cross-family comparison, validates
the full five-cell/three-seed matrix statistics, and requires prompt, answer, base-model, selected
evaluation-record, training-report, and training-manifest identities to agree. It runs from an
immutable `git archive` of the declared revision and emits a nine-row cross-family table, a
bilingual/localization failure page, the complete deterministic failure-case JSONL, and an exact
artifact manifest. The dossier records rather than hides whether the earlier intervention
experiment used the exact selected checkpoint. A mismatch keeps the candidate pre-test-ineligible
until the four interventions are repeated on the selected checkpoint.

The selected-checkpoint interventions, five-cell matrix, and final development dossier have now
all passed. Before any sealed record is read, run
`qwen3vl/slurm/coder_final_benchmark_freeze.sbatch`. The freeze job verifies every development
manifest, fixes the exact five primary candidates, validation-frozen operating thresholds,
base-model revision, generation settings, 10,000-resample source-group bootstrap policy, split
manifest, derivation plan, and train-fitted scalar normalization. It produces a hash-bound protocol
whose state remains `internal_test_accessed=false`; it cannot open or materialize the test split.
The image-only LoRA run root is an explicit required freeze input and is accepted only when both
its training-report and artifact-manifest hashes match the already bound development generation.

Only after that protocol has been reviewed may
`qwen3vl/slurm/coder_final_benchmark_run.sbatch` be submitted. The formal job first verifies the
immutable code revision, protocol, development Dataset, model cache, private detector source tree,
and GPU guard. It then atomically opens access sequence 1, materializes the 1,815 test assets with
the already frozen train normalization, creates physically separated prompt and answer roots, and
evaluates exactly these candidates without model or threshold selection:

- Qwen3-VL zero-shot;
- Qwen3-VL image-only LoRA;
- Qwen3-VL image plus aligned XIC;
- SequencePeakNet; and
- ChromPeakFormer.

The result contains one language-separated main table, the individual hash-bound evaluation
reports and source-group intervals, an evidence registry, and an exact manifest. A crash may resume
only the same protocol and access ID. Completion is idempotent only for the identical final evidence
manifest; changed evidence or a second access event is rejected. Do not run the formal job as a
smoke test and do not submit it before the freeze job and native server-side `bash -n` check pass.

This sealed run is explicitly a frozen multimodal-model benchmark. Qwen structured-output and QC
metrics are included, but they are not a substitute for the separate BioCoder agent gate covering
tool selection, tool arguments, abstention, trajectory quality, and evidence attribution. No agent
promotion claim is permitted from the five-model table alone.

The current cluster advertises `Gres=(null)` for its GPU partitions, so Slurm cannot provide a
GPU TRES reservation for this job. The script records this limitation explicitly and uses a
user-scoped physical-GPU lock plus three startup samples of memory and utilization. This prevents
collisions with cooperating BioCoder jobs but is not claimed as scheduler-enforced exclusivity.
When the cluster enables GPU GRES, replace this fallback with a scheduler-issued single-GPU
allocation before treating the execution contract as portable to other users or clusters.

Multi-task rows are correlated views of the same source assets. The report therefore records
source assets and derived instruction rows separately; instruction count must never be presented
as the number of independent chromatograms or images. Image paths remain relative to the external
asset root, and image bytes are neither copied nor committed.

Model inference must write one complete JSONL prediction record per validation prompt:

```json
{"schema_version":"chrompeak-qwen3vl-prediction-v1","instruction_id":"<24-hex-id>","response":"<raw-model-response>"}
```

Do not point a model process at the instruction Dataset, because that directory also contains the
answer key. First publish a separately hash-bound, prompt-only inference bundle:

```bash
python -m multimodal_science.qwen3vl.build_inference_bundle_cli \
  --instruction-root "<external-dataset-root>/qwen3vl-instructions-v2-bilingual" \
  --instruction-report-sha256 "<expected-64-hex-digest>" \
  --output-dir "<external-dataset-root>/qwen3vl-inference-bundle-v2"
```

The bundle builder opens the instruction report and validation prompts only. Its report attests
that it neither opened nor materialized the answer key or instruction manifest. The inference
runner accepts this bundle and the external image root, but has no instruction-Dataset, answer-key,
or internal-test argument:

```bash
python -m multimodal_science.qwen3vl.run_inference_cli \
  --bundle-root "<external-dataset-root>/qwen3vl-inference-bundle-v2" \
  --bundle-report-sha256 "<expected-64-hex-digest>" \
  --assets-root "<external-assets-root>" \
  --output-dir "<external-run-root>/qwen3vl-zero-shot" \
  --model-name-or-path "<Qwen3-VL-checkpoint>" \
  --model-revision "<immutable-commit-or-revision>" \
  --batch-size 1 \
  --resume
```

The current runner follows the official Transformers Qwen3-VL interface and therefore requires
`transformers>=4.57.0`. It records the resolved model revision, package/runtime metadata, decoding
configuration, prompt/image/response hashes, prompt order, and raw model response for every row.
Interrupted runs retain a verified journal and may resume with exactly the same configuration.
Sample-capped smoke runs remain development-comparison ineligible.

The evaluator joins predictions to the separately hashed answer key only after generation. It
requires exact validation-ID coverage, rejects duplicate or unknown IDs and tampered instruction
artifacts, and scores malformed or schema-invalid model responses as failures rather than dropping
them. Run it without an internal-test input surface:

```bash
python -m multimodal_science.qwen3vl.evaluate_predictions_cli \
  --instruction-root "<external-dataset-root>/qwen3vl-instructions-v2-bilingual" \
  --instruction-report-sha256 "<expected-64-hex-digest>" \
  --predictions "<external-run-root>/qwen3vl-zero-shot/predictions.jsonl" \
  --generation-report "<external-run-root>/qwen3vl-zero-shot/generation_report.json" \
  --generation-report-sha256 "<expected-64-hex-digest>" \
  --output-dir "<external-run-root>/qwen3vl-evaluation" \
  --bootstrap-iterations 1000 \
  --seed 17
```

Presence tasks report binary classification metrics and source-grouped confidence intervals.
Grounding reports strict JSON/schema validity, full-image box IoU, IoU@0.5, horizontal boundary
error, and a source-grouped IoU interval. Scientific QC reports exact-match and field accuracy.
For bilingual v2, the same evaluator additionally reports every task separately for `en` and
`zh-CN`, plus paired cross-language exact consistency and grounding-box consistency. The report
intentionally does not collapse heterogeneous tasks into one combined score. Each run binds the
prediction file, prompts, answer key, instruction manifest, instruction report, and source Dataset
report by SHA-256. Development-comparison eligibility additionally requires a full, uncapped
Transformers generation run, an immutable model identity, and a generation report whose per-row
task, image, image hash, prompt hash, language, pair ID, response, and ordering all match the
evaluation inputs. Evaluator-only and oracle reports remain ineligible; file separation alone
cannot prove clean model generation.

After a provenance-verified bilingual zero-shot evaluation, run the failure-mode audit before
changing prompts, coordinate contracts, or training targets:

```bash
python -m multimodal_science.qwen3vl.audit_zero_shot_cli \
  --generation-report "<external-run-root>/qwen3vl-zero-shot/generation_report.json" \
  --generation-report-sha256 "<expected-64-hex-digest>" \
  --evaluation-report "<external-run-root>/qwen3vl-evaluation/qwen_evaluation_report.json" \
  --evaluation-report-sha256 "<expected-64-hex-digest>" \
  --output-dir "<external-run-root>/qwen3vl-zero-shot-failure-audit" \
  --bootstrap-iterations 1000 \
  --seed 17
```

The audit verifies both upstream report hashes, the prediction artifact, and every evaluation
record before reporting output distributions by task and language. For grounding it compares the
formal source-pixel score with two read-only counterfactuals: a full `0..1000` normalized box and
an `x`-normalized box whose fixed full-height `y` coordinates remain in source pixels. It emits
per-record evidence, source-grouped mean-IoU intervals, and a `sha256sum -c` compatible manifest.
Counterfactual metrics are diagnostic only: they do not replace the immutable formal zero-shot
result and must not be used to select a training protocol on validation data.

## Post-seal v1.1 development controls

The sealed internal-test benchmark is immutable. The following entry points operate only on the
existing leakage-safe train/validation artifacts and always remain development-only.

### Current-sample XIC-only Qwen

Use the formal fusion trainer with `BIOCODER_INPUT_MODALITY=xic_only`. Keep the same image-LoRA
initialization, fusion bundle, train rows, four sensor tokens, seed 17, and uncapped one-epoch
settings used by the selected image + XIC run:

```bash
export BIOCODER_INPUT_MODALITY=xic_only
export BIOCODER_SENSOR_TOKENS=4
export BIOCODER_SEED=17
unset BIOCODER_MAX_STEPS

xic_only_train_job_id="$(sbatch --parsable \
  --job-name=coder \
  --output="<external-run-root>/qwen3vl/slurm/coder-xic-only-train-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-xic-only-train-%j.err" \
  --export=ALL \
  backend/multimodal_science/qwen3vl/slurm/coder_fusion_train.sbatch)"
```

The resulting run name begins `xic-only-full-tokens4-...`. After it completes, set
`BIOCODER_FUSION_ADAPTER_ROOT`, `BIOCODER_FUSION_TRAINING_REPORT_SHA256`, and
`BIOCODER_FUSION_ADAPTER_MANIFEST_SHA256` from that exact output and submit
`coder_fusion_evaluate.sbatch` with `BIOCODER_INPUT_MODALITY=xic_only`. Training and inference use
text-only Qwen processor calls, never forward `pixel_values` or `image_grid_thw`, and fail if a
visual-tower forward hook fires. Source image hashes are still checked as sample-provenance joins;
opening an image for that hash check is not a model input.

This is a controlled current-sample input ablation, not a claim that the shared language adapter
has never seen image-supervised domain training.

### Image LoRA with frozen SequencePeakNet output in the prompt

Set the common immutable model, prompt bundle, Dataset, assets, instruction, and image-LoRA
variables used by full LoRA evaluation. Also set the completed sequence-only run and its hashes:

```bash
export BIOCODER_SEQUENCE_RUN_ROOT="<completed-sequence-only-run>"
export BIOCODER_SEQUENCE_REPORT_SHA256="<scientific-report-digest>"
export BIOCODER_SEQUENCE_MANIFEST_SHA256="<artifact-manifest-digest>"

sequence_prompt_job_id="$(sbatch --parsable \
  --job-name=coder \
  --output="<external-run-root>/qwen3vl/slurm/coder-sequence-prompt-%j.out" \
  --error="<external-run-root>/qwen3vl/slurm/coder-sequence-prompt-%j.err" \
  --export=ALL \
  backend/multimodal_science/qwen3vl/slurm/coder_sequence_prompt_evaluate.sbatch)"
```

The job first builds a manifest-bound target-free bundle from frozen SequencePeakNet validation
predictions. Only probability and normalized interval endpoints become model-visible text. It
then runs all 13,708 bilingual prompts with the unchanged ROI image and image-only LoRA, persists
generation, and opens answers only in the separate evaluator.

### Public development evidence

`build_public_development_evidence_cli` converts the manifest-verified development dossier and
training reports into a path-free public table. It includes the three-seed mean ± sample standard
deviation and individual values, seed-17 one/four/eight-token rows, all four XIC interventions,
gate values, and full uncapped LoRA/fusion wall time. Supplying all three optional auxiliary paths
also adds the random-versus-pretrained-projector comparison:

```bash
python -m multimodal_science.qwen3vl.build_public_development_evidence_cli \
  --development-dossier-root "<selected-checkpoint-development-dossier>" \
  --lora-training-root "<complete-image-lora-training-root>" \
  --fusion-training-root "<complete-image-xic-training-root>" \
  --auxiliary-pretraining-root "<complete-auxiliary-pretraining-root>" \
  --random-projector-evaluation-root "<random-projector-evaluation-root>" \
  --auxiliary-projector-evaluation-root "<auxiliary-projector-evaluation-root>" \
  --xic-only-evaluation-root "<complete-xic-only-evaluation-root>" \
  --sequence-prompt-evaluation-root "<complete-sequence-prompt-evaluation-root>" \
  --output-dir "<external-run-root>/comparisons/public-development-evidence-<revision>"
```

The report is a development artifact and cannot be merged into or used to revise the sealed v1
table. See [`MULTIMODAL_V11_EXPERIMENTS.md`](../../MULTIMODAL_V11_EXPERIMENTS.md) for the measured
wall times, intervention rows, auxiliary result, and completion gate.

On Slurm, export the seven required `BIOCODER_*` roots/revision variables and, optionally, all
three auxiliary evidence roots and both post-seal control evaluation roots, then submit
`qwen3vl/slurm/coder_publish_development_evidence.sbatch`. It stages the exact Git revision, uses
one CPU and no GPU request, verifies the output manifest, checks that the public JSON contains no
machine path, and prints the canonical Markdown table into the job log.

## Current boundary

Model-benchmark v1 is complete for dataset version `raw-072fee8e`. Its frozen protocol SHA-256 is
`0bb30f51f4ab95f3ceee6d577ad2a8d698e224bfa240091011b42525ea6b2b25`; access sequence 1 is
complete, and the final report SHA-256 is
`78e0100e971cb1a6cc37e93775997c54c3e0db12e491ea77f6944706ab4465d2`. The run evaluated all
five frozen candidates on 1,815 assets from 11 independent source mzML groups, performed no
post-test model or threshold selection, and authorizes no additional test access. The result table
and interpretation are maintained in [`MULTIMODAL_RESULTS.md`](../../MULTIMODAL_RESULTS.md).

The source evidence remains outside Git. Verify it in place without reopening protected records:

```bash
biocoder multimodal verify-final \
  --protocol-root "<protocol-root>" \
  --protocol-sha256 "<frozen-protocol-sha256>" \
  --ledger-dir "<completed-access-ledger>" \
  --report-root "<final-report-root>"
```

Build a deterministic public-safe directory and ZIP only after that full chain verifies:

```bash
biocoder multimodal archive-final \
  --protocol-root "<protocol-root>" \
  --protocol-sha256 "<frozen-protocol-sha256>" \
  --ledger-dir "<completed-access-ledger>" \
  --report-root "<final-report-root>" \
  --output-dir "<public-release-root>"

biocoder multimodal verify-release \
  --release-root "<public-release-root>" \
  --archive "<public-release-root>.zip"
```

The release contains aggregate rows and cryptographic identifiers only. It excludes raw mzML,
images, labels, answer keys, predictions, weights, private detector source, and absolute execution
paths. The materialized v1 release manifest is
`82e8f7fed0d470aa4b3ba511977c4e6c71d53a15c913450702dc058a50c0f30d`; its deterministic
ZIP is `a402c31de998f4392ca66798ae8c6db18d227ccc634415d322d82b6b55520481`. The final model
table does not satisfy the separate BioCoder agent gate for tool selection,
arguments, abstention, trajectory quality, evidence attribution, or UI behavior. The research
pipeline is runnable through its CLIs and Slurm scripts; it is not yet a production FastAPI or
LangGraph chromatogram tool. Reinforcement learning is explicitly a v2 experiment and requires a
new sealed test set.
