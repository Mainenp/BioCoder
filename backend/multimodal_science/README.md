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

`train_qwen.jsonl` follows the official Qwen3-VL single-image `image` plus `conversations`
contract, with exactly one `<image>` token in each human message and no visual tokens in model
answers. The format is grounded in the
[official Qwen3-VL fine-tuning documentation](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).
Validation is deliberately not emitted as another SFT file: `validation_prompts.jsonl` contains
model inputs with no answers, while `validation_answers.jsonl` is a separately hashed evaluation
key. `instruction_manifest.jsonl` maps every derived row to the original asset, source mzML group,
image hash, Dataset hash, task, modalities, and supervision source.

Multi-task rows are correlated views of the same source assets. The report therefore records
source assets and derived instruction rows separately; instruction count must never be presented
as the number of independent chromatograms or images. Image paths remain relative to the external
asset root, and image bytes are neither copied nor committed.

Model inference must write one complete JSONL prediction record per validation prompt:

```json
{"schema_version":"chrompeak-qwen3vl-prediction-v1","instruction_id":"<24-hex-id>","response":"<raw-model-response>"}
```

The evaluator joins predictions to the separately hashed answer key only after generation. It
requires exact validation-ID coverage, rejects duplicate or unknown IDs and tampered instruction
artifacts, and scores malformed or schema-invalid model responses as failures rather than dropping
them. Run it without an internal-test input surface:

```bash
python -m multimodal_science.qwen3vl.evaluate_predictions_cli \
  --instruction-root "<external-dataset-root>/qwen3vl-instructions-v1" \
  --instruction-report-sha256 "<expected-64-hex-digest>" \
  --predictions "<external-run-root>/validation_predictions.jsonl" \
  --output-dir "<external-run-root>/qwen3vl-evaluation" \
  --bootstrap-iterations 1000 \
  --seed 17
```

Presence tasks report binary classification metrics and source-grouped confidence intervals.
Grounding reports strict JSON/schema validity, full-image box IoU, IoU@0.5, horizontal boundary
error, and a source-grouped IoU interval. Scientific QC reports exact-match and field accuracy.
The report intentionally does not collapse heterogeneous tasks into one combined score. Each run
binds the prediction file, prompts, answer key, instruction manifest, instruction report, and
source Dataset report by SHA-256. Until a model-inference runner also provides independently
verifiable generation provenance, evaluator-only reports remain ineligible for development
comparisons; file separation alone cannot prove that generation never accessed the answer key.

## Current boundary

This phase has produced deterministic splits, hash-verified extraction jobs, a private-source
adapter, an atomic execution boundary, and a complete train-plus-validation ROI/XIC/COCO index for
dataset version `raw-072fee8e`. The unified Dataset loader, source-grouped metrics, PyTorch residual
1D runner, independent run validator, Qwen3-VL instruction builder, and hash-bound prediction
evaluator are implemented.
A verified external materialization contains 16,170 independent ROI assets, 54,335 train
instructions, and 6,854 answer-separated validation instructions; its report SHA-256 is
`972d2bafe8fad409f2d472732c4a8ab04acd2a42552d17e6f2aad4b388eb560b`. These artifacts remain
outside Git. No comparison-eligible trained baseline or Qwen3-VL result is claimed yet. Full
baseline training, provenance-bound Qwen inference and training, internal-test extraction,
scientific benchmark runs, and agent-tool integration remain downstream milestones in
`MULTIMODAL_ROADMAP.md`.
