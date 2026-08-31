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
selected extraction job is missing.

## Current boundary

This phase creates deterministic splits, hash-verified extraction jobs, a private-source adapter,
and an atomic execution boundary. A real server extraction is still required before claiming ROI
generation is validated. Qwen3-VL training and agent-tool integration remain downstream,
evidence-gated milestones in `MULTIMODAL_ROADMAP.md`.
