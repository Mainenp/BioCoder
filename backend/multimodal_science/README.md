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

The report records whether the runtime provides NumPy, Pandas, SciPy, Matplotlib, and pyOpenMS. It
does not install them. A blocked dependency gate means the plan is valid but extraction has not run.

## Current boundary

This phase creates deterministic splits and hash-verified extraction jobs, but it does not yet
generate ROI images, decode numerical chromatogram arrays, train Qwen3-VL, or integrate an agent
tool. Those remain downstream, evidence-gated milestones in `MULTIMODAL_ROADMAP.md`.
