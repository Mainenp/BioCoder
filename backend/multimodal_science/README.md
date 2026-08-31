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

## Current boundary

This phase does not generate ROI images, decode numerical chromatogram arrays, create dataset
splits, train Qwen3-VL, or integrate an agent tool. Those remain downstream, evidence-gated
milestones in `MULTIMODAL_ROADMAP.md`.
