# ChromPeakFormer Multimodal Test Plan

Status: planned. A requirement is complete only when its linked checks pass with saved evidence.

## Acceptance mapping

| Acceptance ID | Verification groups |
|---|---|
| AC-01 | T01-T02 |
| AC-02 | T03-T05 |
| AC-03 | T06-T08 |
| AC-04 | T09-T11 |
| AC-05 | T12-T14 |
| AC-06 | T15-T16 |
| AC-07 | T17-T19 |
| AC-08 | T20-T22 |
| AC-09 | T23-T25 |
| AC-10 | T26-T27 |

## Contract and unit checks

### T01-T02 — Naming and provenance

- Public interfaces and documents use `ChromPeakFormer` consistently.
- Model cards and reports distinguish the existing detector foundation from new multimodal contributions.

### T03-T05 — Training-substrate boundary

- The multimodal CLI does not call the text/MLX SFT coordinator.
- Multimodal jobs cannot overwrite text-training state.
- Run metadata and registry evidence conform to versioned schemas.

### T06-T08 — Manifest integrity

- All required manifest fields are schema-validated.
- Exact matches receive `fallback_order=0`.
- Weak or fallback matches are assigned to the audit bucket.
- Image, sequence, metadata, ROI window, and label hashes resolve to the same sample identity.
- An alignment or hash failure makes the sample ineligible.
- A complete asset index passes the training-readiness gate before any baseline or Qwen3-VL run.
- Numerical baselines additionally require finite, monotonic, uniquely aligned ROI-cropped XIC
  windows; full-trace point counts cannot be substituted for cropped-window evidence.
- Nonuniform or clustered acquisition axes are resampled from their RT coordinates, never from
  array-index distance.

### T09-T11 — Leakage prevention

- Train, validation, and test sets do not share prohibited source files, experiments, or split groups.
- Audit-bucket samples cannot enter the primary training set or benchmark.
- Legacy image-level random negative splits are rejected by the benchmark gate.

### T12-T14 — Tool and agent state

- Only declared tool states are accepted.
- Every state carries a reason code and provenance payload.
- Agent state and trajectory records distinguish `no_peak` from `tool_error` and other QC states.

## Integration checks

### T15-T16 — B1 demonstration

- A manifest-backed sample can run through ChromPeakFormer and the BioCoder agent.
- The result contains schema-valid measurements, QC status, evidence, and trajectory data.
- B1 interfaces and documentation do not claim that Qwen3-VL domain training is complete.

### T17-T19 — Instruction-data builders

- Builders consume the versioned manifest rather than unrelated CSV files.
- Supervision sources are restricted to `human`, `deterministic_rule`, or `tool_verified`.
- Audit-only samples are rejected from primary training datasets.
- Unified examples bind the image, RT-interpolated sequence, scalar row, target row, source group,
  and provenance hashes to one asset identity.
- Scalar normalization is fit on train only; validation statistics cannot affect transforms.
- Image and sequence peak boundaries must agree in the shared normalized ROI coordinate system.

### T20-T22 — Training integration

- Qwen3-VL-8B LoRA configuration, precision, effective batch size, seed, and gradient accumulation are recorded.
- The 1D encoder output is projected into the expected multimodal representation shape.
- Every run emits an adapter or checkpoint, configuration snapshot, dataset version, logs, and run metadata.

## Evaluation checks

### T23 — Scientific report

The versioned scientific report must contain:

- Dataset and split signatures.
- Eligible and audit counts.
- Detection and boundary metrics.
- Quantification metrics.
- Blank-sample false positives.
- The declared ablation table.

### T24 — Agent report

The versioned agent report must contain:

- Tool-status distribution.
- Tool-selection and QC accuracy.
- Abstention score.
- Structured-output validity.
- Explanation-faithfulness score.
- Evidence-attribution completeness.

### T25 — Registry evidence gate

- Missing scientific evidence blocks promotion.
- Missing agent evidence blocks promotion.
- Audit contamination rejects the candidate.
- Gate decisions record the exact report artifacts used.

## End-to-end checks

### T26 — Full demonstration

1. Select a manifest-backed chromatogram sample.
2. Resolve aligned image, sequence, metadata, and provenance.
3. Invoke the ChromPeakFormer tool.
4. Run multimodal reasoning and QC.
5. Render the ROI, RT interval, area, QC state, evidence, and agent trajectory.

### T27 — Documentation traceability

Sample metrics from the README, model card, and experiment report and verify that each resolves to a run, dataset version, configuration, and evaluation report.

## Training and ablation matrix

| Run | Image | Sequence | Metadata | ChromPeakFormer tool | Domain training |
|---|---:|---:|---:|---:|---|
| Zero-shot ROI | Yes | No | No | No | None |
| Zero-shot image + metadata | Yes | No | Yes | No | None |
| Qwen3-VL LoRA | Yes | No | Optional | No | LoRA |
| Sequence baseline | No | Yes | Optional | No | 1D encoder |
| Image + sequence | Yes | Yes | No | No | LoRA + projector |
| Image + sequence + metadata | Yes | Yes | Yes | No | LoRA + projector |
| Full system | Yes | Yes | Yes | Yes | LoRA + projector |

No multimodal-gain claim is accepted without comparison to the strongest eligible unimodal baseline.

## Observability requirements

- Every tool invocation records status, reason, version, and sample identity.
- Every training run records its dataset version and configuration snapshot.
- Every evaluation records immutable report paths and split signatures.
- Every registry decision records its scientific and agent evidence summary.

## Exit criteria

- AC-01 through AC-10 each have passing evidence.
- Scientific and agent reports are reproducible.
- No audit-only sample contaminates primary training or evaluation.
- Public metrics and claims pass traceability checks.
