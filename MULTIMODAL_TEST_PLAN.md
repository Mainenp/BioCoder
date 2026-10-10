# ChromPeakFormer Multimodal Test Plan

Status: the v1 scientific/model benchmark and its predeclared T01-T23 evidence are complete and
sealed. T24 agent evaluation, T25 agent-side promotion, and the interactive T26 product
demonstration remain open; model-benchmark eligibility alone does not satisfy them. T27 public
traceability is implemented through `MULTIMODAL_RESULTS.md` and `biocoder multimodal` release
verification. The materialized deterministic ZIP passed standalone verification at SHA-256
`a402c31de998f4392ca66798ae8c6db18d227ccc634415d322d82b6b55520481`. Any v2
reinforcement-learning experiment requires a new test protocol. The post-seal, validation-only
T23B controls are also complete and published in `MULTIMODAL_V11_RESULTS.md`; they do not extend
or reopen the T23A benchmark.

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
- Vendor-derived auxiliary jobs use a separate all-or-nothing index: each plan job, provenance
  record, JPEG, and XIC matrix must verify; any missing frame rejects the index rather than creating
  a partial training population.
- The auxiliary index accepts only `channel_driven_inference` plus
  `auxiliary_unlabeled_train`, emits no label or COCO target, fixes metric and benchmark eligibility
  to false, and reports transition traces separately from acquisition frames and source groups.
- Auxiliary signal materialization must re-verify the complete index manifest, every source matrix
  and image digest, crop on the physical RT axis, reproduce the supervised signal normalization,
  and preserve `labels=0`, `metrics_allowed=false`, and `internal_test_accessed=false`.
- Auxiliary projector pretraining may consume only nonconstant training signals. It must not load
  Qwen, the vision tower, validation prompts, validation answers, or internal test data. Its loss is
  not a scientific metric, its output remains comparison-ineligible, and its report must state that
  ROI images and XIC arrays are derived views rather than independent experimental modalities.
- Formal fusion may consume an auxiliary-pretrained projector only when the completed pretraining
  report, exact manifest, projector architecture specification, and weight digest all match. The
  launcher must stage it inside the existing train-only boundary and record the binding in the
  formal fusion report.

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
- Qwen3-VL train records follow the official one-image conversation contract and contain exactly
  one `<image>` token; visual tokens are forbidden in answers.
- The legacy English-only v1 contract remains reproducible. Bilingual v2 assigns one deterministic
  language to each train instruction and emits paired `en`/`zh-CN` validation prompts with distinct
  instruction IDs and one shared semantic `pair_id`.
- Canonical JSON keys and controlled response values do not change with prompt language. Reports
  expose language counts and never count parallel language prompts as independent source assets.
- Validation prompts contain no reference answer. Their answer key is a separate hashed artifact,
  and the builder exposes no internal-test input surface.
- Qwen3-VL evaluation requires exactly one prediction for every validation instruction ID, verifies
  all instruction artifact hashes, records the prediction-file hash, and exposes no internal-test
  input surface.
- Malformed JSON and task-schema violations remain in the denominator as failures. Presence,
  grounding, and QC tasks retain separate metrics and source-grouped uncertainty; no synthetic
  cross-task score is reported.
- Bilingual evaluations report task metrics independently for `en` and `zh-CN`. Every semantic
  validation pair contributes to a cross-language consistency rate; invalid output makes the pair
  inconsistent, and grounding additionally reports prediction-to-prediction box IoU.
- Model generation consumes a separately published prompt-only bundle. Its CLI exposes no
  instruction-Dataset, answer-key, instruction-manifest, or internal-test input surface.
- A resumable generation journal binds each instruction ID to its task, image and prompt hashes,
  language/pair identity, raw response, and response hash in original prompt order.
- Evaluation remains development-comparison ineligible unless it independently verifies full,
  uncapped Transformers generation, `transformers>=4.57.0`, immutable model identity, all artifact
  hashes, the no-answer-access contracts, and every per-row generation record.
- Oracle and evaluator-only reports validate contracts but are never model-quality claims.
- Reports distinguish unique source assets and source mzML groups from correlated multi-task
  instruction rows; derived row counts cannot be claimed as independent samples.
- A zero-shot failure audit accepts only a provenance-verified bilingual evaluation and matching
  generation report, revalidates their hashes and record identities, and exposes no model or
  internal-test input surface.
- Grounding protocol diagnostics retain the formal source-pixel metric while separately scoring
  full `0..1000` and horizontal-only `0..1000` counterfactual interpretations. Counterfactuals are
  never development comparisons and cannot overwrite the formal evaluation.
- Failure-audit reports include per-language output distributions, per-record coordinate evidence,
  source-grouped uncertainty, and a verifiable artifact manifest.

### T20-T22 — Training integration

- The specialist detector consumes COCO images and boxes only through an asset-index-hashed
  train/validation adapter, and source `job_id` overlap is a hard failure.
- Specialist inference emits standard COCO detections with immutable source, checkpoint, Dataset,
  and prediction hashes before any metric is calculated.
- Specialist reports include official COCO AP@[.50:.95], AP50, AP75, recall, image-level peak
  classification, and best-box IoU; validation-selected thresholds are marked development-only.
- The Qwen3-VL LoRA bundle opens only the train prefix and does not open validation prompts,
  validation answers, or internal-test inputs. It binds the instruction report, train rows,
  manifest, Dataset, and asset-index hashes before sampling.
- Sample-capped LoRA bundles use deterministic task/language/label stratification and remain smoke
  evidence rather than development comparisons.
- Qwen3-VL-4B LoRA configuration records BF16 precision, effective batch size, seed, gradient
  accumulation, image resolution, attention implementation, model artifact, trainable parameter
  fraction, and exact `q/k/v/o` adapter targets.
- LoRA training masks image and user tokens from the loss and supervises only the assistant answer.
  The base language weights, vision tower, and visual merger remain frozen.
- A run saves resumable adapter/optimizer/scheduler/RNG checkpoints and a final safe-tensor adapter.
- Resume reconstructs the deterministic epoch order and starts the DataLoader at the first unseen
  batch, rather than decoding and discarding all earlier images again.
- Adapter inference verifies the training report, artifact manifest, every adapter artifact, exact
  base-model hash, and train-only/frozen-base contracts before PEFT model loading.
- Fusion initialization additionally requires the adapter's recorded training-row SHA-256 to match
  the exact `train_qwen.jsonl` artifact joined to XIC signals; equal row counts are insufficient.
- Adapter CLI identity arguments are all-or-none; smoke-trained adapters remain development-only
  contract evidence even when inference covers every validation prompt.
- The scheduled `coder` LoRA smoke locks and selects a physical GPU only when both memory and
  utilization samples pass, performs two BF16 optimizer steps, reloads the persisted adapter
  through the hash-bound inference runner, and leaves both training and bounded generation
  explicitly ineligible for development comparison.
- The scheduled formal runner stages immutable code, the model cache, and hash-verified training
  images on node-local storage; it keeps resumable checkpoints on persistent storage, publishes
  only after manifest verification, and is uncapped unless calibration limits are explicitly set.
  An exclusive run-level lock rejects a duplicate job targeting the same revision and training
  configuration before either process can share history or checkpoint files.
  Training completion alone is not a development metric until answer-separated inference and
  evaluation pass on every validation instruction.
- The scheduled adapter-evaluation runner independently binds the completed training report and
  adapter manifest, stages and copy-verifies all validation images on node-local storage, resumes
  the prompt-only generation journal, and accepts development evidence only after all 13,708
  bilingual prompts pass generation-provenance and answer-separated evaluation contracts. It
  atomically materializes directory manifests after the inference and evaluator CLIs publish their
  native artifacts, so a post-generation scheduler retry reuses predictions instead of rerunning
  the model.
- The cross-family development report accepts only provenance-verified full Qwen generation and
  evaluation pairs, checks that zero-shot has no adapter and LoRA has a completed adapter, and
  binds both to the same immutable base model, instruction artifacts, Dataset, validation assets,
  and source groups as the specialist ablation.
- Cross-family Qwen metrics are primary per-language rows. Paired English and Chinese prompts are
  never summed into an independent scientific-sample count, and Qwen bbox IoU is never relabeled
  as detector COCO AP.
- The 1D encoder output is projected into the expected multimodal representation shape.
- Image/XIC link construction verifies the Dataset, LoRA bundle, and prompt-only bundle hashes;
  binds the complete LoRA train-row and selection-manifest content hashes; rejects
  train/validation source-group overlap; preserves unavailable-signal masks; and never opens
  validation answers. A rebuilt LoRA report is equivalent only when both bound artifacts match;
  legacy v1 fusion bundles retain exact report-hash identity and must be rebuilt for this path.
- The sensor projector maps each 160-point signal to four Qwen-width tokens behind a trainable
  near-closed residual gate. Projector-only tests are not evidence of end-to-end Qwen fusion.
- The fusion CUDA smoke must insert four continuous sensor embeddings before the assistant
  response without changing Qwen's tokenizer or output vocabulary, retain the native image path,
  preserve explicit multimodal RoPE prefix/suffix relations, execute the frozen visual tower,
  prove nonzero gradients for every LoRA attention target and every projector submodule, bind
  before/after parameter-state digests, consume staged train-only input roots from an environment
  that does not expose the original source-root variables, bind
  serialized outputs and the complete base-model file manifest by hash, reload saved weights, and
  remain development-comparison ineligible.
- The first accepted fusion CUDA evidence is job `6626`: both optimizer updates completed, all
  forward/backward and exact-manifest gates passed, and the persisted report and manifest hashes
  are recorded in the roadmap. Its two losses are execution diagnostics only. They cannot be used
  as accuracy evidence, and the Flash Attention warning forbids a bitwise-determinism claim.
- Promotion beyond smoke requires a resumable uncapped train-only fusion run followed by complete
  prompt-only generation and answer-separated evaluation on the same 1,815 validation assets.
  Neither the bounded run nor its training losses may enter a development comparison table.
- The formal fusion trainer must default to no `max_steps`, accept only batch size one until padded
  sensor-token M-RoPE is proven, checkpoint the adapter, projector, optimizer, scheduler, RNG, and
  exact next-batch position together, and resume from the deterministic unseen sample suffix. A
  calibration cap must make `development_training_complete=false`; training alone must never make
  `development_comparison_eligible=true`.
- The formal Slurm boundary must stage only train link rows, train LoRA rows/selections, train XIC
  arrays/examples, hash-verified train images, the complete initial-adapter manifest, and the
  immutable base model. Validation prompts, validation answers, and internal-test paths are not
  accepted by the training CLI or launcher.
- Fused validation generation must reject incomplete/calibration adapters, bind the exact formal
  adapter and projector manifests, align every prompt to one validation XIC row and image digest,
  use greedy batch-one decoding with explicit three-axis M-RoPE positions, and expose no answer-key
  or internal-test argument. The answer-separated evaluator remains a downstream process.
- XIC-use evidence must include aligned, seeded deranged-shuffle, measured-zero, and
  availability-off runs from the same completed adapter. The shuffle operates on independent
  asset rows, has no fixed points, keeps bilingual variants on one donor, and persists its mapping
  hash without consulting answers.
- The four intervention reports must be joined only after verifying the exact base model, fusion
  checkpoint, prompt and answer artifacts, decoding settings, intervention seed, and evaluation
  record identities. Primary uncertainty uses a paired bootstrap over source groups; prompts and
  language variants are never bootstrap units. Selected metrics are recomputed from the hash-bound
  evaluation records and checked against every source report before deltas are accepted.
- Sensor-token ablations are restricted to `1`, `4`, and `8`. Training run identity and reports
  bind the count; auxiliary initialization across counts records source/target counts and the
  adaptive-pooling remap. Every formal training report records initial and final gate probability.
- The reproducibility matrix uses random projector initialization for every row, trains the
  four-token primary at seeds `17/29/43`, and changes only token count for the one- and eight-token
  seed-17 rows. Its Slurm array is capped at two concurrent entries and forbids sample/step caps.
- All five matrix training runs must finish 3,396 optimizer updates, retain one final checkpoint,
  and pass exact artifact-manifest verification before evaluation. Their validation array fixes
  greedy batch-one generation seed `17` for every row; training seed is provenance, not an
  inference hyperparameter.
- Matrix aggregation accepts exactly the five declared cells. It verifies training, generation,
  prediction, evaluation-record, and evaluation-report hashes; recomputes selected metrics from
  the 13,708 records; and requires identical instruction/pair/asset/group/task/language/target
  identities across runs. Four-token seeds report arithmetic mean and sample standard deviation
  (`ddof=1`). Token counts `1/4/8` have only seed 17 and therefore report raw values and directed
  deltas, never a fabricated cross-seed variance.
- English and Chinese are reported separately but remain paired views of 1,815 assets. The matrix
  report includes per-seed English-minus-Chinese gaps and cross-language consistency; neither the
  13,708 prompt count nor 6,854 language rows is treated as an independent-sample count.
- The pre-test development dossier accepts only manifest-verified cross-family, five-cell matrix,
  XIC-intervention, and selected seed-17 evaluation artifacts. It verifies one Dataset identity,
  all five declared matrix cells, seeds `17/29/43`, sample-SD recomputation, prompt/answer/model
  hashes, and evaluation-record identities. It binds the selected evaluation report and records
  to the four-token seed-17 matrix row and generates both the main table and deterministic
  bilingual/localization failure records. Grounding correctness uses IoU >= 0.5 rather than
  coordinate-exact equality.
- The dossier must record whether the intervention analysis uses the exact selected checkpoint.
  Protocol-level evidence from an independently trained checkpoint is retained but cannot silently
  satisfy the selected-candidate causal gate or authorize sealed-test access.
- Every run emits an adapter or checkpoint, configuration snapshot, dataset version, logs, and run metadata.
- Sequence-only and sequence-plus-metadata runs share the same encoder and heads so that their
  ablation changes exactly one input modality.
- Sample-capped smoke runs are ineligible even for development comparisons.
- One-epoch specialist-detector smoke runs carry the same explicit ineligibility marker.
- Full train/validation runs may support ablations but remain final-benchmark and promotion
  ineligible until sealed internal-test and all T23 evidence exists.
- The sequence runner has no internal-test input surface and defaults to CPU unless CUDA is
  explicitly selected.
- A sequence run is not accepted until an independent, checkpoint-safe validator reproduces its
  threshold, classification metrics, physical boundary metrics, and source-grouped bootstrap from
  saved per-asset predictions and verifies every artifact hash.

## Evaluation checks

### T23 — Scientific report

The versioned scientific report must contain:

- Dataset and split signatures.
- Eligible and audit counts.
- Detection and boundary metrics.
- Quantification metrics.
- Blank-sample false positives.
- The declared ablation table.
- Detection thresholds are selected without internal-test labels and frozen before test access.
- Confidence intervals bootstrap complete source mzML groups, not individual compound ROIs.
- Boundary metrics are restricted to labelled positives and include both normalized and physical
  time errors.

### T23A — One-time sealed model benchmark

- Protocol freezing must verify the completed development dossier, exact five-candidate set,
  validation-frozen thresholds, split manifest, derivation plan, train-fitted normalization,
  immutable model artifacts, decoding settings, and 10,000-resample source-group bootstrap policy
  without reading or materializing any internal-test record.
- The sole access ledger must be created atomically before the first protected record is read.
  A second access event is forbidden; crash recovery may resume only the identical protocol and
  access ID. Completion may be retried only with the identical final evidence manifest.
- Prompt-only inference inputs and evaluator-only answers must be physically separate and
  independently hash-bound. Model processes receive no answer-root path.
- Exactly five primary candidates are evaluated: Qwen3-VL zero-shot, image-only LoRA, image+XIC,
  SequencePeakNet, and ChromPeakFormer. No post-test model, token-count, threshold, prompt, or
  decoding selection is allowed.
- Qwen results are reported separately for English and Chinese. Bootstrap units are complete source
  mzML groups; prompt rows and language variants are never treated as independent samples.
- Final outputs must contain the five-candidate table, every bound evaluation report, an evidence
  registry, the access completion record, and exact SHA-256 manifests. Model-benchmark eligibility
  does not imply BioCoder agent promotion.

### T23B — Post-seal v1.1 development controls

- Never reopen the sealed internal test. Both controls are validation-only and must record
  `final_benchmark_eligible=false` and `internal_test_accessed=false`.
- XIC-only Qwen must omit the image message, `pixel_values`, and `image_grid_thw` in training and
  inference. A visual-tower hook must fail the run if any current-sample image forward occurs.
- XIC-only and image + XIC use the same completed image-LoRA language initialization, train rows,
  aligned XIC bundle, seed, sensor-token count, optimizer settings, decoding, and evaluator. The
  claim is therefore about current-sample input contribution, not image-free historical training.
- The SequencePeakNet prompt bundle accepts only a completed sequence-only development run and
  exposes exactly `presence_probability`, `start_normalized`, and `end_normalized` plus sample
  identity. Target, label, correctness, loss, and threshold-selection fields are forbidden.
- Sequence-prompt inference keeps the ROI image and image-only LoRA fixed, appends a language-matched
  prediction-as-evidence note, requires exact coverage of all 1,815 validation assets, and keeps
  the instruction answer root outside the model process.
- Both model processes use greedy decoding and complete all 13,708 bilingual prompts before the
  existing answer-separated evaluator may open validation answers.
- The public development exporter must verify every input manifest and publish: three-seed
  mean/sample-SD with all seed values; one/four/eight-token seed-17 rows; aligned, shuffled, zero,
  and availability-off rows; uncapped LoRA/fusion wall time; gate values; and, when supplied, the
  auxiliary-projector comparison. It must emit no absolute paths.
- Completion evidence: both controls produced 13,708 predictions; the exporter contract and
  manifest verification passed with report SHA-256
  `a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842` and manifest SHA-256
  `fc9539667b1e2f6f129db0de80119b83a9bb8dba4f5f7db5d1344da00ebce3d9`. The report records
  `internal_test_accessed=false` and `final_benchmark_eligible=false`. The deterministic public ZIP
  was generated twice with identical bytes and independently verified at SHA-256
  `cc859013f869480cc82b97cd3edf6f866e9e4023e26c2b80719f8cc1b4ccb094`.

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
| Qwen XIC-only control | No | Yes | No | No | LoRA + projector; shared image-LoRA initialization |
| Image LoRA + sequence-prediction prompt | Yes | Frozen prediction text | No | No | Frozen LoRA and frozen 1D encoder |
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
