# BioCoder multimodal v1.1 development extensions

Status: complete on 2026-10-10. Both development-only controls cover all 13,708 bilingual
validation prompts, their manifests verify, and the path-free public evidence artifact is
materialized. This work did not reopen, replace, or tune against the sealed v1 internal test.

## Why v1.1 exists

The sealed v1 benchmark established that aligned image + XIC fusion is substantially better than
image-only Qwen3-VL LoRA. Two questions remain useful for scientific interpretation:

1. Does the current-sample ROI image add information beyond XIC alone?
2. Does continuous sensor-token fusion outperform the simpler strategy of placing a frozen
   SequencePeakNet prediction in the text prompt?

The v1.1 experiments answer those questions only on the existing leakage-safe validation split.
They are not additional sealed-test candidates, and no v1 test metric may be used to select their
training configuration, prompt, checkpoint, or decoding settings.

## New controlled baselines

| Baseline | Model-visible inputs | Frozen initialization | Development purpose |
| --- | --- | --- | --- |
| XIC-only Qwen | XIC sensor tokens and the bilingual task prompt; no ROI image or pixel tensor | The same completed image-LoRA language adapter used by fusion | Tests whether the current-sample ROI image contributes beyond aligned XIC while holding the language initialization fixed |
| Image LoRA + SequencePeakNet prompt | ROI image, bilingual task prompt, and three frozen SequencePeakNet prediction fields serialized as text | Completed image-only LoRA and completed sequence-only specialist | Compares learned continuous fusion with the strongest simple expert-in-prompt baseline |

The XIC-only run is deliberately described as a *current-sample input ablation*. Its language LoRA
starts from the same domain adapter as the image + XIC model, so it does not claim that no image
supervision ever influenced initialization. During v1.1 XIC-only training and inference, however,
the processor receives no image message, `pixel_values` and `image_grid_thw` are absent, and a
visual-tower hook fails the run if the vision model executes.

The SequencePeakNet prompt bundle is built only from a completed development-only sequence run.
It allowlists `presence_probability`, `start_normalized`, and `end_normalized`; removes target,
loss, label, threshold-selection, and correctness fields; binds the source report, source manifest,
and prediction artifact by SHA-256; and requires exact coverage of the same 1,815 validation
assets. The appended English or Chinese note explicitly says that the values are predictions, not
ground truth. Validation answers remain unavailable until answer-separated evaluation.

## Completed public evidence

The exporter re-verified every source artifact and produced a path-free JSON report, rendered
Markdown, and `sha256sum -c` manifest:

- Public development report SHA-256:
  `a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842`
- Public development manifest SHA-256:
  `fc9539667b1e2f6f129db0de80119b83a9bb8dba4f5f7db5d1344da00ebce3d9`
- Rendered public development Markdown SHA-256:
  `bf22cdb7a74cb7d6ffcfa0fb6ba69b85a6f41a12cbab3a66fb16beff6524c082`
- Deterministic public archive:
  `biocoder-multimodal-v1.1-development-a1ede9932c23.zip`
- Deterministic public archive SHA-256:
  `cc859013f869480cc82b97cd3edf6f866e9e4023e26c2b80719f8cc1b4ccb094`
- Internal-test accessed: `false`
- Final-benchmark eligible: `false`

The complete generated tables are published in
[MULTIMODAL_V11_RESULTS.md](MULTIMODAL_V11_RESULTS.md). They include all three seed values and
mean ± sample SD, the 1/4/8-token ablation, the selected-checkpoint aligned/shuffled/zero/
availability-off interventions, learned gate values, uncapped LoRA and fusion wall times, the
negative auxiliary-initialization result, and both new controlled baselines.

The main findings are deliberately narrow:

- XIC-only Qwen achieved overall presence Macro-F1 `0.9291` and mean IoU `0.7163`. Because its
  language adapter started from image-LoRA weights, this is a current-sample image-input ablation,
  not an image-naive historical-training claim.
- Image LoRA with frozen SequencePeakNet predictions in the prompt achieved overall presence
  Macro-F1 `0.9401` and QC exact match `0.9686`, but mean IoU was only `0.5100`.
- Four sensor tokens gave the strongest localization; eight gave the strongest classification and
  QC. The effect is not monotonic.
- Auxiliary morphology pretraining completed but did not improve the primary downstream
  endpoints, so no positive initialization claim is made.

## Reproduce the canonical public development table

Run this against the persisted development artifacts; use paths outside Git for the output:

```bash
python -m multimodal_science.qwen3vl.build_public_development_evidence_cli \
  --development-dossier-root "<selected-checkpoint-development-dossier>" \
  --lora-training-root "<complete-image-lora-training-root>" \
  --fusion-training-root "<complete-random-projector-fusion-training-root>" \
  --auxiliary-pretraining-root "<complete-auxiliary-pretraining-root>" \
  --random-projector-evaluation-root "<random-projector-evaluation-root>" \
  --auxiliary-projector-evaluation-root "<auxiliary-projector-evaluation-root>" \
  --xic-only-evaluation-root "<complete-xic-only-evaluation-root>" \
  --sequence-prompt-evaluation-root "<complete-sequence-prompt-evaluation-root>" \
  --output-dir "<external-run-root>/comparisons/public-development-evidence-<revision>"

(
  cd "<external-run-root>/comparisons/public-development-evidence-<revision>"
  sha256sum -c artifact_manifest.sha256
)
```

The two post-seal evaluation roots must be supplied together for the completion artifact. The
exporter rejects incomplete seed or token matrices, a non-selected intervention checkpoint,
step-capped training, manifest drift, internal-test access, final-benchmark claims, and partial
auxiliary inputs. Its JSON payload contains no machine-specific paths.

## Completion gate

The gate is satisfied: both new baselines cover all 13,708 bilingual validation prompts, their
generation and answer-separated evaluation manifests verify, the exact metric rows are present in
the generated public evidence table, and no internal-test path or answer key was visible to either
model process. The negative auxiliary-initialization result is retained rather than hidden.
