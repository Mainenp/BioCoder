# BioCoder multimodal v1.1 development extensions

Status: implementation complete; two new development-only GPU evaluations remain to be
materialized. This work does not reopen, replace, or tune against the sealed v1 internal test.

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

## Existing development evidence that must be published

The following measurements already exist in hash-bound artifacts. The
`build_public_development_evidence_cli` command re-verifies those artifacts and emits one path-free
JSON report, Markdown table, and `sha256sum -c` manifest. It never opens the sealed internal test.

### Full uncapped training wall time

| Run | Train instructions | Optimizer updates | Wall seconds | Wall time | Training report SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Image-only Qwen3-VL LoRA | 54,335 | 3,396 | 5,890.844 | 01:38:11 | `ffaba3391e83697d9d9caf4d228c694d7d33b09bd81d72da8a3744376ee94be2` |
| Image + XIC fusion, seed 17 | 54,335 | 3,396 | 22,209.544 | 06:10:10 | `3d78190fb3e10256bccc1a45244ae15d3319202deb568d56a6c805bb339898ad` |

These are measured training-loop wall times from one full, uncapped, one-epoch run. They are not
end-to-end queue, data-extraction, validation-inference, or evaluation times.

### Selected-checkpoint XIC interventions

| Intervention | Presence Macro-F1 | Presence MCC | Presence FPR | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 0.9155 | 0.8313 | 0.1490 | 0.9173 | 0.5969 | 0.7225 | 0.9433 |
| shuffled | 0.6544 | 0.3149 | 0.6010 | 0.6142 | 0.4594 | 0.4553 | 0.8273 |
| zero | 0.8145 | 0.6490 | 0.1281 | 0.8644 | 0.4017 | 0.3481 | 0.8994 |
| availability-off | 0.7788 | 0.5865 | 0.4828 | 0.7729 | 0.4156 | 0.3747 | 0.8970 |

The paired source-group bootstrap report remains the statistical source of truth. In particular,
aligned minus shuffled was `0.2611` for presence Macro-F1 (95% CI `[0.2321, 0.3146]`) and
`0.2672` for grounding IoU@0.5 (`[0.2512, 0.2821]`).

### Sensor gate and auxiliary initialization

The final gate probabilities were `0.018469`, `0.018557`, and `0.018513` for four-token seeds
17/29/43, `0.017967` for one token at seed 17, and `0.018295` for eight tokens at seed 17. A gate
value is an internal residual scale, not a percentage of decisions attributable to XIC.

The unlabeled auxiliary pretraining run completed on 1,610 traces from 77 acquisition frames and
four source groups. Its controlled downstream comparison did not improve the primary endpoints:

| Projector initialization | Presence Macro-F1 | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.9151 | 0.9173 | 0.5973 | 0.7236 | 0.9433 |
| auxiliary pretrained | 0.9129 | 0.9101 | 0.5956 | 0.7207 | 0.9410 |

The exact three-seed mean ± sample standard deviation, all seed values, and the one/four/eight
sensor-token metric rows are intentionally generated from the manifest-verified development
dossier rather than copied by hand. The generated Markdown is the canonical public table.

## Materialize the canonical public development table

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

The two post-seal evaluation roots are optional while GPU runs are pending but must be supplied
together for the completion artifact. The exporter rejects incomplete seed or token matrices, a non-selected intervention checkpoint,
step-capped training, manifest drift, internal-test access, final-benchmark claims, and partial
auxiliary inputs. Its JSON payload contains no machine-specific paths.

## Completion gate

V1.1 is complete only when both new baselines cover all 13,708 bilingual validation prompts,
their generation and answer-separated evaluation manifests verify, the exact metric rows are
added to the generated public evidence table, and no internal-test path or answer key is visible to
the model process. Regardless of outcome, the results must be reported; a negative or tied
ablation is scientifically informative and must not be omitted.
