# BioCoder multimodal v1.1 development evidence

Status: complete on 2026-10-10. This page is leakage-safe validation evidence and is separate
from the immutable v1 sealed internal-test benchmark. V1.1 did not reopen that test set.

## Evidence identity

- Dataset version: `raw-072fee8e`
- Evidence code revision: `db7ee168dc861847458ce5212668634a1245bccf`
- Validation assets: `1,815`
- Bilingual validation prompts: `13,708`
- Independent validation source mzML groups: `11`
- Public development report SHA-256:
  `a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842`
- Public development manifest SHA-256:
  `fc9539667b1e2f6f129db0de80119b83a9bb8dba4f5f7db5d1344da00ebce3d9`
- Internal-test accessed: `false`
- Final-benchmark eligible: `false`

The companion JSON binds every source report and manifest by SHA-256 and contains no
machine-specific paths. Language variants are paired views, not independent scientific samples.

## Four-token fusion reproducibility across three training seeds

Values are mean ± sample standard deviation (`ddof=1`) across training seeds 17, 29, and 43.

| Scope | Metric | Mean ± SD | Seed 17 | Seed 29 | Seed 43 |
| --- | --- | ---: | ---: | ---: | ---: |
| overall | Presence balanced accuracy | 0.9115 ± 0.0037 | 0.9117 | 0.9077 | 0.9152 |
| overall | Presence Macro-F1 | 0.9152 ± 0.0051 | 0.9155 | 0.9099 | 0.9201 |
| overall | Presence MCC | 0.8305 ± 0.0103 | 0.8312 | 0.8198 | 0.8405 |
| overall | Presence FPR | 0.1429 ± 0.0037 | 0.1429 | 0.1466 | 0.1392 |
| overall | Metadata Macro-F1 | 0.9210 ± 0.0050 | 0.9220 | 0.9255 | 0.9156 |
| overall | Metadata MCC | 0.8425 ± 0.0098 | 0.8443 | 0.8513 | 0.8319 |
| overall | Grounding mean IoU | 0.5841 ± 0.0139 | 0.5878 | 0.5687 | 0.5958 |
| overall | Grounding IoU@0.5 | 0.6936 ± 0.0296 | 0.7033 | 0.6604 | 0.7172 |
| overall | QC exact match | 0.9451 ± 0.0033 | 0.9419 | 0.9485 | 0.9449 |
| en | Presence balanced accuracy | 0.9121 ± 0.0035 | 0.9158 | 0.9089 | 0.9116 |
| en | Presence Macro-F1 | 0.9157 ± 0.0038 | 0.9172 | 0.9114 | 0.9185 |
| en | Presence MCC | 0.8316 ± 0.0077 | 0.8345 | 0.8229 | 0.8375 |
| en | Presence FPR | 0.1420 ± 0.0079 | 0.1330 | 0.1453 | 0.1478 |
| en | Metadata Macro-F1 | 0.9184 ± 0.0034 | 0.9197 | 0.9210 | 0.9145 |
| en | Metadata MCC | 0.8373 ± 0.0065 | 0.8396 | 0.8424 | 0.8300 |
| en | Grounding mean IoU | 0.5758 ± 0.0124 | 0.5809 | 0.5617 | 0.5848 |
| en | Grounding IoU@0.5 | 0.6816 ± 0.0305 | 0.6962 | 0.6466 | 0.7019 |
| en | QC exact match | 0.9447 ± 0.0055 | 0.9405 | 0.9510 | 0.9427 |
| zh-CN | Presence balanced accuracy | 0.9110 ± 0.0067 | 0.9077 | 0.9066 | 0.9188 |
| zh-CN | Presence Macro-F1 | 0.9146 ± 0.0067 | 0.9138 | 0.9084 | 0.9217 |
| zh-CN | Presence MCC | 0.8294 ± 0.0134 | 0.8280 | 0.8168 | 0.8435 |
| zh-CN | Presence FPR | 0.1437 ± 0.0116 | 0.1527 | 0.1478 | 0.1305 |
| zh-CN | Metadata Macro-F1 | 0.9236 ± 0.0067 | 0.9244 | 0.9299 | 0.9166 |
| zh-CN | Metadata MCC | 0.8477 ± 0.0133 | 0.8490 | 0.8602 | 0.8338 |
| zh-CN | Grounding mean IoU | 0.5924 ± 0.0156 | 0.5946 | 0.5757 | 0.6068 |
| zh-CN | Grounding IoU@0.5 | 0.7057 ± 0.0294 | 0.7104 | 0.6742 | 0.7324 |
| zh-CN | QC exact match | 0.9455 ± 0.0020 | 0.9433 | 0.9460 | 0.9471 |

## Sensor-token ablation at training seed 17

| Tokens | Presence Macro-F1 | Presence MCC | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact | Final gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8920 | 0.7848 | 0.9028 | 0.4726 | 0.4666 | 0.9300 | 0.017967 |
| 4 | 0.9155 | 0.8312 | 0.9220 | 0.5878 | 0.7033 | 0.9419 | 0.018469 |
| 8 | 0.9211 | 0.8423 | 0.9328 | 0.5221 | 0.5703 | 0.9529 | 0.018295 |

Four tokens gave the best localization, while eight gave the best classification and QC values.
The token effect is not monotonic, and the gate is a learned residual scale rather than a
percentage of decisions attributable to XIC.

## XIC interventions on the selected seed-17 checkpoint

| Intervention | Presence Macro-F1 | Presence MCC | Presence FPR | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 0.9155 | 0.8312 | 0.1429 | 0.9220 | 0.5878 | 0.7033 | 0.9419 |
| shuffled | 0.6888 | 0.3780 | 0.5025 | 0.6539 | 0.4715 | 0.4837 | 0.8303 |
| zero | 0.8281 | 0.6645 | 0.1675 | 0.8572 | 0.4144 | 0.3662 | 0.9022 |
| availability-off | 0.8175 | 0.6429 | 0.3645 | 0.8307 | 0.4097 | 0.3573 | 0.8986 |

The aligned-versus-shuffled collapse demonstrates sample-specific XIC use. The zero and
availability-off rows further show that neither a constant sensor representation nor merely
retaining the fusion interface reproduces the aligned result.

## Full training wall-clock measurements

| Run | Records | Optimizer updates | Wall seconds | HH:MM:SS | Report SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Image-only LoRA | 54,335 | 3,396 | 5,890.844 | 01:38:11 | `ffaba3391e83697d9d9caf4d228c694d7d33b09bd81d72da8a3744376ee94be2` |
| Image + XIC fusion | 54,335 | 3,396 | 22,209.544 | 06:10:10 | `3d78190fb3e10256bccc1a45244ae15d3319202deb568d56a6c805bb339898ad` |

These are uncapped one-epoch training-loop wall times. They exclude queueing, extraction,
validation inference, and evaluation.

## Auxiliary-projector initialization

The zero-label morphology pretraining run completed on 1,610 traces from 77 acquisition frames
and four source groups, but its controlled downstream comparison did not improve the primary
endpoints.

| Initialization | Presence Macro-F1 | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random projector | 0.9151 | 0.9173 | 0.5973 | 0.7236 | 0.9433 |
| Auxiliary pretrained | 0.9129 | 0.9101 | 0.5956 | 0.7207 | 0.9410 |

## Post-seal v1.1 controlled baselines

| Model | Scope | Presence balanced accuracy | Presence Macro-F1 | Presence MCC | Presence FPR | Metadata Macro-F1 | Mean IoU | IoU@0.5 | QC exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XIC-only Qwen | overall | 0.9199 | 0.9291 | 0.8588 | 0.1379 | 0.9340 | 0.7163 | 0.8825 | 0.9515 |
| XIC-only Qwen | en | 0.9225 | 0.9312 | 0.8630 | 0.1330 | 0.9354 | 0.7152 | 0.8829 | 0.9515 |
| XIC-only Qwen | zh-CN | 0.9172 | 0.9269 | 0.8547 | 0.1429 | 0.9325 | 0.7174 | 0.8822 | 0.9515 |
| Image LoRA + SequencePeakNet prompt | overall | 0.9399 | 0.9401 | 0.8802 | 0.0936 | 0.9361 | 0.5100 | 0.5241 | 0.9686 |
| Image LoRA + SequencePeakNet prompt | en | 0.9610 | 0.9511 | 0.9029 | 0.0468 | 0.9158 | 0.4977 | 0.4996 | 0.9669 |
| Image LoRA + SequencePeakNet prompt | zh-CN | 0.9188 | 0.9285 | 0.8579 | 0.1404 | 0.9557 | 0.5223 | 0.5486 | 0.9702 |

The expert-in-prompt baseline is strongest on presence classification and QC but substantially
worse on grounding than XIC-only Qwen. XIC-only Qwen is especially strong on localization. The
XIC-only run started from the shared image-LoRA domain adapter, but its training and inference made
zero current-sample vision-tower calls; it is therefore a current-sample input ablation, not a
claim of image-naive historical training.

## Product verification and archive entry points

The main `biocoder` executable can verify, display, and deterministically archive this evidence
without loading a model or opening any dataset:

```bash
biocoder multimodal verify-development \
  --evidence-root "<public-development-evidence-root>" \
  --report-sha256 "a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842"

biocoder multimodal show-development \
  --evidence-root "<public-development-evidence-root>" \
  --report-sha256 "a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842"

biocoder multimodal archive-development \
  --evidence-root "<public-development-evidence-root>" \
  --report-sha256 "a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842" \
  --archive "<public-development-evidence.zip>"
```

The archive contains only the path-free JSON, rendered Markdown, and their artifact manifest. It
contains no raw chromatograms, images, answer keys, per-record predictions, model weights, private
source, or machine-specific paths.
