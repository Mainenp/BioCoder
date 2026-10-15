# BioCoder LC-MS multimodal benchmark v1

Status: complete and sealed on 2026-10-08. This page reports the frozen five-candidate
internal-test benchmark. It is a model benchmark, not a BioCoder agent-promotion result.

## Evidence identity

- Dataset version: `raw-072fee8e`
- Frozen protocol SHA-256:
  `0bb30f51f4ab95f3ceee6d577ad2a8d698e224bfa240091011b42525ea6b2b25`
- Final benchmark report SHA-256:
  `78e0100e971cb1a6cc37e93775997c54c3e0db12e491ea77f6944706ab4465d2`
- Final evidence manifest SHA-256:
  `3575d0946e7c801931775222e5187f191a0263b407eaab27f9dc1efc1a8f607f`
- Public release manifest SHA-256:
  `82e8f7fed0d470aa4b3ba511977c4e6c71d53a15c913450702dc058a50c0f30d`
- Deterministic public ZIP SHA-256:
  `a402c31de998f4392ca66798ae8c6db18d227ccc634415d322d82b6b55520481`
- One-time access ID: `60f1c499121383188c0c6ec6`
- Internal-test access sequence: `1`
- Additional internal-test access authorized: `false`
- Internal-test assets: `1,815`
- Independent internal-test source mzML groups: `11`
- Final benchmark eligible: `true`

All candidates, thresholds, decoding settings, normalization, and metrics were frozen before the
sole test access. The completed access ledger is bound to the final evidence manifest. The
benchmark must not be rerun for tuning or candidate selection.

## Sealed results

Qwen rows are separated by language. Specialist rows are language-neutral because they do not
consume natural-language prompts.

| Model | Scope | Bal. acc. | Macro-F1 | MCC | FPR | Mean IoU | IoU@0.5 | QC exact | JSON valid | Schema valid | AP | AP50 | AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL zero-shot | en | 0.5000 | 0.4369 | 0.0000 | 1.0000 | 0.1240 | 0.0440 | 0.2242 | 1.0000 | 0.9035 | — | — | — |
| Qwen3-VL zero-shot | zh-CN | 0.5012 | 0.4394 | 0.0437 | 0.9975 | 0.3740 | 0.3203 | 0.2242 | 1.0000 | 1.0000 | — | — | — |
| Qwen3-VL image-only LoRA | en | 0.6970 | 0.7382 | 0.5485 | 0.5946 | 0.4302 | 0.3928 | 0.8953 | 1.0000 | 0.9996 | — | — | — |
| Qwen3-VL image-only LoRA | zh-CN | 0.7540 | 0.7923 | 0.6169 | 0.4693 | 0.4492 | 0.4226 | 0.8788 | 1.0000 | 1.0000 | — | — | — |
| Qwen3-VL image + aligned XIC | en | 0.9270 | 0.9319 | 0.8640 | 0.1204 | 0.5874 | 0.7109 | 0.9504 | 0.9999 | 0.9994 | — | — | — |
| Qwen3-VL image + aligned XIC | zh-CN | 0.9221 | 0.9285 | 0.8573 | 0.1302 | 0.6064 | 0.7230 | 0.9499 | 0.9999 | 0.9999 | — | — | — |
| SequencePeakNet | language-neutral | 0.9676 | 0.9668 | 0.9336 | 0.0491 | 0.7983 | 0.9538 | — | — | — | — | — | — |
| ChromPeakFormer | language-neutral | 0.8428 | 0.8678 | 0.7431 | 0.2875 | 0.7923 | 0.9339 | — | — | — | 0.5427 | 0.8661 | 0.5949 |

No combined cross-task score or post-test model selection is reported. Source-group bootstrap
intervals remain in each hash-bound evaluation report.

## What the result supports

- Aligned XIC materially improves the Qwen model over image-only LoRA on presence, localization,
  and QC, with closely matched English and Chinese performance.
- SequencePeakNet is the strongest specialist for peak presence and interval localization in this
  benchmark. The multimodal Qwen model is not claimed to replace it.
- ChromPeakFormer provides a strong image-detector baseline and the only COCO AP measurements.
- Untuned zero-shot Qwen3-VL is not a usable peak-presence classifier in this setting.
- Pre-test selected-checkpoint interventions showed that the fusion model used aligned XIC rather
  than merely tolerating it. Against shuffled XIC, aligned XIC improved presence Macro-F1 by
  `0.2611` (source-group bootstrap 95% CI `[0.2321, 0.3146]`) and grounding IoU@0.5 by `0.2672`
  (`[0.2512, 0.2821]`). These intervention numbers are validation evidence, not sealed-test rows.

## What the result does not support

- The 13,706 internal-test prompts (and 13,708 development prompts) are not independent
  scientific samples. Statistical uncertainty is grouped by the 11 source mzML files.
- Metrics from different task surfaces must not be collapsed into a single leaderboard score.
- The benchmark does not prove generalization to other laboratories, instruments, acquisition
  methods, compounds, or prevalence regimes.
- The benchmark does not certify the BioCoder agent's tool choice, arguments, abstention,
  trajectory quality, evidence attribution, or user-interface behavior.
- Reinforcement learning is not a missing v1 benchmark step. Any RL work is a new v2 experiment
  and requires train/development-only rewards plus a newly sealed test set; this test set cannot be
  reused for RL selection.

## Product and archive entry points

The main `biocoder` executable exposes the evidence layer without loading Qwen or private data:

```bash
biocoder multimodal verify-final \
  --protocol-root "<protocol-root>" \
  --protocol-sha256 "<frozen-protocol-sha256>" \
  --ledger-dir "<completed-access-ledger>" \
  --report-root "<final-report-root>"

biocoder multimodal archive-final \
  --protocol-root "<protocol-root>" \
  --protocol-sha256 "<frozen-protocol-sha256>" \
  --ledger-dir "<completed-access-ledger>" \
  --report-root "<final-report-root>" \
  --output-dir "<public-release-root>"

biocoder multimodal verify-release \
  --release-root "<public-release-root>" \
  --archive "<public-release-root>.zip"

biocoder multimodal show-results --release-root "<public-release-root>"
```

`archive-final` emits a deterministic ZIP and a manifest-bound directory containing only aggregate
metrics and cryptographic identifiers. It excludes raw chromatograms, ROI images, labels,
per-record predictions, model weights, private detector source, and machine-specific paths.
The materialized v1 capsule was independently verified after creation; its release-manifest and
ZIP hashes are recorded in the evidence identity above.

Interactive model/tool serving inside the BioCoder chat agent remains a separate product gate. The
research pipeline is directly runnable from this repository; it is not yet represented as a
production FastAPI or LangGraph tool.

## Post-seal development extensions

The sealed v1 table above is immutable. The two post-seal, validation-only controls are complete:
current-sample XIC-only Qwen reached overall presence Macro-F1 `0.9291` and mean IoU `0.7163`;
image-LoRA with frozen SequencePeakNet predictions in the prompt reached Macro-F1 `0.9401` and QC
exact match `0.9686`, but mean IoU `0.5100`. These values are not new sealed-test rows.

The complete V1.1 evidence—including three-seed mean ± sample SD, 1/4/8-token results, all four
XIC interventions, gate values, wall times, and the negative auxiliary-projector result—is in
[MULTIMODAL_V11_RESULTS.md](MULTIMODAL_V11_RESULTS.md). Its path-free report SHA-256 is
`a1ede9932c23b69d827232292208973c272be52253883425cd4c9a6930eb2842`, and its manifest SHA-256 is
`fc9539667b1e2f6f129db0de80119b83a9bb8dba4f5f7db5d1344da00ebce3d9`. The experimental protocol
and strict completion boundary remain in
[MULTIMODAL_V11_EXPERIMENTS.md](MULTIMODAL_V11_EXPERIMENTS.md). None of these artifacts modifies,
extends, or reopens the one-time internal-test benchmark.

The materialized deterministic archive is
`biocoder-multimodal-v1.1-development-a1ede9932c23.zip`, SHA-256
`cc859013f869480cc82b97cd3edf6f866e9e4023e26c2b80719f8cc1b4ccb094`. It contains only the
path-free public report, rendered Markdown, and manifest.
