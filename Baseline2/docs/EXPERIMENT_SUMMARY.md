# Baseline 2 — Experiment and result summary

## Experimental setup

- **Architecture:** Single ImageNet-pretrained EfficientNet-B0 feature extractor shared by separate Crop (55) and Disease (175) heads; cross-entropy loss for each task.
- **Training paradigm:** Supervised multi-task transfer learning. **Challenge:** generalisation to crop–disease combinations absent during training, while individual classes remain known.
- **Candidate policy:** 55 × 175 = 9,625 pairs scored by `log P(crop) + log P(disease)`.
- **Development split:** train-fit 6,905, seen-val 695, unseen-val 400.
- **Final evaluation:** 2,000 official seen + 1,448 official unseen; Resize(256, bicubic), CenterCrop(224), ImageNet normalisation, no TTA.

## Experiment timeline

| Series | Purpose | Outcome supported by available artifacts |
|:--|:--|:--|
| Exp01–Exp05 | Stage 1 hyperparameter / sampling comparisons | Exp04 selected as best original checkpoint; per-run validation metrics are listed below |
| **Exp04** | Tempered disease-balanced sampling (α=0.25) | Development seen **66.47%**, unseen **2.25%**, HM **4.35%**, checkpoint epoch **13** |
| FT01–FT04 | Development fine-tuning / freezing / head-retraining trials | Completed; Exp04 remained selected for full-data parent; per-run validation metrics are listed below |
| **Exp06** | Gamma calibration on seen pair scores | Fixed γ=4 selected on development, seen **46.19%**, unseen **6.50%**, HM **11.40%**; post-processing only |
| Exp07 | Two-view test-time augmentation | Development TTA only HM **4.35%**; TTA + γ=4 HM **10.22%**; omitted from final |
| **Stage 2B** | Train on all **8,000 original training images** from Exp04 | 3 additional epochs; low LR `1e-6` backbone and `1e-4` heads; saved final checkpoint; former validation no longer independent |

### Saved development validation metrics

| Run | Best epoch | Seen Top-1 | Unseen Top-1 | Harmonic mean |
|---|---:|---:|---:|---:|
| EXP01 | 11 | 66.19% | 2.00% | 3.88% |
| EXP02 | 13 | 66.47% | 1.75% | 3.41% |
| EXP03 | 14 | 66.47% | 1.25% | 2.45% |
| EXP04 | 13 | 66.47% | 2.25% | 4.35% |
| EXP05 | 13 | 64.75% | 1.75% | 3.41% |
| FT01 | 1 | 66.19% | 2.25% | 4.35% |
| FT02 | 1 | 66.19% | 2.25% | 4.35% |
| FT03 | 3 | 66.33% | 1.50% | 2.93% |
| FT04 | 9 | 64.46% | 1.25% | 2.45% |

The table uses the original validation_metrics.json files. Compact reports are available for [Exp04](../outputs/development/exp04/validation_metrics.json), [Exp06](../outputs/development/exp06/calibration_summary.json), and [Exp07](../outputs/development/exp07/tta_summary.json).

## Official composition results (frozen Stage 2B checkpoint)

| Method | Seen Top-1 | Unseen Top-1 | HM | Seen Top-5 | Unseen Top-5 |
|:--|--:|--:|--:|--:|--:|
| **Original B2 (γ=0, primary)** | **55.60%** | **1.86%** | **3.61%** | 78.85% | 10.15% |
| Optional calibration (γ=4) | 38.70% | 4.77% | 8.49% | 59.35% | 17.33% |

**Uncalibrated Crop/Disease Top-1:** Seen crop **84.65%** / disease **59.65%**; unseen crop **61.60%** / disease **7.94%**. These individual-head metrics are unchanged under calibration.

**B1 teammate-reported official results:** Seen **57.70%**, unseen **2.28%**, HM **4.38%**. This is a reference only, not independently verified from B1's raw metrics. Original B2 is numerically lower: seen −2.10 percentage points, unseen ≈−0.42 pp, HM ≈−0.77 pp. The calibrated extension has a higher HM but a much lower seen score. Differences in preprocessing, sampling and model-selection methods limit architecture-only causal claims.

## Error analysis / interpretation

The original unseen disease head achieved **7.94% Top-1**, versus unseen crop Top-1 of **61.60%**. Thus disease recognition, and its interaction with crop compositionality, is a bottleneck. Calibration subtracts a preselected γ=4 score penalty from all **323 training-seen pairs** among the 9,625 candidates; the other **9,302** pairs remain unchanged. It improves unseen composition Top-1 and HM, but reduces seen Top-1. The development-only mask had 313 pairs; do not reuse it for final evaluation.

**Test hygiene:** These official scores are final reporting evidence; no additional tuning or winner-selection may be based on them. Confusion matrix PNGs are compressed 25-class + 'Other' visual summaries; full raw confusion CSVs and per-image errors are preserved in the owner's original local output folder.
