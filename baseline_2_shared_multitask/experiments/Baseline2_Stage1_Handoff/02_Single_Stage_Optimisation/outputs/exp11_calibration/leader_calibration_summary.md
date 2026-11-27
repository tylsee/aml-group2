# Exp11 — Post-hoc Calibration Findings (DEVELOPMENT ONLY)

**Scope:** Exp09 & Exp10 ResNet-50 V2 stage-1 checkpoints, original fixed development splits, all 9,625 candidates. No retraining, Stage 2 or Official Test.
**Important:** hyperparameters were swept on the same development validation images used to report results; peak development scores are exploratory and may be optimistic.

## Main findings
### exp09
- Baseline: Seen 72.95% / Unseen 4.00% / HM 7.58% (16/400)
- Highest measured development HM: Seen 46.04% / Unseen 6.75% / HM 11.77% (27/400) at method=seen_pair_gamma, γ=6, τ=0.
- Strict Seen+Unseen improvement found in measured grid: **NO**.

### exp10
- Baseline: Seen 73.09% / Unseen 4.00% / HM 7.58% (16/400)
- Highest measured development HM: Seen 46.33% / Unseen 8.25% / HM 14.01% (33/400) at method=seen_pair_gamma, γ=6, τ=0.
- Strict Seen+Unseen improvement found in measured grid: **NO**.

## Methods and interpretation
- Uniform seen-pair penalty γ >= 0 is a trade-off: it cannot increase seen Top-1 for a fixed set of raw scores; use the full seen-unseen curve.
- Disease logit adjustment uses π(d) from training-fit ONLY and applies z_d - τ log π(d). This may improve minority disease recall but can hurt healthy/common classes.
- The combined γ/τ grid is exploratory and limited; it is not a hyperparameter guarantee.
- Do not infer generalisation beyond these ten held-out development compositions without independent assessment.
- Candidate restriction to 333 was NOT tested. Multi-seed re-training was NOT performed.

## Files
- calibration_sweep.csv: full evaluated settings, including disease and macro disease recall.
- calibration_summary.csv: control and highest measured development-HM settings.
- seen_unseen_calibration_tradeoff.png: trade-off scatter plot.
- exp09/exp10 development raw logits: NPZ files in this directory.

## References (background, not claims of implementation)
- Chao et al. (2016), An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild.
- Menon et al. (2021), Long-Tail Learning via Logit Adjustment.
