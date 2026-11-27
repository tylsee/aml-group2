# Baseline 2 — Exp12 consolidated calibration and error findings

**Status:** Completed post-hoc scoring only; NO training, Stage 2, official test, or 333-pair restriction.
**Sources:** Fixed Exp09 & Exp10 ResNet-50 stage-1 checkpoints via cached Exp11 development logits.
**Important:** Calibration sweeps on reused development data are exploratory, not independent generalisation estimates.
**Reproduction warning:** Exp11 logits give Exp10 seen 73.09% vs original training-time 73.24% (one image/695). New comparisons are internally consistent on cached logits.

## Summary (full development; exploratory)
### exp09
- **Baseline:** baseline crop_tau=0, disease_tau=0, gamma=0: Seen 72.95%; Unseen 4.00%; HM 7.58%
- **Highest tune-subset HM:** seen_pair_gamma crop_tau=0, disease_tau=0, gamma=6: Seen 46.04%; Unseen 6.75%; HM 11.77%
- **Seen-preserving (≤2 pp tune Seen drop):** prior+gamma crop_tau=0.25, disease_tau=0, gamma=1: Seen 71.80%; Unseen 4.50%; HM 8.47%
- **Strict dual-improvement settings (full development):** 0
- See `selected_settings_internal_check.csv` for exploratory tune/check metrics and warnings.

### exp10
- **Baseline:** baseline crop_tau=0, disease_tau=0, gamma=0: Seen 73.09%; Unseen 4.00%; HM 7.58%
- **Highest tune-subset HM:** seen_pair_gamma crop_tau=0, disease_tau=0, gamma=6: Seen 46.33%; Unseen 8.25%; HM 14.01%
- **Seen-preserving (≤2 pp tune Seen drop):** crop+disease_prior crop_tau=0.5, disease_tau=0.2, gamma=0: Seen 72.23%; Unseen 4.50%; HM 8.47%
- **Strict dual-improvement settings (full development):** 0
- See `selected_settings_internal_check.csv` for exploratory tune/check metrics and warnings.

## Explanation of the γ=6 mechanism
A fixed seen-pair penalty changes only ranking: unseen candidates are promoted relative to training-seen candidates.
It does not retrain the disease recognition head or add new symptom features.
The table below counts actual examples recovered and broken (not merely net gain):

- exp09/val_seen: recovered 0, lost 187, net -187
- exp09/val_unseen: recovered 11, lost 0, net 11
- exp10/val_seen: recovered 0, lost 186, net -186
- exp10/val_unseen: recovered 17, lost 0, net 17

## Evidence files
- `all_calibration_settings.csv`: complete prespecified grid across crop/disease prior adjustments and mild γ.
- `selected_settings_internal_check.csv`: tune-subset choices and internal check-subset accuracy (NOT independent test).
- `gamma6_prediction_transitions.csv` / `gamma6_transition_counts.csv`: corrected vs broken images.
- `selected_per_disease_unseen.csv`: per-class disease and pair accuracy on held-out compositions.
- `calibration_seen_unseen_scatter.png`: seen/unseen trade-offs.

## Next controlled training experiments (NOT executed)
1. Exp13: stronger disease-tempered sampling α=0.5 compared with Exp09/Exp10 (seed 42, all other settings held constant).
2. Exp14: effective-number class-balanced disease loss (Cui et al.) without changing the natural shuffle sampler, separate experiment.
3. Exp15+: fixed multi-seed verification (42, 123, 2026) for preselected promising settings; report mean±SD.
Do NOT assume combining interventions improves results; isolate factors and validate on an independent protocol.

## Current evidence in one sentence
Exp10 tempered sampling barely changed development composition HM; Exp11 γ=6 raised unseen accuracy but substantially reduced seen accuracy. Exp12 investigates class-level corrections as a possible less destructive alternative.
