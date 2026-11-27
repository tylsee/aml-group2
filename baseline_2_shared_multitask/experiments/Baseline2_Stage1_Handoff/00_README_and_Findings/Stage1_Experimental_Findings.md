# ResNet-50 Stage 1 experimental findings

Development validation only. These results are not Official Test scores. Training was not rerun, and metrics below are transcribed from saved validation_metrics.json files. Experiment_Comparison.csv retains the full stored precision and evidence paths.

| Experiment | Seen % | Unseen % | HM % | Best epoch | Training method |
|---|---:|---:|---:|---:|---|
| b2_resnet50_exp09_b1matched_seed42 | 72.95 | 4.00 | 7.58 | 12 | B1-matched ResNet50-V2: WD=1e-3, train/val interpolation aligned, 4 loader workers |
| exp10 | 73.24 | 4.00 | 7.59 | 13 | Exp09 matched settings; ONLY disease-frequency tempered sampling (alpha=0.25, replacement=True) |
| exp13_tempered_alpha05 | 73.24 | 3.75 | 7.13 | 28 | Compared with Exp09 natural shuffle: ONLY disease-frequency tempered sampling alpha=0.5 with replacement |
| exp14_effective_number_disease | 72.66 | 3.75 | 7.13 | 16 | Compared with Exp09 natural shuffle: ONLY disease-head effective-number class-balanced CE beta=0.99 |
| exp15_disease_weight15 | 72.52 | 4.25 | 8.03 | 12 | Vs Exp09: ONLY disease task CE loss weight changed from 1.0 to 1.5 |
| exp16_disease_weight20 | 72.52 | 3.75 | 7.13 | 12 | Vs Exp09: ONLY disease task CE loss weight changed from 1.0 to 2.0 |
| exp17_crt_disease_alpha05 | 72.52 | 5.00 | 9.35 | 20 | cRT extension from Exp09: frozen backbone/crop; reset disease head + tempered disease sampling alpha=0.5 |
| exp18_crt_seed123 | 71.08 | 5.50 | 10.21 | 8 | Repeat Exp17 cRT recipe, change random seed 42 -> 123 (same Exp09 seed42 frozen parent) |
| exp18_crt_seed2026 | 72.09 | 5.25 | 9.79 | 15 | Repeat Exp17 cRT recipe, change random seed 42 -> 2026 (same Exp09 seed42 frozen parent) |

## Independent verification and discrepancies

The five previously recorded Seen and Unseen values match the saved files at two decimal places. HM values also match except Exp17: the saved fraction is 0.09354988399071926, or 9.354988399071926%, which rounds to **9.35%**, not the previously recorded 9.36%. The original files are preserved, including any differing summaries.

Across classifier-stage seeds 42, 123 and 2026, mean saved HM is **9.78405%**, sample standard deviation **0.42750 percentage points** (n=3, ddof=1); population SD is 0.34905 pp. Thus approximately 9.78% ± 0.43 pp is supported as mean ± sample SD. This calculation summarizes existing results; no predictions or evaluation metrics were regenerated.

## Observations and limitations

Exp15 (disease task-loss weight 1.5) is the strongest observed single-stage training configuration among these controlled training comparisons. Weight 2.0, stronger sampling (alpha 0.5), and effective-number weighting did not improve HM over Exp09. Exp11/12 are separately preserved calibration/exploration results and must not be conflated with uncalibrated training results in this table.

Exp17/18 are additional cRT experiments: Exp09 backbone and crop head are frozen, disease head is reset and trained with alpha 0.5 disease-frequency sampling. All three seeds share the same Exp09 seed-42 backbone; this verifies classifier-stage seed variability only, not full end-to-end multi-seed robustness. The three configurations record the same parent SHA-256. cRT combines representation freezing, head reset and sampling changes, so it is not a one-factor ablation.

Unseen composition accuracy remains only 5.00–5.50% (20–22 of 400 images), despite seen accuracy around 71–73%. This is weak compositional generalisation, not evidence of broadly successful unseen recognition. The development set was reused for selection; small differences and three classifier seeds do not establish statistical significance.

Stage 2 full-data fine-tuning and Official Test for this ResNet-50 branch are not established by this handoff. The workspace separately contains older EfficientNet-related Stage 2/Official Test files and README claims; those are excluded and must not be attributed to these experiments.

## Next steps for the Proposed New Approach

Review the disease-head bottleneck, rare disease errors, per-class support and confusion matrices. Use the shared frozen-parent comparison to motivate changes to representations or compositional modelling; test full end-to-end seeds before making robustness claims. Keep development selection separate from a future locked final evaluation. No Stage 2 or Official Test action was performed during preparation.
