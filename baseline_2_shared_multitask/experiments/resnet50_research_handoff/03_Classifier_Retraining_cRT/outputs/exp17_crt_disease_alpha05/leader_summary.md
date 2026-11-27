# exp17_crt_disease_alpha05 — Development-only training results

- Intervention: cRT extension from Exp09: frozen backbone/crop; reset disease head + tempered disease sampling alpha=0.5
- Source: saved Exp09 ResNet50 V2 Stage-1 checkpoint; freeze backbone and crop head (including BatchNorm).
- Reset and train only disease linear head, using train-fit disease sampler alpha=0.5 with replacement.
- Batch16, image224, seed42, disease head LR1e-3, AdamW wd1e-3, maximum30 epochs, patience15, 9,625 candidate pairs.
- Best epoch: 20 (selected on development composition HM)
- Seen composition: 72.52%
- Unseen composition: 5.00%
- HM: 9.35%
- Unseen crop: 70.50%; unseen disease: 24.00%
- Unseen correct: 20/400

## Difference from Exp09 (percentage points)
- seen_comp: -0.43 pp
- unseen_comp: +1.00 pp
- HM: +1.77 pp
- unseen_crop: +0.00 pp
- unseen_disease: +2.00 pp

## Weak unseen disease classes (>=5 validation images)
```text
         disease_name  train_images  n_validation  top1_accuracy most_common_wrong_prediction
         spider_mites             5            70       0.000000                  target_spot
    angular_leaf_spot             7            18       0.000000                    bean_rust
        monillia_laxa             8             9       0.000000                    aphis_spp
bacterial_leaf_streak            12             5       0.000000              septoria_blotch
     bacterial_blight            27           117       0.008547                   brown_spot
          late_blight           103            70       0.300000                 early_blight
       powdery_mildew           150            60       0.466667                  sooty_mould
              healthy          1417            51       0.901961                       mosaic
```

## Protocol and limits
- No Stage 2, official test, calibration or re-training of reference checkpoints.
- Single seed (42); development set used many times. These are exploratory comparisons, not independent test scores.
- Do not interpret changes as statistically robust without multi-seed / independent evaluation.
- This is a bundled cRT experiment: frozen representation + disease head reset + balanced sampling; NOT a one-factor ablation.
- Original official test is not accessed or used for selection.
