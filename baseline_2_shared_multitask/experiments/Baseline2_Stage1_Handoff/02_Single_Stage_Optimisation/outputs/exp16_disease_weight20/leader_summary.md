# exp16_disease_weight20 — Development-only training results

- Intervention: Vs Exp09: ONLY disease task CE loss weight changed from 1.0 to 2.0
- All other key Exp09 settings fixed: ResNet50_V2, batch16, size224, seed42, LR 1e-5/1e-3, AdamW, weight decay 1e-3, 9,625 candidate pairs.
- Best epoch: 12 (selected on development composition HM)
- Seen composition: 72.52%
- Unseen composition: 3.75%
- HM: 7.13%
- Unseen crop: 69.50%; unseen disease: 22.50%
- Unseen correct: 15/400

## Difference from Exp09 (percentage points)
- seen_comp: -0.43 pp
- unseen_comp: -0.25 pp
- HM: -0.45 pp
- unseen_crop: -1.00 pp
- unseen_disease: +0.50 pp

## Weak unseen disease classes (>=5 validation images)
```text
         disease_name  train_images  n_validation  top1_accuracy most_common_wrong_prediction
     bacterial_blight            27           117       0.000000                   brown_spot
         spider_mites             5            70       0.000000                  target_spot
    angular_leaf_spot             7            18       0.000000                    bean_rust
bacterial_leaf_streak            12             5       0.000000                         rust
        monillia_laxa             8             9       0.111111           eriosoma_lanigerum
          late_blight           103            70       0.242857                 early_blight
       powdery_mildew           150            60       0.416667                  sooty_mould
              healthy          1417            51       0.921569                       mosaic
```

## Protocol and limits
- No Stage 2, official test, calibration or re-training of reference checkpoints.
- Single seed (42); development set used many times. These are exploratory comparisons, not independent test scores.
- Do not interpret changes as statistically robust without multi-seed / independent evaluation.
- This controlled experiment changes ONLY disease task-loss weight from Exp09; Exp13/14 are separate controlled comparisons.
- Original official test is not accessed or used for selection.
