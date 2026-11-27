# exp14_effective_number_disease — Development-only training results

- Intervention: Compared with Exp09 natural shuffle: ONLY disease-head effective-number class-balanced CE beta=0.99
- All other key Exp09 settings fixed: ResNet50_V2, batch16, size224, seed42, LR 1e-5/1e-3, AdamW, weight decay 1e-3, 9,625 candidate pairs.
- Best epoch: 16 (selected on development composition HM)
- Seen composition: 72.66%
- Unseen composition: 3.75%
- HM: 7.13%
- Unseen crop: 68.25%; unseen disease: 23.00%
- Unseen correct: 15/400

## Difference from Exp09 (percentage points)
- seen_comp: -0.29 pp
- unseen_comp: -0.25 pp
- HM: -0.45 pp
- unseen_crop: -2.25 pp
- unseen_disease: +1.00 pp

## Weak unseen disease classes (>=5 validation images)
| disease_name          |   train_images |   n_validation |   top1_accuracy | most_common_wrong_prediction   |
|:----------------------|---------------:|---------------:|----------------:|:-------------------------------|
| spider_mites          |              5 |             70 |      0          | healthy                        |
| angular_leaf_spot     |              7 |             18 |      0          | bean_rust                      |
| bacterial_leaf_streak |             12 |              5 |      0          | septoria_blotch                |
| bacterial_blight      |             27 |            117 |      0.00854701 | brown_spot                     |
| monillia_laxa         |              8 |              9 |      0.111111   | canker                         |
| late_blight           |            103 |             70 |      0.271429   | early_blight                   |
| powdery_mildew        |            150 |             60 |      0.4        | healthy                        |
| healthy               |           1417 |             51 |      0.921569   | mosaic                         |

## Protocol and limits
- No Stage 2, official test, calibration or re-training of reference checkpoints.
- Single seed (42); development set used many times. These are exploratory comparisons, not independent test scores.
- Do not interpret changes as statistically robust without multi-seed / independent evaluation.
- Exp13 tests alpha 0.5 vs natural / alpha 0.25; Exp14 separately tests CB disease loss while restoring natural shuffle.
- Original official test is not accessed or used for selection.
