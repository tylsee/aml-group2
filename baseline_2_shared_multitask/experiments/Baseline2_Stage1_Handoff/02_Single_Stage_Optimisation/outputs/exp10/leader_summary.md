# Baseline 2 Exp10 — Results for team leader (development only)

## Objective and controlled change
- Same multi-task ResNet-50 ImageNet-1K V2, two heads, 224 image size, augmentation, 9,625 candidates, seed 42, weight decay 1e-3, batch 16, 4 workers, 6,905/695/400 split as Exp09.
- Exp09 uses shuffled sampling; Exp10 changes ONLY the train sampler to disease-frequency weights n_d^(-0.25), with replacement and 6,905 draws per epoch. Exp10 training examples may repeat; validation unchanged.
- Stage 1 checkpoint selected by development composition HM; official test not accessed; Stage 2 not run.

## Main metrics — best development checkpoints
| experiment | best_epoch | seen_composition | unseen_composition | composition_HM | unseen_crop | unseen_disease | unseen_correct_of_400 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp09 natural shuffle | 12 | 72.95% | 4.00% | 7.58% | 70.50% | 22.00% | 16 |
| Exp10 tempered alpha=0.25 | 13 | 73.24% | 4.00% | 7.59% | 69.25% | 23.25% | 16 |

## Exp10 error diagnosis
- Unseen crop Top-1: 69.25%; unseen disease Top-1: 23.25%; composition Top-1: 4.00%.
- Disease-only wrong: 261 / 400; crop-only wrong: 77 / 400; both wrong: 46 / 400.
- Predicted training-seen pairs on held-out combinations: 84.50%.
- Exp09 comparison: unseen disease 22.00%, disease-only errors 266, predicted seen-pair rate 85.25%.

## Weakest unseen disease classes (support >=5)
| disease_name | train_images | n_validation | n_correct | top1_accuracy | most_common_wrong_prediction |
| --- | --- | --- | --- | --- | --- |
| bacterial_leaf_streak | 12 | 5 | 0 | 0.00% | rust |
| angular_leaf_spot | 7 | 18 | 0 | 0.00% | bean_rust |
| spider_mites | 5 | 70 | 0 | 0.00% | healthy |
| bacterial_blight | 27 | 117 | 1 | 0.85% | brown_spot |
| monillia_laxa | 8 | 9 | 1 | 11.11% | aphis_spp |
| late_blight | 103 | 70 | 19 | 27.14% | early_blight |
| powdery_mildew | 150 | 60 | 25 | 41.67% | healthy |
| healthy | 1417 | 51 | 47 | 92.16% | anthracnose |

## Exp09 to Exp10 per-disease changes (same validation images)
**Largest development improvements, only classes with n>=5:**
| disease_name | train_images | n_validation_exp10 | exp09_accuracy | exp10_accuracy | accuracy_delta_pp |
| --- | --- | --- | --- | --- | --- |
| late_blight | 103 | 70 | 22.86% | 27.14% | 4.29 |
| powdery_mildew | 150 | 60 | 40.00% | 41.67% | 1.67 |
| bacterial_blight | 27 | 117 | 0.00% | 0.85% | 0.85 |
| spider_mites | 5 | 70 | 0.00% | 0.00% | 0.00 |
| angular_leaf_spot | 7 | 18 | 0.00% | 0.00% | 0.00 |
| bacterial_leaf_streak | 12 | 5 | 0.00% | 0.00% | 0.00 |
**Largest development declines, only classes with n>=5:**
| disease_name | train_images | n_validation_exp10 | exp09_accuracy | exp10_accuracy | accuracy_delta_pp |
| --- | --- | --- | --- | --- | --- |
| spider_mites | 5 | 70 | 0.00% | 0.00% | 0.00 |
| angular_leaf_spot | 7 | 18 | 0.00% | 0.00% | 0.00 |
| bacterial_leaf_streak | 12 | 5 | 0.00% | 0.00% | 0.00 |
| monillia_laxa | 8 | 9 | 11.11% | 11.11% | 0.00 |
| healthy | 1417 | 51 | 92.16% | 92.16% | 0.00 |
| bacterial_blight | 27 | 117 | 0.00% | 0.85% | 0.85 |

## Class imbalance evidence from TRAIN-FIT (not test)
- crop: 55 classes/pairs; min 2, median 38, max 1007 images.
- disease: 175 classes/pairs; min 1, median 8, max 1417 images.
- composition: 313 classes/pairs; min 1, median 6, max 500 images.

## Interpretation / limits
- Higher class balancing may help some rare diseases but can hurt crop accuracy and may oversample single examples; this experiment alone does not prove a causal explanation for unseen-pair errors.
- Results are on one development split and one seed. Compare paired per-class errors and repeat with different fixed seeds before claiming robust improvements.
- Candidate restriction to 333 was NOT tested; requires an approved independently specified catalogue. Changing candidate count is a separate inference-only protocol, not training.

## Saved evidence
- training_history.csv, training_curves.png, best_development.pt, validation_metrics.json, task_error_breakdown.csv, class counts, per-class accuracy, confusion matrices and summary CSV are in outputs/exp10/.
