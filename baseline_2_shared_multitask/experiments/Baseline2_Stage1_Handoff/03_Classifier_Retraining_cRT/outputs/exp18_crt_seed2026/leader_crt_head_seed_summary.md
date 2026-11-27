# Exp18 — cRT head-seed conditional repeat (development only)

- Current run: exp18_crt_seed2026, seed=2026
- Parent: SAME trained Exp09 seed42 ResNet50 backbone/crop head as original Exp17.
- cRT recipe: frozen backbone/crop; reset disease linear head; tempered disease sampler alpha=0.5.
- Head/augmentation/sampler seeds: 42 (Exp17), 123 and 2026 (Exp18).
- The shared frozen parent makes these conditional head-seed repeats, NOT fully independent full-model seeds.
- Same fixed 6,905 / 695 / 400 development split and all 9,625 candidate pairs.
- Best checkpoint selected within each run by development seen/unseen composition HM.
- No Stage 2, official test, or test-driven parameter selection.

## Saved checkpoint comparison

```text
        experiment  seed  best_epoch  seen_comp  unseen_comp       HM  unseen_crop  unseen_disease  unseen_correct_of_400
Exp17 cRT original    42          20   0.725180       0.0500 0.093550        0.705          0.2400                     20
  Exp18 cRT repeat   123           8   0.710791       0.0550 0.102100        0.705          0.2300                     22
  Exp18 cRT repeat  2026          15   0.720863       0.0525 0.097872        0.705          0.2375                     21
```

## Across available seeds (mean and sample SD; values are fractions)

```text
                    mean       std
seen_comp       0.718945  0.007384
unseen_comp     0.052500  0.002500
HM              0.097841  0.004275
unseen_crop     0.705000  0.000000
unseen_disease  0.235833  0.005204
```

- All three seeds complete: True

## Weakest unseen disease classes in this run (support >=5)

```text
         disease_name  train_images  n_validation  top1_accuracy most_common_wrong_prediction
         spider_mites             5            70       0.000000                  target_spot
    angular_leaf_spot             7            18       0.000000                    bean_rust
        monillia_laxa             8             9       0.000000           eriosoma_lanigerum
bacterial_leaf_streak            12             5       0.000000              septoria_blotch
     bacterial_blight            27           117       0.008547                   brown_spot
          late_blight           103            70       0.271429                 early_blight
       powdery_mildew           150            60       0.483333                  sooty_mould
              healthy          1417            51       0.901961                       mosaic
```

## Interpretation

- A higher development HM at one seed does not establish robust improvement.
- Check both variation in disease Top-1 and variation in exact unseen pair successes.
- Train separately seeded Exp09 backbones if full end-to-end seed stability is required.
