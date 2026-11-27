# V12b — Frozen encoders, full-8,000-image pair-head refit

Original source: [results_CDC_CP_V12b_FROZEN_FULL8000_v1](https://drive.google.com/drive/folders/1cpbYlwsP9pex2kU14qCfFZ2ohPTZHpA0).

## Files

- [run_identity.json](run_identity.json): model-source hashes, seed, hyperparameters and 55-crop / 175-disease configuration.
- [reports/](reports/): pair-head phase A/B histories, internal selection record and training-curve figure.
- [evaluation/](evaluation/): complete small summary of subsequent **post-hoc exploratory official test** (not independent model selection).

The saved [selected epoch file](reports/selected_full_refit_epochs.json) records **9 selected epochs**, chosen by internal training holdout cross-entropy. A [locked bundle](../../checkpoints/README.md) remains on Google Drive.

**Critical:** The source [full refit summary](reports/full_refit_summary.json) says that training/refit finished **without an official test at that point**. The separate `evaluation/` files were produced subsequently and mark the official test **post-hoc exploratory**. They must not be described as unbiased model selection.
