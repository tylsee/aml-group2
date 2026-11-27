# Baseline 1 — Final model navigation

This folder is an **index** to the original saved final-selection and evaluation records; it does not duplicate or silently substitute a checkpoint.

- [Saved final selection](../experiments/consolidated_results/final/final_selection.json) — original crop/disease run IDs and selected epochs.
- [Consolidated official metrics](../experiments/consolidated_results/official_metrics.json).
- [Saved full-data fine-tuning and official evaluation](../experiments/consolidated_results/fine_tuning/Stage2_C4_D1_Full8000_10Epochs/official_test_final/).
- [ResNet-50 research and official evaluation](../experiments/resnet50/ResNet50_V2_Experiment_Results/).
- [ConvNeXt-Tiny study](../experiments/convnext_tiny/).

**Do not treat the highest exploratory test score as the formally selected final model.** Check the final-selection metadata and the report's chosen submission protocol. Model checkpoint weights intentionally are not included if they were excluded from the repository upload; provide an external checkpoint link if required.
