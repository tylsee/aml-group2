# aml-group2

COS30082 Applied Machine Learning — DPD-Small out-of-distribution plant disease identification.

This repository contains experiment notebooks, results, diagnostic plots, configurations and documented evaluation artifacts for **independent CNN baselines**, **shared-backbone multi-task CNN baselines**, and **proposed methods**.

## Repository navigation

| Area | Contributor | Start here |
| --- | --- | --- |
| [Baseline 1: independent CNNs](baseline_1_independent_cnn/) | Amanda | [experiments](baseline_1_independent_cnn/experiments/) · [final/evaluation index](baseline_1_independent_cnn/final/) |
| [Baseline 2: shared multi-task CNNs](baseline_2_shared_multitask/) | Jia Han | [experiments](baseline_2_shared_multitask/experiments/) · [final/evaluation index](baseline_2_shared_multitask/final/) |
| [Proposed CDC-CP](new_proposed_approach/emily_cdc_cp/) | Emily | [historical experiments](new_proposed_approach/emily_cdc_cp/experiments/) · [locked/evaluation records](new_proposed_approach/emily_cdc_cp/final/) |
| [Proposed ConvNeXt/CLIP](new_proposed_approach/norman_convnext_clip/) | Norman | [source and initial notebook](new_proposed_approach/norman_convnext_clip/experiments/) · [completed notebook, results and report tables](new_proposed_approach/norman_convnext_clip/final/) |
| [Proposed CDC-CP experiments (V12b-focused)](new_proposed_approach/tyra_cdc_cp_experiments/) | Tyra | [run/evaluation evidence](new_proposed_approach/tyra_cdc_cp_experiments/experiments/) · [Drive checkpoints](new_proposed_approach/tyra_cdc_cp_experiments/checkpoints/) |
| [Data](data/) | Shared | [split manifest](data/split_manifest.json) |
| [Comparisons](comparisons/) | Shared | Cross-approach navigation and reporting policy |
| [Report](report/) | Shared | Submission destination (not yet populated) |
| [Presentation](presentation/) | Shared | Slides and video destination (not yet populated) |

## Where to look

- Each approach has an `experiments/` folder. Tyra's V12b folder is a curated selection of original Drive artifacts; other approaches retain their imported experiment subtrees.
- The `final/` folders contain final-selection navigation and available evaluation artifacts. Norman's `final/` folder includes the executed notebook, complete tracked result records and reporting tables imported from `new-approach`. These do **not** fabricate or replace omitted trained model checkpoints.
- See [REORGANIZATION_MANIFEST.md](REORGANIZATION_MANIFEST.md) for the original folder reorganization, and [NORMAN_IMPORT_MANIFEST.md](NORMAN_IMPORT_MANIFEST.md) for exact source-branch import mapping and verification.
- The tracked dataset metadata is under [data](data/). Raw DPD-Small images and large model checkpoints are not committed; Tyra's experiment links to the actual checkpoint location on Google Drive.

## Scientific and reproducibility cautions

- Compare **seen / pseudo-unseen development validation** separately from **official seen / unseen tests**. Do not select epochs or calibration parameters using official-test feedback.
- Some experiment notebooks expect an original Google Drive / Colab directory structure or adjacent source files. This repository organization change **does not rewrite executable code**, so consult each experiment's run configuration before reproducing it.
- Experiment folders may contain repeated historical files intentionally. Nothing has been deduplicated or removed.
- Use the original checkpoint links from the experiment owners if final model weights are not hosted in GitHub.

## File preservation

The reorganization was built using Git tree SHA references for the pre-existing experiment directories. Every original tracked experiment file is retained without changing its contents; only directory paths, added navigation documents, and this expanded README differ.

The original repository commits and the `new-approach` source branch remain accessible for provenance. Experimental artifacts were not edited during the Norman import.
