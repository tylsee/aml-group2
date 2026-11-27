# aml-group2

COS30082 Applied Machine Learning — DPD-Small out-of-distribution plant disease identification.

This repository contains experiment notebooks, results, diagnostic plots, configurations and documented evaluation artifacts for **independent CNN baselines**, **shared-backbone multi-task CNN baselines**, and **proposed methods**.

## Repository navigation

| Area | Contributor | Start here |
| --- | --- | --- |
| [Baseline 1: independent CNNs](baseline_1_independent_cnn/) | Amanda | [experiments](baseline_1_independent_cnn/experiments/) · [final/evaluation index](baseline_1_independent_cnn/final/) |
| [Baseline 2: shared multi-task CNNs](baseline_2_shared_multitask/) | Jia Han | [experiments](baseline_2_shared_multitask/experiments/) · [final/evaluation index](baseline_2_shared_multitask/final/) |
| [Proposed CDC-CP](new_proposed_approach/emily_cdc_cp/) | Emily | [historical experiments](new_proposed_approach/emily_cdc_cp/experiments/) · [locked/evaluation records](new_proposed_approach/emily_cdc_cp/final/) |
| [Proposed ConvNeXt/CLIP](new_proposed_approach/norman_convnext_clip/) | Norman | **Awaiting upload** — not present in original GitHub tree |
| [Data](data/) | Shared | [split manifest](data/split_manifest.json) |
| [Comparisons](comparisons/) | Shared | Cross-approach navigation and reporting policy |
| [Report](report/) | Shared | Submission destination (not yet populated) |
| [Presentation](presentation/) | Shared | Slides and video destination (not yet populated) |

## Where to look

- Every approach has an `experiments/` folder with complete tracked original experiment subtrees.
- The `final/` folders contain **navigation README files** pointing to saved final-selection metadata, officially evaluated artifacts, or clearly identified placeholders. They do **not** fabricate or replace trained model checkpoints.
- See [REORGANIZATION_MANIFEST.md](REORGANIZATION_MANIFEST.md) for the exact old → new path-prefix mapping and original Git commit.
- The tracked dataset metadata is under [data](data/). The raw DPD-Small images and large model checkpoints were not present in this repository and are **not** silently recreated.

## Scientific and reproducibility cautions

- Compare **seen / pseudo-unseen development validation** separately from **official seen / unseen tests**. Do not select epochs or calibration parameters using official-test feedback.
- Some experiment notebooks expect an original Google Drive / Colab directory structure or adjacent source files. This repository organization change **does not rewrite executable code**, so consult each experiment's run configuration before reproducing it.
- Experiment folders may contain repeated historical files intentionally. Nothing has been deduplicated or removed.
- Use the original checkpoint links from the experiment owners if final model weights are not hosted in GitHub.

## File preservation

The reorganization was built using Git tree SHA references for the pre-existing experiment directories. Every original tracked experiment file is retained without changing its contents; only directory paths, added navigation documents, and this expanded README differ.

The original `main` commit remains part of Git history, and the changes are proposed via a review branch rather than applied directly to `main`.
