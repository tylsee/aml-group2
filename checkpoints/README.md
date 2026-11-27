# AML Group 2 — Checkpoint inventory

Verified checkpoint locations as of 27 November 2026. This repository directory contains an inventory, **not checkpoint binaries**.

**Already hosted on GitHub:** five Norman ConvNeXt/CLIP model weights and SHA256SUMS.txt: [new-approach-models release](https://github.com/tylsee/aml-group2/releases/tag/new-approach-models).

**Still in Google Drive:** seven checkpoint files. The connected GitHub tools cannot upload Releases assets or LFS binaries, so these seven are not yet hosted on GitHub.

## Checkpoints waiting for GitHub Release upload

| Approach | Checkpoint | MiB | Status | Source |
| --- | --- | ---: | --- | --- |
| Baseline 1 | `crop_transfer_epoch_03.pt` | 90.4 | Drive, selection unconfirmed | [Google Drive](https://drive.google.com/file/d/1f9PJtVOTqFdRkxnVyah6Ad2xYBx_OrR2/view?usp=drivesdk) |
| Baseline 1 | `disease_transfer_epoch_06.pt` | 91.3 | Drive, selection unconfirmed | [Google Drive](https://drive.google.com/file/d/1rDoFf4ri4KoSHhSCo9cLlLl_4K9qdNno/view?usp=drivesdk) |
| Baseline 2 | `final_full_data.pt` | 275.0 | Drive, backbone final; approach selection unconfirmed | [Google Drive](https://drive.google.com/file/d/13CEhyhi9zizvx-3Diy_yL6D5dh7Er4Mk/view?usp=drivesdk) |
| Baseline 2 | `final_epoch10.pt` | 107.0 | Drive, backbone final; approach selection unconfirmed | [Google Drive](https://drive.google.com/file/d/16xHZtoezyT0kxLECxUaZQ6FIm4rFTsi4/view?usp=drivesdk) |
| Emily CDC-CP | `stage2_final_epoch03.pt` | 26.8 | Drive, locked final epoch 3 | [Google Drive](https://drive.google.com/file/d/1WBsZrYcZgBkocr6Rz1Y9mSt3Q-jdelbl/view?usp=drivesdk) |
| Tyra CDC-CP | `v12b_frozen_cnn_full8000_locked_bundle.pt` | 39.2 | Drive, V12b locked bundle; evaluation post-hoc | [Google Drive](https://drive.google.com/file/d/1H3HKUU3EP3azob5nNfOagFH8Roz1Hph0/view?usp=drivesdk) |
| Tyra CDC-CP | `locked_final_model.pt` | 39.0 | Drive, separate locked final protocol | [Google Drive](https://drive.google.com/file/d/1oUjSEt1RidiIZ89akj8kE_3w2P-HJykh/view?usp=drivesdk) |

## Norman — already uploaded to GitHub Releases

| Member | Checkpoint | MiB | Download |
| --- | --- | ---: | --- |
| Crop ConvNeXt-Tiny | `crop_b1_best_train_acc.pt` | 106.3 | [GitHub Release asset](https://github.com/tylsee/aml-group2/releases/download/new-approach-models/crop_b1_best_train_acc.pt) |
| Crop CLIP ViT-B16 | `crop_clip_best_train_acc.pt` | 329.0 | [GitHub Release asset](https://github.com/tylsee/aml-group2/releases/download/new-approach-models/crop_clip_best_train_acc.pt) |
| Disease ConvNeXt-Tiny | `disease_b1_best_train_acc.pt` | 106.7 | [GitHub Release asset](https://github.com/tylsee/aml-group2/releases/download/new-approach-models/disease_b1_best_train_acc.pt) |
| Disease CLIP ViT-B16 | `disease_clip_best_train_acc.pt` | 329.2 | [GitHub Release asset](https://github.com/tylsee/aml-group2/releases/download/new-approach-models/disease_clip_best_train_acc.pt) |
| Disease ConvNeXt-Small | `disease_small_reg_best_train_acc.pt` | 189.3 | [GitHub Release asset](https://github.com/tylsee/aml-group2/releases/download/new-approach-models/disease_small_reg_best_train_acc.pt) |

## Model selection cautions

- Baseline 1: the final-selection JSON names C1_reference / D1_reference. The two available ResNet transfer files have **not** been confirmed as those formal selections.
- Baseline 2: full-data ResNet and ConvNeXt weights were located, but no single approach-level final model is declared. An EfficientNet final `.pt` was not found in the inspected Stage 2 output.
- Emily: the locked Stage-2 protocol specifies epoch 3. The P3 prototypes are saved separately as an NPZ, not as a trained-model PT.
- Tyra: the V12b frozen bundle and later locked 320-pair model are **different protocols**. V12b official results are post-hoc exploratory.
- Norman: the five Stage-2 weights are already GitHub Release assets. Do not duplicate them in Git history.

All Drive source links require their owners' sharing permissions. Do not assume a lecturer can open them.

To complete the GitHub-hosted binary collection, follow [RELEASE_UPLOAD_INSTRUCTIONS.md](RELEASE_UPLOAD_INSTRUCTIONS.md). Exact source links, byte sizes, GitHub SHA-256 digests where available, and suggested asset filenames are in [checkpoint_inventory.csv](checkpoint_inventory.csv).
