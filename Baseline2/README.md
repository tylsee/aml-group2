# COS30082 Applied Machine Learning · Baseline 2

Start with [baseline2.ipynb](baseline2.ipynb) for the shared EfficientNet-B0 architecture, separate crop and disease heads, Exp04 development recipe and final fit. The [experiment summary](docs/EXPERIMENT_SUMMARY.md) compares the development runs.

| Official composition metrics | Seen Top-1 | Unseen Top-1 | Harmonic mean |
|---|---:|---:|---:|
| Original B2, γ = 0 (primary) | 55.60% | 1.86% | 3.61% |
| B2 + calibration, γ = 4 | 38.70% | 4.77% | 8.49% |

Both modes use the same frozen Stage 2B checkpoint. Exp04 was the selected development training run. Exp06 selected the calibration penalty on development validation. Exp07 tested two-view augmentation on development validation. The official evaluation used a single view.

- [Exp04 development metrics](outputs/development/exp04/validation_metrics.json) and [training curves](outputs/development/exp04/training_curves.png)
- [Exp06 notebook](notebooks/experiments/B2_EXP06_seen_pair_calibration.ipynb) and [calibration sweep](outputs/development/exp06/gamma_sweep.csv)
- [Exp07 notebook](notebooks/experiments/B2_EXP07_two_view_tta.ipynb) and [TTA summary](outputs/development/exp07/tta_summary.json)
- [Final training configuration](outputs/final_training/config.json) and [official comparison](outputs/official/comparison.json)

## Setup

The notebooks and saved reports can be read directly. To launch Jupyter from the project root in a separate environment with Python 3.12 installed, use PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

The requirements retain the earlier PyTorch 2.6.0 and torchvision 0.21.0 pins. The completed Exp04 and final fits recorded Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8. Use the saved run configuration when matching the recorded environment matters.

The [handover](docs/HANDOVER.md) lists the local artifacts. The notebooks default to the sibling DPD_Small_dataset directory. Set the B2_IMAGE_ROOT environment variable to the folder containing the training images if they are elsewhere. Relative overrides resolve from the project root. Training, diagnostics, experiment inference and official testing are disabled by default.
