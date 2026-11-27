# LeafLens final ensemble upgrade (preserves training gallery)

This patch was built from the submitted **LeafLens_Current_Source.zip**, which already contains the working gallery. Stop Streamlit, back up app.py, leaflens/dual_model.py and config.example.json, then extract the patch into LeafLens_Simple. The patch does not contain or overwrite config.json, data/train.csv, the gallery module, the CSS, photographs, or model weights.

## macOS

```bash
brew install gh
# If not already authenticated:
gh auth login
python -m pip install -r requirements-model.txt
python scripts/setup_final_ensemble.py
python scripts/check_final_ensemble.py
python -m streamlit run app.py
```

The setup script verifies all four files against the pinned release byte counts and SHA-256 values; it reuses valid files in models/ or Downloads/, and downloads missing files from the private GitHub release using your authenticated CLI. It updates config.json only after all four files are verified, preserves training_image_root and backs up the original config. To use manually downloaded assets instead, run `python scripts/setup_final_ensemble.py --local-only`. The old two-model files are not deleted. No additional training is needed. The new default mode is convnext_clip_ensemble; the existing legacy adapter remains available only if explicitly selected in config.json.

## Windows (PowerShell)

```powershell
winget install --id GitHub.cli
# Open a new terminal if gh is not yet on PATH; authenticate if needed:
gh auth login
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe scripts/setup_final_ensemble.py
.\.venv\Scripts\python.exe scripts/check_final_ensemble.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use existing venv rather than creating it again if already installed. The release page (also suitable for manual download) is https://github.com/tylsee/aml-group2/releases/tag/new-approach-models. No Git LFS upload is necessary if the release is used. The four checkpoint files are approximately 1 GB together.

## Submission

Keep the detailed marker README in the repository at https://github.com/tylsee/aml-group2/blob/main/gui/LeafLens/README.md. Publish the tested GUI source, including scripts/setup_final_ensemble.py and scripts/check_final_ensemble.py. Exclude `.venv`, `*.pt`, `config.json`, personal paths, training image binaries, caches and backup files. Markers download model assets from the existing release and training photographs from Google Drive as documented in the marker README. The verification script performs strict architecture and label checks; `--image /path/to/leaf.jpg` additionally performs a real image smoke test. All 17 supplied synthetic/unit tests passed in the preparation environment; actual checkpoint loading and real-image rendering were not exercised there. The former two-model screenshots may change under the final ensemble.
