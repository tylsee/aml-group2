# LeafLens — installation and assessment guide

COS30082 Applied Machine Learning | Group 2 | DPD-Small out-of-distribution plant disease identification

LeafLens is a local Streamlit application that accepts a plant image and predicts the crop and disease using Norman's selected **Stage 2 ConvNeXt–CLIP ensemble** trained on all 8,000 original training images. Its reported unrestricted official-test composition harmonic means are 16.47% (Top-1) and 44.05% (Top-5). These are research evaluation results, not new measurements from the GUI. This is not the supplementary CDC-CP extension.

**No retraining, Colab session, API key or dedicated GPU is required.** The application runs locally. The training-image gallery requires the original photographs separately.

## 1. Before installation

- macOS (Intel or Apple Silicon) or 64-bit Windows 10/11; Python 3.12 is recommended.
- Install [Python 3.12](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads) and [Git LFS](https://git-lfs.com/) if obtaining checkpoint binaries via Git. On Windows, select the Python launcher and Add Python to PATH when installing.
- VS Code is optional: [download](https://code.visualstudio.com/download). Any terminal and modern browser can be used.
- A computer with 8 GB RAM or more is recommended; 16 GB is preferable for four-model inference and attribution. Reserve several GB of disk space for dependencies, about 1 GB for the checkpoints, and additional space for the image dataset.
- Internet access is needed initially for software, dependencies and asset downloads. After setup, inference uses local weights.
- The repository is private: sign in with an authorised GitHub account. The repository owner must grant marker access separately. Google Drive access is independent of GitHub access.

## 2. Obtain the application

The recommended route is Git clone. First install Git LFS if using the tracked weights, then clone. If Git is unavailable, use **Code → Download ZIP** on the [repository](https://github.com/tylsee/aml-group2), extract it, and open the extracted gui/LeafLens directory. GitHub source ZIPs may contain only LFS pointer files; download the actual checkpoints using Section 4 if needed.

### macOS — Terminal

~~~bash
git lfs install
git clone https://github.com/tylsee/aml-group2.git
cd aml-group2/gui/LeafLens
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ -f requirements-model.txt ]; then
  python -m pip install -r requirements-model.txt
else
  python -m pip install -r requirements.txt
fi
~~~

For an existing clean clone, run git pull --ff-only origin main from the repository root before entering gui/LeafLens. If Python 3.12 is unavailable, install it first. If Git LFS is unavailable on macOS, Homebrew users can install it with brew install git-lfs.

### Windows — PowerShell

~~~powershell
git lfs install
git clone https://github.com/tylsee/aml-group2.git
Set-Location aml-group2/gui/LeafLens
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if (Test-Path requirements-model.txt) {
    .\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
} else {
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
~~~

Using the virtual environment's Python executable directly avoids changing PowerShell execution policies. If desired, activate the environment with .\.venv\Scripts\Activate.ps1. In Windows Command Prompt, use .venv\Scripts\activate.bat. If the py launcher is unavailable, install Python 3.12 or use its full executable path. No system-wide package installation is necessary.

## 3. Required checkpoints — final four-model ensemble

Exactly four Stage 2 best-training-accuracy checkpoints are required. Five models were trained, but disease_b1 (disease ConvNeXt-Tiny) is not a member of the selected P3 ensemble.

| Checkpoint filename in models/ | Recognition task | Exact bytes | Original Google Drive file |
| --- | --- | ---: | --- |
| crop_b1_best_train_acc.pt | Crop ConvNeXt-Tiny | 111507819 | [Download](https://drive.google.com/file/d/1M3TDdOwoSV1dExCLZrfq51Zv3dE4xHnG/view) |
| crop_clip_best_train_acc.pt | Crop CLIP ViT-B/16 | 344964043 | [Download](https://drive.google.com/file/d/1FsoJNAK5hIp7v4_ozgGyJ0hxVYcN2FoJ/view) |
| disease_clip_best_train_acc.pt | Disease CLIP ViT-B/16 | 345238603 | [Download](https://drive.google.com/file/d/1CAu1Sg6Lg1LcduU1WWYoC6ec6G_MxW96/view) |
| disease_small_reg_best_train_acc.pt | Disease ConvNeXt-Small | 198466235 | [Download](https://drive.google.com/file/d/1OJM_3ORwzqFfOHO4EipxjnhoQGZJQlxE/view) |

Total: 1,000,176,700 bytes (approximately 1 GB).

## 4. Obtain model weights (choose ONE method)

### A. Git LFS, if weights are tracked in this checkout

From gui/LeafLens:

~~~bash
git lfs pull --include="gui/LeafLens/models/*.pt" --exclude=""
git lfs ls-files
~~~

Check that the four actual files exist and match the sizes above. If the repository has not yet received the LFS commit, or a download is blocked by LFS quota/access, use B or C. A small text file beginning with the Git LFS specification URL is a pointer, not a model checkpoint.

### B. GitHub Releases

Open [new-approach-models](https://github.com/tylsee/aml-group2/releases/tag/new-approach-models) and download the four assets listed above plus SHA256SUMS.txt for optional checksum verification. Do not download the disease_b1 asset for this GUI. Move the four files into gui/LeafLens/models/ with their exact filenames.

Alternatively, using the authenticated [GitHub CLI](https://cli.github.com/):

~~~bash
gh auth login
gh release download new-approach-models -R tylsee/aml-group2 -D models -p crop_b1_best_train_acc.pt -p crop_clip_best_train_acc.pt -p disease_clip_best_train_acc.pt -p disease_small_reg_best_train_acc.pt -p SHA256SUMS.txt
~~~

### C. Google Drive — original checkpoints

Open [Norman's final Stage 2 folder](https://drive.google.com/drive/folders/1RHYlG9DIoE3oGIbqIKJghx18CmF76wf5) or the four direct file links in Section 3. Each original model subfolder contains a file named best_train_acc.pt. Download each separately and **rename it using its parent model folder** to the exact filename in the table, before copying it into models/. Do not let the browser append (1), (2), etc. to the names. For large files, Google Drive may show a preview or virus-scan warning; select Download only if using the verified project link. Never use a Stage 1 best.pt or supplementary CDC-CP checkpoint.

For downloaded release assets, the following commands copy the files from the default Downloads folder (adjust the path if necessary).

macOS:

~~~bash
mkdir -p models
cp ~/Downloads/crop_b1_best_train_acc.pt models/
cp ~/Downloads/crop_clip_best_train_acc.pt models/
cp ~/Downloads/disease_clip_best_train_acc.pt models/
cp ~/Downloads/disease_small_reg_best_train_acc.pt models/
~~~

Windows PowerShell:

~~~powershell
New-Item -ItemType Directory -Force models | Out-Null
Copy-Item "$HOME\Downloads\crop_b1_best_train_acc.pt" models/
Copy-Item "$HOME\Downloads\crop_clip_best_train_acc.pt" models/
Copy-Item "$HOME\Downloads\disease_clip_best_train_acc.pt" models/
Copy-Item "$HOME\Downloads\disease_small_reg_best_train_acc.pt" models/
~~~

## 5. Google Drive dataset — required for the training-image gallery

The GUI predicts from a user-supplied photo without requiring the complete dataset. To demonstrate the assessment requirement of displaying relevant training images, also obtain the original DPD-Small photographs.

- [DPD-Small project dataset folder](https://drive.google.com/drive/folders/1jbwvz6-BATK6XBUHh0ho1eYtJq0qFk7a)
- [Actual DPD-Small image directory](https://drive.google.com/drive/folders/1WVRirfUT5uC2zqEUH7vlaDCOaPrct3Tm)
- [Original DPD_Small_train.csv (8,000 records)](https://drive.google.com/file/d/1f8HPgmKYNizELvQ140CSQRc35POm5v1R/view)
- [Curated assessment evidence and experiment index](https://drive.google.com/drive/folders/1bdAckBN8emEA3C8yBRxikyIKPDp_D8Js)
- [Full individual experimental archive index](../../GOOGLE_DRIVE_ARCHIVE.md)

In Drive, select the image folder and choose Download. Drive may split a large folder into multiple ZIP archives; download and extract every part, keeping the original image filenames. Locate the directory actually containing the photographs, often named DPD_Small_dataset. Do not rename images or substitute test labels for the training manifest.

For the updated gallery-enabled version, data/train.csv is already supplied. If missing, download the original training CSV above and save it as data/train.csv. Do not replace it with a seen/unseen test CSV.

Choose either:

**Option 1 (portable):** place the actual photographs inside gui/LeafLens/data/train_images/. The default configuration will read this location.

**Option 2 (no duplicate files):** copy config.example.json to config.json and edit training_image_root to point to the existing extracted directory. Retain every other setting, including model_mode = convnext_clip_ensemble and the four ensemble checkpoint paths.

macOS:

~~~bash
cp config.example.json config.json
~~~

Example JSON field (replace only the value in the existing file):

~~~json
"training_image_root": "/Users/YOUR_USERNAME/Downloads/DPD_Small/DPD_Small_dataset"
~~~

Windows PowerShell:

~~~powershell
Copy-Item config.example.json config.json
~~~

Example JSON field (forward slashes avoid JSON backslash escaping):

~~~json
"training_image_root": "C:/Users/YOUR_USERNAME/Downloads/DPD_Small/DPD_Small_dataset"
~~~

Use the actual folder on the marker's machine, not the example username. If the photographs are directly in DPD_Small, omit the final DPD_Small_dataset component. If a config.json already exists, edit that one rather than overwriting it. Restart Streamlit after editing.

The gallery only opens image filenames listed in the original training CSV even when the directory also contains test photographs. Seen predicted combinations show up to six exact class matches. Unseen predicted combinations show separately labelled same-disease/different-crop and same-crop/different-disease examples (up to three each). References are label-matched, not visual nearest-neighbour matches. Missing files are reported without blocking inference.

**Version note:** the gallery-enabled source has leaflens/training_gallery.py, data/train.csv and scripts/check_final_ensemble.py. The earlier consolidated GUI edition in this folder uses ensemble.py, metadata.json and verify.py instead; it does not include the gallery. If the earlier edition is still checked out, obtain the updated submission source before assessing the gallery feature.

## 6. Verify and start the GUI

Run from gui/LeafLens after completing the above steps.

macOS (activated virtual environment):

~~~bash
python scripts/check_final_ensemble.py
python -m streamlit run app.py
~~~

Windows PowerShell (activation not required):

~~~powershell
.\.venv\Scripts\python.exe scripts/check_final_ensemble.py
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

For the older consolidated edition only, substitute verify.py for scripts/check_final_ensemble.py. The new verification script checks the expected sizes, reconstructs all four checkpoints strictly, and verifies 8,000 training rows, 55 crop classes and 175 disease classes. To optionally verify release hashes, pass --sha256-file followed by the path to SHA256SUMS.txt. To perform a real-image smoke test, pass --image followed by the path to an actual plant photograph.

Open [http://localhost:8501](http://localhost:8501/) in a web browser. If the browser does not open automatically, paste the address manually. Leave the terminal running during assessment. Stop the application with Ctrl+C. To restart later, return to the same folder and run the same Streamlit command using the existing virtual environment.

## 7. Using LeafLens (assessment walkthrough)

1. Upload a plant photograph in a format accepted by the uploader.
2. Select Analyse Image and allow the four models to complete inference.
3. Inspect the predicted crop, disease, and their model scores.
4. Review the Top-3 disease predictions and composition status.
5. Switch between normal view and AI Vision to examine influential image regions. The heatmap is an input-gradient attribution from the disease ensemble, not verified lesion segmentation.
6. Scroll down to Relevant training images. For a seen predicted class, inspect exact class references. For a predicted unseen combination, inspect the explicitly labelled concept-level references.
7. Test an external seen and unseen composition to demonstrate practical behaviour. Correct recognition is not guaranteed, especially for unfamiliar disease–host combinations.

Composition status describes the **predicted** crop–disease pair relative to the training labels; it does not establish whether the prediction or input ground truth is correct. Display scores are normalised model scores, not calibrated clinical or diagnostic probabilities.

## 8. Architecture and evaluation

- Crop: ConvNeXt-Tiny + CLIP ViT-B/16.
- Disease: CLIP ViT-B/16 + ConvNeXt-Small.
- Deterministic inference: resize short edge to 256, centre crop 224, model-family normalisation, horizontal-flip probability averaging per member, equal-weight mean of member log-probabilities per task.
- Add crop and disease scores to rank all 9,625 possible compositions; no restricted catalogue or calibration is used for primary inference.
- Official research evaluation: 2,000 seen and 1,448 unseen images. Final ensemble Top-1 composition accuracy: 70.55% seen, 9.32% unseen, 16.47% harmonic mean; Top-5 HM: 44.05%.
- Disease recognition is the main unseen-composition limitation. The four-model ensemble also requires more computation and memory than the two-model version.

See [single_task.py](../../new_proposed_approach/norman_convnext_clip/experiments/cdc-cp/cdccp/single_task.py), [ensemble.py](../../new_proposed_approach/norman_convnext_clip/experiments/cdc-cp/cdccp/ensemble.py), [checkpoint inventory](../../checkpoints/) and [experiment evidence](../../GOOGLE_DRIVE_ARCHIVE.md).

## 9. Troubleshooting

| Problem | Resolution |
| --- | --- |
| GitHub 404 / permission denied | Sign in and confirm the repository owner granted access. Drive downloads have separate permissions. |
| Git LFS pointer or truncated checkpoint | Use Section 4B or 4C, confirm exact file name and size, then rerun the verification script. |
| Missing Python module | Install requirements-model.txt in the same virtual environment used to launch Streamlit; for the earlier edition use requirements.txt. |
| PowerShell activation blocked | Use the direct .venv Python executable commands above; no execution-policy change is needed. |
| Incorrect model / checkpoint path | Confirm model_mode and ensemble_checkpoints in config.json, or use defaults. The four exact Stage 2 files are mandatory. |
| Gallery missing | Check that the gallery-enabled source is present (see Section 5), the patch has been applied to app.py, and Streamlit was restarted. |
| Gallery contains no photos | Check that training_image_root points to actual extracted photos and data/train.csv is present. Keep original filenames. |
| Slow analysis or memory pressure | Close other applications, use one image at a time; CPU inference and gradient attribution are computationally intensive. |
| Port 8501 occupied | Add --server.port 8502 to the Streamlit command and open http://localhost:8502. |

## 10. Reproducibility and scope

The marker can run the software on their own computer using the provided code, checkpoints and dataset; no additional training is necessary. This is a local research prototype, not a hosted web deployment or a validated agricultural diagnostic product. The macOS gallery has been reported working by the developer; the Windows commands are provided for equivalent setup and have not been executed on a Windows machine in this review. Real-image verification on the marker's machine is recommended. The experiment evidence remains available through the GitHub and Google Drive links above.
