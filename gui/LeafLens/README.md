# LeafLens — final four-model ensemble

Streamlit demonstration of the selected ConvNeXt–CLIP Stage-2 ensemble (8,000 training images). This is not the supplementary CDC-CP extension. The published official-test composition HM is 16.47% Top-1 and 44.05% Top-5; these are research results, not a fresh GUI evaluation.

## Run locally (macOS / Linux, Python 3.12)

From a terminal:

```bash
git clone https://github.com/tylsee/aml-group2.git
cd aml-group2/gui/LeafLens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the four assets from [GitHub Releases](https://github.com/tylsee/aml-group2/releases/tag/new-approach-models). Place them in `models/`, keeping their exact filenames. If already downloaded to ~/Downloads:

```bash
mkdir -p models
cp ~/Downloads/crop_b1_best_train_acc.pt models/
cp ~/Downloads/crop_clip_best_train_acc.pt models/
cp ~/Downloads/disease_clip_best_train_acc.pt models/
cp ~/Downloads/disease_small_reg_best_train_acc.pt models/
python verify.py
python -m streamlit run app.py
```

Alternatively, authenticate GitHub CLI with `gh auth login` and download from the release:

```bash
gh release download new-approach-models -R tylsee/aml-group2 -D models -p crop_b1_best_train_acc.pt -p crop_clip_best_train_acc.pt -p disease_clip_best_train_acc.pt -p disease_small_reg_best_train_acc.pt
```

Open http://localhost:8501. No training images or retraining required. Existing repository users can run `git pull --ff-only origin main` first. If clone requires GitHub authentication, use a signed-in GitHub CLI or configured Git credentials. Keep the repository private if required by the course.

## Features and inference

- Crop: ConvNeXt-Tiny + CLIP ViT-B/16; disease: CLIP ViT-B/16 + ConvNeXt-Small.
- Resize 256 / centre crop 224, family-specific normalisation, horizontal-flip probability averaging, equal-weight mean of member log-probabilities per task. Unrestricted additive scoring over 9,625 pairs (independent argmax is equivalent for Top-1).
- Top-3 disease scores use the same fused ranking. Display scores are normalised geometric pooling, not calibrated diagnostic probabilities.
- AI View shows input-gradient saliency from both disease models and both flip views. It does not confirm lesion boundaries.
- Predicted composition status uses the 323 training-seen pairs derived from the supplied 8,000-row original training CSV. The compact ID-ordered metadata contains no images; the source CSV SHA-256 is recorded.
- The existing release assets are pinned by byte size and SHA-256. No old weights or automatic fallback. Local CPU inference may be slow and needs several GB of RAM.

## Validation and limits

`python verify.py --image /path/to/leaf.jpg` performs a real-image smoke test after downloading the exact weights. `python -m pytest -q` checks synthetic scoring and metadata; it does not reproduce official accuracy. The uploaded LeafLens source was used as the design and inference reference; this repository edition consolidates the GUI into a dedicated folder and retains its green/cream styling, upload, AI View, Top-3, and composition status. Actual release-byte loading has not been executed in the assistant environment. The app is designed for local Streamlit, not a published Streamlit Cloud deployment. Cloud hosting requires private release access and sufficient memory, and is not configured here.

Research reference: [single_task.py](../../new_proposed_approach/norman_convnext_clip/experiments/cdc-cp/cdccp/single_task.py), [ensemble.py](../../new_proposed_approach/norman_convnext_clip/experiments/cdc-cp/cdccp/ensemble.py).
