"""Build kaggle_final/P6_colab.ipynb, the P6 notebook for one Colab GPU.

P6 averages the log-probabilities of four single-task models. crop_b1 and disease_clip come from P5 and are not retrained
(their saved outputs on Drive are only read). crop_clip and disease_small_reg are trained again with the P5 stage-1 and
stage-2 settings. The logic lives in cdccp/ensemble.py; these cells call it.

Usage: python scripts/make_p6_colab_notebook.py [--zip cdc-cp_code_p6.zip] [--out P6_colab.ipynb]
The notebook carries the sha256 of the code zip, taken from the zip itself, so the zip must be packed first
(python scripts/pack_for_colab.py <zip>).
"""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

DEFAULT_ZIP = Path(r"D:\dev\uni\AML\kaggle_final\cdc-cp_code_p6.zip")
DEFAULT_OUT = Path(r"D:\dev\uni\AML\kaggle_final\P6_colab.ipynb")
MD, CODE = "markdown", "code"
PLACEHOLDER_SHA = "@@CODE_SHA@@"
PLACEHOLDER_FINGERPRINT = "@@FINGERPRINT@@"

FINGERPRINT = r"""
code_files = {f.relative_to(CODE_DIR).as_posix(): f for f in CODE_DIR.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
code_hash = hashlib.sha256()
for rel in sorted(code_files):
    code_hash.update(rel.encode() + b'\0' + code_files[rel].read_bytes())
CODE_SHA = code_hash.hexdigest()
""".strip("\n")

CELLS = [
    (MD, r"""
# P6 on Colab: the mean of four single-task models (two from P5, two retrained), seed 42

**Before you run (Google Drive, folder `My Drive/DPD-Small/`)**
- `DPD_Small_dataset/` (all 11,448 images), `DPD_Small_train.csv`, `DPD_Small_test_seen.csv`, `DPD_Small_test_unseen.csv`
- `cdc-cp_code_p6.zip` (the project code; cell 2 stops in minute 1 if it is not the version this notebook was built for)
- `p5_results/` (the P5 results folder, left where it is; this notebook only reads six files in it)

**Run**: Runtime > Change runtime type > any GPU (L4 is fine) > Save. Then Runtime > **Run all**.
Allow Google Drive access when Colab asks. Keep this tab open until the last cell prints "Finished".
Results are saved to `My Drive/DPD-Small/p6_results/` after stage 1, after stage 2 and at the end.

**What P6 is (fixed before any test number)**
- Crop scores = mean of log-softmax of `crop_b1` and `crop_clip` (weight 0.5 each).
- Disease scores = mean of log-softmax of `disease_clip` and `disease_small_reg` (weight 0.5 each).
- Flip averaging on all four models, on dev and on test.
- `crop_b1` and `disease_clip` are the P5 final models. They are not retrained. Their dev and test log-probabilities are read from `p5_results/`.
- `crop_clip` and `disease_small_reg` are trained again with the P5 settings: stage 1 at most 25 epochs, early stop after 5 epochs
  with no new best dev HM, seed 42, then stage 2 for 10 epochs on all 8,000 images. The P5 stage-1 checkpoints were not kept.

**What this notebook does**
1. **Stage 1** (development split only): train `disease_small_reg` and `crop_clip`, one in each of two slots on the one GPU.
2. **Dev outputs**: flip log-probabilities of both stage-1 best models on the dev split, saved.
3. **Stage 2**: train both on all 8,000 images, again one per slot.
4. **Calibration values, from dev only**, fitted on the four-model dev outputs with the two P5 rules.
   Rule 1: the gamma with the best dev unseen pair accuracy while dev seen pair accuracy drops at most 3 points (r = 1).
   Rule 2: the weight r (0.5 to 2) and the gamma with the best dev pair HM. Information only: the dev unseen numbers
   without the two healthy dev unseen pairs, and the mean over pairs.
5. **Lock**: the fingerprint (sha256) of both final models, the P5 files this run reuses, both test CSVs, the settings, the code and the
   calibration values, saved before the test files are read.
6. **Official test, once**: the marker is written first (after label-free checks of the P5 files). Then P5's own stage-2 `crop_b1` model
   predicts every test row in the current CSV order and must agree row by row with the P5 test log-probabilities saved on Drive (this binds
   the row order to P5's run). Then the P5 test table is recomputed from the saved P5 log-probabilities and the test labels and must equal
   P5's own table. Then the RAW result of P6 (r = 1, no gamma), then each locked calibration.

**Rules fixed before the run**: nothing is chosen from a test number. The test runs once (a second run stops, also on a new Colab
machine, because the Drive results folder records it). Its result is reported even if it is below P5 or Baseline 1.
The marker file is written when the test cell starts, so if the P5 checks at the top of that cell fail, nothing of P6 was scored,
and the marker has to be removed by hand on purpose (see P6_colab_steps.txt).

**Time (a guess, not measured)**: about 65-70 min on an L4 (a guess from the P5 measured epoch times). Stage 1 prints the real time per epoch.
"""),
    (MD, r"""
## Settings

| Setting | `crop_clip` | `disease_small_reg` | `crop_b1`, `disease_clip` |
|---|---|---|---|
| Source | trained again here | trained again here | P5 final models, outputs read from Drive |
| Backbone | CLIP ViT-B/16 image encoder, text-anchored crop head | ConvNeXt-Small, ImageNet weights, P4 recipe `reg` | not used here |
| Stage 1 | at most 25 epochs, early stop after 5 epochs with no new best dev HM, batch 16, cosine (T_max 30) | same | P5 |
| Stage 2 | 10 epochs on 8,000 images, keep the highest train accuracy | same | P5 |
| Test transform | resize 256, centre crop 224, plus the mirror image (flip averaging) | same | P5 |
| P6 score | crop: mean of log-softmax of `crop_b1` and `crop_clip`; disease: mean of log-softmax of `disease_clip` and `disease_small_reg`; pair score = crop + r x disease | | |
"""),
    (CODE, r"""
# 1. Settings: the seed, which model runs in which GPU slot, and the folders.
#    P6_SMOKE is read only here. It is empty on Colab. A folder path in it switches to the quick local test run
#    (fake Drive folder, tiny epochs, no pretrained weights).
import json, os
from pathlib import Path

SMOKE_ROOT = os.environ.get('P6_SMOKE', '')
SMOKE = bool(SMOKE_ROOT)
SEED = 42
STAGE1_EPOCHS = 25                # at most 25 epochs per model
PATIENCE = 5                      # stop a model after 5 epochs with no new best dev HM (cosine T_max stays 30)
STAGE2_EPOCHS = 10
WORKERS = max(2, min(12, ((os.cpu_count() or 4) - 2) // 2))   # image-loading processes per job (2 jobs share the CPUs)
GPU_JOBS = {(0, 0): ['disease_small_reg'],     # GPU 0, slot 0
            (0, 1): ['crop_clip']}             # GPU 0, slot 1: at the same time as slot 0
MAX_TRAIN_BATCHES = None          # None = full epochs
PRETRAINED = True                 # CLIP and ImageNet weights
AMP = True                        # half precision on the GPU (faster); cell 4 checks it gives finite numbers here
FLIP = True                       # average each image with its mirror image (dev and test)
DEVICE_NAME = 'cuda'
EXPECT_IMAGES = 11448             # 8,000 train + 3,448 test images
EXPECT_TEST = (2000, 1448)        # rows of DPD_Small_test_seen.csv and DPD_Small_test_unseen.csv
TEST_DISJOINT_FROM_TRAIN = True
INPUT = Path('/content/data')     # local copy of the Drive files (cells 1b and 2b)
OUT = Path('/content/p6')
DRIVE_DIR = Path('/content/drive/MyDrive/DPD-Small')   # your Drive folder with the data and the code zip
CODE_DIR = Path('/tmp/cdc-cp')
# The six P5 files this run reuses, pinned by full sha256 (computed from the P5 folder; crop_b1_model is the sha256 in the P5 lock).
# Cell 3 stops if a copy on Drive differs.
P5_EXPECTED_SHA = {'lock': '783b913f02fd42bc7a0a8ef1cab9468a56be78ba51901b75c350588f2032385f',
                   'marker': 'af44acc2118239386db0f59f574a8d91e434ea914237a4abc13de938f8f2bc71',
                   'dev_logprobs': '89b2cd043b6920d05ef4add01fff44958cc689ee6e0ed6b52d6cc6dd2d34ed08',
                   'test_logprobs': 'b017be11d34d8e804e8db6159882cc156816dc6d217bc36b797e3492e64a2652',
                   'test_results': 'a9e67f1d7b4821ceafe7e71a130e47b25098567ac87b73b6496d77917bdff1b9',
                   'crop_b1_model': '427e9d8e7cb0b1dc4cfe002e5bfce9069e4693204d451f32d1233b74eb132df1'}
if SMOKE:
    import torch
    root = Path(SMOKE_ROOT)
    STAGE1_EPOCHS, PATIENCE, STAGE2_EPOCHS, MAX_TRAIN_BATCHES = 1, 1, 1, 2
    WORKERS, PRETRAINED, AMP = 0, False, False
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
    EXPECT_IMAGES, EXPECT_TEST, TEST_DISJOINT_FROM_TRAIN = 8000, (695, 400), False
    INPUT, OUT, CODE_DIR, DRIVE_DIR = root / 'data', root / 'p6', root / 'code' / 'cdc-cp', root / 'drive' / 'DPD-Small'
    P5_EXPECTED_SHA = json.loads((root / 'p5_expected_sha.json').read_text())   # the smoke hashes its own fake P5 files
DRIVE_RESULTS = DRIVE_DIR / 'p6_results'   # P6 results go here and nowhere else
P5_RESULTS = DRIVE_DIR / 'p5_results'      # P5 results: read only
CODE_ZIP = 'cdc-cp_code_p6.zip'

# Baseline 1 official test numbers (Amanda's notebook). Only printed next to ours at the end. Never used to choose anything.
B1_TEST = {'Crop': (89.30, 73.07, 96.20, 84.74), 'Disease': (66.25, 11.12, 88.45, 26.73),
           'Composition': (63.65, 6.56, 84.00, 18.85)}   # seen Top-1, unseen Top-1, seen Top-5, unseen Top-5
"""),
    (CODE, r"""
# 1b. Connect Google Drive, check the files, and copy the CSVs and the code zip to this machine's disk.
import shutil, time
if not SMOKE:
    from google.colab import drive
    drive.mount('/content/drive')

P5_FILES = {'lock': P5_RESULTS / 'lock.json',
            'marker': P5_RESULTS / 'official_test_started.json',
            'dev_logprobs': P5_RESULTS / 'results' / 'dev_flip_logprobs.npz',
            'test_logprobs': P5_RESULTS / 'results' / 'official_test_logprobs.npz',
            'test_results': P5_RESULTS / 'results' / 'official_test_results.csv',
            'crop_b1_model': P5_RESULTS / 'stage2' / 'crop_b1' / 'best_train_acc.pt'}
COPY_TO_LOCAL = ['DPD_Small_train.csv', 'DPD_Small_test_seen.csv', 'DPD_Small_test_unseen.csv', CODE_ZIP]
missing = [name for name in ['DPD_Small_dataset', *COPY_TO_LOCAL] if not (DRIVE_DIR / name).exists()]
assert not missing, f'missing in {DRIVE_DIR}: {missing}'
missing_p5 = [name for name, path in P5_FILES.items() if not path.is_file()]
assert not missing_p5, f'missing P5 files in {P5_RESULTS}: {missing_p5} (leave the p5_results folder where it is)'
assert not list(DRIVE_DIR.glob('p6_results*/official_test_started.json')), (
    'the official test already ran for P6 (marker in a p6_results folder on Drive); it must not run twice')
assert not list(OUT.parent.glob('p6*/official_test_started.json')), (
    'the official test already ran for P6 (marker in a local p6 folder on this machine); it must not run twice')
if DRIVE_RESULTS.exists():   # results of an earlier, unfinished try: keep them under a new name, never delete
    old = DRIVE_RESULTS.with_name(f"p6_results_unfinished_{time.strftime('%Y%m%d_%H%M%S')}")
    DRIVE_RESULTS.rename(old)
    print('earlier unfinished results kept in', old)
if OUT.exists():   # a failed earlier try in this same Colab machine: keep it under a new name (stage 1 needs a fresh folder)
    OUT.rename(OUT.with_name(f"p6_unfinished_{time.strftime('%Y%m%d_%H%M%S')}"))
(INPUT / 'DPD-Small').mkdir(parents=True, exist_ok=True)   # one folder level, as on Drive (cell 3 looks for the images next to the train CSV)
for name in COPY_TO_LOCAL:
    shutil.copy2(DRIVE_DIR / name, INPUT / 'DPD-Small' / name)
print('copied to', INPUT / 'DPD-Small', ':', COPY_TO_LOCAL)
LOCAL_CROP_B1 = INPUT / 'p5_crop_b1' / 'best_train_acc.pt'   # P5's stage-2 crop model, used by the row-order check in cell 11
LOCAL_CROP_B1.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(P5_FILES['crop_b1_model'], LOCAL_CROP_B1)
print('copied P5 crop_b1 model to', LOCAL_CROP_B1)
"""),
    (CODE, r"""
# 2. Check the GPU and unpack the project code (cdc-cp_code_p6.zip) from the local copy.
import hashlib, json, os, shutil, subprocess, sys, zipfile
import torch

if DEVICE_NAME == 'cuda':
    assert torch.cuda.is_available(), 'Runtime > Change runtime type > any GPU (L4 is fine)'
    N_GPU = torch.cuda.device_count()
    GPU_NAME = torch.cuda.get_device_name(0)
    assert all(gpu < N_GPU for gpu, _ in GPU_JOBS), f'GPU_JOBS names a GPU this machine does not have ({N_GPU} GPU)'
else:
    N_GPU, GPU_NAME = 0, 'cpu'
print(N_GPU, 'x', GPU_NAME, '| torch', torch.__version__)
print('CPUs:', os.cpu_count(), '| image-loading workers per job:', WORKERS)

# The code zip was copied from Drive in cell 1b. Only this project's code has cdccp/gpu_runner.py.
zips = sorted(INPUT.rglob(CODE_ZIP))
folders = sorted(p.parent.parent for p in INPUT.rglob('gpu_runner.py') if p.parent.name == 'cdccp')
assert len(zips) + len(folders) == 1, f'put {CODE_ZIP} in My Drive/DPD-Small exactly once, found zips {zips} and folders {folders}'
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
if zips:
    unzip_dir = CODE_DIR.parent / 'p6_code_unzip'
    if unzip_dir.exists():
        shutil.rmtree(unzip_dir)
    with zipfile.ZipFile(zips[0]) as archive:
        archive.extractall(unzip_dir)
    found = sorted(p.parent.parent for p in unzip_dir.rglob('gpu_runner.py') if p.parent.name == 'cdccp')
    assert len(found) == 1, f'expected one cdccp/gpu_runner.py in the zip, found {found}'
    shutil.move(str(found[0]), CODE_DIR)
else:
    shutil.copytree(folders[0], CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
# One fingerprint of the code that runs: every file's path and bytes, in sorted order (same for zip or folder).
@@FINGERPRINT@@
print('code:', (zips or folders)[0], '| sha256 of code files', CODE_SHA[:12])
# This notebook was built for one exact version of the code. An older or newer code zip stops here, in minute 1.
EXPECTED_CODE_SHA = '@@CODE_SHA@@'
assert CODE_SHA == EXPECTED_CODE_SHA, (f'code input {CODE_SHA[:12]} is not the version this notebook was built for '
                                       f'({EXPECTED_CODE_SHA[:12]}): put the matching {CODE_ZIP} in My Drive/DPD-Small')
"""),
    (CODE, r"""
# 2b. Copy the images from Drive to this machine's disk (training never reads images from Drive: too slow).
#     Every image (train and test) is opened once here, so a damaged copy stops the run now, not at the official test.
from cdccp.colab_io import check_images, copy_file, copy_tree, save_results

n_copied = copy_tree(DRIVE_DIR / 'DPD_Small_dataset', INPUT / 'DPD-Small' / 'DPD_Small_dataset')
assert n_copied >= EXPECT_IMAGES, f'Drive folder DPD_Small_dataset has {n_copied} files; expected {EXPECT_IMAGES} (8,000 train + 3,448 test)'
print('images that open:', check_images(INPUT / 'DPD-Small' / 'DPD_Small_dataset'))
"""),
    (CODE, r"""
# 3. Find the 8,000 training images and the development split. The official test CSVs are only located here, not read.
#    The six P5 files are only read: each sha256 must equal the pinned value (cell 1), and the P5 dev log-probabilities are loaded.
#    The test CSVs on Drive must be older than the P5 marker (only file times are read). That is evidence, not proof, that they are
#    the files P5 read (a copy can keep an old file time). Cell 11 binds the row order to P5's own crop model.
import pandas as pd
from cdccp import ensemble as pm
from cdccp import single_task as st

train_csv = next(p for p in INPUT.rglob('*') if p.name.lower() == 'dpd_small_train.csv')
assert hashlib.sha256(train_csv.read_bytes()).hexdigest().startswith('9f1b729a'), 'unexpected DPD_Small_train.csv'
train_all = st.read_labelled_csv(train_csv)
assert len(train_all) == 8000
first = train_all.filename.iloc[0]
IMAGE_DIR = next(d for d in [train_csv.parent, *train_csv.parent.parent.rglob('*')] if d.is_dir() and (d / first).is_file())
missing = [f for f in train_all.filename if not (IMAGE_DIR / f).is_file()]
assert not missing, f'{len(missing)} training images missing, e.g. {missing[:3]}'

SPLITS = CODE_DIR / 'data' / 'splits_v2'
SPLIT_SHA = {'train_fit.csv': 'fe02d6099c00', 'val_seen.csv': 'ab8d18a13b28', 'val_unseen.csv': 'b9eae2a57d83'}
for name, sha in SPLIT_SHA.items():
    assert hashlib.sha256((SPLITS / name).read_bytes()).hexdigest().startswith(sha), f'{name} is not the B1 development split'
train_fit, val_seen, val_unseen = st.load_dev(SPLITS)
CROP_IDS, DISEASE_IDS = st.class_ids(train_fit, val_seen, val_unseen)

OFFICIAL = {}
for split in ('seen', 'unseen'):
    hits = [p for p in INPUT.rglob('*') if p.name.lower() == f'dpd_small_test_{split}.csv']
    assert len(hits) == 1, f'expected one DPD_Small_test_{split}.csv, found {hits}'
    OFFICIAL[split] = hits[0]
print('images:', IMAGE_DIR, '| dev split', len(train_fit), len(val_seen), len(val_unseen),
      '| classes', len(CROP_IDS), len(DISEASE_IDS))
print('official test CSVs found (not read yet):', *OFFICIAL.values())

DRIVE_TEST_CSVS = {split: DRIVE_DIR / f'DPD_Small_test_{split}.csv' for split in ('seen', 'unseen')}
pm.check_test_csvs_predate_marker(DRIVE_TEST_CSVS.values(), P5_FILES['marker'])
P5_SHA = {name: pm.sha256_of(path) for name, path in P5_FILES.items()}
pm.check_pinned_hashes(P5_SHA, P5_EXPECTED_SHA)
assert pm.sha256_of(LOCAL_CROP_B1) == P5_SHA['crop_b1_model'], 'the local copy of the P5 crop_b1 model is not the file on Drive'
P5_LOCK = json.loads(P5_FILES['lock'].read_text())
pm.check_p5_lock(P5_LOCK)
assert P5_LOCK['files']['crop_b1']['sha256'] == P5_SHA['crop_b1_model'], 'the P5 lock names another crop_b1 model than the file on Drive'
P5_DEV = pm.read_logprobs(P5_FILES['dev_logprobs'], pm.P5_ARMS.values(), {'seen': len(val_seen), 'unseen': len(val_unseen)})
print('P5 files (read only):', {name: sha[:12] for name, sha in P5_SHA.items()})
print('P5 lock: crop model', P5_LOCK['info']['crop_winner'], '| disease model', P5_LOCK['info']['winner'],
      '| code', P5_LOCK['info']['code_sha256'][:12])
"""),
    (CODE, r"""
# 4. Print the settings this run really uses (read from the code). Nothing is trained here.
print('Baseline 1 recipe:', st.B1)
print('Models:', st.ARMS, '| controls:', st.CONTROLS)
for aug in st.AUGS:
    print(f'--- train transform "{aug}":', st.train_transform(aug))
print('--- test transform:', st.eval_transform())

# Check GPU_JOBS now, not after hours of training: the two new models, each once.
TRAINED = [arm for arms in GPU_JOBS.values() for arm in arms]
assert sorted(TRAINED) == sorted(pm.NEW_ARMS.values()), f'GPU_JOBS must train exactly {list(pm.NEW_ARMS.values())}, got {TRAINED}'
print('Stage 1 will train:', TRAINED)
DEVICE = torch.device(DEVICE_NAME)

# Download the ImageNet weights once, here, so a download problem stops the run in minute 1 and not mid-training.
if PRETRAINED:
    for backbone in sorted({st.ARMS[arm].backbone for arm in TRAINED}):
        if backbone in st.BACKBONES:
            st.BACKBONES[backbone][1].IMAGENET1K_V1.get_state_dict(progress=False)
        else:
            print('pretrained forward pass shape:', st.check_backbone(backbone))
        print('pretrained weights ready:', backbone)

# Half precision must give finite numbers on this GPU for the batch sizes the runs use (16 to train, 32 to evaluate,
# and shorter last batches). Some GPUs (for example the GTX 16xx cards) give NaN for batches of 15-17.
if AMP:
    for backbone in sorted({st.ARMS[arm].backbone for arm in TRAINED if st.ARMS[arm].backbone in st.BACKBONES}):
        constructor, weights = st.BACKBONES[backbone]
        probe = constructor(weights=weights.IMAGENET1K_V1).to(DEVICE).eval()
        probe_gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for n in (1, 8, 15, 16, 17, 32):
                with torch.autocast(DEVICE.type):
                    out = probe(torch.randn(n, 3, 224, 224, generator=probe_gen).to(DEVICE)).float()
                assert torch.isfinite(out).all(), f'half precision gives NaN/inf on this GPU ({backbone}, batch {n}): set AMP = False in cell 1'
        del probe, out
        torch.cuda.empty_cache()
        print(f'half precision check passed for {backbone}, batches 1, 8, 15, 16, 17, 32')

# The row-order check in cell 11 (after the marker) allows at most pm.ROW_MAX_ABS_DIFF between P5's saved crop log-probabilities
# and a rerun of P5's crop model. Measure the rerun noise here, before the marker: P5's crop model on 256 dev images with this
# run's AMP setting against full precision. This gap must stay under a quarter of that limit.
noise_model = st.load_trained(LOCAL_CROP_B1, DEVICE)
noise_rows = val_seen.head(256)
NOISE_NOW = st.predict(noise_model, noise_rows, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)
NOISE_FULL = st.predict(noise_model, noise_rows, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=False, workers=WORKERS, flip=FLIP)
del noise_model
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
NOISE_LIMIT = pm.ROW_MAX_ABS_DIFF / 4
try:
    NOISE = pm.check_rows_match_saved({'dev': NOISE_NOW}, {'dev': NOISE_FULL}, tolerance=NOISE_LIMIT)['dev']
except ValueError as err:
    raise RuntimeError(f'on this GPU, AMP = {AMP} changes the P5 crop log-probabilities of dev images by more than {NOISE_LIMIT}, '
                       'so the row-order check in cell 11 could refuse correct rows after the marker. Stop here and report this.') from err
print(f"rerun noise of P5's crop model on {NOISE['rows']} dev images (AMP = {AMP} vs full precision): max {NOISE['max']:.5f}, "
      f"median {NOISE['median']:.5f}, 99th percentile {NOISE['p99']:.5f}; must stay under {NOISE_LIMIT}")

# CLIP checks, minute 1: the text model (tokenizer download), the CLIP weights on dev images (zero-shot, before any
# training, must beat 2x chance: broken weights or broken prompts give chance), and half precision.
DEV_ALL = (train_fit, val_seen, val_unseen)
CLASS_NAMES = {'crop': st.names_in_order('crop', DEV_ALL, CROP_IDS, True),
               'disease': st.names_in_order('disease', DEV_ALL, DISEASE_IDS, True)}
for task in sorted({st.ARMS[arm].task for arm in TRAINED if st.ARMS[arm].backbone == 'clip_vit_b16'}):
    print(f'CLIP text features for {task}:', st.clip_text_check(task, CLASS_NAMES[task], pretrained=PRETRAINED))
    zero_shot = st.clip_zero_shot(task, val_seen, IMAGE_DIR, CLASS_NAMES[task], DEVICE, n=len(val_seen), pretrained=PRETRAINED)
    chance = 100 / len(CLASS_NAMES[task])
    print(f'CLIP zero-shot {task} Top-1 on dev seen ({len(val_seen)} images, no training): {zero_shot:.2f}% (chance {chance:.2f}%)')
    if PRETRAINED:
        assert zero_shot > 2 * chance, f'CLIP zero-shot {task} is {zero_shot:.2f}%, not above 2x chance: check the CLIP weights'
if AMP and any(st.ARMS[arm].backbone == 'clip_vit_b16' for arm in TRAINED):
    print('CLIP half precision check passed for batches', st.clip_fp16_probe(DEVICE, pretrained=PRETRAINED))
    torch.cuda.empty_cache()

# The image folder holds the training images and the test images (checked by count, no test CSV is read).
n_images = sum(1 for p in IMAGE_DIR.iterdir() if p.is_file())
assert n_images >= EXPECT_IMAGES, f'{IMAGE_DIR} holds {n_images} files; expected at least {EXPECT_IMAGES}'
print('image files:', n_images)
"""),
    (CODE, r"""
# 5. Helper (tested code in cdccp/gpu_runner.py): run a list of jobs in each GPU slot at the same time (one process per slot).
#    If any job fails, the time budget runs out, or you press Stop, every job stops at once and the cell fails with the log tail.
import time
from cdccp.gpu_runner import require_fresh, run_on_gpus

STAGE1_BUDGET_HOURS = 8   # a guard only: stage 1 is expected to take well under 2 h
REPORT_SECONDS = 120      # print the newest log line of each job every 2 minutes
"""),
    (MD, r"""
## Stage 1: train the two new models on the development split (at most 25 epochs each, early stop after 5 epochs with no new best dev HM)
"""),
    (CODE, r"""
# 6. Stage 1: each GPU slot trains its model. Each model keeps its best epoch by dev HM.
#    The output folder must be fresh: old model folders are never reused (they could come from a quick test run).
STAGE1 = OUT / 'stage1'
require_fresh(STAGE1, TRAINED)
start = time.time()

def stage1_command(arm):
    cmd = [sys.executable, 'scripts/train_job.py', 'stage1', '--arm', arm, '--splits', str(SPLITS), '--images', str(IMAGE_DIR),
           '--out', str(STAGE1), '--epochs', str(STAGE1_EPOCHS), '--patience', str(PATIENCE), '--workers', str(WORKERS), '--device', DEVICE_NAME]
    if not PRETRAINED:
        cmd.append('--no-pretrained')
    if MAX_TRAIN_BATCHES:
        cmd += ['--max-train-batches', str(MAX_TRAIN_BATCHES)]
    if not AMP:
        cmd += ['--no-amp']
    return cmd


try:
    run_on_gpus(GPU_JOBS, stage1_command, OUT / 'logs_stage1', CODE_DIR, budget_hours=STAGE1_BUDGET_HOURS,
                report_seconds=REPORT_SECONDS)
finally:
    save_results(OUT, DRIVE_RESULTS)   # logs and histories reach Drive even if a job failed
print(f'Stage 1 finished in {(time.time() - start) / 60:.0f} min')
"""),
    (CODE, r"""
# 7. Stage 1 board and curves for the two new models (information: no model is chosen, both are used).
import matplotlib.pyplot as plt

RESULTS = OUT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
board = st.stage1_board(STAGE1, TRAINED)
board.to_csv(RESULTS / 'dev_stage1_board.csv', index=False)
display(board.round(4))

fig, axes = plt.subplots(len(TRAINED), 2, figsize=(13, 3.2 * len(TRAINED)), squeeze=False)
for row, arm in zip(axes, TRAINED):
    h = pd.read_csv(STAGE1 / arm / 'history.csv')
    for col, label in (('train_loss', 'train'), ('seen_loss', 'dev seen'), ('unseen_loss', 'dev unseen')):
        row[0].plot(h.epoch, h[col], marker='o', ms=3, label=label)
    for col, label in (('train_acc', 'train'), ('seen_acc', 'dev seen'), ('unseen_acc', 'dev unseen')):
        row[1].plot(h.epoch, 100 * h[col], marker='o', ms=3, label=label)
    best = json.loads((STAGE1 / arm / 'done.json').read_text())['best_epoch']   # the epoch the code really kept
    for ax, title in zip(row, ('loss', 'Top-1 accuracy (%)')):
        ax.axvline(best, color='grey', ls='--', lw=1)
        ax.set_title(f'{arm}: {title} (dashed line = kept epoch {best})')
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / 'stage1_curves.png', dpi=150)
plt.show()
"""),
    (CODE, r"""
# 8. Flip dev outputs of the two new models, from their stage-1 best checkpoints (the same flip averaging as the test).
import numpy as np

assert CROP_IDS == list(range(len(CROP_IDS))) and DISEASE_IDS == list(range(len(DISEASE_IDS))), 'column j must be class id j'
NEW_DEV = {}
for arm in TRAINED:
    model = st.load_trained(STAGE1 / arm / 'best.pt', DEVICE)
    for split, df in (('seen', val_seen), ('unseen', val_unseen)):
        NEW_DEV[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
np.savez_compressed(RESULTS / 'dev_flip_logprobs_p6_new.npz', **{f'{a}_{s}': v for (a, s), v in NEW_DEV.items()})
for arm in TRAINED:
    shutil.copy2(STAGE1 / arm / 'best_validation.npz', RESULTS / f'dev_best_validation_{arm}.npz')
    shutil.copy2(STAGE1 / arm / 'history.csv', RESULTS / f'dev_history_{arm}.csv')
save_results(OUT, DRIVE_RESULTS)
"""),
    (MD, r"""
## Stage 2: 10 more epochs on all 8,000 images (both new models)
"""),
    (CODE, r"""
# 9. Stage 2: both new models, one per GPU slot, at the same time.
STAGE2 = OUT / 'stage2'
FINAL_ARMS = [pm.NEW_ARMS['crop'], pm.NEW_ARMS['disease']]
jobs = {(0, 0): [FINAL_ARMS[0]], (0, 1): [FINAL_ARMS[1]]}
require_fresh(STAGE2, FINAL_ARMS)
start = time.time()

def stage2_command(arm):
    cmd = [sys.executable, 'scripts/train_job.py', 'stage2', '--arm', arm, '--stage1', str(STAGE1), '--train-csv', str(train_csv),
           '--images', str(IMAGE_DIR), '--out', str(STAGE2), '--epochs', str(STAGE2_EPOCHS), '--workers', str(WORKERS), '--device', DEVICE_NAME]
    if MAX_TRAIN_BATCHES:
        cmd += ['--max-train-batches', str(MAX_TRAIN_BATCHES)]
    if not AMP:
        cmd += ['--no-amp']
    return cmd


try:
    run_on_gpus(jobs, stage2_command, OUT / 'logs_stage2', CODE_DIR, report_seconds=REPORT_SECONDS)
finally:
    save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',))   # the two final models reach Drive here
print(f'Stage 2 finished in {(time.time() - start) / 60:.0f} min')
for arm in FINAL_ARMS:
    display(pd.read_csv(STAGE2 / arm / 'history.csv').round(4))
"""),
    (CODE, r"""
# 10. Calibration values from dev only, then the lock, BEFORE the test files are read.
#     The lock records the sha256 of both test CSVs (the bytes are hashed, nothing is parsed), so cell 11 can prove they did not change.
#     Both rules use the flip dev outputs of the four models (P5 files for crop_b1 and disease_clip, the new files for the other two).
#     Rule 1 (main calibrated result, same as P1 to P5): gamma is taken off the score of every seen pair; the largest
#     dev unseen pair accuracy while dev seen pair accuracy stays within 3 points of raw, r = 1.
#     Rule 2: the weight r of log P(disease) and the gamma with the best dev pair HM (r in 0.5-2; a tie goes to the r
#     closest to 1, then the smaller r, then the smaller gamma). The test cell checks the fingerprints again.
from cdccp import calibration as pg
from cdccp.catalogue import load_catalogue

CAT_DEV_PATH = CODE_DIR / 'data' / 'mappings' / 'catalogue_dev.json'
CAT_FINAL_PATH = CODE_DIR / 'data' / 'mappings' / 'catalogue_final.json'
CAT_DEV = load_catalogue(CAT_DEV_PATH)
DEV_PARTS = pm.p6_parts(P5_DEV, NEW_DEV, (val_seen, val_unseen))
DEV_CAL = pm.dev_calibration(*DEV_PARTS, CAT_DEV)
GAMMA, GAMMA_HM, R_HM = DEV_CAL['gamma_keep'], DEV_CAL['gamma_hm'], DEV_CAL['r_hm']
pd.DataFrame([DEV_CAL]).to_csv(RESULTS / 'dev_calibration.csv', index=False)
print(f"Rule 1, keep seen (seen drops at most 3 points on dev, r 1): gamma = {GAMMA:.3f}. Dev pair Top-1: "
      f"seen {DEV_CAL['A_S_raw']:.2f} -> {DEV_CAL['A_S_keep']:.2f}, unseen {DEV_CAL['A_U_raw']:.2f} -> {DEV_CAL['A_U_keep']:.2f}")
print(f"Rule 2, best dev HM: r = {R_HM:g}, gamma = {GAMMA_HM:.3f}. Dev pair Top-1: seen {DEV_CAL['A_S_hm']:.2f}, "
      f"unseen {DEV_CAL['A_U_hm']:.2f}, HM {DEV_CAL['H_hm']:.2f}")

# Information only, nothing is chosen from it: dev unseen pair Top-1 without the two healthy unseen pairs and the mean over pairs,
# for P5 (crop_b1 + disease_clip) and for P6. The dev unseen set has 400 images: 1 image = 0.25 points.
P5_DEV_PARTS = tuple(pg.parts_from_logprobs(*pm.member_heads(P5_DEV, pm.P5_ARMS)[s], df) for s, df in zip(pm.SPLITS, (val_seen, val_unseen)))
P5_DEV_CAL = pm.dev_calibration(*P5_DEV_PARTS, CAT_DEV)
dev_numbers = pd.concat([pm.dev_unseen_numbers(P5_DEV_PARTS[1], CAT_DEV, P5_DEV_CAL, 'P5 (crop_b1 + disease_clip)'),
                         pm.dev_unseen_numbers(DEV_PARTS[1], CAT_DEV, DEV_CAL, 'P6 (four models)')], ignore_index=True)
dev_numbers.to_csv(RESULTS / 'dev_unseen_numbers.csv', index=False)
display(dev_numbers.round(2))

FINAL = {arm: STAGE2 / arm / 'best_train_acc.pt' for arm in FINAL_ARMS}
LOCK = st.write_lock({**{arm: str(p) for arm, p in FINAL.items()}, 'train_csv': str(train_csv),
                      'catalogue': str(CAT_FINAL_PATH), 'catalogue_dev': str(CAT_DEV_PATH),
                      **{f'p5_{name}': str(path) for name, path in P5_FILES.items()}, 'p5_crop_b1_model_local': str(LOCAL_CROP_B1),
                      **{f'test_csv_{split}': str(path) for split, path in OFFICIAL.items()},
                      **{f'drive_test_csv_{split}': str(path) for split, path in DRIVE_TEST_CSVS.items()}},
                     {'method': 'P6', 'crop_members': [pm.P5_ARMS['crop'], pm.NEW_ARMS['crop']],
                      'disease_members': [pm.P5_ARMS['disease'], pm.NEW_ARMS['disease']], 'weights': list(pm.WEIGHTS),
                      'ensemble_rule': 'head = weighted mean of log_softmax of the members, weights 0.5 and 0.5, flip averaging on all four',
                      'seed': SEED, 'code_sha256': CODE_SHA, 'stage1_epochs': STAGE1_EPOCHS, 'patience': PATIENCE,
                      'stage2_epochs': STAGE2_EPOCHS, 'gpu': GPU_NAME, 'flip': FLIP, 'gamma': GAMMA,
                      'gamma_rule': 'dev keep-seen: max dev unseen pair Top-1 with dev seen pair Top-1 >= raw - 3, r 1, flip dev outputs of the four models',
                      'gamma_hm': GAMMA_HM, 'r_hm': R_HM,
                      'gamma_hm_rule': 'dev best pair HM over r in (0.5, 0.75, 1, 1.25, 1.5, 2) x gamma; tie -> smallest |log r|, smaller r, smaller gamma; flip dev outputs of the four models',
                      'p5_sha256': P5_SHA},
                     OUT / 'lock.json')
print(json.dumps(LOCK, indent=1))
for split in DRIVE_TEST_CSVS:
    assert LOCK['files'][f'test_csv_{split}']['sha256'] == LOCK['files'][f'drive_test_csv_{split}']['sha256'], (
        f'the local and the Drive DPD_Small_test_{split}.csv differ: do not run the test')
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',))
"""),
    (MD, r"""
## Official test (runs once)
"""),
    (CODE, r"""
# 11. Official test, once: check the lock, check the P5 files without any label, write the marker, read the test CSVs, check that the
#     rows are in the order P5 used (P5's own crop model must reproduce the saved P5 test log-probabilities row by row), recompute the
#     P5 test table (it must equal P5's own), then score the two new models and the four-model mean.
LOCK_INFO = st.verify_lock(OUT / 'lock.json')['info']   # the lock is checked here, before the marker; cell 11b uses LOCK_INFO
STARTED = OUT / 'official_test_started.json'   # the test runs once: a second run of this cell stops here
assert not STARTED.exists(), f'the official test already ran in this output folder ({STARTED}); it must not run twice'
assert not list(OUT.parent.glob('p6*/official_test_started.json')), 'the official test already ran (marker in a local p6 folder)'
assert not list(DRIVE_DIR.glob('p6_results*/official_test_started.json')), 'the official test already ran (marker on Drive)'

# Checks that need no label, so a failure here leaves no marker and the notebook can run again.
P5_TEST = pm.read_logprobs(P5_FILES['test_logprobs'], pm.P5_ARMS.values(), {'seen': EXPECT_TEST[0], 'unseen': EXPECT_TEST[1]})
P5_SAVED_TABLE = pm.check_results_layout(P5_FILES['test_results'])   # kept in memory: no Drive read after the marker
ALLOWED_333 = pm.pair_mask(CAT_FINAL_PATH, CROP_IDS, DISEASE_IDS)

STARTED.write_text(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'lock_sha256': hashlib.sha256((OUT / 'lock.json').read_bytes()).hexdigest()}))
copy_file(STARTED, DRIVE_RESULTS / STARTED.name, retries=10, wait=30)
test = {split: st.read_labelled_csv(path) for split, path in OFFICIAL.items()}
pm.check_test_frames(test, train_all, EXPECT_TEST, disjoint=TEST_DISJOINT_FROM_TRAIN)

# First: P5's own stage-2 crop model (sha256 pinned in cell 3) predicts every test row in the order of these CSVs, with flip averaging
# and the AMP setting P5 used. Each row must agree with the log-probabilities P5 saved for that row (largest difference over the 55
# classes at most 1.0). Other images or another row order differ by far more, so this stops here, before any P6 number exists.
row_model = st.load_trained(LOCAL_CROP_B1, DEVICE)
ROW_CHECK_NOW = {}
for split, df in test.items():
    ROW_CHECK_NOW[split] = st.predict(row_model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)
del row_model
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
ROW_CHECK = pm.check_rows_match_saved(ROW_CHECK_NOW, {split: P5_TEST[pm.P5_ARMS['crop'], split] for split in pm.SPLITS})
(RESULTS / 'p5_row_check.json').write_text(json.dumps(ROW_CHECK, indent=1))
for split, numbers in ROW_CHECK.items():
    print(f"row order check, {split} ({numbers['rows']} rows): largest per-row difference to the saved P5 crop log-probabilities "
          f"max {numbers['max']:.5f} (row {numbers['worst_row']}), median {numbers['median']:.5f}, 99th percentile {numbers['p99']:.5f}; limit {pm.ROW_MAX_ABS_DIFF}")

# Second: the saved P5 test log-probabilities and these test labels must give P5's own table (read into memory before the marker).
_, P5_TABLE = pm.official_table(pm.member_heads(P5_TEST, pm.P5_ARMS), test, CROP_IDS, DISEASE_IDS, ALLOWED_333)
pm.assert_matches_table(P5_TABLE, P5_SAVED_TABLE)
print('P5 table recomputed from the saved P5 log-probabilities equals p5 official_test_results.csv: the rows line up.')

LOGP_NEW = {}   # flip averaging: gives every table
for arm, path in FINAL.items():
    model = st.load_trained(path, DEVICE)
    for split, df in test.items():
        LOGP_NEW[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS, flip=FLIP)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

TEST_PARTS = pm.p6_parts(P5_TEST, LOGP_NEW, (test['seen'], test['unseen']))
HEADS = pm.parts_heads(TEST_PARTS)
METRICS, results = pm.official_table(HEADS, test, CROP_IDS, DISEASE_IDS, ALLOWED_333)
results.to_csv(RESULTS / 'official_test_results.csv', index=False)
np.savez_compressed(RESULTS / 'official_test_logprobs_p6_new.npz', **{f'{a}_{s}': v for (a, s), v in LOGP_NEW.items()})
np.savez_compressed(RESULTS / 'official_test_logprobs_p6_ensemble.npz',
                    **{f'{head}_{s}': HEADS[s][i] for s in pm.SPLITS for i, head in enumerate(('crop', 'disease'))})
display(results.round(2))
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)   # official test results on Drive now

# Appendix, after the main results are saved: the same two models without flip averaging. No table uses these.
LOGP_NEW_NOFLIP = {}
for arm, path in FINAL.items():
    model = st.load_trained(path, DEVICE)
    for split, df in test.items():
        LOGP_NEW_NOFLIP[arm, split] = st.predict(model, df, IMAGE_DIR, CROP_IDS, DISEASE_IDS, DEVICE, amp=AMP, workers=WORKERS)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
np.savez_compressed(RESULTS / 'official_test_logprobs_p6_new_noflip.npz', **{f'{a}_{s}': v for (a, s), v in LOGP_NEW_NOFLIP.items()})
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)
"""),
    (CODE, r"""
# 11b. Raw and calibrated in one table (the layout of the team's table). Same four models, same test outputs.
#      Raw, heads only = Baseline 1 rule. Raw, pair list = top pair of the 333 pairs. Calibrated = the same after a
#      locked dev calibration (rule 1 keep seen at r 1, rule 2 best dev HM with its r). Reference = never name a seen pair.
#      The RAW rows are the main result; calibrated rows are the second result. LOCK_INFO is the lock checked in cell 11, before the marker.
topline = pd.DataFrame(pg.topline_rows(TEST_PARTS[0], TEST_PARTS[1], load_catalogue(CAT_FINAL_PATH),
                                       LOCK_INFO['gamma'], gamma_hm=LOCK_INFO['gamma_hm'], r_hm=LOCK_INFO['r_hm']))
topline.to_csv(RESULTS / 'official_test_raw_and_calibrated.csv', index=False)
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)
display(topline.round(2))
"""),
    (CODE, r"""
# 12. Baseline 1, P5 and P6 under the same pair rule (all 9,625 pairs). P5 and P6 use flip averaging, Baseline 1 does not.
#     P5 here is recomputed from its saved log-probabilities (it equals P5's own table, checked in cell 11).
#     Baseline 1 numbers are from Amanda's notebook.
vs = pm.compare_table(P5_TABLE, results, B1_TEST)
vs.to_csv(RESULTS / 'p6_vs_p5_vs_b1.csv', index=False)
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)
display(vs.round(2))
print('All are one seed (42). This is the one official test run of P6.')
"""),
    (CODE, r"""
# 13. The 10 unseen test pairs: how often the top pair has the right crop and the right disease (all 9,625 pairs).
per_pair = pm.unseen_pair_table(HEADS['unseen'][0], HEADS['unseen'][1], test['unseen'], CROP_IDS, DISEASE_IDS)
per_pair.to_csv(RESULTS / 'unseen_pairs.csv')
display(per_pair.round(1))
"""),
    (CODE, r"""
# 14. Confusion matrix for the seen test images (the brief asks for one): true seen pair (rows) vs predicted pair (columns),
#     333-pair list. Each row is divided by its image count, so the diagonal shows the accuracy of each pair.
catalogue = json.loads(CAT_FINAL_PATH.read_text())['pairs']
conf, rows_pairs, pairs = pm.seen_confusion(HEADS['seen'][0], HEADS['seen'][1], test['seen'], catalogue, CROP_IDS, DISEASE_IDS, ALLOWED_333)
np.save(RESULTS / 'confusion_seen_333.npy', conf)
fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.get_cmap('Blues').copy()
cmap.set_under('white')
ax.imshow(conf, cmap=cmap, vmin=0.3, vmax=1, aspect='auto', interpolation='nearest')
ax.axvline(sum(p['is_seen'] for p in catalogue) - 0.5, color='red', lw=1)
ax.set_xlabel('predicted pair (333 list; right of the red line = the 10 unseen pairs)')
ax.set_ylabel(f'true seen pair ({len(rows_pairs)} pairs in the seen test)')
ax.set_title(f'P6 seen test confusion (333 list): pair Top-1 = {100 * METRICS[pm.LIST_333, "seen"]["composition_top1"]:.2f}%')
fig.savefig(RESULTS / 'confusion_seen_333.png', dpi=150)
plt.show()
"""),
    (CODE, r"""
# 15. Save everything to Drive (tables, graphs, test log-probabilities, dev outputs, lock, and the two final models),
#     then finish the Drive upload (Colab uploads in the background) before saying "Finished".
for p in sorted(RESULTS.iterdir()):
    print(p.name)
save_results(OUT, DRIVE_RESULTS, model_folders=('stage2',), retries=10, wait=30)
if not SMOKE:
    drive.flush_and_unmount()
print('Finished. Results are in My Drive/DPD-Small/p6_results. You can close this tab.')
"""),
]


def code_sha_of_dir(code_dir):
    """The fingerprint the notebook computes in cell 2, for an unpacked cdc-cp folder."""
    scope = {"CODE_DIR": Path(code_dir), "hashlib": hashlib}
    exec(FINGERPRINT, scope)
    return scope["CODE_SHA"]


def code_sha_of_zip(zip_path):
    """The fingerprint of the cdc-cp folder inside a code zip (the folder holding cdccp/gpu_runner.py)."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp)
        found = sorted(p.parent.parent for p in Path(tmp).rglob("gpu_runner.py") if p.parent.name == "cdccp")
        if len(found) != 1:
            raise ValueError(f"expected one cdccp/gpu_runner.py in {zip_path}, found {found}")
        return code_sha_of_dir(found[0])


def build(out, code_sha):
    if len(code_sha) != 64 or set(code_sha) - set("0123456789abcdef"):
        raise ValueError(f"code_sha must be 64 hex characters, got {code_sha!r}")
    cells = []
    for kind, source in CELLS:
        text = source.strip("\n").replace(PLACEHOLDER_FINGERPRINT, FINGERPRINT).replace(PLACEHOLDER_SHA, code_sha)
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python"}, "accelerator": "GPU",
                                             "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    Path(out).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return len(cells)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the P6 Colab notebook")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="the packed code zip; its fingerprint goes into the notebook")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    sha = code_sha_of_zip(args.zip)
    count = build(args.out, sha)
    print(f"wrote {args.out} ({count} cells), code fingerprint {sha}")


if __name__ == "__main__":
    main()
