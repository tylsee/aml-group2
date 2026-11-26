"""CDC-CP phases P1-P3: single-task models per task, averaged per task, then one pair score.

    P1: crop_b1 (ConvNeXt-Tiny) + disease_b1 (ConvNeXt-Tiny)
    P2: crop_b1 + disease_clip (CLIP ViT-B/16)
    P3: crop = mean of log_softmax(crop_b1, crop_clip); disease = mean of log_softmax(disease_clip, disease_small_reg)
    S(c, d) = crop(c) + r * disease(d) - gamma * [pair (c, d) was seen in training]

This module holds what one fresh run of all five models needs; the scoring is cdccp.ensemble and cdccp.calibration, the
training is cdccp.single_task.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import StochasticDepth

from . import calibration as pg
from . import ensemble as pm
from . import single_task as st

# phase -> task -> models; the i-th crop model and the i-th disease model form ensemble member i (equal weights)
PHASES = {"P1": {"crop": ("crop_b1",), "disease": ("disease_b1",)},
          "P2": {"crop": ("crop_b1",), "disease": ("disease_clip",)},
          "P3": {"crop": ("crop_b1", "crop_clip"), "disease": ("disease_clip", "disease_small_reg")}}
ALL_MEMBERS = tuple(dict.fromkeys(arm for phase in PHASES.values() for task in ("crop", "disease") for arm in phase[task]))
TRAIN_CSV_SHA = "9f1b729a"
SPLIT_SHA = {"train_fit.csv": "fe02d6099c00", "val_seen.csv": "ab8d18a13b28", "val_unseen.csv": "b9eae2a57d83"}

# Official test, all 9,625 pairs (Baseline 1 rule): task -> (seen Top-1, unseen Top-1, seen Top-5, unseen Top-5), in %.
EARLIER_RUNS = {
    "Baseline 1": {"Crop": (89.30, 73.07, 96.20, 84.74), "Disease": (66.25, 11.12, 88.45, 26.73),
                   "Composition": (63.65, 6.56, 84.00, 18.85)},
}


def code_fingerprint(code_dir):
    """sha256 over every file's relative path and bytes (sorted), __pycache__ left out."""
    code_dir = Path(code_dir)
    files = {f.relative_to(code_dir).as_posix(): f for f in code_dir.rglob("*") if f.is_file() and "__pycache__" not in f.parts}
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode() + b"\0" + files[rel].read_bytes())
    return digest.hexdigest()


def check_inputs(train_csv, splits_dir):
    """Raise unless the train CSV and the three dev split files are the Baseline 1 files (sha256 prefixes)."""
    wrong = [] if pm.sha256_of(train_csv).startswith(TRAIN_CSV_SHA) else [str(train_csv)]
    wrong += [name for name, sha in SPLIT_SHA.items() if not pm.sha256_of(Path(splits_dir) / name).startswith(sha)]
    if wrong:
        raise ValueError(f"these files are not the Baseline 1 train CSV / development split: {wrong}")


def check_jobs(gpu_jobs):
    """Raise unless the GPU job lists train each of the five models exactly once."""
    trained = [arm for arms in gpu_jobs.values() for arm in arms]
    if sorted(trained) != sorted(ALL_MEMBERS):
        raise ValueError(f"GPU_JOBS must train each of {list(ALL_MEMBERS)} once, got {trained}")
    return trained


def stochastic_depth_of(arm):
    """The stochastic depth the member's network really has (the largest drop rate of its blocks); 0.0 for CLIP, which has none.

    A ConvNeXt built without an explicit value keeps the torchvision default, so the value is read from a built network.
    """
    task, _, backbone, recipe = st.ARMS[arm]
    if backbone not in st.BACKBONES:
        return 0.0
    model = st.build_model(task, st.B1["dropout"][task], False, backbone=backbone,
                           stochastic_depth_prob=st.RECIPES[recipe]["stochastic_depth"][backbone])
    return max(float(m.p) for m in model.modules() if isinstance(m, StochasticDepth))


def member_settings():
    """One row per member with its training settings, read from cdccp.single_task (the values the jobs use)."""
    rows = []
    for arm in ALL_MEMBERS:
        task, aug, backbone, recipe = st.ARMS[arm]
        rc = st.RECIPES[recipe]
        (lr1_backbone, lr1_head), (lr2_backbone, lr2_head) = st.stage_lrs(recipe, 1), st.stage_lrs(recipe, 2)
        clip = backbone == "clip_vit_b16"
        rows.append({"model": arm, "task": task, "backbone": backbone, "augmentation": aug,
                     "head": "frozen text prompts + learned scale" if clip else f"dropout {st.B1['dropout'][task]} + linear",
                     "stage 1 lr (backbone / head)": f"{lr1_backbone:g} / {lr1_head:g}",
                     "stage 2 lr (backbone / head)": f"{lr2_backbone:g} / {lr2_head:g}",
                     "weight decay": rc["weight_decay"], "label smoothing": rc["label_smoothing"],
                     "stochastic depth": stochastic_depth_of(arm)})
    return pd.DataFrame(rows)


def job_command(stage, arm, *, images, out, epochs, workers, device, amp=True, max_batches=None,
                splits=None, patience=None, pretrained=True, stage1=None, train_csv=None):
    """The scripts/train_job.py command line for one member: stage 1 (dev split) or stage 2 (all 8,000 images)."""
    cmd = [sys.executable, "scripts/train_job.py", stage, "--arm", arm, "--images", str(images), "--out", str(out),
           "--epochs", str(epochs), "--workers", str(workers), "--device", device]
    if stage == "stage1":
        cmd += ["--splits", str(splits), "--patience", str(patience)]
        if not pretrained:
            cmd.append("--no-pretrained")
    elif stage == "stage2":
        cmd += ["--stage1", str(stage1), "--train-csv", str(train_csv)]
    else:
        raise ValueError(f"stage must be 'stage1' or 'stage2', got {stage!r}")
    if max_batches:
        cmd += ["--max-train-batches", str(max_batches)]
    if not amp:
        cmd.append("--no-amp")
    return cmd


def preflight(members, device, amp, pretrained, val_seen, image_dir, class_names):
    """Checks that take a minute now instead of failing after hours of training. Raises on a problem, prints each result.

    ImageNet weights download, half precision gives finite numbers on this GPU, the CLIP text model loads, and the CLIP
    weights beat 2x chance zero-shot on dev seen (broken weights or prompts give chance).
    """
    backbones = {st.ARMS[arm].backbone for arm in members}
    for backbone in sorted(backbones & set(st.BACKBONES)):
        constructor, weights = st.BACKBONES[backbone]
        if pretrained:
            weights.IMAGENET1K_V1.get_state_dict(progress=False)
            print("ImageNet weights ready:", backbone)
        if amp:
            probe = constructor(weights=weights.IMAGENET1K_V1 if pretrained else None).to(device).eval()
            generator = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for n in st.PROBE_BATCHES:
                    with torch.autocast(device.type):
                        out = probe(torch.randn(n, 3, 224, 224, generator=generator).to(device)).float()
                    if not torch.isfinite(out).all():
                        raise RuntimeError(f"half precision gives NaN/inf on this GPU ({backbone}, batch {n}): set AMP = False")
            del probe
            print(f"half precision OK for {backbone}, batches {st.PROBE_BATCHES}")
    if "clip_vit_b16" in backbones:
        for task in sorted({st.ARMS[arm].task for arm in members if st.ARMS[arm].backbone == "clip_vit_b16"}):
            st.clip_text_check(task, class_names[task], pretrained=pretrained)
            score = st.clip_zero_shot(task, val_seen, image_dir, class_names[task], device, n=len(val_seen), pretrained=pretrained)
            chance = 100 / len(class_names[task])
            print(f"CLIP zero-shot {task} on dev seen, before training: {score:.2f}% (chance {chance:.2f}%)")
            if pretrained and score <= 2 * chance:
                raise RuntimeError(f"CLIP zero-shot {task} is {score:.2f}%, not above 2x chance: check the CLIP weights")
        if amp:
            print("CLIP half precision OK for batches", st.clip_fp16_probe(device, pretrained=pretrained))
    if device.type == "cuda":
        torch.cuda.empty_cache()


def stage2_summary(stage2_dir, members=ALL_MEMBERS):
    """One row per member: the stage-1 epoch it started from, the stage-2 epoch kept and its training accuracy (%)."""
    rows = []
    for arm in members:
        done = json.loads((Path(stage2_dir) / arm / "done.json").read_text())
        rows.append({"model": arm, "started from stage-1 epoch": int(done["source_epoch"]), "kept stage-2 epoch": int(done["best_epoch"]),
                     "train accuracy (%)": 100 * float(done["best_train_acc"])})
    return pd.DataFrame(rows)


def predict_members(paths, frames, image_dir, crop_ids, disease_ids, device, amp, workers, flip=True):
    """{(member, split): log-probabilities} for each checkpoint in paths ({member: path}) and frames ({split: labelled frame})."""
    out = {}
    for arm, path in paths.items():
        model = st.load_trained(path, device)
        for split, df in frames.items():
            out[arm, split] = st.predict(model, df, image_dir, crop_ids, disease_ids, device, amp=amp, workers=workers, flip=flip)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def phase_parts(logp, dfs, phase):
    """(seen, unseen) parts of one phase from {(model, split): log-probabilities}: per task, the equal-weight mean of the
    log_softmax of its models. dfs = (seen frame, unseen frame) in the row order of the arrays."""
    tasks = PHASES[phase]
    missing = [(arm, split) for task in ("crop", "disease") for arm in tasks[task] for split in pm.SPLITS if (arm, split) not in logp]
    if missing:
        raise ValueError(f"log-probabilities missing for {missing}")
    members = [tuple(pg.parts_from_logprobs(logp[crop, split], logp[disease, split], df) for split, df in zip(pm.SPLITS, dfs))
               for crop, disease in zip(tasks["crop"], tasks["disease"])]
    return pg.ensemble(members)


def _row(run, task):
    seen1, unseen1, seen5, unseen5 = run[task]
    return seen1, unseen1, 100 * st.harmonic(seen1 / 100, unseen1 / 100), 100 * st.harmonic(seen5 / 100, unseen5 / 100)


def table_as_run(table, pair_list=pm.ALL_PAIRS):
    """{task: (seen Top-1, unseen Top-1, seen Top-5, unseen Top-5)} from an official_table frame, for one pair list."""
    run = {}
    for task in pm.TASKS:
        values = []
        for k in (1, 5):
            hit = table[(table["pair list"] == pair_list) & (table.Task == task.title()) & (table["Top-K"] == k)]
            if len(hit) != 1:
                raise ValueError(f"the table has {len(hit)} rows for {pair_list}, {task}, Top-{k}")
            values.append((float(hit["Seen (%)"].iloc[0]), float(hit["Unseen (%)"].iloc[0])))
        run[task.title()] = (values[0][0], values[0][1], values[1][0], values[1][1])
    return run


def compare_runs(tables, earlier=None):
    """Raw results, all 9,625 pairs (Baseline 1 rule), one row per run: the earlier runs, then {name: official_table frame}."""
    runs = {**(EARLIER_RUNS if earlier is None else earlier), **{name: table_as_run(t) for name, t in tables.items()}}
    rows = []
    for name, run in runs.items():
        crop, disease, pair = (_row(run, task) for task in ("Crop", "Disease", "Composition"))
        rows.append({"model": name, "crop HM": crop[2], "disease seen": disease[0], "disease unseen": disease[1],
                     "disease HM": disease[2], "pair seen": pair[0], "pair unseen": pair[1], "pair HM": pair[2],
                     "pair Top-5 HM": pair[3]})
    return pd.DataFrame(rows).set_index("model")
