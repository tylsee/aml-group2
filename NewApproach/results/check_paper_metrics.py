"""Checks paper_metrics.py on the development split (no test labels) and against the cdccp code and saved run outputs.

Run: python check_paper_metrics.py. Exits non-zero on the first failed check.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CDC = HERE.parent.parent / "cdc-cp"
RES = HERE / "final_model_results" / "results"
SPLITS = CDC / "data" / "splits_v2"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CDC))

import paper_metrics as pm
from cdccp import calibration as cb
from cdccp import evaluate as ev
from cdccp import final_model as fm
from cdccp import single_task as st
from cdccp.catalogue import load_catalogue
from cdccp.scoring import head_score

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


check("phase definitions equal cdccp.final_model.PHASES", pm.PHASES == fm.PHASES)

# The saved test heads of each phase equal paper_metrics' own phase mean (no labels needed).
models = np.load(RES / "official_test_logprobs_models.npz")
for phase in pm.PHASES:
    saved = np.load(RES / f"official_test_logprobs_{phase}.npz")
    for split in ("seen", "unseen"):
        crop, disease = pm.phase_heads(models, phase, split)
        diff = max(np.abs(np.exp(crop) - np.exp(pm.log_softmax(saved[f"crop_{split}"]))).max(),
                   np.abs(np.exp(disease) - np.exp(pm.log_softmax(saved[f"disease_{split}"]))).max())
        check(f"{phase} {split}: phase heads match saved test heads", diff < 1e-5, f"max prob diff {diff:.2e}")

# Dev: the pair list built from train_fit + val_unseen equals catalogue_dev.json.
train_fit, val_seen, val_unseen = st.load_dev(SPLITS)
labels = {"seen": pm.read_labels(SPLITS / "val_seen.csv"), "unseen": pm.read_labels(SPLITS / "val_unseen.csv")}
train_pairs = set(zip(*pm.read_labels(SPLITS / "train_fit.csv")))
seen_mask, closed = pm.masks(train_pairs, set(zip(*labels["unseen"])))
cat_dev = load_catalogue(CDC / "data" / "mappings" / "catalogue_dev.json")
cat_pairs = {(int(c), int(d)): bool(s) for c, d, s in zip(cat_dev.crop_id, cat_dev.disease_id, cat_dev.is_seen)}
built = {(c, d): bool(seen_mask[c, d]) for c, d in zip(*np.nonzero(closed))}
check("dev closed list equals catalogue_dev.json (pairs and seen flags)", built == cat_pairs, f"{len(built)} vs {len(cat_pairs)}")

dev_npz = np.load(RES / "dev_flip_logprobs.npz")
with open(RES / "dev_calibration.csv", newline="") as f:
    dev_cal = {r["phase"]: r for r in csv.DictReader(f)}
results, sides = pm.evaluate(dev_npz, labels, train_pairs, r=1.0)
dev_logp = {(arm, split): dev_npz[f"{arm}_{split}"] for arm in fm.ALL_MEMBERS for split in ("seen", "unseen")}

for row in results:
    phase, world = row["phase"], row["world"]
    if world != "closed":
        continue
    cal, side = dev_cal[phase], sides[phase, world]
    check(f"{phase} dev raw seen = dev_calibration A_S_raw", close(row["raw_seen"], cal["A_S_raw"]), f"{row['raw_seen']:.4f}")
    check(f"{phase} dev raw unseen = dev_calibration A_U_raw", close(row["raw_unseen"], cal["A_U_raw"]), f"{row['raw_unseen']:.4f}")
    g = float(cal["gamma_keep"])
    a_s, a_u = pm.accuracy_at(side["seen"], g), pm.accuracy_at(side["unseen"], g)
    check(f"{phase} dev rule-1 point reproduced", close(a_s, cal["A_S_keep"]) and close(a_u, cal["A_U_keep"]), f"{a_s:.2f}/{a_u:.2f}")

    # Independent route: cdccp's own scores and sweep at the exact curve gammas.
    parts = fm.phase_parts(dev_logp, (val_seen, val_unseen), phase)
    cat = cb.report_order(cat_dev)
    ss, su = head_score(parts[0], cat), head_score(parts[1], cat)
    gammas, c_s, c_u, c_h = pm.curve(side["seen"], side["unseen"])
    picks = np.unique(np.linspace(0, len(gammas) - 1, 60).astype(int))
    rows = ev.sweep(ss, parts[0], su, parts[1], cat, gammas[picks])
    check(f"{phase} dev curve matches cdccp sweep at 60 gammas",
          np.allclose(rows[:, 1], c_s[picks]) and np.allclose(rows[:, 2], c_u[picks]))
    full = ev.sweep(ss, parts[0], su, parts[1], cat, gammas)
    check(f"{phase} dev AUC equals cdccp curve_auc on the same points", close(100 * ev.curve_auc(full), row["auc"], 1e-9),
          f"{row['auc']:.4f}")
    grid_best = cb.best_hm_gamma(parts[0], parts[1], cat_dev)["H_hm"]
    check(f"{phase} dev exact best HM >= cdccp 201-point grid best HM", row["best_hm"] >= grid_best - 1e-9,
          f"{row['best_hm']:.2f} vs {grid_best:.2f}")
    if float(cal["r_hm"]) == 1.0:
        check(f"{phase} dev grid best HM = dev_calibration H_hm (r = 1)", close(grid_best, cal["H_hm"], 1e-6))

# The dev file the report reads equals this fresh computation.
with open(HERE / "paper_metrics_dev.csv", newline="") as f:
    dev_file = {(r["phase"], r["world"]): r for r in csv.DictReader(f)}
for row in results:
    saved = dev_file[row["phase"], row["world"]]
    same = all(close(saved[k], v, 1e-9) for k, v in row.items() if k not in ("phase", "world"))
    check(f"{row['phase']} dev {row['world']}: paper_metrics_dev.csv equals a fresh run", same)


def brute(npz, labels_, train_pairs_, phase, world, gamma):
    """Plain argmax over the full crop x disease grid after gamma is taken off the seen pairs (no shortcuts)."""
    seen_m, closed_m = pm.masks(train_pairs_, set(zip(*labels_["unseen"])))
    allowed = closed_m if world == "closed" else np.ones_like(closed_m)
    out = []
    for split in ("seen", "unseen"):
        crop, disease = pm.phase_heads(npz, phase, split)
        grid = crop[:, :, None] + disease[:, None, :] - gamma * seen_m[None]
        grid = np.where(allowed[None], grid, -np.inf).reshape(len(crop), -1)
        c, d = labels_[split]
        out.append(100 * (grid.argmax(axis=1) == c * pm.N_DISEASES + d).mean())
    return out


for phase in pm.PHASES:
    for world in ("closed", "open"):
        gammas, c_s, c_u, _ = pm.curve(*(sides[phase, world][s] for s in ("seen", "unseen")))
        picks = np.unique(np.linspace(0, len(gammas) - 1, 7).astype(int))
        ok = all(np.allclose(brute(dev_npz, labels, train_pairs, phase, world, gammas[i]), (c_s[i], c_u[i])) for i in picks)
        check(f"{phase} dev {world}: curve equals brute-force argmax at 7 gammas", ok)

# Open world at gamma 0 is the plain argmax of each head (the Baseline 1 rule).
for phase in pm.PHASES:
    parts = fm.phase_parts(dev_logp, (val_seen, val_unseen), phase)
    row = next(r for r in results if r["phase"] == phase and r["world"] == "open")
    for split, part, col in (("seen", parts[0], "raw_seen"), ("unseen", parts[1], "raw_unseen")):
        c, d = labels[split]
        both = (part["crop_logits"].argmax(1) == c) & (part["disease_logits"].argmax(1) == d)
        check(f"{phase} dev open-world raw {split} = both heads right", close(100 * both.mean(), row[col]), f"{row[col]:.2f}")

# Test output (paper_metrics_test.csv, if it exists): its end points must equal the notebook's own test tables.
test_csv = HERE / "paper_metrics_test.csv"
if test_csv.exists():
    with open(test_csv, newline="") as f:
        paper = {(r["phase"], r["world"]): r for r in csv.DictReader(f)}
    with open(RES / "official_test_raw_and_calibrated.csv", newline="") as f:
        top = {(r["phase"], r["Model"], r["Test split"]): float(r["Pair Top-1"]) for r in csv.DictReader(f)}
    for phase in pm.PHASES:
        closed_row, open_row = paper[phase, "closed"], paper[phase, "open"]
        for col, model, split in (("raw_seen", "Raw, pair list", "Seen pairs"), ("raw_unseen", "Raw, pair list", "Unseen pairs")):
            check(f"{phase} test closed {col} = notebook '{model}'", close(closed_row[col], top[phase, model, split]))
        for col, model, split in (("raw_seen", "Raw, heads only", "Seen pairs"), ("raw_unseen", "Raw, heads only", "Unseen pairs")):
            check(f"{phase} test open {col} = notebook '{model}'", close(open_row[col], top[phase, model, split]))
        check(f"{phase} test closed best unseen = notebook reference row (never a seen pair)",
              close(closed_row["best_unseen"], top[phase, "Reference: unseen pairs only", "Unseen pairs"]))
        for world, row in (("closed", closed_row), ("open", open_row)):
            s, u = float(row["raw_seen"]), float(row["raw_unseen"])
            ok = (float(row["best_seen"]) >= s and float(row["best_unseen"]) >= u
                  and float(row["best_hm"]) >= ev.harmonic(s, u) - 1e-9)
            check(f"{phase} test {world}: best seen, best unseen and best HM are not below the raw point", ok)
    # With the test labels (allowed by the user for this post-test metric): a fresh run equals the file, the notebook's
    # r = 1 calibrated test points lie exactly on the curve, and mid-curve points equal a brute-force argmax.
    data = HERE.parent.parent / "data" / "DPD_Small"
    test_labels = {s: pm.read_labels(data / f"DPD_Small_test_{s}.csv") for s in ("seen", "unseen")}
    test_train = set(zip(*pm.read_labels(data / "DPD_Small_train.csv")))
    check("test: 323 training pairs and 10 unseen pairs, no overlap",
          len(test_train) == 323 and len(set(zip(*test_labels["unseen"]))) == 10
          and not set(zip(*test_labels["unseen"])) & test_train)
    test_results, test_sides = pm.evaluate(models, test_labels, test_train, r=1.0)
    for row in test_results:
        saved = paper[row["phase"], row["world"]]
        same = all(close(saved[k], v, 1e-9) for k, v in row.items() if k not in ("phase", "world"))
        check(f"{row['phase']} test {row['world']}: paper_metrics_test.csv equals a fresh run", same)
    with open(RES / "official_test_raw_and_calibrated.csv", newline="") as f:
        calibrated = [r for r in csv.DictReader(f) if r["Model"].startswith("Calibrated") and float(r["r"]) == 1.0]
    for r in calibrated:
        side = test_sides[r["phase"], "closed"][{"Seen pairs": "seen", "Unseen pairs": "unseen"}[r["Test split"]]]
        check(f"{r['phase']} test '{r['Model']}' {r['Test split']} lies on the closed curve",
              close(pm.accuracy_at(side, float(r["gamma"])), r["Pair Top-1"]), r["Pair Top-1"])
    for phase in pm.PHASES:
        for world in ("closed", "open"):
            gammas, c_s, c_u, _ = pm.curve(*(test_sides[phase, world][s] for s in ("seen", "unseen")))
            picks = np.unique(np.linspace(0, len(gammas) - 1, 7).astype(int))
            ok = all(np.allclose(brute(models, test_labels, test_train, phase, world, gammas[i]), (c_s[i], c_u[i]))
                     for i in picks)
            check(f"{phase} test {world}: curve equals brute-force argmax at 7 gammas", ok)
else:
    print("SKIP test-output checks: paper_metrics_test.csv not found")

print(json.dumps({"failed": failures}))
sys.exit(1 if failures else 0)
