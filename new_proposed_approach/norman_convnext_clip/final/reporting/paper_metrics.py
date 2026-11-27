"""Paper-style CZSL numbers for P1, P2 and P3 from saved log-probabilities: best seen, best unseen, best HM and AUC.

This is the reporting protocol of the CZSL papers (TMN, CGE, CompCos): sweep a bias gamma that is taken off every seen
pair, from "always a seen pair" to "always an unseen pair", and report the whole curve. It reads saved scores only; it does
not train, does not change gamma, and does not choose a model.

Two candidate sets:
  closed world: the training pairs plus the unseen pairs of the evaluated unseen split (333 on the official test).
  open world:   all 55 x 175 = 9,625 crop-disease pairs (no feasibility filter).

Usage (one line):
  python paper_metrics.py --scores official_test_logprobs_models.npz --train DPD_Small_train.csv
      --seen DPD_Small_test_seen.csv --unseen DPD_Small_test_unseen.csv --out paper_metrics_test.csv
Dev check (labels from the development split only):
  python paper_metrics.py --scores dev_flip_logprobs.npz --train train_fit.csv --seen val_seen.csv
      --unseen val_unseen.csv --out paper_metrics_dev.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

N_CROPS, N_DISEASES = 55, 175
COLUMNS = ("filename", "crop_id", "disease_id", "composition_id", "crop_name", "disease_name", "composition_name")
PHASES = {"P1": {"crop": ("crop_b1",), "disease": ("disease_b1",)},
          "P2": {"crop": ("crop_b1",), "disease": ("disease_clip",)},
          "P3": {"crop": ("crop_b1", "crop_clip"), "disease": ("disease_clip", "disease_small_reg")}}
CHUNK = 512


def read_labels(path):
    """(crop_id, disease_id) arrays. Header row if the first cell is 'filename', else the official headerless 7 columns."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0][0].strip().lower() == "filename":
        header, rows = [c.strip() for c in rows[0]], rows[1:]
    else:
        header = list(COLUMNS)
    rows = [r for r in rows if r]
    crop, disease = header.index("crop_id"), header.index("disease_id")
    return np.array([int(r[crop]) for r in rows]), np.array([int(r[disease]) for r in rows])


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def phase_heads(npz, phase, split):
    """Per task, the equal-weight mean of the log-softmax of the phase's models (as cdccp.final_model.phase_parts)."""
    tasks = PHASES[phase]
    crop = np.mean([log_softmax(npz[f"{m}_{split}"]) for m in tasks["crop"]], axis=0)
    disease = np.mean([log_softmax(npz[f"{m}_{split}"]) for m in tasks["disease"]], axis=0)
    return log_softmax(crop), log_softmax(disease)


def side_scores(crop, disease, true_c, true_d, seen_mask, allowed_mask, r):
    """Per image: best seen score and whether that seen pick is right; the same for unseen; and which comes first in
    crop-major order (the tie rule of np.argmax over the (crop id, disease id) ordered pair list)."""
    flat_seen = (seen_mask & allowed_mask).ravel()
    flat_unseen = (~seen_mask & allowed_mask).ravel()
    true_flat = true_c * N_DISEASES + true_d
    out = {k: [] for k in ("s_max", "s_ok", "s_idx", "u_max", "u_ok", "u_idx")}
    for start in range(0, len(crop), CHUNK):
        grid = (crop[start:start + CHUNK, :, None] + r * disease[start:start + CHUNK, None, :]).reshape(-1, N_CROPS * N_DISEASES)
        for side, mask in (("s", flat_seen), ("u", flat_unseen)):
            masked = np.where(mask[None, :], grid, -np.inf)
            idx = masked.argmax(axis=1)
            out[f"{side}_max"].append(masked[np.arange(len(idx)), idx])
            out[f"{side}_idx"].append(idx)
            out[f"{side}_ok"].append(idx == true_flat[start:start + CHUNK])
    return {k: np.concatenate(v) for k, v in out.items()}


def accuracy_at(side, gamma):
    """Top-1 (%) at one gamma, with the exact argmax tie rule."""
    seen_score = side["s_max"] - gamma
    pick_seen = (seen_score > side["u_max"]) | ((seen_score == side["u_max"]) & (side["s_idx"] < side["u_idx"]))
    return 100 * np.where(pick_seen, side["s_ok"], side["u_ok"]).mean()


def curve(seen_side, unseen_side):
    """Every distinct operating point of the gamma sweep: gamma at the midpoints between the sorted gaps, plus both ends."""
    gaps = np.unique(np.concatenate([seen_side["s_max"] - seen_side["u_max"], unseen_side["s_max"] - unseen_side["u_max"]]))
    gammas = np.concatenate([[gaps[0] - 1.0], (gaps[:-1] + gaps[1:]) / 2, [gaps[-1] + 1.0]])
    seen_gap = seen_side["s_max"] - seen_side["u_max"]
    unseen_gap = unseen_side["s_max"] - unseen_side["u_max"]
    # No gamma equals a gap, so an image picks its seen side exactly when gap > gamma.
    s_hits = np.sort(seen_gap[seen_side["s_ok"]])
    s_miss_u = np.sort(seen_gap[seen_side["u_ok"]])
    u_hits_u = np.sort(unseen_gap[unseen_side["u_ok"]])
    u_hits_s = np.sort(unseen_gap[unseen_side["s_ok"]])
    above = lambda sorted_gaps: len(sorted_gaps) - np.searchsorted(sorted_gaps, gammas, side="right")
    below = lambda sorted_gaps: np.searchsorted(sorted_gaps, gammas, side="left")
    a_s = 100 * (above(s_hits) + below(s_miss_u)) / len(seen_gap)
    a_u = 100 * (below(u_hits_u) + above(u_hits_s)) / len(unseen_gap)
    hm = np.where(a_s + a_u > 0, 2 * a_s * a_u / np.maximum(a_s + a_u, 1e-12), 0.0)
    return gammas, a_s, a_u, hm


def auc(a_s, a_u):
    """Area under seen against unseen accuracy, in % of the unit square (CZSL papers report AUC x 100)."""
    order = np.lexsort((-a_s, a_u))
    xs, ys = a_u[order] / 100, a_s[order] / 100
    return 100 * float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2))


def masks(train_pairs, unseen_pairs):
    seen = np.zeros((N_CROPS, N_DISEASES), dtype=bool)
    closed = np.zeros_like(seen)
    for c, d in train_pairs:
        seen[c, d] = closed[c, d] = True
    for c, d in unseen_pairs:
        closed[c, d] = True
    return seen, closed


def evaluate(npz, labels, train_pairs, r=1.0):
    unseen_pairs = set(zip(*labels["unseen"]))
    if unseen_pairs & train_pairs:
        raise SystemExit("unseen split shares pairs with the training CSV")
    seen_mask, closed = masks(train_pairs, unseen_pairs)
    results, sides = [], {}
    for phase in PHASES:
        for world, allowed in (("closed", closed), ("open", np.ones_like(closed))):
            per_split = {}
            for split in ("seen", "unseen"):
                crop, disease = phase_heads(npz, phase, split)
                c, d = labels[split]
                per_split[split] = side_scores(crop, disease, c, d, seen_mask, allowed, r)
            gammas, a_s, a_u, hm = curve(per_split["seen"], per_split["unseen"])
            best = int(np.argmax(hm))
            sides[phase, world] = per_split
            results.append({"phase": phase, "world": world, "pairs": int(allowed.sum()), "r": r,
                            "best_seen": a_s.max(), "best_unseen": a_u.max(), "best_hm": hm[best],
                            "seen_at_best_hm": a_s[best], "unseen_at_best_hm": a_u[best], "gamma_at_best_hm": gammas[best],
                            "auc": auc(a_s, a_u), "raw_seen": accuracy_at(per_split["seen"], 0.0),
                            "raw_unseen": accuracy_at(per_split["unseen"], 0.0), "curve_points": len(gammas)})
    return results, sides


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, help="npz with <model>_seen and <model>_unseen log-probabilities")
    p.add_argument("--train", required=True, help="labelled CSV whose pairs are the seen (training) pairs")
    p.add_argument("--seen", required=True)
    p.add_argument("--unseen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--r", type=float, default=1.0, help="disease weight; papers use 1")
    a = p.parse_args()
    npz = np.load(a.scores, allow_pickle=False)
    labels = {"seen": read_labels(a.seen), "unseen": read_labels(a.unseen)}
    for split, (c, _) in labels.items():
        n = npz[f"crop_b1_{split}"].shape[0]
        if n != len(c):
            raise SystemExit(f"{split}: {n} score rows but {len(c)} label rows")
    train_pairs = set(zip(*read_labels(a.train)))
    results, _ = evaluate(npz, labels, train_pairs, a.r)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"seen pairs {len(train_pairs)} | seen rows {len(labels['seen'][0])} | unseen rows {len(labels['unseen'][0])}")
    print(f"{'phase':5} {'world':6} {'pairs':>5} {'raw S':>6} {'raw U':>6} {'best S':>6} {'best U':>6} {'best HM':>7} {'AUC':>6}")
    for r in results:
        print(f"{r['phase']:5} {r['world']:6} {r['pairs']:5d} {r['raw_seen']:6.2f} {r['raw_unseen']:6.2f} {r['best_seen']:6.2f} "
              f"{r['best_unseen']:6.2f} {r['best_hm']:7.2f} {r['auc']:6.2f}")
    print("written", Path(a.out).resolve())


if __name__ == "__main__":
    main()
