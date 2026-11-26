"""Ensembles of single-task models and the official-test tables.

An ensemble head is the mean of the log_softmax of its models, for example
crop head    = mean of log_softmax(crop_b1) and log_softmax(crop_clip)               (weights 0.5, 0.5)
disease head = mean of log_softmax(disease_clip) and log_softmax(disease_small_reg)  (weights 0.5, 0.5)

The weights and the model names are fixed before any test number exists. Everything here is a pure function of arrays and
labelled frames, so the notebook cells only call it. It reuses cdccp.calibration (ensemble, calibration, topline rows) and
cdccp.single_task (composition metrics). It also holds the checks for reusing the saved outputs of an earlier run
(lock, pinned hashes, row order), which a fresh run does not call.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as pg
from . import single_task as st
from .scoring import head_score

SPLITS = ("seen", "unseen")
P5_ARMS = {"crop": "crop_b1", "disease": "disease_clip"}
NEW_ARMS = {"crop": "crop_clip", "disease": "disease_small_reg"}
WEIGHTS = (0.5, 0.5)
ALL_PAIRS = "all 9,625 pairs (Baseline 1 rule)"
LIST_333 = "333-pair list"
TASKS = ("crop", "disease", "composition")
TABLE_COLUMNS = ["pair list", "Task", "Top-K", "Seen (%)", "Unseen (%)", "HM (%)"]
HEALTHY = "healthy"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_p5_lock(lock):
    """The P5 lock must describe the pair P6 reuses: crop_b1 and disease_clip, flip on, seed 42."""
    info = lock["info"]
    expect = {"crop_winner": P5_ARMS["crop"], "winner": P5_ARMS["disease"], "flip": True, "seed": 42}
    wrong = {key: info.get(key) for key, value in expect.items() if info.get(key) != value}
    if wrong:
        raise ValueError(f"the P5 lock does not describe crop_b1 + disease_clip with flip and seed 42: {wrong}")
    return info


def check_pinned_hashes(actual, expected):
    """Raise unless every file has exactly the sha256 pinned for it (both maps: name -> full hex digest)."""
    if set(actual) != set(expected):
        raise ValueError(f"pinned files are {sorted(expected)}, hashed files are {sorted(actual)}")
    wrong = sorted(name for name in expected if actual[name] != expected[name])
    if wrong:
        raise ValueError(f"these files differ from the pinned sha256: {wrong}; the P5 copy on Drive is not the P5 run's file")


def check_test_csvs_predate_marker(csv_paths, marker_path):
    """Raise unless every test CSV was last modified strictly before the P5 official-test marker was written.

    The P5 run wrote its marker just before it read the test CSVs, so a CSV older than the marker is evidence, not proof,
    that it is the file P5 read: a copy can keep an old file time. check_rows_match_saved is the check that binds the row order.
    Only file times are read, never the CSV content.
    """
    marker_time = Path(marker_path).stat().st_mtime
    late = [str(path) for path in csv_paths if not Path(path).stat().st_mtime < marker_time]
    if late:
        raise ValueError(f"these test CSVs are not older than the P5 marker {marker_path}: {late}. They may differ from the files P5 "
                         "read, so the saved P5 test outputs may not line up with them: do not run the test")


ROW_MAX_ABS_DIFF = 1.0


def check_rows_match_saved(recomputed, saved, tolerance=ROW_MAX_ABS_DIFF):
    """Bind the row order of the test CSVs to the saved P5 outputs.

    recomputed and saved are {split: array (rows x classes)}: log-probabilities of one model predicted now, in the current CSV
    order, and the log-probabilities P5 saved for the same model. A row where any class differs by more than tolerance
    (default 1.0) raises ValueError naming the split, the row, and the largest difference. A different image gives a far larger
    difference; half-precision rerun noise is far below the tolerance.
    Returns {split: {"rows", "max", "median", "p99", "worst_row"}} where max, median and p99 are over the per-row largest
    absolute difference.
    """
    if set(recomputed) != set(saved):
        raise ValueError(f"recomputed splits {sorted(recomputed)} differ from saved splits {sorted(saved)}")
    stats = {}
    for split in sorted(saved):
        new, old = np.asarray(recomputed[split], dtype=np.float64), np.asarray(saved[split], dtype=np.float64)
        if new.shape != old.shape or new.ndim != 2:
            raise ValueError(f"{split}: recomputed shape {new.shape} differs from the saved P5 shape {old.shape}")
        gap = np.abs(new - old)
        if not np.isfinite(gap).all():
            raise ValueError(f"{split}: the recomputed or the saved log-probabilities are not all finite")
        per_row = gap.max(axis=1)
        worst = int(per_row.argmax())
        stats[split] = {"rows": int(len(per_row)), "max": float(per_row[worst]), "median": float(np.median(per_row)),
                        "p99": float(np.percentile(per_row, 99)), "worst_row": worst}
        if per_row[worst] > tolerance:
            late = int((per_row > tolerance).sum())
            raise ValueError(f"{split}: row {worst} of the current test CSV differs by {per_row[worst]:.3f} (limit {tolerance}) from the "
                             f"saved P5 log-probabilities, and {late} of {len(per_row)} rows are over the limit: the CSV rows are not "
                             "in the order P5 used (or are other images). Do not run the test")
    return stats


def check_results_layout(csv_path):
    """Raise unless a saved table has the P5 layout: the six columns, 12 rows in the order of official_table, finite numbers.

    Returns the saved table, so a caller can keep it in memory.
    """
    saved = pd.read_csv(csv_path)
    keys = [(name, task.title(), k) for name in (ALL_PAIRS, LIST_333) for task in TASKS for k in (1, 5)]
    if list(saved.columns) != TABLE_COLUMNS:
        raise ValueError(f"{csv_path} has columns {list(saved.columns)}, expected {TABLE_COLUMNS}")
    got = list(zip(saved["pair list"], saved["Task"], saved["Top-K"]))
    if got != keys:
        raise ValueError(f"{csv_path} rows are {got}, expected {keys}")
    if not np.isfinite(saved[["Seen (%)", "Unseen (%)", "HM (%)"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{csv_path} holds numbers that are not finite")
    return saved


def read_logprobs(path, arms, rows):
    """{(arm, split): float32 array} from an npz whose keys are exactly <arm>_<split> for the arms given.

    rows maps split to the number of rows. Each array must have that many rows, one column per class of the arm's task,
    only finite values, and rows that are log-probabilities (they sum to 1 as probabilities).
    """
    keys = {f"{arm}_{split}": (arm, split) for arm in arms for split in SPLITS}
    out = {}
    with np.load(path) as z:
        if set(z.files) != set(keys):
            raise ValueError(f"{path} holds keys {sorted(z.files)}, expected {sorted(keys)}")
        for key, (arm, split) in keys.items():
            array = np.asarray(z[key])
            classes = st.CLASS_COUNTS[0 if st.ARMS[arm].task == "crop" else 1]
            if array.shape != (rows[split], classes):
                raise ValueError(f"{path}: {key} has shape {array.shape}, expected {(rows[split], classes)}")
            if not np.isfinite(array).all():
                raise ValueError(f"{path}: {key} is not all finite")
            mass = np.exp(array.astype(np.float64)).sum(axis=1)
            if not np.allclose(mass, 1.0, atol=1e-3):
                raise ValueError(f"{path}: rows of {key} are not log-probabilities (probabilities sum to {mass.min():.4f}..{mass.max():.4f})")
            out[arm, split] = array
    return out


def member_heads(source, arms):
    """{split: (crop, disease)} log-probabilities of one crop arm and one disease arm, as they were saved."""
    return {split: (source[arms["crop"], split], source[arms["disease"], split]) for split in SPLITS}


def p6_parts(p5_logp, new_logp, dfs):
    """(seen, unseen) parts of the P6 ensemble.

    p5_logp and new_logp hold {(arm, split): log-probabilities}; dfs = (seen frame, unseen frame), in the row order of the arrays.
    pg.ensemble refuses members that disagree on filenames, labels or composition ids.
    """
    members = []
    for source, arms in ((p5_logp, P5_ARMS), (new_logp, NEW_ARMS)):
        heads = member_heads(source, arms)
        members.append(tuple(pg.parts_from_logprobs(*heads[split], df) for split, df in zip(SPLITS, dfs)))
    return pg.ensemble(members, weights=WEIGHTS)


def parts_heads(parts):
    """{split: (crop, disease)} from a (seen, unseen) pair of parts."""
    return {split: (part["crop_logits"], part["disease_logits"]) for split, part in zip(SPLITS, parts)}


def pair_mask(catalogue_path, crop_ids, disease_ids, expect=333):
    """Boolean mask over the crop x disease grid (crop-major) that marks the pairs of the catalogue."""
    pairs = json.loads(Path(catalogue_path).read_text())["pairs"]
    crop_index, disease_index = _positions(crop_ids), _positions(disease_ids)
    mask = np.zeros(len(crop_ids) * len(disease_ids), dtype=bool)
    for pair in pairs:
        mask[crop_index[pair["crop_id"]] * len(disease_ids) + disease_index[pair["disease_id"]]] = True
    if expect is not None and mask.sum() != expect:
        raise ValueError(f"the catalogue marks {int(mask.sum())} pairs, expected {expect}")
    return mask


def _positions(ids):
    return {int(i): n for n, i in enumerate(ids)}


def check_test_frames(test, train_all, sizes, disjoint=True):
    """The two test frames have the expected sizes and share no image with each other or (when disjoint) with training."""
    got = (len(test["seen"]), len(test["unseen"]))
    if got != tuple(sizes):
        raise ValueError(f"test sizes (seen, unseen) are {got}, expected {tuple(sizes)}")
    seen, unseen = set(test["seen"].filename), set(test["unseen"].filename)
    if seen & unseen:
        raise ValueError(f"{len(seen & unseen)} images are in both the seen and the unseen test")
    if disjoint and (set(train_all.filename) & (seen | unseen)):
        raise ValueError("test images overlap training images")


def official_table(heads, dfs, crop_ids, disease_ids, allowed_333):
    """(metrics, table): one row per pair list, task and Top-K, the layout of official_test_results.csv.

    heads is {split: (crop, disease)} log-probabilities with column j = class id j of crop_ids or disease_ids; dfs is the
    {split: labelled frame} in the same row order. Rows 'all 9,625 pairs' are the Baseline 1 rule.
    """
    crop_index, disease_index = _positions(crop_ids), _positions(disease_ids)
    metrics, rows = {}, []
    for name, allowed in ((ALL_PAIRS, None), (LIST_333, allowed_333)):
        for split in SPLITS:
            df = dfs[split]
            crop, disease = heads[split]
            true_crop = np.array([crop_index[c] for c in df.crop_id])
            true_disease = np.array([disease_index[d] for d in df.disease_id])
            metrics[name, split] = st.composition_metrics(crop, disease, true_crop, true_disease, allowed)
        for task in TASKS:
            for k in (1, 5):
                s, u = (100 * metrics[name, split][f"{task}_top{k}"] for split in SPLITS)
                rows.append({"pair list": name, "Task": task.title(), "Top-K": k, "Seen (%)": s, "Unseen (%)": u,
                             "HM (%)": 100 * st.harmonic(s / 100, u / 100)})
    return metrics, pd.DataFrame(rows, columns=TABLE_COLUMNS)


def assert_matches_table(table, saved, atol=1e-6):
    """Raise unless table equals the saved table (same rows, same order, numbers within atol percentage points).

    saved is the saved table as a frame, or the path of its CSV.
    """
    csv_path = "the saved P5 table" if isinstance(saved, pd.DataFrame) else saved
    saved = saved if isinstance(saved, pd.DataFrame) else pd.read_csv(saved)
    if list(saved.columns) != list(table.columns) or len(saved) != len(table):
        raise ValueError(f"{csv_path} has columns {list(saved.columns)} and {len(saved)} rows; the recomputed table has "
                         f"{list(table.columns)} and {len(table)} rows")
    for column in ("pair list", "Task", "Top-K"):
        if not (saved[column].to_numpy() == table[column].to_numpy()).all():
            raise ValueError(f"{csv_path}: column {column} differs from the recomputed table")
    wrong = []
    for column in ("Seen (%)", "Unseen (%)", "HM (%)"):
        gap = np.abs(saved[column].to_numpy(dtype=float) - table[column].to_numpy(dtype=float))
        wrong += [(int(i), column, float(saved[column][i]), float(table[column][i])) for i in np.flatnonzero(~(gap <= atol))]
    if wrong:
        raise ValueError(f"the table recomputed from the saved log-probabilities differs from {csv_path} "
                         f"(row, column, saved, recomputed): {wrong[:6]}")


def compare_table(p5_table, p6_table, b1_test, pair_list=ALL_PAIRS):
    """Baseline 1, P5 and P6 under the same pair rule. b1_test[task] = (seen Top-1, unseen Top-1, seen Top-5, unseen Top-5) in %."""
    rows = []
    for task in TASKS:
        b1 = b1_test[task.title()]
        for k, (b1_seen, b1_unseen) in ((1, (b1[0], b1[1])), (5, (b1[2], b1[3]))):
            row = {"Task": task.title(), "Top-K": k, "B1 seen (%)": b1_seen, "P5 seen (%)": None, "P6 seen (%)": None,
                   "B1 unseen (%)": b1_unseen, "P5 unseen (%)": None, "P6 unseen (%)": None,
                   "B1 HM (%)": 100 * st.harmonic(b1_seen / 100, b1_unseen / 100), "P5 HM (%)": None, "P6 HM (%)": None}
            for label, table in (("P5", p5_table), ("P6", p6_table)):
                hit = table[(table["pair list"] == pair_list) & (table.Task == task.title()) & (table["Top-K"] == k)]
                if len(hit) != 1:
                    raise ValueError(f"{label} table has {len(hit)} rows for {pair_list}, {task}, Top-{k}")
                row[f"{label} seen (%)"], row[f"{label} unseen (%)"], row[f"{label} HM (%)"] = (
                    float(hit["Seen (%)"].iloc[0]), float(hit["Unseen (%)"].iloc[0]), float(hit["HM (%)"].iloc[0]))
            rows.append(row)
    return pd.DataFrame(rows)


def unseen_pair_table(crop, disease, df, crop_ids, disease_ids):
    """Per unseen test pair, over all 9,625 pairs: how often the top pair has the right crop, right disease, right pair (%)."""
    df = df.reset_index(drop=True)
    scores = (np.asarray(crop, dtype=np.float64)[:, :, None] + np.asarray(disease, dtype=np.float64)[:, None, :]).reshape(len(df), -1)
    top = scores.argmax(1)
    pred_crop = np.array(crop_ids)[top // len(disease_ids)]
    pred_disease = np.array(disease_ids)[top % len(disease_ids)]
    right_crop, right_disease = pred_crop == df.crop_id.to_numpy(), pred_disease == df.disease_id.to_numpy()
    per_pair = (df.assign(right_crop=right_crop, right_disease=right_disease, right_pair=right_crop & right_disease)
                  .groupby("composition_name")
                  .agg(images=("filename", "size"), right_crop=("right_crop", "mean"), right_disease=("right_disease", "mean"),
                       right_pair=("right_pair", "mean"))
                  .sort_values("images", ascending=False))
    per_pair[["right_crop", "right_disease", "right_pair"]] *= 100
    return per_pair


def seen_confusion(crop, disease, df, catalogue_pairs, crop_ids, disease_ids, allowed_333):
    """(conf, rows_pairs, pairs): true seen pair (rows) against the predicted pair of the 333 list (columns), each row divided by its image count.

    pairs is the 333 list with the seen pairs first, then the unseen ones; rows_pairs are the pairs the seen test images have.
    """
    df = df.reset_index(drop=True)
    pairs = [(p["crop_id"], p["disease_id"]) for p in sorted(catalogue_pairs, key=lambda p: (not p["is_seen"], p["crop_id"], p["disease_id"]))]
    col = {pair: i for i, pair in enumerate(pairs)}
    rows_pairs = sorted(set(zip(df.crop_id, df.disease_id)), key=lambda p: col[p])
    row = {pair: i for i, pair in enumerate(rows_pairs)}
    scores = (np.asarray(crop, dtype=np.float64)[:, :, None] + np.asarray(disease, dtype=np.float64)[:, None, :]).reshape(len(df), -1)
    scores[:, ~np.asarray(allowed_333, dtype=bool)] = -np.inf
    pred = scores.argmax(1)
    conf = np.zeros((len(rows_pairs), len(pairs)))
    for true_pair, p in zip(zip(df.crop_id, df.disease_id), pred):
        conf[row[true_pair], col[crop_ids[p // len(disease_ids)], disease_ids[p % len(disease_ids)]]] += 1
    return conf / conf.sum(1, keepdims=True), rows_pairs, pairs


def dev_calibration(seen, unseen, cat):
    """Rule 1 (keep seen, r 1) and rule 2 (best dev pair HM over r and gamma) on dev parts. All four numbers must be finite."""
    cal = {**pg.keep_gamma(seen, unseen, cat), **pg.best_hm_r_gamma(seen, unseen, cat)}
    for key in ("gamma_keep", "gamma_hm", "r_hm"):
        if not np.isfinite(cal[key]):
            raise ValueError(f"calibration value {key} is {cal[key]}: do not run the test")
    return cal


def healthy_unseen_pairs(cat):
    """Catalogue indices of the unseen pairs whose name contains 'healthy'."""
    return np.array([i for i, (name, seen) in enumerate(zip(cat.names, cat.is_seen)) if HEALTHY in name.lower() and not seen], dtype=np.int64)


def unseen_pair_accuracy(unseen, cat, gamma, r=1.0):
    """Per unseen pair on dev: images, hits and Top-1 (%) of the pair list at gamma and r (ties as in topline_rows)."""
    ordered = pg.report_order(cat)
    top = np.argmax(head_score(unseen, ordered, r=r, g=gamma), axis=1)
    truth = ordered.index_of(unseen["comp_id"])
    healthy = {ordered.names[i] for i in healthy_unseen_pairs(ordered)}
    rows = []
    for index in np.unique(truth):
        mine = truth == index
        hits = int((top[mine] == index).sum())
        rows.append({"pair": ordered.names[index], "healthy": ordered.names[index] in healthy, "images": int(mine.sum()),
                     "hits": hits, "accuracy (%)": 100.0 * hits / int(mine.sum())})
    return pd.DataFrame(rows)


def dev_unseen_numbers(unseen, cat, cal, model):
    """Dev unseen pair Top-1 for the raw rule and both calibration rules: all images, without the healthy pairs, and the mean over pairs.

    Information only. cal comes from dev_calibration. 'pair-macro' is the mean of the per-pair accuracies.
    """
    rules = (("raw (gamma 0, r 1)", 0.0, 1.0), ("rule 1 keep seen", cal["gamma_keep"], 1.0), ("rule 2 best dev HM", cal["gamma_hm"], cal["r_hm"]))
    rows = []
    for label, gamma, r in rules:
        per = unseen_pair_accuracy(unseen, cat, gamma, r)
        rest = per[~per.healthy]
        rows.append({"model": model, "rule": label, "gamma": gamma, "r": r, "unseen pairs": len(per),
                     "unseen pair Top-1, all images (%)": 100.0 * per.hits.sum() / per.images.sum(),
                     "healthy pairs left out": int(per.healthy.sum()),
                     "unseen pair Top-1, without healthy pairs (%)": 100.0 * rest.hits.sum() / rest.images.sum(),
                     "pair-macro mean, all pairs (%)": float(per["accuracy (%)"].mean()),
                     "pair-macro mean, without healthy pairs (%)": float(rest["accuracy (%)"].mean())})
    return pd.DataFrame(rows)
