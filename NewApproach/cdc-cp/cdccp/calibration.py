"""Score single-task crop and disease models and their ensembles, raw and with the gamma calibration rules.

Everything works on "parts": a dict with crop_logits [n, 55], disease_logits [n, 175], crop_id, disease_id, comp_id
(int64) and fn (str). The heads may be logits or log-probabilities, because log_softmax is idempotent. These are the
dicts cdccp.evaluate, cdccp.scoring and final.Scorer("head") already take.

keep_gamma ("keep seen"): the largest dev unseen accuracy with dev seen accuracy within 3 points of raw.
best_hm_r_gamma ("best dev HM"): the gamma and disease weight r with the best dev HM.
pick_candidate chooses among candidate models by their dev H_keep. The test set takes no part in any choice.
"""
from pathlib import Path

import numpy as np

from .bn import recompute_bn
from .catalogue import Catalogue
from .checkpoint import load_checkpoint
from .data import eval_loader, load_split
from .evaluate import gamma_grid, harmonic, head_score, pair_breakdown, sweep
from .extract import extract
from .final import Scorer, full_catalogue, score_split_pair, seen_pair_set, with_full_ids
from .models import build_model
from .single_task import DEV_SIZES, CLASS_COUNTS, class_ids, load_dev

PART_KEYS = ("crop_logits", "disease_logits", "crop_id", "disease_id", "comp_id", "fn")
ID_KEYS = ("fn", "crop_id", "disease_id", "comp_id")
BLOCKS = ("locked_raw", "locked_calibrated", "full_raw")
CANDIDATE_ORDER = ("A", "B", "C")
R_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
TRIVIAL_GAMMA = 1e6  # larger than any log-probability gap, so no seen pair can be the top pair


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _part(crop, disease, df_or_arrays):
    return {
        "crop_logits": _log_softmax(crop),
        "disease_logits": _log_softmax(disease),
        "crop_id": np.asarray(df_or_arrays["crop_id"], dtype=np.int64),
        "disease_id": np.asarray(df_or_arrays["disease_id"], dtype=np.int64),
        "comp_id": np.asarray(df_or_arrays["comp_id"], dtype=np.int64),
        "fn": np.asarray(df_or_arrays["fn"]).astype(str),
    }


def _load_arm(path, n_seen, n_unseen, n_classes, arm):
    with np.load(path) as z:
        seen, unseen = np.asarray(z["seen"]), np.asarray(z["unseen"])
    if seen.shape != (n_seen, n_classes) or unseen.shape != (n_unseen, n_classes):
        raise ValueError(f"{arm}: expected seen {(n_seen, n_classes)} and unseen {(n_unseen, n_classes)}, "
                         f"got {seen.shape} and {unseen.shape}")
    if not (np.isfinite(seen).all() and np.isfinite(unseen).all()):
        raise ValueError(f"{arm}: dev outputs in {path} are not all finite")
    return seen, unseen


def dev_parts_p4(stage1_dir, crop_arm, disease_arm, splits_dir, expect_sizes=DEV_SIZES, expect_classes=CLASS_COUNTS):
    """(seen, unseen) parts from <stage1_dir>/<arm>/best_validation.npz of the crop arm and the disease arm."""
    train_fit, val_seen, val_unseen = load_dev(splits_dir, expect_sizes)
    crops, diseases = class_ids(train_fit, val_seen, val_unseen, expect=expect_classes)
    if crops != list(range(len(crops))) or diseases != list(range(len(diseases))):
        raise ValueError("class ids are not 0..n-1, so a column position is not a class id")
    n_crop, n_disease = len(crops), len(diseases)
    crop_seen, crop_unseen = _load_arm(Path(stage1_dir) / crop_arm / "best_validation.npz", len(val_seen), len(val_unseen), n_crop, crop_arm)
    dis_seen, dis_unseen = _load_arm(Path(stage1_dir) / disease_arm / "best_validation.npz", len(val_seen), len(val_unseen), n_disease, disease_arm)
    out = []
    for crop, disease, df in ((crop_seen, dis_seen, val_seen), (crop_unseen, dis_unseen, val_unseen)):
        out.append(_part(crop, disease, {"crop_id": df.crop_id, "disease_id": df.disease_id,
                                         "comp_id": df.composition_id, "fn": df.filename}))
    return tuple(out)


def dev_parts_p2(val_npz_path):
    """(seen, unseen) parts from a cdccp val_XX.npz. Embedding keys are ignored."""
    with np.load(val_npz_path) as z:
        return tuple(
            _part(z[f"{split}_crop_logits"], z[f"{split}_disease_logits"],
                  {"crop_id": z[f"{split}_crop_id"], "disease_id": z[f"{split}_disease_id"],
                   "comp_id": z[f"{split}_comp_id"], "fn": z[f"{split}_fn"]})
            for split in ("seen", "unseen"))


def _ensemble_one(parts_list, weights, normalise=True):
    first = parts_list[0]
    for member in parts_list[1:]:
        for key in ID_KEYS:
            if not np.array_equal(first[key], member[key]):
                raise ValueError(f"ensemble members disagree on {key}")
    out = {key: first[key] for key in ID_KEYS}
    for head in ("crop_logits", "disease_logits"):
        out[head] = sum(w * (_log_softmax(m[head]) if normalise else m[head]) for w, m in zip(weights, parts_list))
    return out


def _check_weights(weights, n):
    if weights is None:
        return [1.0 / n] * n
    weights = [float(w) for w in weights]
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} members")
    if any(w < 0 for w in weights):
        raise ValueError("weights must not be negative")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights sum to {sum(weights)}, not 1")
    return weights


def ensemble(parts_list, weights=None, normalise=True):
    """Weighted mean of log_softmax(head) per head. The members must hold the same rows in the same order.

    parts_list is a list of parts, or a list of (seen, unseen) pairs, in which case a (seen, unseen) pair comes back.
    normalise=False takes members whose heads are already averaged log-probabilities and does not renormalise them.
    """
    parts_list = list(parts_list)
    if not parts_list:
        raise ValueError("no members to ensemble")
    weights = _check_weights(weights, len(parts_list))
    if isinstance(parts_list[0], tuple):
        return tuple(_ensemble_one([m[i] for m in parts_list], weights, normalise) for i in range(2))
    return _ensemble_one(parts_list, weights, normalise)


def family_ensemble(groups):
    """Average inside each group with equal weights, then average the groups with equal weights.

    Each group is a list of parts or of (seen, unseen) pairs, as in ensemble().
    """
    groups = [list(g) for g in groups]
    if not groups or any(not g for g in groups):
        raise ValueError("every group needs at least one member")
    return ensemble([ensemble(g) for g in groups], normalise=False)


def report_order(cat):
    """The same catalogue with its pairs in (crop id, disease id) order.

    np.argmax takes the first of equal scores, so on this order a tie goes to the lowest (crop id, disease id) pair:
    the rule of topline_rows and composition_metrics. Calibration uses it so that it fits the rule the table reports."""
    o = np.lexsort((cat.disease_id, cat.crop_id))
    return Catalogue(comp_id=cat.comp_id[o], crop_id=cat.crop_id[o], disease_id=cat.disease_id[o],
                     is_seen=cat.is_seen[o], names=[cat.names[i] for i in o])


def keep_gamma(seen, unseen, cat, r=1.0, tol=3.0):
    """pair_breakdown plus H_keep, the harmonic mean of the keep-seen seen and unseen accuracies.

    Pairs are scored in report_order, so a tied score resolves as in topline_rows."""
    out = pair_breakdown(seen, unseen, report_order(cat), tol=tol, r=r)
    out["H_keep"] = harmonic(out["A_S_keep"], out["A_U_keep"])
    return out


def best_hm_gamma(seen, unseen, cat, r=1.0):
    """The gamma with the highest pair HM (seen and unseen pair Top-1) on these outputs; a tie goes to the smaller gamma.

    The usual CZSL rule. Fitted on dev only, it is the second calibration rule next to keep_gamma's keep-seen rule.
    Pairs are scored in report_order, so a tied score resolves as in topline_rows."""
    cat = report_order(cat)
    ss, su = head_score(seen, cat, r=r), head_score(unseen, cat, r=r)
    rows = sweep(ss, seen, su, unseen, cat, gamma_grid(ss, su, cat))
    best = rows[np.lexsort((rows[:, 0], -rows[:, 3]))[0]]
    return {"gamma_hm": float(best[0]), "A_S_hm": float(best[1]), "A_U_hm": float(best[2]), "H_hm": float(best[3])}


def best_hm_r_gamma(seen, unseen, cat, r_grid=R_GRID):
    """The (r, gamma) with the highest pair HM on these outputs, fitted on dev only.

    For each r the sweep is best_hm_gamma's (gamma grid from that r's scores, pairs in report_order). A tie in HM goes
    to the r with the smallest |log r|, then to the smaller r (0.5 before 2.0), then to the smaller gamma."""
    r_grid = [float(r) for r in r_grid]
    if not r_grid:
        raise ValueError("r_grid is empty")
    if not all(np.isfinite(r) and r > 0 for r in r_grid):
        raise ValueError(f"r_grid must hold finite positive values, got {r_grid}")
    for split, part in (("seen", seen), ("unseen", unseen)):
        if not (np.isfinite(part["crop_logits"]).all() and np.isfinite(part["disease_logits"]).all()):
            raise ValueError(f"{split} scores are not all finite")
    cat = report_order(cat)
    best_key, best = None, None
    for r in r_grid:
        ss, su = head_score(seen, cat, r=r), head_score(unseen, cat, r=r)
        for gamma, a_s, a_u, h in sweep(ss, seen, su, unseen, cat, gamma_grid(ss, su, cat)):
            key = (-float(h), abs(float(np.log(r))), r, float(gamma))
            if best_key is None or key < best_key:
                best_key, best = key, {"r_hm": r, "gamma_hm": float(gamma), "A_S_hm": float(a_s),
                                       "A_U_hm": float(a_u), "H_hm": float(h)}
    return best


def calibrated_scores(parts, cat, gamma, r=1.0):
    """(seen_scores, unseen_scores) at gamma, for a confusion matrix. parts is a (seen, unseen) pair."""
    seen, unseen = parts
    _, seen_scores, unseen_scores = score_split_pair(Scorer("head", r=r), seen, unseen, cat, gamma)
    return seen_scores, unseen_scores


def test_metrics(seen, unseen, cat, gamma, r=1.0):
    """The three blocks final.evaluate_model reports for a head scorer, plus the calibrated scores."""
    n_crop, n_disease = seen["crop_logits"].shape[1], seen["disease_logits"].shape[1]
    scorer = Scorer("head", r=r)
    full = full_catalogue(n_crop, n_disease, seen_pair_set(cat))
    raw, _, _ = score_split_pair(scorer, seen, unseen, cat, 0.0)
    cal, cal_s, cal_u = score_split_pair(scorer, seen, unseen, cat, gamma)
    full_m, _, _ = score_split_pair(scorer, with_full_ids(seen, n_disease), with_full_ids(unseen, n_disease), full, 0.0)
    return {
        "locked_raw": {"gamma": 0.0, **raw},
        "locked_calibrated": {"gamma": gamma, **cal},
        "full_raw": {"gamma": 0.0, **full_m},
        "scores": {"seen": cal_s, "unseen": cal_u},
    }


def parts_from_logprobs(crop, disease, df):
    """Parts for one split from crop [n, 55] and disease [n, 175] scores and a labelled DataFrame
    (columns crop_id, disease_id, composition_id, filename). Column j of each head must be class id j."""
    return _part(crop, disease, {"crop_id": df["crop_id"], "disease_id": df["disease_id"],
                                 "comp_id": df["composition_id"], "fn": df["filename"]})


def topline_rows(seen, unseen, cat, gamma, r=1.0, gamma_hm=None, r_hm=1.0):
    """Crop-only, disease-only and pair Top-1 (%) per split, three ways, from the same two heads:

    "Raw, heads only": argmax of each head on its own (the Baseline 1 rule, all crop x disease pairs).
    "Raw, pair list": the top pair of the pair list at gamma 0.
    "Calibrated, pair list": the top pair of the pair list after gamma is taken off every seen pair.
    "Calibrated (best dev HM), pair list": the same with gamma_hm and r_hm (best_hm_r_gamma on dev), when gamma_hm is
    given. Every other row scores with r. The "r" column holds the r each row used.
    "Reference: unseen pairs only": a trivial rule that never names a seen pair (gamma = TRIVIAL_GAMMA). It shows how
    much of the calibrated unseen gain comes only from naming the unseen pair of the predicted crop.
    For the pair-list rows, crop-only and disease-only are the crop and the disease of that top pair. A tie goes to
    the lowest (crop id, disease id), the same as single_task.composition_metrics.
    """
    for name, value in (("gamma", gamma), ("gamma_hm", 0.0 if gamma_hm is None else gamma_hm)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if not (np.isfinite(r_hm) and r_hm > 0):
        raise ValueError(f"r_hm must be finite and positive, got {r_hm}")
    for split, part in (("seen", seen), ("unseen", unseen)):
        if not (np.isfinite(part["crop_logits"]).all() and np.isfinite(part["disease_logits"]).all()):
            raise ValueError(f"{split} scores are not all finite")
    rows = []
    for split, part in (("Seen pairs", seen), ("Unseen pairs", unseen)):
        crop_ok = np.argmax(part["crop_logits"], axis=1) == part["crop_id"]
        disease_ok = np.argmax(part["disease_logits"], axis=1) == part["disease_id"]
        rows.append((split, "Raw, heads only", 0.0, r, crop_ok, disease_ok))
    order = np.lexsort((cat.disease_id, cat.crop_id))
    rules = [("Raw, pair list", 0.0, r), ("Calibrated, pair list", float(gamma), r)]
    if gamma_hm is not None:
        rules.append(("Calibrated (best dev HM), pair list", float(gamma_hm), float(r_hm)))
    rules.append(("Reference: unseen pairs only", TRIVIAL_GAMMA, r))
    for label, g, row_r in rules:
        seen_scores, unseen_scores = calibrated_scores((seen, unseen), cat, g, row_r)
        for split, part, scores in (("Seen pairs", seen, seen_scores), ("Unseen pairs", unseen, unseen_scores)):
            top = order[np.argmax(scores[:, order], axis=1)]
            rows.append((split, label, g, row_r, cat.crop_id[top] == part["crop_id"], cat.disease_id[top] == part["disease_id"]))
    return [{"Test split": split, "Model": label, "gamma": g, "r": float(row_r), "Crop-only Top-1": 100 * float(c.mean()),
             "Disease-only Top-1": 100 * float(d.mean()), "Pair Top-1": 100 * float((c & d).mean())}
            for split, label, g, row_r, c, d in rows]


def metrics_rows(name, result):
    """Flat rows model, catalogue, gamma, split, metric, value for one test_metrics result."""
    rows = []
    for catalogue in BLOCKS:
        block = result[catalogue]
        for split in ("seen", "unseen", "combined"):
            for metric, value in block[split].items():
                rows.append({"model": name, "catalogue": catalogue, "gamma": block["gamma"], "split": split,
                             "metric": metric, "value": float(value)})
    return rows


def p2_test_parts(checkpoint_path, seen_df, unseen_df, cfg, device, use_cache=False):
    """(seen, unseen) parts of a cdccp multi-task checkpoint, as final.evaluate_model scores it.

    The caller supplies the rows, so this never reads an official test CSV.
    """
    meta = load_checkpoint(checkpoint_path)
    run_cfg = {**meta["config"], "paths": cfg["paths"], "loader": cfg["loader"]}
    model = build_model(run_cfg, pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    train_df = load_split(run_cfg, "train")
    bn_df = train_df.sample(min(run_cfg["bn"]["subset"], len(train_df)), random_state=run_cfg["seed"]).sort_values("filename")
    recompute_bn(model, eval_loader(bn_df, run_cfg, run_cfg["bn"]["batch"], use_cache=use_cache), device)
    out = []
    for df in (seen_df, unseen_df):
        raw = extract(model, df, run_cfg, device, use_cache)
        out.append(_part(raw["crop_logits"], raw["disease_logits"], raw))
    return tuple(out)


def pick_candidate(board, order=CANDIDATE_ORDER):
    """Name with the highest dev H_keep. A tie goes to the name that comes first in order."""
    missing = [name for name in order if name not in board]
    if missing:
        raise ValueError(f"board lacks candidates {missing}")
    best = None
    for name in order:
        if best is None or board[name] > board[best]:
            best = name
    return best
