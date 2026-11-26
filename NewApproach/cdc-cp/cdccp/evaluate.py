"""Shared metrics. Every model's reported numbers come from this file."""
import numpy as np

from .scoring import head_score, rank

OFFICIAL_ROLES = ("official_seen", "official_unseen")


class Refused(RuntimeError):
    pass


def check_rows(df, role, meta, lock=None):
    """Refuse evaluations the protocol forbids. role is seen, unseen, official_seen or official_unseen."""
    if role in OFFICIAL_ROLES and lock is None:
        raise Refused("official test rows need a valid lock")
    trained = set(df.filename) & set(meta["training_files"])
    if trained:
        raise Refused(f"{len(trained)} rows are training files of this checkpoint")
    if role in ("unseen", "official_unseen"):
        pairs = set(int(c) for c in df.composition_id) & set(meta["training_pairs"])
        if pairs:
            raise Refused(f"unseen-role rows use trained pairs {sorted(pairs)}")


def harmonic(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def topk_hits(scores, y, k):
    return (rank(scores)[:, :k] == y[:, None]).any(axis=1)


def split_metrics(scores, npz, cat, crop_scores=None, disease_scores=None):
    """Top-1 and Top-5 on one split, as percentages. Head scores default to the logits."""
    y = cat.index_of(npz["comp_id"])
    crop = npz["crop_logits"] if crop_scores is None else crop_scores
    disease = npz["disease_logits"] if disease_scores is None else disease_scores
    return {
        "comp_top1": 100 * topk_hits(scores, y, 1).mean(),
        "comp_top5": 100 * topk_hits(scores, y, 5).mean(),
        "crop_top1": 100 * topk_hits(crop, npz["crop_id"], 1).mean(),
        "crop_top5": 100 * topk_hits(crop, npz["crop_id"], 5).mean(),
        "disease_top1": 100 * topk_hits(disease, npz["disease_id"], 1).mean(),
        "disease_top5": 100 * topk_hits(disease, npz["disease_id"], 5).mean(),
    }


def combined_metrics(seen, unseen):
    """Harmonic mean and gap from two split_metrics results, which must share one score and gamma."""
    a_s, a_u = seen["comp_top1"], unseen["comp_top1"]
    return {"A_S": a_s, "A_U": a_u, "H": harmonic(a_s, a_u), "gap": a_s - a_u}


def confusion(scores, npz, cat):
    """[K,K] integer counts, indexed by catalogue position. Rows are true pairs."""
    matrix = np.zeros((len(cat), len(cat)), dtype=np.int64)
    np.add.at(matrix, (cat.index_of(npz["comp_id"]), rank(scores)[:, 0]), 1)
    return matrix


def seen_rows(matrix, cat):
    """Published view: one row per seen pair, all K columns."""
    return matrix[cat.is_seen]


def gamma_grid(scores_seen, scores_unseen, cat, n=201):
    """Gamma 0 plus quantiles of (best seen score - best unseen score) across both splits."""
    all_scores = np.vstack([scores_seen, scores_unseen])
    gaps = all_scores[:, cat.is_seen].max(axis=1) - all_scores[:, ~cat.is_seen].max(axis=1)
    return np.concatenate([[0.0], np.quantile(gaps, np.linspace(0, 1, n))])


def sweep(scores_seen, npz_seen, scores_unseen, npz_unseen, cat, gammas):
    """Seen and unseen Top-1 for each gamma, applied to scores computed at gamma = 0."""
    y_s, y_u = cat.index_of(npz_seen["comp_id"]), cat.index_of(npz_unseen["comp_id"])
    rows = []
    for g in gammas:
        shift = g * cat.is_seen
        a_s = 100 * (np.argmax(scores_seen - shift, axis=1) == y_s).mean()
        a_u = 100 * (np.argmax(scores_unseen - shift, axis=1) == y_u).mean()
        rows.append((float(g), a_s, a_u, harmonic(a_s, a_u)))
    return np.array(rows)


def curve_auc(rows):
    """Area under A_S against A_U, on a 0-1 scale, for rows from sweep()."""
    a_s, a_u = rows[:, 1] / 100, rows[:, 2] / 100
    order = np.lexsort((-a_s, a_u))
    return float(np.trapezoid(a_s[order], a_u[order]) if hasattr(np, "trapezoid") else np.trapz(a_s[order], a_u[order]))


def _pair_counts(hits, pairs, n_pairs):
    return np.bincount(pairs, hits.astype(float), n_pairs), np.bincount(pairs, minlength=n_pairs).astype(float)


def bootstrap_delta(seen_control, seen_variant, seen_pair, unseen_control, unseen_variant, unseen_pair, draws=2000, seed=42):
    """Paired bootstrap of variant H minus control H, resampling pairs. Hits are boolean per image."""
    rng = np.random.default_rng(seed)
    _, s_inv = np.unique(seen_pair, return_inverse=True)
    _, u_inv = np.unique(unseen_pair, return_inverse=True)
    n_s, n_u = s_inv.max() + 1, u_inv.max() + 1
    s_c, s_n = _pair_counts(seen_control, s_inv, n_s)
    s_v, _ = _pair_counts(seen_variant, s_inv, n_s)
    u_c, u_n = _pair_counts(unseen_control, u_inv, n_u)
    u_v, _ = _pair_counts(unseen_variant, u_inv, n_u)
    pick_s = rng.integers(0, n_s, (draws, n_s))
    pick_u = rng.integers(0, n_u, (draws, n_u))

    def h(seen_hits, unseen_hits):
        a = 100 * seen_hits[pick_s].sum(1) / s_n[pick_s].sum(1)
        b = 100 * unseen_hits[pick_u].sum(1) / u_n[pick_u].sum(1)
        return 2 * a * b / np.maximum(a + b, 1e-12)

    return h(s_v, u_v) - h(s_c, u_c)


def shortcut_metrics(scores, disease_scores, npz, cat):
    """Crop-shortcut diagnostics as percentages.

    shortcut_rate: of the rows whose disease argmax is wrong, the share whose predicted disease forms a seen pair with the true crop.
    shortcut_chance: the same share if the wrong disease were drawn uniformly (mean seen diseases of the true crop over n_disease).
    oracle_disease_top1: disease Top-1 when the catalogue search is limited to the true crop.
    """
    n_disease = disease_scores.shape[1]
    crop, disease = np.asarray(npz["crop_id"]), np.asarray(npz["disease_id"])
    seen_pairs = {(int(c), int(d)) for c, d in zip(cat.crop_id[cat.is_seen], cat.disease_id[cat.is_seen])}
    seen_count = {}
    for c, _ in seen_pairs:
        seen_count[c] = seen_count.get(c, 0) + 1
    predicted = np.argmax(disease_scores, axis=1)
    wrong = predicted != disease
    if wrong.any():
        hits = [(int(c), int(p)) in seen_pairs for c, p in zip(crop[wrong], predicted[wrong])]
        shortcut_rate = 100 * float(np.mean(hits))
    else:
        shortcut_rate = 0.0
    shortcut_chance = 100 * float(np.mean([seen_count.get(int(c), 0) for c in crop])) / n_disease
    oracle_hits = []
    for crop_value in np.unique(crop):
        rows = np.flatnonzero(crop == crop_value)
        columns = np.flatnonzero(cat.crop_id == crop_value)
        if len(columns) == 0:
            oracle_hits.extend([False] * len(rows))
            continue
        best = columns[np.argmax(scores[np.ix_(rows, columns)], axis=1)]
        oracle_hits.extend((cat.disease_id[best] == disease[rows]).tolist())
    return {
        "shortcut_rate": shortcut_rate,
        "shortcut_chance": shortcut_chance,
        "oracle_disease_top1": 100 * float(np.mean(oracle_hits)),
    }


def pair_breakdown(seen, unseen, cat, tol=3.0, r=1.0):
    """Unseen pair accuracy split into its parts, as percentages, for one epoch's seen and unseen outputs.

    crop, disease: head Top-1 on the unseen rows. both_heads: rows where both heads are right, which is the pair a
    model names with no help from the seen-pair penalty. A_U_raw: pair Top-1 at gamma 0. A_U_keep: pair Top-1 at the
    keep-seen gamma (the largest unseen accuracy with seen accuracy within tol points of raw, as in calib_board).
    keep_disease_right: of the keep-seen unseen pair hits, the share whose disease head is also right. A low share
    means the hits come from a right crop plus the penalty, because each crop has about one unseen pair.
    """
    ss, su = head_score(seen, cat, r=r), head_score(unseen, cat, r=r)
    rows = sweep(ss, seen, su, unseen, cat, gamma_grid(ss, su, cat))
    raw = rows[0]
    keep = rows[rows[:, 1] >= raw[1] - tol]
    k = keep[np.lexsort((-keep[:, 1], -keep[:, 2]))[0]]
    hits = np.argmax(su - k[0] * cat.is_seen, axis=1) == cat.index_of(unseen["comp_id"])
    crop_ok = np.argmax(unseen["crop_logits"], axis=1) == unseen["crop_id"]
    disease_ok = np.argmax(unseen["disease_logits"], axis=1) == unseen["disease_id"]
    return {
        "crop": 100 * float(crop_ok.mean()),
        "disease": 100 * float(disease_ok.mean()),
        "both_heads": 100 * float((crop_ok & disease_ok).mean()),
        "A_S_raw": float(raw[1]), "A_U_raw": float(raw[2]),
        "gamma_keep": float(k[0]), "A_S_keep": float(k[1]), "A_U_keep": float(k[2]),
        "keep_disease_right": 100 * float((hits & disease_ok).sum() / hits.sum()) if hits.any() else float("nan"),
    }
