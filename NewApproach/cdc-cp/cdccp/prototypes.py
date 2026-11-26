"""Class prototype banks built from unit embeddings."""
import numpy as np

FORMS = ("pooled", "balanced")


def _unit(rows):
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norm, 1e-12)).astype(np.float32)


def _check_present(labels, n_class, what):
    missing = sorted(set(range(n_class)) - set(int(v) for v in labels))
    if missing:
        raise ValueError(f"{what} classes with no images: {missing}")


def _pooled(u, labels, n_class):
    sums = np.zeros((n_class, u.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, u)
    return _unit(sums / np.bincount(labels, minlength=n_class)[:, None])


def _balanced(u, labels, groups, n_class):
    """Mean over groups of the per-(label, group) mean, so no single group dominates a class."""
    pair_ids, inverse = np.unique(labels * (int(groups.max()) + 1) + groups, return_inverse=True)
    pair_sums = np.zeros((len(pair_ids), u.shape[1]), dtype=np.float64)
    np.add.at(pair_sums, inverse, u)
    pair_means = pair_sums / np.bincount(inverse)[:, None]
    pair_labels = pair_ids // (int(groups.max()) + 1)
    sums = np.zeros((n_class, u.shape[1]), dtype=np.float64)
    np.add.at(sums, pair_labels, pair_means)
    return _unit(sums / np.bincount(pair_labels, minlength=n_class)[:, None])


def build_banks(npz, form="pooled", n_crop=55, n_disease=175):
    """Unit prototype rows {"crop": [n_crop,128], "disease": [n_disease,128]} from extract() output.

    pooled: per-class mean of the unit embeddings.
    balanced: the disease row averages over crops of the per-(crop, disease) mean, and the crop row averages over diseases.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}, expected one of {FORMS}")
    crop, disease = np.asarray(npz["crop_id"], dtype=np.int64), np.asarray(npz["disease_id"], dtype=np.int64)
    _check_present(crop, n_crop, "crop")
    _check_present(disease, n_disease, "disease")
    u_crop, u_disease = np.asarray(npz["u_crop"], dtype=np.float64), np.asarray(npz["u_disease"], dtype=np.float64)
    if form == "pooled":
        return {"crop": _pooled(u_crop, crop, n_crop), "disease": _pooled(u_disease, disease, n_disease)}
    return {"crop": _balanced(u_crop, crop, disease, n_crop), "disease": _balanced(u_disease, disease, crop, n_disease)}
