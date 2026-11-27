import numpy as np


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def combine(crop_scores, disease_scores, cat, r=1.0, g=0.0):
    """S(c,d) = crop + r * disease - g * [pair seen], for every catalogue pair. Shape [N,K]."""
    return crop_scores[:, cat.crop_id] + r * disease_scores[:, cat.disease_id] - g * cat.is_seen


def head_score(npz, cat, r=1.0, g=0.0):
    return combine(_log_softmax(npz["crop_logits"]), _log_softmax(npz["disease_logits"]), cat, r, g)


def proto_cosines(npz, banks):
    """Cosines to each bank. Embeddings and banks must already be unit length."""
    return npz["u_crop"] @ banks["crop"].T, npz["u_disease"] @ banks["disease"].T


def proto_score(npz, banks, cat, r=1.0, g=0.0):
    crop, disease = proto_cosines(npz, banks)
    return combine(crop, disease, cat, r, g)


def rank(scores):
    """Descending catalogue indices. Ties keep catalogue order."""
    return np.argsort(-scores, axis=1, kind="stable")
