"""A1: hand-computed fixture, invariances, chance level, and evaluation refusals."""
import numpy as np
import pandas as pd
import pytest

from cdccp.catalogue import Catalogue, load_catalogue
from cdccp.checkpoint import training_meta
from cdccp.data import load_split
from cdccp.evaluate import Refused, bootstrap_delta, check_rows, combined_metrics, confusion, curve_auc, harmonic, seen_rows, split_metrics, sweep, topk_hits
from cdccp.gate import accept
from cdccp.scoring import combine, head_score, proto_score

CAT = Catalogue(
    comp_id=np.array([10, 11, 12, 13]),
    crop_id=np.array([0, 0, 1, 1]),
    disease_id=np.array([0, 1, 0, 1]),
    is_seen=np.array([True, True, True, False]),
    names=["a", "b", "c", "d"],
)
CROP = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.5]])
DISEASE = np.array([[0.0, 1.0], [0.5, 2.0], [4.0, 0.0]])


def test_hand_computed_scores():
    expected = np.array([
        [2.0, 3.0, 0.0, 1.0],
        [0.5, 2.0, 3.5, 5.0],
        [5.0, 1.0, 5.5, 1.5],
    ])
    assert np.allclose(combine(CROP, DISEASE, CAT), expected)
    expected_gamma = expected - np.array([1.0, 1.0, 1.0, 0.0])
    assert np.allclose(combine(CROP, DISEASE, CAT, g=1.0), expected_gamma)
    assert np.allclose(combine(CROP, DISEASE, CAT, r=2.0)[0], [2.0, 4.0, 0.0, 2.0])


def test_hand_computed_metrics():
    npz = {"comp_id": np.array([11, 13, 10]), "crop_id": np.array([0, 1, 0]), "disease_id": np.array([1, 1, 0]), "crop_logits": CROP, "disease_logits": DISEASE}
    m = split_metrics(combine(CROP, DISEASE, CAT), npz, CAT)
    # Top-1 picks are b, d, c, so the first two are right and the third is wrong. All are within Top-5 of 4 pairs.
    assert m["comp_top1"] == pytest.approx(200 / 3)
    assert m["comp_top5"] == 100
    assert m["crop_top1"] == pytest.approx(200 / 3)
    assert m["disease_top1"] == 100
    matrix = confusion(combine(CROP, DISEASE, CAT), npz, CAT)
    assert matrix[1, 1] == 1 and matrix[3, 3] == 1 and matrix[0, 2] == 1 and matrix.sum() == 3
    assert seen_rows(matrix, CAT).shape == (3, 4)


def test_harmonic():
    assert harmonic(0, 0) == 0
    assert harmonic(60, 20) == pytest.approx(30)
    assert combined_metrics({"comp_top1": 60}, {"comp_top1": 20}) == {"A_S": 60, "A_U": 20, "H": pytest.approx(30), "gap": 40}


def test_rescale_invariance_and_gamma_sign():
    scores = combine(CROP, DISEASE, CAT, r=1.5, g=0.7)
    y = np.array([0, 1, 2])
    for k in (1, 2, 3):
        assert (topk_hits(scores, y, k) == topk_hits(scores * 7.0 + 3.0, y, k)).all()
    plus_unseen = combine(CROP, DISEASE, CAT, r=1.5) + 0.7 * (~CAT.is_seen)
    assert (np.argsort(-scores, axis=1) == np.argsort(-plus_unseen, axis=1)).all()


def test_random_scores_are_at_chance():
    rng = np.random.default_rng(0)
    k = 323
    cat = Catalogue(np.arange(k), np.arange(k) % 55, np.arange(k) % 175, np.arange(k) >= 10, [""] * k)
    y = rng.integers(0, k, 20000)
    scores = rng.normal(size=(20000, k))
    assert 100 * topk_hits(scores, y, 1).mean() == pytest.approx(100 / k, abs=0.15)
    assert 100 * topk_hits(scores, y, 5).mean() == pytest.approx(500 / k, abs=0.3)


def test_head_and_proto_scores_use_catalogue_pairs():
    npz = {"crop_logits": CROP, "disease_logits": DISEASE, "u_crop": np.eye(2)[[0, 1, 0]], "u_disease": np.eye(2)[[1, 1, 0]]}
    assert head_score(npz, CAT).shape == (3, 4)
    banks = {"crop": np.eye(2), "disease": np.eye(2)}
    assert np.allclose(proto_score(npz, banks, CAT)[0], [1, 2, 0, 1])


def test_sweep_moves_between_seen_and_unseen_only():
    scores = combine(CROP, DISEASE, CAT)
    seen = {"comp_id": np.array([11, 10])}
    unseen = {"comp_id": np.array([13])}
    rows = sweep(scores[[0, 2]], seen, scores[[1]], unseen, CAT, [-100.0, 0.0, 100.0])
    assert rows[0, 2] == 0 and rows[2, 1] == 0 and rows[2, 2] == 100
    assert 0 <= curve_auc(rows) <= 1


def test_bootstrap_of_identical_hits_is_zero():
    s = np.array([True, False, True, True]); sp = np.array([0, 0, 1, 2])
    u = np.array([True, False, False]); up = np.array([0, 1, 2])
    assert np.allclose(bootstrap_delta(s, s, sp, u, u, up, draws=50), 0)


def test_gate_accepts_a_clear_gain_and_rejects_no_gain():
    n_pairs = 10
    pair = np.repeat(np.arange(n_pairs), 20)
    seen = np.ones_like(pair, dtype=bool)
    weak = (np.arange(len(pair)) % 5) == 0
    strong = (np.arange(len(pair)) % 5) <= 2
    groups = {"all": np.ones_like(pair, dtype=bool)}
    make = lambda hits: {"seen_hits": seen, "seen_pair": pair, "unseen_hits": hits, "unseen_pair": pair, "subgroups": groups}
    assert accept(make(strong), make(weak), delta=1)["accept"]
    assert not accept(make(weak), make(weak), delta=1)["accept"]
    assert not accept(make(weak), make(strong), delta=1)["accept"]
    with_empty = lambda hits: {**make(hits), "subgroups": {**groups, "empty": np.zeros_like(pair, dtype=bool)}}
    assert accept(with_empty(strong), with_empty(weak), delta=1)["accept"]


def test_refusals_on_real_splits(cfg):
    fit, seen, unseen = (load_split(cfg, n) for n in ("train_fit", "val_seen", "val_unseen"))
    dev = training_meta(fit)
    check_rows(seen, "seen", dev)
    check_rows(unseen, "unseen", dev)
    with pytest.raises(Refused):
        check_rows(fit.head(5), "seen", dev)

    fold = {"cassava", "blueberry", "bean", "apple", "wheat"}
    hold, add = unseen[unseen.crop_name.isin(fold)], unseen[~unseen.crop_name.isin(fold)]
    rehearsal = training_meta(pd.concat([fit, add]))
    check_rows(hold, "unseen", rehearsal)
    with pytest.raises(Refused):
        check_rows(add, "unseen", rehearsal)

    final = training_meta(load_split(cfg, "train"))
    with pytest.raises(Refused):
        check_rows(unseen, "unseen", final)


def test_official_rows_need_a_lock():
    frame = pd.DataFrame({"filename": ["x.jpg"], "composition_id": [1]})
    meta = {"training_files": [], "training_pairs": []}
    with pytest.raises(Refused):
        check_rows(frame, "official_seen", meta)
    check_rows(frame, "official_seen", meta, lock={"ok": True})


def test_dev_catalogue_covers_validation(cfg):
    cat = load_catalogue(cfg["paths"]["catalogue_dev"])
    for name, flag in (("val_seen", True), ("val_unseen", False)):
        index = cat.index_of(load_split(cfg, name).composition_id)
        assert (cat.is_seen[index] == flag).all()
