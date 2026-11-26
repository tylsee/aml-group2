"""P3 evaluation: shortcut metrics on a hand-made catalogue and prototype banks."""
import numpy as np
import pytest

from cdccp.catalogue import Catalogue
from cdccp.evaluate import shortcut_metrics
from cdccp.prototypes import build_banks

# crop 0: diseases 0,1 seen, 2 unseen. crop 1: disease 0 seen, 3 unseen. crop 2: disease 3 seen.
CAT = Catalogue(
    comp_id=np.arange(6),
    crop_id=np.array([0, 0, 0, 1, 1, 2]),
    disease_id=np.array([0, 1, 2, 0, 3, 3]),
    is_seen=np.array([True, True, False, True, False, True]),
    names=list("abcdef"),
)


def one_hot(idx, n):
    return np.eye(n)[idx]


def test_shortcut_metrics_known_answers():
    # row 0: true (crop 0, disease 2), predicts disease 1 -> wrong, (0,1) is seen -> shortcut
    # row 1: true (crop 1, disease 3), predicts disease 2 -> wrong, (1,2) not in catalogue -> not a shortcut
    # row 2: true (crop 1, disease 0), predicts disease 0 -> right
    # row 3: true (crop 2, disease 3), predicts disease 0 -> wrong, (2,0) not seen -> not a shortcut
    npz = {"crop_id": np.array([0, 1, 1, 2]), "disease_id": np.array([2, 3, 0, 3])}
    disease_scores = one_hot([1, 2, 0, 0], 4)
    scores = np.zeros((4, 6))
    scores[0, 1] = 1.0  # best pair for the true crop 0 is disease 1: wrong
    scores[1, 4] = 1.0  # crop 1 best is (1,3): right
    scores[2, 4] = 1.0  # crop 1 best is (1,3) but truth is disease 0: wrong
    scores[3, 0] = 5.0  # a crop 0 pair scores highest overall, but crop 2 only has (2,3): right
    m = shortcut_metrics(scores, disease_scores, npz, CAT)
    assert m["shortcut_rate"] == pytest.approx(100 / 3)
    # seen diseases per true crop: 2, 1, 1, 1 over 4 diseases
    assert m["shortcut_chance"] == pytest.approx(100 * (2 + 1 + 1 + 1) / 4 / 4)
    assert m["oracle_disease_top1"] == pytest.approx(50.0)


def test_shortcut_rate_is_zero_without_wrong_rows():
    npz = {"crop_id": np.array([0, 1]), "disease_id": np.array([1, 0])}
    m = shortcut_metrics(np.zeros((2, 6)), one_hot([1, 0], 4), npz, CAT)
    assert m["shortcut_rate"] == 0.0


def banks_input(crop, disease, u_crop, u_disease):
    return {"crop_id": np.array(crop), "disease_id": np.array(disease), "u_crop": np.array(u_crop, float), "u_disease": np.array(u_disease, float)}


def test_single_image_prototype_equals_embedding():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 4))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    npz = banks_input([0, 1, 2], [0, 1, 2], u, u[::-1])
    for form in ("pooled", "balanced"):
        banks = build_banks(npz, form, n_crop=3, n_disease=3)
        assert banks["crop"].dtype == np.float32
        assert np.allclose(banks["crop"], u, atol=1e-6)
        assert np.allclose(banks["disease"], u[::-1], atol=1e-6)


def test_balanced_differs_from_pooled_and_equals_mean_of_pair_means():
    # disease 0 has 3 images on crop 0 and 1 image on crop 1
    a, b, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    u_disease = [a, a, a, b, c]
    npz = banks_input([0, 0, 0, 1, 0], [0, 0, 0, 0, 1], u_disease, u_disease)
    pooled = build_banks(npz, "pooled", n_crop=2, n_disease=2)
    balanced = build_banks(npz, "balanced", n_crop=2, n_disease=2)
    assert not np.allclose(pooled["disease"][0], balanced["disease"][0])
    expected = (a + b) / 2
    expected /= np.linalg.norm(expected)
    assert np.allclose(balanced["disease"][0], expected, atol=1e-6)
    pooled_expected = (3 * a + b) / 4
    assert np.allclose(pooled["disease"][0], pooled_expected / np.linalg.norm(pooled_expected), atol=1e-6)
    assert np.allclose(np.linalg.norm(balanced["disease"], axis=1), 1.0, atol=1e-6)


def test_missing_class_raises():
    npz = banks_input([0, 0], [0, 1], np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match=r"crop classes with no images: \[1\]"):
        build_banks(npz, "pooled", n_crop=2, n_disease=2)
    with pytest.raises(ValueError, match=r"disease classes with no images: \[2\]"):
        build_banks(npz, "balanced", n_crop=1, n_disease=3)


def test_p3_score_row_uses_each_scorers_own_disease_scores():
    import importlib.util

    from cdccp.catalogue import load_catalogue
    from cdccp.config import REPO, load_config

    spec = importlib.util.spec_from_file_location("p3_eval", REPO / "scripts" / "p3_eval.py")
    p3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3)
    cat = load_catalogue(load_config()["paths"]["catalogue_dev"])
    seen_i, unseen_i = np.flatnonzero(cat.is_seen)[:3], np.flatnonzero(~cat.is_seen)[:3]
    npz = lambda idx: {"comp_id": cat.comp_id[idx], "crop_id": cat.crop_id[idx], "disease_id": cat.disease_id[idx]}
    seen, unseen = npz(seen_i), npz(unseen_i)
    rng = np.random.default_rng(0)
    ss, su = rng.normal(size=(3, len(cat))), rng.normal(size=(3, len(cat)))
    right = np.full((3, 175), -9.0)
    right[np.arange(3), unseen["disease_id"]] = 9.0
    wrong = np.full((3, 175), -9.0)
    crop_seen_disease = [cat.disease_id[cat.is_seen & (cat.crop_id == c)][0] for c in unseen["crop_id"]]
    wrong[np.arange(3), crop_seen_disease] = 9.0  # each wrong guess is a disease the true crop had in training
    a = p3.score_row("a", ss, su, right, seen, unseen, cat, 3.0)
    b = p3.score_row("b", ss, su, wrong, seen, unseen, cat, 3.0)
    assert a["unseen disease"] == 100.0 and b["unseen disease"] == 0.0
    assert a["unseen shortcut rate"] == 0.0 and a["H raw"] == b["H raw"]
    assert b["unseen shortcut rate"] == 100.0
