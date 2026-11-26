"""pair_breakdown splits unseen pair accuracy into crop, disease, both heads and the seen-pair penalty."""
import math

import numpy as np
import pytest

from cdccp.catalogue import Catalogue
from cdccp.evaluate import pair_breakdown
from cdccp.selection import compare_arms
from test_compare_arms import paired

# crops 0, 1; diseases 0, 1. Seen pairs (0,0), (0,1), (1,1); the unseen pair is (1,0).
CAT = Catalogue(comp_id=np.array([10, 11, 12, 13]), crop_id=np.array([0, 0, 1, 1]), disease_id=np.array([0, 1, 1, 0]),
                is_seen=np.array([True, True, True, False]), names=["a", "b", "c", "d"])


def rows(crop_logits, disease_logits, crop, disease, comp):
    return {"crop_logits": np.array(crop_logits, float), "disease_logits": np.array(disease_logits, float),
            "crop_id": np.array(crop), "disease_id": np.array(disease), "comp_id": np.array(comp)}


def test_breakdown_separates_both_heads_from_hits_made_by_the_penalty():
    seen = rows([[9, 0], [0, 9]], [[9, 0], [0, 9]], [0, 1], [0, 1], [10, 12])      # sure, right seen rows
    unseen = rows([[0, 5], [0, 5]], [[5, 0], [0, 2]], [1, 1], [0, 0], [13, 13])    # A: both heads right, B: disease wrong
    b = pair_breakdown(seen, unseen, CAT)
    assert (b["crop"], b["disease"], b["both_heads"]) == (100, 50, 50)
    assert (b["A_S_raw"], b["A_U_raw"]) == (100, 50)
    assert (b["A_S_keep"], b["A_U_keep"]) == (100, 100) and b["gamma_keep"] > 0
    assert b["keep_disease_right"] == 50     # B is a pair hit only because crop 1 has one unseen pair


def test_breakdown_keeps_seen_accuracy_and_reports_no_share_without_hits():
    seen = rows([[9, 0], [0, 9]], [[9, 0], [0, 0.3]], [0, 1], [0, 1], [10, 12])   # row 2 beats the unseen pair by only 0.3
    unseen = rows([[0, 9]], [[0, 2]], [1], [0], [13])                              # needs a penalty above 2 to be right
    b = pair_breakdown(seen, unseen, CAT, tol=0.0)
    assert (b["A_S_raw"], b["A_S_keep"], b["A_U_keep"]) == (100, 100, 0)
    assert math.isnan(b["keep_disease_right"])


def test_compare_arms_z_widens_the_bar_and_must_be_positive():
    s = paired([20, 21, 19], [27, 28, 26])
    _, _, mean, limit2, verdict2 = compare_arms(s, "base", "patch")
    _, _, _, limit3, verdict3 = compare_arms(s, "base", "patch", z=3.0)
    assert limit3 == pytest.approx(limit2 * 1.5) and verdict2 == "better" and (verdict3 == "better") == (mean > limit3)
    for z in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="z must be positive"):
            compare_arms(s, "base", "patch", z=z)
