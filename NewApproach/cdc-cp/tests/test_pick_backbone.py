import math

import pandas as pd
import pytest

from cdccp.selection import pick_backbone


def board(rows, head="h256"):
    return pd.DataFrame([{"run": f"sweep_{b}_{head}_seed{s}", "backbone": b, "unseen disease": u, "H keep": 40.0, "seen disease": 75.0}
                         for b, s, u in rows])


def full(scores):
    return board([(b, s, u) for b, us in scores.items() for s, u in zip((42, 43, 44), us)])


def test_clear_winner_is_not_a_tie():
    t, win, second, tie, limit = pick_backbone(full({"a": (30, 31, 29), "b": (20, 21, 19)}), ["a", "b"])
    assert (win, second, tie) == ("a", "b", False) and limit < 10
    assert t.loc["a", "unseen_disease"] == pytest.approx(30) and t.loc["a", "per_seed"] == "30.0 / 31.0 / 29.0"


def test_noisy_small_gap_is_a_tie():
    _, win, second, tie, _ = pick_backbone(full({"a": (26, 17, 24), "b": (21, 22, 20)}), ["a", "b"])
    assert (win, second, tie) == ("a", "b", True)


def test_steady_seeds_still_tie_on_image_noise():
    _, win, _, tie, limit = pick_backbone(full({"a": (22.9, 23.1, 23.0), "b": (22.1, 22.0, 22.2)}), ["a", "b"])
    assert win == "a" and tie and limit == pytest.approx(2 * math.sqrt(1e4 * (0.23 * 0.77 + 0.221 * 0.779) / 400 + 0.1 ** 2 / 3 * 2), rel=1e-3)


def test_missing_seed_is_refused():
    with pytest.raises(ValueError, match="sweep_b_h256_seed44"):
        pick_backbone(board([("a", 42, 30), ("a", 43, 31), ("a", 44, 29), ("b", 42, 20), ("b", 43, 21)]), ["a", "b"])


def test_missing_backbone_is_refused():
    with pytest.raises(ValueError, match="sweep_c_h256_seed42"):
        pick_backbone(full({"a": (30, 31, 29), "b": (20, 21, 19)}), ["a", "b", "c"])


def test_fewer_than_three_seeds_are_refused():
    with pytest.raises(ValueError, match="3 distinct seeds"):
        pick_backbone(full({"a": (30, 31, 29), "b": (20, 21, 19)}), ["a", "b"], seeds=(42, 43))


def test_unscored_run_is_refused():
    with pytest.raises(ValueError, match="sweep_b_h256_seed43"):
        pick_backbone(full({"a": (30, 31, 29), "b": (20, float("nan"), 19)}), ["a", "b"])


def test_duplicate_run_is_refused():
    b = full({"a": (30, 31, 29), "b": (20, 21, 19)})
    with pytest.raises(ValueError, match="more than one row"):
        pick_backbone(pd.concat([b, b.iloc[[0]]]), ["a", "b"])


def test_run_of_another_head_is_refused():
    b = pd.concat([full({"a": (30, 31, 29)}), board([("b", 42, 20), ("b", 43, 21), ("b", 44, 19)], head="h256gelu")])
    with pytest.raises(ValueError, match="sweep_b_h256gelu_seed42"):
        pick_backbone(b, ["a", "b"])


def test_extra_seeds_and_backbones_outside_the_plan_are_ignored():
    b = pd.concat([full({"a": (30, 30, 30), "b": (20, 20, 20), "z": (90, 90, 90)}), board([("a", 45, 0)])])
    t, win, _, _, _ = pick_backbone(b, ["a", "b"])
    assert win == "a" and list(t.index) == ["a", "b"] and t.loc["a", "unseen_disease"] == pytest.approx(30) and t.loc["a", "n"] == 3


def test_per_seed_follows_the_requested_seed_order():
    t, *_ = pick_backbone(full({"a": (30, 31, 29), "b": (20, 21, 19)}), ["a", "b"], seeds=(44, 42, 43))
    assert t.loc["a", "per_seed"] == "29.0 / 30.0 / 31.0"
