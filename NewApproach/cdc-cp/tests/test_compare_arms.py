import math

import pandas as pd
import pytest

from cdccp.selection import compare_arms, run_scores


def hist(unseen, seen=None):
    seen = seen or [50.0] * len(unseen)
    return pd.DataFrame({"epoch": range(1, len(unseen) + 1), "unseen_disease_top1": unseen, "seen_disease_top1": seen})


def scores(rows):
    return pd.DataFrame([{"arm": a, "seed": s, "unseen_disease": u, "seen_disease": 60.0} for a, s, u in rows])


def paired(ctrl, treat, seeds=(42, 43, 44)):
    return scores([("base", s, u) for s, u in zip(seeds, ctrl)] + [("patch", s, u) for s, u in zip(seeds, treat)])


def test_run_scores_uses_the_best_three_epoch_average_not_one_lucky_epoch():
    r = run_scores(hist([10, 30, 10, 20, 22, 21, 15], seen=[40, 41, 42, 43, 44, 45, 46]))
    assert r["best_epoch"] == 2 and r["best_unseen_disease"] == 30
    assert r["pick_epoch"] == 5 and r["unseen_disease"] == pytest.approx(21) and r["seen_disease"] == pytest.approx(44)
    assert r["epochs"] == 7


def test_run_scores_works_on_an_early_stopped_run_and_unsorted_rows():
    r = run_scores(hist([12, 18, 17, 16, 15]).iloc[::-1])
    assert r["best_epoch"] == 2 and r["epochs"] == 5
    assert r["pick_epoch"] == 3 and r["unseen_disease"] == pytest.approx((18 + 17 + 16) / 3)


def test_run_scores_uses_full_three_epoch_windows_only():
    r = run_scores(hist([30, 30, 0, 0]))
    assert r["pick_epoch"] == 2 and r["unseen_disease"] == pytest.approx(20) and r["best_epoch"] == 1
    assert run_scores(hist([5, 9]))["unseen_disease"] == pytest.approx(7)


def test_run_scores_refuses_empty_and_nan():
    with pytest.raises(ValueError, match="no epochs"):
        run_scores(hist([]))
    with pytest.raises(ValueError, match="NaN"):
        run_scores(hist([10, float("nan")]))


def test_clear_gain_is_better():
    table, diffs, mean, limit, verdict = compare_arms(paired((15, 16, 14), (25, 26, 24)), "base", "patch")
    assert verdict == "better" and mean == pytest.approx(10) and list(diffs) == [10, 10, 10]
    assert table.index.tolist() == ["base", "patch"] and limit < 10


def test_clear_loss_is_worse():
    assert compare_arms(paired((25, 26, 24), (15, 16, 14)), "base", "patch")[-1] == "worse"


def test_small_gap_is_no_clear_change_and_limit_adds_image_noise():
    _, _, mean, limit, verdict = compare_arms(paired((15, 16, 14), (17, 18, 16)), "base", "patch")
    assert verdict == "no clear change" and mean == pytest.approx(2)
    assert limit == pytest.approx(2 * math.sqrt(1e4 * (0.15 * 0.85 + 0.17 * 0.83) / 400), rel=1e-6)


def test_noisy_seed_diffs_widen_the_limit():
    _, diffs, _, limit, verdict = compare_arms(paired((15, 15, 15), (30, 15, 22)), "base", "patch")
    assert verdict == "no clear change" and limit > 2 * math.sqrt(diffs.var(ddof=1) / 3)


def test_diffs_are_paired_by_seed_not_by_row_order():
    df = paired((10, 20, 30), (11, 21, 31)).sample(frac=1, random_state=0)
    _, diffs, _, _, _ = compare_arms(df, "base", "patch")
    assert list(diffs) == [1, 1, 1] and list(diffs.index) == [42, 43, 44]


def test_missing_duplicate_and_bad_inputs_are_refused():
    full = paired((15, 16, 14), (25, 26, 24))
    with pytest.raises(ValueError, match="patch_seed44"):
        compare_arms(full.iloc[:-1], "base", "patch")
    with pytest.raises(ValueError, match="more than one row"):
        compare_arms(pd.concat([full, scores([("base", 42, 15)])]), "base", "patch")
    with pytest.raises(ValueError, match="3 distinct seeds"):
        compare_arms(paired((15, 16), (25, 26), seeds=(42, 43)), "base", "patch", seeds=(42, 43))
    with pytest.raises(ValueError, match="each once"):
        compare_arms(full, "base", "patch", seeds=(42, 43, 44, 42))
    with pytest.raises(ValueError, match="must differ"):
        compare_arms(full, "base", "base")
    nan = full.copy()
    nan.loc[0, "unseen_disease"] = float("nan")
    with pytest.raises(ValueError, match="without an unseen disease score"):
        compare_arms(nan, "base", "patch")


def test_other_arms_and_seeds_are_ignored():
    df = pd.concat([paired((15, 16, 14), (25, 26, 24)), scores([("other", 42, 99), ("base", 45, 0), ("patch", 45, 99)])])
    assert compare_arms(df, "base", "patch")[2] == pytest.approx(10)
