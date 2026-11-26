import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cdccp.ensemble as pm
from cdccp import calibration as pg
from cdccp import single_task as st
from cdccp.catalogue import Catalogue, load_catalogue
from cdccp.config import REPO

P5_RESULTS = Path(r"D:\dev\uni\AML\runs\p5_colab_L4\p5_results")
CAT_DEV = REPO / "data" / "mappings" / "catalogue_dev.json"
CAT_FINAL = REPO / "data" / "mappings" / "catalogue_final.json"
SPLITS_DIR = REPO / "data" / "splits_v2"
HAVE_P5 = (P5_RESULTS / "results" / "dev_flip_logprobs.npz").is_file() and (P5_RESULTS / "lock.json").is_file()


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def frame(crop_id, disease_id, n_disease, prefix="f"):
    crop_id, disease_id = list(crop_id), list(disease_id)
    return pd.DataFrame({"filename": [f"{prefix}{i}.jpg" for i in range(len(crop_id))], "crop_id": crop_id, "disease_id": disease_id,
                         "composition_id": [c * n_disease + d for c, d in zip(crop_id, disease_id)],
                         "composition_name": [f"c{c}_d{d}" for c, d in zip(crop_id, disease_id)]})


def random_world(seed=0, n_crop=4, n_disease=5, n=40):
    rng = np.random.default_rng(seed)
    dfs = tuple(frame(rng.integers(0, n_crop, n), rng.integers(0, n_disease, n), n_disease, prefix=p) for p in "su")
    arrays = {}
    for arm, task in (("crop_b1", "crop"), ("crop_clip", "crop"), ("disease_clip", "disease"), ("disease_small_reg", "disease")):
        classes = n_crop if task == "crop" else n_disease
        for split in pm.SPLITS:
            arrays[arm, split] = log_softmax(rng.normal(size=(n, classes)) * 3).astype(np.float32)
    return dfs, arrays


def split_sources(arrays):
    p5 = {k: v for k, v in arrays.items() if k[0] in ("crop_b1", "disease_clip")}
    new = {k: v for k, v in arrays.items() if k[0] in ("crop_clip", "disease_small_reg")}
    return p5, new


def test_members_and_weights_are_the_fixed_design():
    assert pm.P5_ARMS == {"crop": "crop_b1", "disease": "disease_clip"}
    assert pm.NEW_ARMS == {"crop": "crop_clip", "disease": "disease_small_reg"}
    assert pm.WEIGHTS == (0.5, 0.5)


def test_p6_parts_are_the_mean_of_log_softmax_of_the_four_members():
    dfs, arrays = random_world()
    p5, new = split_sources(arrays)
    seen, unseen = pm.p6_parts(p5, new, dfs)
    for split, part in zip(pm.SPLITS, (seen, unseen)):
        crop = 0.5 * log_softmax(arrays["crop_b1", split].astype(np.float64)) + 0.5 * log_softmax(arrays["crop_clip", split].astype(np.float64))
        disease = 0.5 * log_softmax(arrays["disease_clip", split].astype(np.float64)) + 0.5 * log_softmax(arrays["disease_small_reg", split].astype(np.float64))
        np.testing.assert_allclose(part["crop_logits"], crop, atol=1e-12)
        np.testing.assert_allclose(part["disease_logits"], disease, atol=1e-12)
    assert list(seen["fn"]) == list(dfs[0].filename)
    assert list(unseen["comp_id"]) == list(dfs[1].composition_id)


def test_p6_parts_with_the_same_member_twice_gives_that_member():
    dfs, arrays = random_world(1)
    p5, _ = split_sources(arrays)
    twin = {("crop_clip", s): p5["crop_b1", s] for s in pm.SPLITS} | {("disease_small_reg", s): p5["disease_clip", s] for s in pm.SPLITS}
    seen, _ = pm.p6_parts(p5, twin, dfs)
    single = pg.parts_from_logprobs(p5["crop_b1", "seen"], p5["disease_clip", "seen"], dfs[0])
    np.testing.assert_allclose(seen["crop_logits"], single["crop_logits"], atol=1e-12)
    np.testing.assert_allclose(seen["disease_logits"], single["disease_logits"], atol=1e-12)


def write_npz(path, arrays):
    np.savez_compressed(path, **{f"{a}_{s}": v for (a, s), v in arrays.items()})


def test_read_logprobs_roundtrip_and_refusals(tmp_path):
    arrays = {k: v for k, v in random_world()[1].items() if k[0] in ("crop_b1", "disease_clip")}
    # the toy world has 4 crops and 5 diseases; read_logprobs checks the real 55 and 175 class counts
    write_npz(tmp_path / "a.npz", arrays)
    with pytest.raises(ValueError, match="shape"):
        pm.read_logprobs(tmp_path / "a.npz", ("crop_b1", "disease_clip"), {"seen": 40, "unseen": 40})
    rng = np.random.default_rng(0)
    real = {}
    for arm, classes in (("crop_b1", 55), ("disease_clip", 175)):
        for split, n in (("seen", 6), ("unseen", 4)):
            real[arm, split] = log_softmax(rng.normal(size=(n, classes))).astype(np.float32)
    write_npz(tmp_path / "ok.npz", real)
    got = pm.read_logprobs(tmp_path / "ok.npz", ("crop_b1", "disease_clip"), {"seen": 6, "unseen": 4})
    assert sorted(got) == sorted(real) and all(np.array_equal(got[k], real[k]) for k in real)
    with pytest.raises(ValueError, match="shape"):
        pm.read_logprobs(tmp_path / "ok.npz", ("crop_b1", "disease_clip"), {"seen": 7, "unseen": 4})
    with pytest.raises(ValueError, match="keys"):
        pm.read_logprobs(tmp_path / "ok.npz", ("crop_b1", "crop_clip"), {"seen": 6, "unseen": 4})
    bad = dict(real)
    bad["crop_b1", "seen"] = real["crop_b1", "seen"].copy()
    bad["crop_b1", "seen"][0, 0] = np.nan
    write_npz(tmp_path / "nan.npz", bad)
    with pytest.raises(ValueError, match="finite"):
        pm.read_logprobs(tmp_path / "nan.npz", ("crop_b1", "disease_clip"), {"seen": 6, "unseen": 4})
    bad["crop_b1", "seen"] = real["crop_b1", "seen"] + 1.0
    write_npz(tmp_path / "shift.npz", bad)
    with pytest.raises(ValueError, match="log-probabilities"):
        pm.read_logprobs(tmp_path / "shift.npz", ("crop_b1", "disease_clip"), {"seen": 6, "unseen": 4})


def brute_top(crop, disease, true_crop, true_disease, allowed, k):
    n_disease = disease.shape[1]
    hits = 0
    for i in range(len(crop)):
        scored = [(crop[i, c] + disease[i, d], -(c * n_disease + d)) for c in range(crop.shape[1]) for d in range(n_disease)
                  if allowed is None or allowed[c * n_disease + d]]
        best = sorted(scored, reverse=True)[:k]
        hits += any(-b[1] == true_crop[i] * n_disease + true_disease[i] for b in best)
    return hits / len(crop)


def test_official_table_matches_a_brute_force_count():
    dfs, arrays = random_world(2)
    seen_df, unseen_df = dfs
    heads = {s: (arrays["crop_b1", s], arrays["disease_clip", s]) for s in pm.SPLITS}
    allowed = np.zeros(20, dtype=bool)
    allowed[::3] = True
    allowed[[c * 5 + d for c, d in zip(seen_df.crop_id[:8], seen_df.disease_id[:8])]] = True
    allowed[[c * 5 + d for c, d in zip(unseen_df.crop_id[:8], unseen_df.disease_id[:8])]] = True
    metrics, table = pm.official_table(heads, {"seen": seen_df, "unseen": unseen_df}, [0, 1, 2, 3], [0, 1, 2, 3, 4], allowed)
    assert list(table.columns) == pm.TABLE_COLUMNS and len(table) == 12
    assert list(table["pair list"].unique()) == [pm.ALL_PAIRS, pm.LIST_333]
    for name, mask in ((pm.ALL_PAIRS, None), (pm.LIST_333, allowed)):
        for split, df in (("seen", seen_df), ("unseen", unseen_df)):
            crop, disease = heads[split]
            for k in (1, 5):
                expect = brute_top(crop.astype(float), disease.astype(float), df.crop_id.to_numpy(), df.disease_id.to_numpy(), mask, k)
                assert metrics[name, split][f"composition_top{k}"] == pytest.approx(expect)
                row = table[(table["pair list"] == name) & (table.Task == "Composition") & (table["Top-K"] == k)].iloc[0]
                assert row["Seen (%)" if split == "seen" else "Unseen (%)"] == pytest.approx(100 * expect)
    row = table.iloc[0]
    assert row["HM (%)"] == pytest.approx(2 * row["Seen (%)"] * row["Unseen (%)"] / (row["Seen (%)"] + row["Unseen (%)"]))


def test_recompute_check_passes_on_the_same_rows_and_fails_on_other_row_order(tmp_path):
    dfs, arrays = random_world(3)
    frames = {"seen": dfs[0], "unseen": dfs[1]}
    heads = {s: (arrays["crop_b1", s], arrays["disease_clip", s]) for s in pm.SPLITS}
    _, table = pm.official_table(heads, frames, [0, 1, 2, 3], [0, 1, 2, 3, 4], None)
    saved = tmp_path / "p5.csv"
    table.to_csv(saved, index=False)
    pm.assert_matches_table(table, saved)
    order = np.random.default_rng(0).permutation(40)
    shuffled = {s: (heads[s][0][order], heads[s][1][order]) for s in pm.SPLITS}
    _, other = pm.official_table(shuffled, frames, [0, 1, 2, 3], [0, 1, 2, 3, 4], None)
    with pytest.raises(ValueError, match="differs"):
        pm.assert_matches_table(other, saved)
    changed = table.copy()
    changed.loc[3, "Seen (%)"] += 0.5
    with pytest.raises(ValueError, match="differs"):
        pm.assert_matches_table(changed, saved)
    with pytest.raises(ValueError, match="rows"):
        pm.assert_matches_table(table.iloc[:-1], saved)
    swapped = table.copy()
    swapped["Task"] = swapped["Task"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="Task"):
        pm.assert_matches_table(swapped, saved)


def test_compare_table_picks_the_same_rule_rows():
    def table(offset):
        rows = []
        for name in (pm.ALL_PAIRS, pm.LIST_333):
            for task in ("Crop", "Disease", "Composition"):
                for k in (1, 5):
                    s, u = 50 + offset + k + (10 if name == pm.LIST_333 else 0), 20 + offset + k
                    rows.append({"pair list": name, "Task": task, "Top-K": k, "Seen (%)": s, "Unseen (%)": u, "HM (%)": 2 * s * u / (s + u)})
        return pd.DataFrame(rows, columns=pm.TABLE_COLUMNS)

    b1 = {"Crop": (89.3, 73.07, 96.2, 84.74), "Disease": (66.25, 11.12, 88.45, 26.73), "Composition": (63.65, 6.56, 84.0, 18.85)}
    out = pm.compare_table(table(0), table(5), b1)
    assert len(out) == 6
    row = out[(out.Task == "Composition") & (out["Top-K"] == 5)].iloc[0]
    assert (row["B1 seen (%)"], row["B1 unseen (%)"]) == (84.0, 18.85)
    assert (row["P5 seen (%)"], row["P5 unseen (%)"]) == (55.0, 25.0)
    assert (row["P6 seen (%)"], row["P6 unseen (%)"]) == (60.0, 30.0)
    assert row["P6 HM (%)"] == pytest.approx(2 * 60 * 30 / 90)
    assert row["B1 HM (%)"] == pytest.approx(100 * st.harmonic(0.84, 0.1885))


def test_unseen_pair_table_and_seen_confusion_on_a_known_case():
    crop = np.log(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]))
    disease = np.log(np.array([[0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]]))
    df = frame([0, 1, 2, 1], [0, 1, 1, 1], 2)
    table = pm.unseen_pair_table(crop, disease, df, [0, 1, 2], [0, 1])
    assert table.loc["c0_d0", ["images", "right_crop", "right_disease", "right_pair"]].tolist() == [1, 100.0, 100.0, 100.0]
    assert table.loc["c1_d1"].tolist() == [2, 50.0, 100.0, 50.0]
    assert table.loc["c2_d1", "right_disease"] == 0.0
    allowed = np.ones(6, dtype=bool)
    pairs = [{"crop_id": c, "disease_id": d, "is_seen": c != 2} for c in range(3) for d in range(2)]
    conf, rows_pairs, order = pm.seen_confusion(crop[:3], disease[:3], df.iloc[:3], pairs, [0, 1, 2], [0, 1], allowed)
    assert rows_pairs == [(0, 0), (1, 1), (2, 1)] and order[-2:] == [(2, 0), (2, 1)]
    assert conf.shape == (3, 6) and np.allclose(conf.sum(1), 1.0)
    assert conf[0, order.index((0, 0))] == 1.0 and conf[2, order.index((2, 0))] == 1.0


def test_pair_mask_counts_333_on_the_final_catalogue():
    mask = pm.pair_mask(CAT_FINAL, list(range(55)), list(range(175)))
    assert mask.sum() == 333 and mask.shape == (55 * 175,)
    with pytest.raises(ValueError, match="expected 10"):
        pm.pair_mask(CAT_FINAL, list(range(55)), list(range(175)), expect=10)


def test_check_test_frames_sizes_and_overlaps():
    seen, unseen = frame([0, 1], [0, 1], 2, "s"), frame([1], [0], 2, "u")
    train = frame([0], [0], 2, "t")
    pm.check_test_frames({"seen": seen, "unseen": unseen}, train, (2, 1))
    with pytest.raises(ValueError, match="sizes"):
        pm.check_test_frames({"seen": seen, "unseen": unseen}, train, (2000, 1448))
    with pytest.raises(ValueError, match="overlap"):
        pm.check_test_frames({"seen": seen, "unseen": unseen}, seen.iloc[:1], (2, 1))
    pm.check_test_frames({"seen": seen, "unseen": unseen}, seen.iloc[:1], (2, 1), disjoint=False)
    with pytest.raises(ValueError, match="both"):
        pm.check_test_frames({"seen": seen, "unseen": seen.iloc[:1]}, train, (2, 1), disjoint=False)


def test_check_p5_lock_and_sha256(tmp_path):
    good = {"info": {"crop_winner": "crop_b1", "winner": "disease_clip", "flip": True, "seed": 42}}
    assert pm.check_p5_lock(good) is good["info"]
    for key, value in (("crop_winner", "crop_clip"), ("winner", "disease_small_reg"), ("flip", False), ("seed", 43)):
        bad = {"info": {**good["info"], key: value}}
        with pytest.raises(ValueError, match=key):
            pm.check_p5_lock(bad)
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 1000)
    assert pm.sha256_of(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def toy_dev_world():
    rng = np.random.default_rng(5)
    names = ["a_x", "b_healthy", "c_y", "d_healthy", "e_z", "f_q"]
    cat = Catalogue(comp_id=np.arange(6), crop_id=np.array([0, 1, 2, 3, 4, 5]), disease_id=np.array([0, 1, 2, 1, 3, 4]),
                    is_seen=np.array([True, False, True, False, True, False]), names=names)

    def make(is_seen, n):
        idx = rng.choice(np.flatnonzero(cat.is_seen == is_seen), n)
        crop = rng.normal(size=(n, 6)) * 2
        disease = rng.normal(size=(n, 5)) * 2
        crop[np.arange(n), cat.crop_id[idx]] += 2.5
        disease[np.arange(n), cat.disease_id[idx]] += 1.5
        return pg._part(crop, disease, {"crop_id": cat.crop_id[idx], "disease_id": cat.disease_id[idx], "comp_id": cat.comp_id[idx],
                                        "fn": [f"i{i}" for i in range(n)]})
    return cat, make(True, 80), make(False, 80)


def test_dev_numbers_agree_with_calibration_and_split_out_the_healthy_pairs():
    cat, seen, unseen = toy_dev_world()
    cal = pm.dev_calibration(seen, unseen, cat)
    table = pm.dev_unseen_numbers(unseen, cat, cal, "toy")
    raw, keep, hm = table.iloc[0], table.iloc[1], table.iloc[2]
    assert raw["unseen pair Top-1, all images (%)"] == pytest.approx(cal["A_U_raw"])
    assert keep["unseen pair Top-1, all images (%)"] == pytest.approx(cal["A_U_keep"])
    assert hm["unseen pair Top-1, all images (%)"] == pytest.approx(cal["A_U_hm"])
    assert list(table["healthy pairs left out"]) == [2, 2, 2] and list(table["unseen pairs"]) == [3, 3, 3]
    per = pm.unseen_pair_accuracy(unseen, cat, cal["gamma_keep"], 1.0)
    assert sorted(per.pair) == ["b_healthy", "d_healthy", "f_q"]
    rest = per[~per.healthy]
    assert keep["unseen pair Top-1, without healthy pairs (%)"] == pytest.approx(100 * rest.hits.sum() / rest.images.sum())
    assert keep["pair-macro mean, all pairs (%)"] == pytest.approx(per["accuracy (%)"].mean())
    assert keep["pair-macro mean, without healthy pairs (%)"] == pytest.approx(rest["accuracy (%)"].mean())
    assert (per[per.healthy].pair.str.contains("healthy")).all() and not rest.pair.str.contains("healthy").any()


def test_dev_calibration_refuses_a_non_finite_value(monkeypatch):
    cat, seen, unseen = toy_dev_world()
    nan_seen = {**seen, "crop_logits": seen["crop_logits"] * np.nan}
    with pytest.raises(ValueError, match="not all finite"):
        pm.dev_calibration(nan_seen, unseen, cat)
    good = pm.dev_calibration(seen, unseen, cat)
    assert np.isfinite([good["gamma_keep"], good["gamma_hm"], good["r_hm"]]).all()
    for key, found in (("gamma_keep", np.nan), ("gamma_hm", np.inf), ("r_hm", np.nan)):
        patched = {k: good[k] for k in ("gamma_keep", "gamma_hm", "r_hm")}
        patched[key] = found
        monkeypatch.setattr(pg, "keep_gamma", lambda *a, _p=patched, **k: {"gamma_keep": _p["gamma_keep"]})
        monkeypatch.setattr(pg, "best_hm_r_gamma", lambda *a, _p=patched, **k: {"gamma_hm": _p["gamma_hm"], "r_hm": _p["r_hm"]})
        with pytest.raises(ValueError, match=f"calibration value {key} is .*do not run the test"):
            pm.dev_calibration(seen, unseen, cat)
        monkeypatch.undo()


def test_the_comparison_default_is_the_all_9625_pairs_rule():
    assert pm.ALL_PAIRS == "all 9,625 pairs (Baseline 1 rule)"
    assert 55 * 175 == 9625
    assert inspect.signature(pm.compare_table).parameters["pair_list"].default == pm.ALL_PAIRS
    rows = pm.official_table({s: (np.zeros((2, 4)), np.zeros((2, 5))) for s in pm.SPLITS},
                             {s: frame([0, 1], [0, 1], 5) for s in pm.SPLITS}, [0, 1, 2, 3], [0, 1, 2, 3, 4], None)[1]
    assert rows["pair list"].iloc[0] == pm.ALL_PAIRS


def test_test_csvs_must_be_older_than_the_p5_marker(tmp_path):
    marker = tmp_path / "official_test_started.json"
    seen, unseen = tmp_path / "seen.csv", tmp_path / "unseen.csv"
    for path in (marker, seen, unseen):
        path.write_text("x")
    os.utime(marker, (2_000_000_000, 2_000_000_000))
    os.utime(seen, (1_999_999_000, 1_999_999_000))
    os.utime(unseen, (1_999_999_999.5, 1_999_999_999.5))
    pm.check_test_csvs_predate_marker([seen, unseen], marker)
    os.utime(unseen, (2_000_000_000, 2_000_000_000))
    with pytest.raises(ValueError, match="not older than the P5 marker"):
        pm.check_test_csvs_predate_marker([seen, unseen], marker)
    os.utime(unseen, (2_000_000_100, 2_000_000_100))
    with pytest.raises(ValueError, match="unseen.csv"):
        pm.check_test_csvs_predate_marker([seen, unseen], marker)
    with pytest.raises(FileNotFoundError):
        pm.check_test_csvs_predate_marker([seen, tmp_path / "gone.csv"], marker)


def test_file_time_check_is_described_as_evidence_not_proof():
    doc = pm.check_test_csvs_predate_marker.__doc__
    assert "evidence, not proof" in doc and "has not changed" not in doc


def row_world(rows=60, classes=55, seed=11):
    rng = np.random.default_rng(seed)
    saved = {split: log_softmax(rng.normal(size=(rows, classes)) * 4).astype(np.float32) for split in pm.SPLITS}
    return saved


def test_row_check_passes_identical_and_noisy_rows_and_reports_the_numbers():
    saved = row_world()
    stats = pm.check_rows_match_saved({s: a.copy() for s, a in saved.items()}, saved)
    assert set(stats) == set(pm.SPLITS)
    for numbers in stats.values():
        assert numbers["max"] == 0.0 and numbers["median"] == 0.0 and numbers["p99"] == 0.0 and numbers["rows"] == 60
    rng = np.random.default_rng(1)
    noisy = {s: a + rng.normal(scale=0.01, size=a.shape).astype(np.float32) for s, a in saved.items()}
    stats = pm.check_rows_match_saved(noisy, saved)
    for split, numbers in stats.items():
        per_row = np.abs(noisy[split].astype(np.float64) - saved[split].astype(np.float64)).max(axis=1)
        assert numbers["max"] == pytest.approx(per_row.max()) and numbers["median"] == pytest.approx(np.median(per_row))
        assert numbers["p99"] == pytest.approx(np.percentile(per_row, 99)) and numbers["worst_row"] == int(per_row.argmax())
        assert 0 < numbers["median"] <= numbers["p99"] <= numbers["max"] < 0.2


def test_row_check_limit_is_one_over_the_largest_class_difference():
    assert pm.ROW_MAX_ABS_DIFF == 1.0
    saved = {s: np.zeros((60, 55)) for s in pm.SPLITS}
    edge = {s: a.copy() for s, a in saved.items()}
    edge["seen"][3, 7] = 1.0
    pm.check_rows_match_saved(edge, saved)
    over = {s: a.copy() for s, a in saved.items()}
    over["seen"][3, 7] = 1.01
    with pytest.raises(ValueError, match=r"seen: row 3 .* differs by 1\.0"):
        pm.check_rows_match_saved(over, saved)
    over_unseen = {s: a.copy() for s, a in saved.items()}
    over_unseen["unseen"][41, 54] = -2.5
    with pytest.raises(ValueError, match=r"unseen: row 41 .* differs by 2\.5.*1 of 60 rows"):
        pm.check_rows_match_saved(over_unseen, saved)


def test_row_check_refuses_two_swapped_rows_and_a_different_order():
    saved = row_world()
    swapped = {s: a.copy() for s, a in saved.items()}
    swapped["seen"][[5, 20]] = swapped["seen"][[20, 5]]
    with pytest.raises(ValueError, match=r"seen: row (5|20) of the current test CSV differs by .*2 of 60 rows are over the limit"):
        pm.check_rows_match_saved(swapped, saved)
    shuffled = {s: a[np.random.default_rng(0).permutation(60)] for s, a in saved.items()}
    with pytest.raises(ValueError, match="not in the order P5 used"):
        pm.check_rows_match_saved(shuffled, saved)


def test_row_check_refuses_shape_split_and_non_finite_problems():
    saved = row_world()
    with pytest.raises(ValueError, match=r"seen: recomputed shape \(59, 55\) differs from the saved P5 shape \(60, 55\)"):
        pm.check_rows_match_saved({**saved, "seen": saved["seen"][:59]}, saved)
    with pytest.raises(ValueError, match="shape"):
        pm.check_rows_match_saved({**saved, "unseen": saved["unseen"][:, :54]}, saved)
    with pytest.raises(ValueError, match="splits"):
        pm.check_rows_match_saved({"seen": saved["seen"]}, saved)
    broken = {s: a.copy() for s, a in saved.items()}
    broken["unseen"][2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pm.check_rows_match_saved(broken, saved)


def test_table_check_accepts_the_saved_table_as_a_frame(tmp_path):
    dfs, arrays = random_world(6)
    heads = {s: (arrays["crop_b1", s], arrays["disease_clip", s]) for s in pm.SPLITS}
    _, table = pm.official_table(heads, {"seen": dfs[0], "unseen": dfs[1]}, [0, 1, 2, 3], [0, 1, 2, 3, 4], None)
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    kept = pm.check_results_layout(path)
    path.unlink()
    pm.assert_matches_table(table, kept)
    changed = kept.copy()
    changed.loc[0, "Seen (%)"] += 1
    with pytest.raises(ValueError, match="the saved P5 table"):
        pm.assert_matches_table(table, changed)


def test_pinned_hashes_must_match_exactly():
    pins = {"lock": "a" * 64, "marker": "b" * 64}
    pm.check_pinned_hashes(dict(pins), pins)
    with pytest.raises(ValueError, match=r"\['marker'\]"):
        pm.check_pinned_hashes({"lock": "a" * 64, "marker": "c" * 64}, pins)
    with pytest.raises(ValueError, match="pinned files"):
        pm.check_pinned_hashes({"lock": "a" * 64}, pins)
    with pytest.raises(ValueError, match="pinned files"):
        pm.check_pinned_hashes({**pins, "extra": "d" * 64}, pins)


def test_results_layout_check(tmp_path):
    dfs, arrays = random_world(4)
    heads = {s: (arrays["crop_b1", s], arrays["disease_clip", s]) for s in pm.SPLITS}
    _, table = pm.official_table(heads, {"seen": dfs[0], "unseen": dfs[1]}, [0, 1, 2, 3], [0, 1, 2, 3, 4], None)
    good = tmp_path / "good.csv"
    table.to_csv(good, index=False)
    pm.check_results_layout(good)
    short = tmp_path / "short.csv"
    table.iloc[:-1].to_csv(short, index=False)
    with pytest.raises(ValueError, match="rows are"):
        pm.check_results_layout(short)
    renamed = tmp_path / "renamed.csv"
    table.rename(columns={"HM (%)": "HM"}).to_csv(renamed, index=False)
    with pytest.raises(ValueError, match="columns"):
        pm.check_results_layout(renamed)
    swapped = table.copy()
    swapped.loc[[0, 1], "Task"] = ["Disease", "Disease"]
    swapped.to_csv(tmp_path / "swapped.csv", index=False)
    with pytest.raises(ValueError, match="rows are"):
        pm.check_results_layout(tmp_path / "swapped.csv")
    nan = table.copy()
    nan.loc[2, "Seen (%)"] = np.nan
    nan.to_csv(tmp_path / "nan.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        pm.check_results_layout(tmp_path / "nan.csv")


def test_the_real_p5_results_csv_has_the_layout_and_the_pinned_hash():
    csv = P5_RESULTS / "results" / "official_test_results.csv"
    assert csv.is_file(), csv
    pm.check_results_layout(csv)
    assert pm.sha256_of(csv) == "a9e67f1d7b4821ceafe7e71a130e47b25098567ac87b73b6496d77917bdff1b9"


@pytest.mark.skipif(not HAVE_P5, reason="P5 results folder is missing")
def test_p5_dev_outputs_give_the_p5_lock_gamma_and_the_known_healthy_free_number():
    tf, vs, vu = st.load_dev(SPLITS_DIR)
    cat = load_catalogue(CAT_DEV)
    lock = json.loads((P5_RESULTS / "lock.json").read_text())
    pm.check_p5_lock(lock)
    p5 = pm.read_logprobs(P5_RESULTS / "results" / "dev_flip_logprobs.npz", pm.P5_ARMS.values(), {"seen": len(vs), "unseen": len(vu)})
    parts = tuple(pg.parts_from_logprobs(*pm.member_heads(p5, pm.P5_ARMS)[s], df) for s, df in zip(pm.SPLITS, (vs, vu)))
    cal = pm.dev_calibration(*parts, cat)
    assert cal["gamma_keep"] == pytest.approx(lock["info"]["gamma"], abs=1e-9)
    assert cal["gamma_hm"] == pytest.approx(lock["info"]["gamma_hm"], abs=1e-9)
    table = pm.dev_unseen_numbers(parts[1], cat, cal, "P5")
    assert table.iloc[1]["unseen pair Top-1, without healthy pairs (%)"] == pytest.approx(38.1, abs=0.05)
    assert table.iloc[1]["healthy pairs left out"] == 2


@pytest.mark.skipif(not HAVE_P5, reason="P5 results folder is missing")
def test_dev_probe_of_the_four_members_gives_the_numbers_the_user_approved_p6_on():
    tf, vs, vu = st.load_dev(SPLITS_DIR)
    cat = load_catalogue(CAT_DEV)
    p5 = pm.read_logprobs(P5_RESULTS / "results" / "dev_flip_logprobs.npz", pm.P5_ARMS.values(), {"seen": len(vs), "unseen": len(vu)})
    new = {}
    for arm in pm.NEW_ARMS.values():
        with np.load(P5_RESULTS / "results" / f"dev_best_validation_{arm}.npz") as z:
            new.update({(arm, s): z[s] for s in pm.SPLITS})
    parts = pm.p6_parts(p5, new, (vs, vu))
    cal = pm.dev_calibration(*parts, cat)
    table = pm.dev_unseen_numbers(parts[1], cat, cal, "P6 probe")
    assert table.iloc[0]["unseen pair Top-1, all images (%)"] == pytest.approx(10.5, abs=0.05)
    assert table.iloc[1]["unseen pair Top-1, without healthy pairs (%)"] == pytest.approx(47.9, abs=0.05)
