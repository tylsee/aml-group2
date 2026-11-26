import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cdccp.calibration as p4
from cdccp.bn import recompute_bn
from cdccp.catalogue import Catalogue, load_catalogue
from cdccp.checkpoint import load_checkpoint, save_checkpoint, training_meta
from cdccp.config import REPO, image_dir, load_config
from cdccp.data import COLUMNS, eval_loader, load_split
from cdccp.evaluate import harmonic, pair_breakdown
from cdccp.extract import extract
from cdccp.models import build_model

P2_VAL = Path(r"D:\dev\uni\AML\runs\p2e\runs\p2c_plain_seed42\val_10.npz")
DATA_ROOT = Path(r"D:\dev\uni\AML\data\DPD_Small")
CAT_DEV = REPO / "data" / "mappings" / "catalogue_dev.json"
CAT_FINAL = REPO / "data" / "mappings" / "catalogue_final.json"


def toy_cat():
    return Catalogue(comp_id=np.array([0, 1, 2, 3]), crop_id=np.array([0, 0, 1, 1]), disease_id=np.array([0, 1, 0, 1]),
                     is_seen=np.array([True, False, False, True]), names=["a", "b", "c", "d"])


def part(crop, disease, crop_id, disease_id, comp_id, fn=None):
    n = len(crop_id)
    return {"crop_logits": np.asarray(crop, dtype=float), "disease_logits": np.asarray(disease, dtype=float),
            "crop_id": np.asarray(crop_id, dtype=np.int64), "disease_id": np.asarray(disease_id, dtype=np.int64),
            "comp_id": np.asarray(comp_id, dtype=np.int64), "fn": np.array(fn or [f"f{i}.jpg" for i in range(n)])}


def toy_parts():
    seen = part([[5, 0], [0, 5]], [[5, 0], [0, 5]], [0, 1], [0, 1], [0, 3])
    unseen = part([[5, 0], [0, 5]], [[2, 0], [3, 0]], [0, 1], [1, 0], [1, 2])
    return seen, unseen


def random_parts(rng, cat, n_seen, n_unseen, n_crop, n_disease):
    def make(is_seen, n):
        idx = rng.choice(np.flatnonzero(cat.is_seen == is_seen), n)
        return part(rng.normal(size=(n, n_crop)) * 3, rng.normal(size=(n, n_disease)) * 3, cat.crop_id[idx], cat.disease_id[idx], cat.comp_id[idx])
    return make(True, n_seen), make(False, n_unseen)


def test_keep_gamma_equals_pair_breakdown_on_synthetic():
    rng = np.random.default_rng(42)
    cat = toy_cat()
    seen, unseen = random_parts(rng, cat, 60, 60, 2, 2)
    got = p4.keep_gamma(seen, unseen, cat)
    expect = pair_breakdown(seen, unseen, cat, tol=3.0, r=1.0)
    for key, value in expect.items():
        assert got[key] == pytest.approx(value, nan_ok=True)
    assert got["H_keep"] == pytest.approx(harmonic(expect["A_S_keep"], expect["A_U_keep"]))


@pytest.mark.skipif(not P2_VAL.is_file(), reason="P2 dev output is missing")
def test_keep_gamma_on_real_p2_file():
    seen, unseen = p4.dev_parts_p2(P2_VAL)
    cat = load_catalogue(CAT_DEV)
    got = p4.keep_gamma(seen, unseen, cat)
    print({k: round(v, 3) for k, v in got.items()})
    expect = pair_breakdown(seen, unseen, cat)
    for key, value in expect.items():
        assert got[key] == pytest.approx(value, nan_ok=True)
    assert got["H_keep"] == pytest.approx(56.3, abs=0.1)
    assert got["A_S_keep"] == pytest.approx(74.0, abs=0.1)
    assert got["A_U_keep"] == pytest.approx(45.5, abs=0.1)
    assert got["gamma_keep"] == pytest.approx(11.3, abs=0.1)


def test_ensemble_identical_members_give_same_scores():
    rng = np.random.default_rng(1)
    cat = toy_cat()
    seen, unseen = random_parts(rng, cat, 8, 8, 2, 2)
    out = p4.ensemble([seen, seen, seen])
    lp = p4._log_softmax
    np.testing.assert_allclose(out["crop_logits"], lp(seen["crop_logits"]), atol=1e-12)
    np.testing.assert_allclose(out["disease_logits"], lp(seen["disease_logits"]), atol=1e-12)
    assert list(out["fn"]) == list(seen["fn"])


def test_ensemble_weighted_mean_of_log_probs():
    rng = np.random.default_rng(2)
    a, b = part(rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), [0, 1, 0], [0, 1, 1], [0, 3, 1]), None
    b = {**a, "crop_logits": rng.normal(size=(3, 2)), "disease_logits": rng.normal(size=(3, 2))}
    out = p4.ensemble([a, b], weights=[0.25, 0.75])
    np.testing.assert_allclose(out["crop_logits"], 0.25 * p4._log_softmax(a["crop_logits"]) + 0.75 * p4._log_softmax(b["crop_logits"]))


def test_ensemble_rejects_bad_members_and_weights():
    seen, _ = toy_parts()
    other_fn = {**seen, "fn": np.array(["x.jpg", "y.jpg"])}
    other_comp = {**seen, "comp_id": np.array([3, 0])}
    other_crop = {**seen, "crop_id": np.array([1, 0])}
    other_disease = {**seen, "disease_id": np.array([1, 0])}
    for bad in (other_fn, other_comp, other_crop, other_disease):
        with pytest.raises(ValueError):
            p4.ensemble([seen, bad])
    with pytest.raises(ValueError):
        p4.ensemble([seen, seen], weights=[1.0])
    with pytest.raises(ValueError):
        p4.ensemble([seen, seen], weights=[0.6, 0.6])
    with pytest.raises(ValueError):
        p4.ensemble([seen, seen], weights=[1.5, -0.5])
    with pytest.raises(ValueError):
        p4.ensemble([])


def test_family_ensemble_is_half_and_half_across_groups():
    rng = np.random.default_rng(3)
    base = part(np.zeros((4, 3)), np.zeros((4, 5)), [0, 1, 2, 0], [0, 1, 2, 3], [0, 1, 2, 3])
    members = [{**base, "crop_logits": rng.normal(size=(4, 3)), "disease_logits": rng.normal(size=(4, 5))} for _ in range(4)]
    p4_member, seeds = members[0], members[1:]
    out = p4.family_ensemble([[p4_member], seeds])
    lp = p4._log_softmax
    for head in ("crop_logits", "disease_logits"):
        expect = 0.5 * lp(p4_member[head]) + 0.5 * np.mean([lp(m[head]) for m in seeds], axis=0)
        np.testing.assert_allclose(out[head], expect, atol=1e-12)
    pair = p4.family_ensemble([[(base, base)], [(base, base), (base, base)]])
    assert isinstance(pair, tuple) and len(pair) == 2
    with pytest.raises(ValueError):
        p4.family_ensemble([[p4_member], []])




def write_split(directory, name, rows):
    frame = pd.DataFrame([(f"{name}_{i}.jpg", c, d, 10 * c + d, f"crop{c}", f"dis{d}", f"c{c}d{d}") for i, (c, d) in enumerate(rows)], columns=COLUMNS)
    frame.to_csv(directory / f"{name}.csv", index=False)


def toy_stage1(tmp_path, fit=((0, 0), (1, 1), (2, 2), (0, 3)), seen=((0, 0), (1, 1)), unseen=((0, 1), (1, 2))):
    splits = tmp_path / "splits"
    splits.mkdir()
    write_split(splits, "train_fit", fit)
    write_split(splits, "val_seen", seen)
    write_split(splits, "val_unseen", unseen)
    rng = np.random.default_rng(4)
    stage1 = tmp_path / "stage1"
    arrays = {}
    for arm, n_classes in (("crop_b1", 3), ("disease_b1", 4)):
        (stage1 / arm).mkdir(parents=True)
        arrays[arm] = {"seen": rng.normal(size=(2, n_classes)), "unseen": rng.normal(size=(2, n_classes))}
        np.savez(stage1 / arm / "best_validation.npz", **arrays[arm])
    return splits, stage1, arrays


def test_dev_parts_p4_reads_arms_in_csv_order(tmp_path):
    splits, stage1, arrays = toy_stage1(tmp_path)
    seen, unseen = p4.dev_parts_p4(stage1, "crop_b1", "disease_b1", splits, expect_sizes=(4, 2, 2), expect_classes=(3, 4))
    np.testing.assert_allclose(seen["crop_logits"], p4._log_softmax(arrays["crop_b1"]["seen"]))
    np.testing.assert_allclose(unseen["disease_logits"], p4._log_softmax(arrays["disease_b1"]["unseen"]))
    assert list(seen["fn"]) == ["val_seen_0.jpg", "val_seen_1.jpg"]
    assert list(unseen["comp_id"]) == [1, 12]
    assert list(unseen["crop_id"]) == [0, 1] and list(unseen["disease_id"]) == [1, 2]


def test_dev_parts_p4_rejects_non_position_ids(tmp_path):
    splits, stage1, _ = toy_stage1(tmp_path, fit=((1, 0), (2, 1), (3, 2), (1, 3)), seen=((1, 0), (2, 1)), unseen=((1, 1), (2, 2)))
    with pytest.raises(ValueError):
        p4.dev_parts_p4(stage1, "crop_b1", "disease_b1", splits, expect_sizes=(4, 2, 2), expect_classes=(3, 4))


def test_dev_parts_p4_rejects_wrong_shape(tmp_path):
    splits, stage1, _ = toy_stage1(tmp_path)
    np.savez(stage1 / "crop_b1" / "best_validation.npz", seen=np.zeros((3, 3)), unseen=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        p4.dev_parts_p4(stage1, "crop_b1", "disease_b1", splits, expect_sizes=(4, 2, 2), expect_classes=(3, 4))


def test_test_metrics_matches_hand_computation():
    cat = toy_cat()
    seen, unseen = toy_parts()
    result = p4.test_metrics(seen, unseen, cat, gamma=3.0)
    raw, cal, full = result["locked_raw"], result["locked_calibrated"], result["full_raw"]
    assert raw["gamma"] == 0.0 and cal["gamma"] == 3.0 and full["gamma"] == 0.0
    assert raw["seen"]["comp_top1"] == 100 and raw["unseen"]["comp_top1"] == 50
    assert raw["combined"]["H"] == pytest.approx(2 * 100 * 50 / 150)
    assert raw["combined"]["gap"] == 50
    assert cal["seen"]["comp_top1"] == 100 and cal["unseen"]["comp_top1"] == 100
    assert cal["combined"]["H"] == 100
    assert full["combined"] == raw["combined"]
    assert raw["unseen"]["crop_top1"] == 100 and raw["unseen"]["disease_top1"] == 50
    assert result["scores"]["seen"].shape == (2, 4) and result["scores"]["unseen"].shape == (2, 4)
    assert np.argmax(result["scores"]["unseen"][0]) == 1
    rows = p4.metrics_rows("toy", result)
    assert len(rows) == 3 * (6 + 6 + 4)
    assert {r["catalogue"] for r in rows} == set(p4.BLOCKS)
    assert {"model", "catalogue", "gamma", "split", "metric", "value"} == set(rows[0])
    h = [r for r in rows if r["catalogue"] == "locked_raw" and r["metric"] == "H"]
    assert len(h) == 1 and h[0]["split"] == "combined" and h[0]["value"] == pytest.approx(200 / 3)
    a_s_cal = [r for r in rows if r["catalogue"] == "locked_calibrated" and r["metric"] == "A_S"]
    assert a_s_cal[0]["gamma"] == 3.0


def test_test_metrics_full_catalogue_has_9625_columns(monkeypatch):
    cat = load_catalogue(CAT_FINAL)
    rng = np.random.default_rng(5)
    seen, unseen = random_parts(rng, cat, 12, 12, 55, 175)
    shapes = []
    real = p4.score_split_pair

    def spy(scorer, s, u, catalogue, gamma):
        out = real(scorer, s, u, catalogue, gamma)
        shapes.append(out[1].shape)
        return out

    monkeypatch.setattr(p4, "score_split_pair", spy)
    result = p4.test_metrics(seen, unseen, cat, gamma=2.0)
    assert len(cat) == 333 and int(cat.is_seen.sum()) == 323
    assert shapes == [(12, 333), (12, 333), (12, 9625)]
    assert result["locked_calibrated"]["gamma"] == 2.0


def test_calibrated_scores_match_test_metrics_scores():
    cat = toy_cat()
    parts = toy_parts()
    s, u = p4.calibrated_scores(parts, cat, 3.0)
    result = p4.test_metrics(*parts, cat, 3.0)
    np.testing.assert_array_equal(s, result["scores"]["seen"])
    np.testing.assert_array_equal(u, result["scores"]["unseen"])


def test_pick_candidate_tie_rules():
    assert p4.pick_candidate({"A": 50.0, "B": 55.0, "C": 52.0}) == "B"
    assert p4.pick_candidate({"A": 50.0, "B": 50.0, "C": 50.0}) == "A"
    assert p4.pick_candidate({"A": 49.0, "B": 50.0, "C": 50.0}) == "B"
    assert p4.pick_candidate({"A": 49.0, "B": 50.0, "C": 51.0}) == "C"
    assert p4.pick_candidate({"A": 51.0, "B": 50.0, "C": 51.0}) == "A"
    with pytest.raises(ValueError):
        p4.pick_candidate({"A": 1.0, "B": 2.0})


@pytest.mark.skipif(not (DATA_ROOT / "DPD_Small_dataset").is_dir(), reason="image folder is missing")
def test_p2_test_parts_matches_direct_bn_and_extract(tmp_path):
    cfg = load_config(overrides=[f"paths.data_root={DATA_ROOT.as_posix()}", "loader.num_workers=0", "loader.eval_batch=8"])
    run_cfg = load_config(REPO / "configs" / "p1.yaml", ["pretrained=False", "loader.num_workers=0", "loader.eval_batch=8", "bn.subset=6", "bn.batch=8",
                                                          "amp=False", "model.crop_classes=3", "model.disease_classes=4"])
    run_cfg["name"] = "tiny"
    run_cfg = json.loads(json.dumps(run_cfg))
    train_df = pd.read_csv(DATA_ROOT / "DPD_Small_train.csv", header=None, names=COLUMNS)
    seen_df, unseen_df = train_df.iloc[:5].reset_index(drop=True), train_df.iloc[5:9].reset_index(drop=True)
    assert image_dir(cfg).is_dir()

    torch.manual_seed(42)
    model = build_model(run_cfg, pretrained=False)
    path = tmp_path / "stage2_final.pt"
    save_checkpoint(path, model, {**training_meta(train_df), "stage": 2, "config": run_cfg, "seed": run_cfg["seed"]})

    device = torch.device("cpu")
    got_seen, got_unseen = p4.p2_test_parts(path, seen_df, unseen_df, cfg, device, use_cache=False)

    meta = load_checkpoint(path)
    direct_cfg = {**meta["config"], "paths": cfg["paths"], "loader": cfg["loader"]}
    direct = build_model(direct_cfg, pretrained=False)
    load_checkpoint(path, direct)
    bn_df = load_split(direct_cfg, "train").sample(min(direct_cfg["bn"]["subset"], len(train_df)), random_state=direct_cfg["seed"]).sort_values("filename")
    recompute_bn(direct, eval_loader(bn_df, direct_cfg, direct_cfg["bn"]["batch"], use_cache=False), device)
    for got, df in ((got_seen, seen_df), (got_unseen, unseen_df)):
        want = extract(direct, df, direct_cfg, device, False)
        assert got["crop_logits"].shape == (len(df), 3) and got["disease_logits"].shape == (len(df), 4)
        assert list(got["fn"]) == list(df.filename)
        np.testing.assert_array_equal(got["comp_id"], df.composition_id.values)
        np.testing.assert_allclose(got["crop_logits"], p4._log_softmax(want["crop_logits"]), atol=1e-6)
        np.testing.assert_allclose(got["disease_logits"], p4._log_softmax(want["disease_logits"]), atol=1e-6)
        np.testing.assert_allclose(np.exp(got["crop_logits"]).sum(axis=1), 1.0, atol=1e-9)


def _rows_by(rows):
    return {(r["Test split"], r["Model"]): (r["Crop-only Top-1"], r["Disease-only Top-1"], r["Pair Top-1"]) for r in rows}


def test_topline_rows_hand_computed_toy():
    # Unseen row 0 has the right crop but the disease head picks disease 0, so at gamma 0 the top pair is the seen
    # pair (0, 0). Taking 10 off every seen pair moves it to the unseen pair (0, 1), which is right.
    # Seen rows (disease head 6 vs 0, crop head 5 vs 0, so no ties): after gamma 10 row 0 scores the unseen pairs
    # (0, 1) at -6.009 and (1, 0) at -5.009, so it names (1, 0): wrong crop, right disease. Row 1 names (0, 1) the
    # same way. So the calibrated seen row is crop 0, disease 100, pair 0.
    _, unseen = toy_parts()
    seen = part([[5, 0], [0, 5]], [[6, 0], [0, 6]], [0, 1], [0, 1], [0, 3])
    rows = p4.topline_rows(seen, unseen, toy_cat(), gamma=10.0)
    got = _rows_by(rows)
    assert len(got) == 8
    assert got["Seen pairs", "Raw, heads only"] == (100, 100, 100)
    assert got["Unseen pairs", "Raw, heads only"] == (100, 50, 50)
    assert got["Seen pairs", "Raw, pair list"] == (100, 100, 100)
    assert got["Unseen pairs", "Raw, pair list"] == (100, 50, 50)
    assert got["Seen pairs", "Calibrated, pair list"] == (0, 100, 0)
    assert got["Unseen pairs", "Calibrated, pair list"] == (100, 100, 100)
    assert got["Seen pairs", "Reference: unseen pairs only"] == (0, 100, 0)
    assert got["Unseen pairs", "Reference: unseen pairs only"] == (100, 100, 100)
    gammas = {r["Model"]: r["gamma"] for r in rows}
    assert gammas == {"Raw, heads only": 0.0, "Raw, pair list": 0.0, "Calibrated, pair list": 10.0,
                      "Reference: unseen pairs only": p4.TRIVIAL_GAMMA}


def test_topline_rows_break_ties_like_composition_metrics():
    # Catalogue order is NOT (crop, disease) order: index 0 is (1, 1). Every pair ties, so a tie rule by catalogue
    # index would name (1, 1); composition_metrics (cell 11) names the lowest (crop, disease), which is (0, 0).
    from cdccp.single_task import composition_metrics
    cat = Catalogue(comp_id=np.array([0, 1, 2, 3]), crop_id=np.array([1, 0, 1, 0]), disease_id=np.array([1, 1, 0, 0]),
                    is_seen=np.array([True, False, False, True]), names=["a", "b", "c", "d"])
    seen = part([[0, 0]], [[0, 0]], [0], [0], [3])
    unseen = part([[0, 0]], [[0, 0]], [0], [1], [1])
    got = _rows_by(p4.topline_rows(seen, unseen, cat, gamma=0.0))
    allowed = np.zeros(4, dtype=bool)
    allowed[cat.crop_id * 2 + cat.disease_id] = True
    for split, p in (("Seen pairs", seen), ("Unseen pairs", unseen)):
        ref = composition_metrics(p["crop_logits"], p["disease_logits"], p["crop_id"], p["disease_id"], allowed)
        assert got[split, "Raw, pair list"][2] == pytest.approx(100 * ref["composition_top1"])
    assert got["Seen pairs", "Raw, pair list"] == (100, 100, 100)
    assert got["Unseen pairs", "Raw, pair list"] == (100, 0, 0)


def test_topline_rows_agree_with_pair_breakdown_on_real_catalogue():
    cat = load_catalogue(CAT_FINAL)
    rng = np.random.default_rng(7)
    seen, unseen = random_parts(rng, cat, 300, 300, 55, 175)
    pb = pair_breakdown(seen, unseen, cat)
    got = _rows_by(p4.topline_rows(seen, unseen, cat, gamma=pb["gamma_keep"]))
    assert got["Unseen pairs", "Raw, heads only"][0] == pytest.approx(pb["crop"])
    assert got["Unseen pairs", "Raw, heads only"][1] == pytest.approx(pb["disease"])
    assert got["Unseen pairs", "Raw, heads only"][2] == pytest.approx(pb["both_heads"])
    assert got["Seen pairs", "Raw, pair list"][2] == pytest.approx(pb["A_S_raw"])
    assert got["Unseen pairs", "Raw, pair list"][2] == pytest.approx(pb["A_U_raw"])
    assert got["Seen pairs", "Calibrated, pair list"][2] == pytest.approx(pb["A_S_keep"])
    assert got["Unseen pairs", "Calibrated, pair list"][2] == pytest.approx(pb["A_U_keep"])


def test_parts_from_logprobs_maps_columns_and_normalises():
    df = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "crop_id": [1, 0], "disease_id": [0, 2], "composition_id": [7, 9]})
    got = p4.parts_from_logprobs(np.array([[0.0, 2.0], [3.0, 1.0]]), np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 5.0]]), df)
    assert set(got) == set(p4.PART_KEYS)
    assert got["crop_id"].tolist() == [1, 0] and got["disease_id"].tolist() == [0, 2] and got["comp_id"].tolist() == [7, 9]
    assert got["fn"].tolist() == ["a.jpg", "b.jpg"]
    assert np.allclose(np.exp(got["crop_logits"]).sum(1), 1) and np.allclose(np.exp(got["disease_logits"]).sum(1), 1)
    assert np.allclose(got["disease_logits"][0], np.log(1 / 3))


def test_topline_rows_refuse_non_finite_gamma_or_scores():
    seen, unseen = toy_parts()
    with pytest.raises(ValueError, match="gamma must be finite"):
        p4.topline_rows(seen, unseen, toy_cat(), gamma=float("nan"))
    bad = dict(unseen, disease_logits=unseen["disease_logits"].copy())
    bad["disease_logits"][0, 0] = np.nan
    with pytest.raises(ValueError, match="unseen scores are not all finite"):
        p4.topline_rows(seen, bad, toy_cat(), gamma=1.0)


def test_load_arm_refuses_non_finite_dev_outputs(tmp_path):
    path = tmp_path / "best_validation.npz"
    seen = np.zeros((4, 3))
    seen[1, 2] = np.nan
    np.savez(path, seen=seen, unseen=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="not all finite"):
        p4._load_arm(path, 4, 2, 3, "disease_b1")


def test_best_hm_gamma_is_the_best_grid_point_and_beats_keep_seen():
    cat = load_catalogue(CAT_FINAL)
    seen, unseen = random_parts(np.random.default_rng(3), cat, 300, 300, 55, 175)
    got = p4.best_hm_gamma(seen, unseen, cat)
    pb = pair_breakdown(seen, unseen, cat)
    assert got["H_hm"] >= harmonic(pb["A_S_keep"], pb["A_U_keep"]) - 1e-9
    assert got["H_hm"] == pytest.approx(harmonic(got["A_S_hm"], got["A_U_hm"]))
    rows = _rows_by(p4.topline_rows(seen, unseen, cat, gamma=pb["gamma_keep"], gamma_hm=got["gamma_hm"]))
    assert len(rows) == 10
    assert rows["Seen pairs", "Calibrated (best dev HM), pair list"][2] == pytest.approx(got["A_S_hm"])
    assert rows["Unseen pairs", "Calibrated (best dev HM), pair list"][2] == pytest.approx(got["A_U_hm"])


def test_best_hm_gamma_toy_and_tie_goes_to_smaller_gamma():
    # The result must be the highest-H point of the same grid, and the smallest gamma among equal-H points.
    seen, unseen = toy_parts()
    got = p4.best_hm_gamma(seen, unseen, toy_cat())
    ss = p4.head_score(seen, toy_cat())
    su = p4.head_score(unseen, toy_cat())
    rows = p4.sweep(ss, seen, su, unseen, toy_cat(), p4.gamma_grid(ss, su, toy_cat()))
    best_h = rows[:, 3].max()
    assert got["H_hm"] == pytest.approx(best_h)
    assert got["gamma_hm"] == pytest.approx(rows[rows[:, 3] == best_h, 0].min())


def test_topline_rows_reject_non_finite_gamma_hm():
    seen, unseen = toy_parts()
    with pytest.raises(ValueError, match="gamma_hm must be finite"):
        p4.topline_rows(seen, unseen, toy_cat(), gamma=1.0, gamma_hm=float("inf"))


def test_calibration_breaks_ties_like_the_reported_table():
    # Catalogue order puts the seen pair first, but the unseen pair has the lower (crop id, disease id).
    # The unseen row ties both pairs at gamma 0; the table names the unseen pair, so gamma 0 already gives HM 100.
    cat = Catalogue(comp_id=np.array([0, 1]), crop_id=np.array([1, 0]), disease_id=np.array([1, 0]),
                    is_seen=np.array([True, False]), names=["seen", "unseen"])
    seen = part([[0, 5]], [[0, 5]], [1], [1], [0])
    unseen = part([[0, 0]], [[0, 0]], [0], [0], [1])
    hm = p4.best_hm_gamma(seen, unseen, cat)
    assert (hm["gamma_hm"], hm["A_S_hm"], hm["A_U_hm"], hm["H_hm"]) == (0.0, 100.0, 100.0, 100.0)
    keep = p4.keep_gamma(seen, unseen, cat)
    assert (keep["gamma_keep"], keep["A_S_raw"], keep["A_U_raw"], keep["A_U_keep"]) == (0.0, 100.0, 100.0, 100.0)
    rows = _rows_by(p4.topline_rows(seen, unseen, cat, gamma=keep["gamma_keep"], gamma_hm=hm["gamma_hm"]))
    for model in ("Raw, pair list", "Calibrated, pair list", "Calibrated (best dev HM), pair list"):
        assert rows["Seen pairs", model][2] == 100.0 and rows["Unseen pairs", model][2] == 100.0


def r_toy_parts():
    # Pairs of toy_cat: 0 (0,0) seen, 1 (0,1) unseen, 2 (1,0) unseen, 3 (1,1) seen. With two classes per head, a score
    # gap between pairs is a logit gap. Seen image A (truth pair 0): crop gap 2, disease gap 8. Unseen image B (truth
    # pair 1): crop gap 4, disease gap 3 toward the wrong disease.
    # At gamma g and weight r, A stays on pair 0 iff g < min(8r, 2). B moves to pair 1 iff g > 3r.
    # Both hold only for 3r < min(8r, 2), i.e. r < 2/3, so among the grid only r = 0.5 reaches HM 100.
    seen = part([[2, 0]], [[8, 0]], [0], [0], [0])
    unseen = part([[4, 0]], [[3, 0]], [0], [1], [1])
    return seen, unseen


def test_best_hm_r_gamma_picks_r_that_beats_r_one():
    seen, unseen = r_toy_parts()
    cat = toy_cat()
    at_one = p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(1.0,))
    assert (at_one["r_hm"], at_one["gamma_hm"], at_one["A_S_hm"], at_one["A_U_hm"], at_one["H_hm"]) == (1.0, 0.0, 100.0, 0.0, 0.0)
    got = p4.best_hm_r_gamma(seen, unseen, cat)
    assert got["r_hm"] == 0.5
    assert (got["A_S_hm"], got["A_U_hm"], got["H_hm"]) == (100.0, 100.0, 100.0)
    assert 1.5 < got["gamma_hm"] < 2.0
    assert p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(0.75, 1.0, 2.0))["H_hm"] == 0.0


def test_topline_rows_use_r_hm_only_on_the_hm_row():
    seen, unseen = r_toy_parts()
    cat = toy_cat()
    rows = p4.topline_rows(seen, unseen, cat, gamma=0.0, gamma_hm=1.75, r_hm=0.5)
    pair = {(r["Test split"], r["Model"]): r["Pair Top-1"] for r in rows}
    assert pair["Seen pairs", "Calibrated (best dev HM), pair list"] == 100.0
    assert pair["Unseen pairs", "Calibrated (best dev HM), pair list"] == 100.0
    assert {r["Model"]: r["r"] for r in rows} == {
        "Raw, heads only": 1.0, "Raw, pair list": 1.0, "Calibrated, pair list": 1.0,
        "Calibrated (best dev HM), pair list": 0.5, "Reference: unseen pairs only": 1.0}
    same_r = {(r["Test split"], r["Model"]): r["Pair Top-1"] for r in p4.topline_rows(seen, unseen, cat, gamma=0.0, gamma_hm=1.75)}
    assert same_r["Seen pairs", "Calibrated (best dev HM), pair list"] == 100.0
    assert same_r["Unseen pairs", "Calibrated (best dev HM), pair list"] == 0.0
    shifted = p4.topline_rows(seen, unseen, cat, gamma=0.0, r=1.5)
    assert {r["r"] for r in shifted} == {1.5}


def test_best_hm_r_gamma_ties_go_to_smallest_abs_log_r():
    # The disease head is flat, so r moves no score gap: every r reaches the same HM at gamma 0.
    cat = toy_cat()
    seen = part([[5, 0]], [[0, 0]], [0], [0], [0])
    unseen = part([[0, 5]], [[0, 0]], [1], [0], [2])
    assert p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(0.5, 1.0, 2.0))["r_hm"] == 1.0
    assert p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(2.0, 1.0, 0.5))["r_hm"] == 1.0
    got = p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(0.75, 1.25))
    assert got["r_hm"] == 1.25 and abs(np.log(1.25)) < abs(np.log(0.75))
    assert got["H_hm"] == 100.0 and got["gamma_hm"] == 0.0
    assert p4.best_hm_r_gamma(seen, unseen, cat)["r_hm"] == 1.0


def test_best_hm_r_gamma_with_r_one_equals_best_hm_gamma_on_real_catalogue():
    cat = load_catalogue(CAT_FINAL)
    seen, unseen = random_parts(np.random.default_rng(11), cat, 150, 150, 55, 175)
    want = p4.best_hm_gamma(seen, unseen, cat)
    got = p4.best_hm_r_gamma(seen, unseen, cat, r_grid=(1.0,))
    assert got == {"r_hm": 1.0, **want}
    full = p4.best_hm_r_gamma(seen, unseen, cat)
    assert full["r_hm"] in p4.R_GRID and full["H_hm"] >= want["H_hm"]


def test_topline_hm_row_matches_best_hm_r_gamma_on_real_catalogue():
    cat = load_catalogue(CAT_FINAL)
    seen, unseen = random_parts(np.random.default_rng(12), cat, 120, 120, 55, 175)
    best = p4.best_hm_r_gamma(seen, unseen, cat)
    rows = p4.topline_rows(seen, unseen, cat, gamma=1.0, gamma_hm=best["gamma_hm"], r_hm=best["r_hm"])
    hm = {r["Test split"]: r for r in rows if r["Model"] == "Calibrated (best dev HM), pair list"}
    assert hm["Seen pairs"]["Pair Top-1"] == pytest.approx(best["A_S_hm"])
    assert hm["Unseen pairs"]["Pair Top-1"] == pytest.approx(best["A_U_hm"])
    assert hm["Seen pairs"]["r"] == best["r_hm"] == hm["Unseen pairs"]["r"]
    assert {r["r"] for r in rows if r["Model"] != "Calibrated (best dev HM), pair list"} == {1.0}


def test_best_hm_r_gamma_and_topline_reject_bad_r_and_scores():
    seen, unseen = r_toy_parts()
    cat = toy_cat()
    for bad in ((), (0.0,), (-1.0,), (1.0, float("nan")), (float("inf"),)):
        with pytest.raises(ValueError, match="r_grid"):
            p4.best_hm_r_gamma(seen, unseen, cat, r_grid=bad)
    broken = dict(unseen, crop_logits=unseen["crop_logits"].copy())
    broken["crop_logits"][0, 0] = np.inf
    with pytest.raises(ValueError, match="unseen scores are not all finite"):
        p4.best_hm_r_gamma(seen, broken, cat)
    for bad_r in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="r_hm must be finite and positive"):
            p4.topline_rows(seen, unseen, cat, gamma=1.0, gamma_hm=1.0, r_hm=bad_r)


def test_best_hm_r_gamma_reciprocal_r_tie_goes_to_smaller_r_before_smaller_gamma():
    # r 0.5 and r 2.0 have the same |log r|. Here both reach H 40, and r 2.0 does it at the smaller gamma.
    # The pre-registered order is |log r|, then the smaller r, then the smaller gamma, so r 0.5 must win.
    cat = toy_cat()
    rng = np.random.default_rng(4)

    def make(ids, n):
        idx = rng.choice(ids, n)
        return part(rng.integers(-3, 4, (n, 2)), rng.integers(-3, 4, (n, 2)), cat.crop_id[idx], cat.disease_id[idx], cat.comp_id[idx])

    seen, unseen = make(np.flatnonzero(cat.is_seen), 6), make(np.flatnonzero(~cat.is_seen), 6)
    half, double = p4.best_hm_r_gamma(seen, unseen, cat, (0.5,)), p4.best_hm_r_gamma(seen, unseen, cat, (2.0,))
    assert half["H_hm"] == double["H_hm"] == pytest.approx(40.0)
    assert double["gamma_hm"] < half["gamma_hm"]
    for grid in ((0.5, 2.0), (2.0, 0.5)):
        assert p4.best_hm_r_gamma(seen, unseen, cat, grid) == half
