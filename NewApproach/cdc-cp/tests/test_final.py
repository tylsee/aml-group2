"""The lock and the final test, on tiny untrained models and FAKE official CSVs written to tmp folders.

The fake CSVs reuse real image files from the development splits with invented labels. The real official test CSVs are never opened.
"""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

import cdccp.final as final
from cdccp.catalogue import Catalogue, load_catalogue
from cdccp.checkpoint import save_checkpoint, training_meta
from cdccp.config import REPO, image_dir, load_config
from cdccp.data import COLUMNS
from cdccp.evaluate import Refused
from cdccp.final import (full_catalogue, gammas_from_boards, load_official, make_lock, score_split_pair, Scorer, verify_lock,
                         with_full_ids)
from cdccp.models import build_model


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


final_script = load_script("final_test")
make_lock_script = load_script("make_lock")

SEEN = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 0), (2, 1)]
UNSEEN = [(0, 1), (1, 2), (2, 3)]
EXTRA_UNSEEN = [(2, 0)]
REAL_IMAGES = image_dir(load_config())


def comp(c, d):
    return 100 + c * 4 + d


def val_names(name):
    return list(pd.read_csv(REPO / "data" / "splits_v2" / f"{name}.csv").filename)


def run_config(extra=()):
    cfg = load_config(REPO / "configs" / "p1.yaml", ["pretrained=False", "loader.num_workers=0", "loader.eval_batch=8", "bn.subset=6", "bn.batch=8",
                                                      "amp=False", *extra])
    cfg["name"] = "fake"
    return json.loads(json.dumps(cfg))


def machine_cfg(data_root):
    cfg = load_config(overrides=["loader.num_workers=0", "loader.eval_batch=8"])
    cfg["paths"]["data_root"] = str(data_root)
    cfg["paths"]["image_subdir"] = str(REAL_IMAGES)
    cfg["paths"]["catalogue_final"] = str(data_root / "does_not_exist.json")
    return cfg


def machine_overrides(data_root):
    return [f"paths.data_root={Path(data_root).as_posix()}", f"paths.image_subdir={Path(REAL_IMAGES).as_posix()}", "loader.num_workers=0",
            "loader.eval_batch=8", "paths.catalogue_final=nowhere.json"]


def fake_frame(files, labels):
    rows = [(f, c, d, k, f"crop{c}", f"disease{d}", f"crop{c}_disease{d}") for f, (c, d, k) in zip(files, labels)]
    return pd.DataFrame(rows, columns=COLUMNS)


def write_csvs(data_root, train, test_seen, test_unseen):
    data_root.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("test_seen", test_seen), ("test_unseen", test_unseen)):
        df.to_csv(data_root / f"DPD_Small_{name}.csv", header=False, index=False)


def save_fake_checkpoint(path, cfg, train_df, seed):
    torch.manual_seed(seed)
    model = build_model(cfg)
    save_checkpoint(path, model, {**training_meta(train_df), "stage": 2, "config": cfg, "seed": cfg["seed"]})


def write_catalogue(path, entries):
    pairs = [{"comp_id": k, "crop_id": c, "disease_id": d, "name": f"c{c}d{d}", "is_seen": s} for c, d, k, s in sorted(entries, key=lambda e: e[2])]
    path.write_text(json.dumps({"version": 1, "sources": {}, "pairs": pairs}), encoding="utf-8")


def write_board(path, **gammas):
    pd.DataFrame({"scorer": list(gammas), "gamma keep": list(gammas.values())}).to_csv(path, index=False)


class World:
    pass


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World()
    w.root, w.data_root, w.lockdir = root, root / "data", root / "lock"
    names = val_names("val_seen")[:18]
    train = fake_frame(names[:6], [(c, d, comp(c, d)) for c, d in SEEN])
    seen = fake_frame(names[6:12], [(c, d, comp(c, d)) for c, d in SEEN])
    unseen = fake_frame(names[12:18], [(c, d, comp(c, d)) for c, d in (UNSEEN * 2)])
    write_csvs(w.data_root, train, seen, unseen)
    w.train, w.seen, w.unseen = train, seen, unseen
    w.catalogue = root / "catalogue_small.json"
    write_catalogue(w.catalogue, [(c, d, comp(c, d), True) for c, d in SEEN] + [(c, d, comp(c, d), False) for c, d in UNSEEN + EXTRA_UNSEEN])
    cfg = run_config(["model.crop_classes=3", "model.disease_classes=4"])
    w.run_cfg = cfg
    for folder, seed in (("head_a", 1), ("proto_a", 2)):
        save_fake_checkpoint(w.lockdir / folder / "stage2_final.pt", cfg, train, seed)
    boards = root / "boards"
    boards.mkdir()
    write_board(boards / "s1.csv", **{"head": 0.4, "proto-pooled": 0.2})
    write_board(boards / "s2.csv", **{"head": 0.8, "proto-pooled": 0.6})
    w.boards = boards
    w.gamma_head = gammas_from_boards([boards / "s1.csv", boards / "s2.csv"], "head")
    w.gamma_proto = gammas_from_boards([boards / "s1.csv", boards / "s2.csv"], "proto-pooled")
    w.lock_path = w.lockdir / "lock.json"
    make_lock(models_spec(w.lockdir), {"head_a": w.gamma_head, "proto_a": w.gamma_proto}, w.catalogue, w.lock_path)
    w.cfg = machine_cfg(w.data_root)
    return w


def models_spec(lockdir):
    return [{"name": "head_a", "checkpoint": lockdir / "head_a" / "stage2_final.pt", "kind": "head"},
            {"name": "proto_a", "checkpoint": lockdir / "proto_a" / "stage2_final.pt", "kind": "proto-pooled"}]


def copy_lock(world, destination):
    shutil.copytree(world.lockdir, destination)
    return destination / "lock.json"


def edit_lock(lock_path, change):
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    change(data)
    lock_path.write_text(json.dumps(data), encoding="utf-8")


def test_load_official_refuses_without_a_verified_lock(world):
    with pytest.raises(Refused):
        load_official(world.cfg, "test_seen", None)
    with pytest.raises(Refused):
        load_official(world.cfg, "test_unseen", json.loads(world.lock_path.read_text(encoding="utf-8")))
    lock = verify_lock(world.lock_path)
    assert len(load_official(world.cfg, "test_seen", lock)) == 6
    assert len(load_official(world.cfg, "test_unseen", lock)) == 6
    with pytest.raises(ValueError):
        load_official(world.cfg, "train", lock)


def test_verify_lock_refuses_a_changed_checkpoint_byte(world, tmp_path):
    lock_path = copy_lock(world, tmp_path / "copy")
    assert verify_lock(lock_path).sha256
    ckpt = tmp_path / "copy" / "head_a" / "stage2_final.pt"
    data = bytearray(ckpt.read_bytes())
    data[len(data) // 2] ^= 1
    ckpt.write_bytes(bytes(data))
    with pytest.raises(Refused, match="sha256 mismatch"):
        verify_lock(lock_path)


def test_verify_lock_refuses_a_changed_catalogue(world, tmp_path):
    lock_path = copy_lock(world, tmp_path / "copy")
    catalogue = tmp_path / "copy" / "catalogue.json"
    catalogue.write_text(catalogue.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(Refused, match="catalogue"):
        verify_lock(lock_path)


def test_verify_lock_refuses_hand_made_and_broken_locks(world, tmp_path):
    hand = tmp_path / "hand" / "lock.json"
    hand.parent.mkdir()
    hand.write_text(json.dumps({"models": {"x": {"checkpoint": "x.pt", "sha256": "0", "kind": "head", "gamma": 0.1}}}), encoding="utf-8")
    with pytest.raises(Refused):
        verify_lock(hand)
    hand.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(Refused):
        verify_lock(hand)
    hand.write_text("not json", encoding="utf-8")
    with pytest.raises(Refused):
        verify_lock(hand)
    with pytest.raises(Refused, match="missing"):
        verify_lock(tmp_path / "nowhere" / "lock.json")
    lock_path = copy_lock(world, tmp_path / "copy")
    (tmp_path / "copy" / "proto_a" / "stage2_final.pt").unlink()
    with pytest.raises(Refused, match="missing"):
        verify_lock(lock_path)


def test_verify_lock_refuses_a_bad_gamma_or_marked_run(world, tmp_path):
    lock_path = copy_lock(world, tmp_path / "a")
    edit_lock(lock_path, lambda d: d["models"]["head_a"].update(gamma="0.5"))
    with pytest.raises(Refused, match="gamma"):
        verify_lock(lock_path)
    lock_path = copy_lock(world, tmp_path / "b")
    edit_lock(lock_path, lambda d: d.update(test_run=True))
    with pytest.raises(Refused):
        verify_lock(lock_path)


def test_a_moved_lock_folder_still_verifies_and_resolves_in_its_new_place(world, tmp_path):
    moved = copy_lock(world, tmp_path / "elsewhere" / "input" / "lock")
    lock = verify_lock(moved)
    assert lock.dir == moved.parent.resolve()
    assert lock.path(lock["models"]["head_a"]["checkpoint"]).is_file()
    assert lock.path(lock["models"]["head_a"]["checkpoint"]).is_relative_to(moved.parent.resolve())
    stored = json.loads(world.lock_path.read_text(encoding="utf-8"))
    assert stored["models"]["head_a"]["checkpoint"] == "head_a/stage2_final.pt" and stored["catalogue"]["file"] == "catalogue.json"
    assert all(not Path(e["checkpoint"]).is_absolute() for e in stored["models"].values())


@pytest.mark.parametrize("bad", ["ABS", "../outside.pt", "head_a/../../outside.pt", "..\\outside.pt", "/etc/hosts", "C:\\Windows\\win.ini"])
def test_absolute_or_escaping_checkpoint_paths_are_refused(world, tmp_path, bad):
    lock_path = copy_lock(world, tmp_path / "copy")
    outside = tmp_path / "outside.pt"
    shutil.copyfile(tmp_path / "copy" / "head_a" / "stage2_final.pt", outside)
    value = str(outside.resolve()) if bad == "ABS" else bad
    edit_lock(lock_path, lambda d: d["models"]["head_a"].update(checkpoint=value, bank_source=value, sha256=final.file_sha256(outside)))
    with pytest.raises(Refused, match="absolute|leaves"):
        verify_lock(lock_path)


def test_absolute_or_escaping_catalogue_path_is_refused(world, tmp_path):
    lock_path = copy_lock(world, tmp_path / "copy")
    outside = tmp_path / "catalogue_outside.json"
    shutil.copyfile(tmp_path / "copy" / "catalogue.json", outside)
    for value in (str(outside.resolve()), "../catalogue_outside.json"):
        edit_lock(lock_path, lambda d: d["catalogue"].update(file=value, sha256=final.file_sha256(outside)))
        with pytest.raises(Refused, match="absolute|leaves"):
            verify_lock(lock_path)


def test_make_lock_refuses_a_checkpoint_outside_the_lock_folder_and_an_existing_lock(world, tmp_path):
    ckpt = world.lockdir / "head_a" / "stage2_final.pt"
    with pytest.raises(ValueError, match="not inside"):
        make_lock([{"name": "m", "checkpoint": ckpt, "kind": "head"}], {"m": 0.1}, world.catalogue, tmp_path / "other" / "lock.json")
    with pytest.raises(FileExistsError):
        make_lock(models_spec(world.lockdir), {"head_a": 0.1, "proto_a": 0.1}, world.catalogue, world.lock_path)
    with pytest.raises(ValueError, match="kind"):
        make_lock([{"name": "m", "checkpoint": ckpt, "kind": "svm"}], {"m": 0.1}, world.catalogue, world.lockdir / "x.json")
    with pytest.raises(ValueError, match="gamma"):
        make_lock([{"name": "m", "checkpoint": ckpt, "kind": "head"}], {"m": float("nan")}, world.catalogue, world.lockdir / "x.json")


def test_lock_records_calibrated_gammas_catalogue_copy_and_untouched_test_flag(world):
    lock = verify_lock(world.lock_path)
    assert lock["test_run"] is False and "created" in lock
    assert lock["models"]["head_a"]["gamma"] == pytest.approx(0.6) == pytest.approx(world.gamma_head)
    assert lock["models"]["proto_a"]["gamma"] == pytest.approx(0.4)
    assert lock["models"]["proto_a"]["bank_source"] == lock["models"]["proto_a"]["checkpoint"]
    assert lock["catalogue"]["sha256"] == final.file_sha256(world.lockdir / "catalogue.json")
    assert lock["catalogue"]["pairs"] == 10 and lock["catalogue"]["seen"] == 6


def test_gamma_is_the_mean_over_boards_and_refuses_missing_or_nan(world, tmp_path):
    assert gammas_from_boards([world.boards / "s1.csv", world.boards / "s2.csv"], "head") == pytest.approx(0.6)
    assert gammas_from_boards([world.boards / "s1.csv"], "head") == pytest.approx(0.4)
    with pytest.raises(Refused, match="proto-balanced"):
        gammas_from_boards([world.boards / "s1.csv"], "proto-balanced")
    write_board(tmp_path / "nan.csv", head=float("nan"))
    with pytest.raises(Refused, match="finite"):
        gammas_from_boards([world.boards / "s1.csv", tmp_path / "nan.csv"], "head")
    with pytest.raises(Refused):
        gammas_from_boards([], "head")


def test_full_catalogue_has_every_pair_and_its_composition_top1_needs_both_heads_right():
    seen = [(0, 0), (54, 174), (3, 7)]
    full = full_catalogue(55, 175, seen)
    assert len(full) == 9625 and len(set(full.comp_id)) == 9625
    assert int(full.is_seen.sum()) == 3 and full.is_seen[full.index_of([0 * 175 + 0, 54 * 175 + 174, 3 * 175 + 7])].all()
    with pytest.raises(ValueError):
        full_catalogue(55, 175, [(55, 0)])
    rng = np.random.default_rng(0)
    n = 40
    npz = {"crop_logits": rng.normal(size=(n, 55)), "disease_logits": rng.normal(size=(n, 175))}
    crop_pred, disease_pred = npz["crop_logits"].argmax(1), npz["disease_logits"].argmax(1)
    crop_true, disease_true = crop_pred.copy(), disease_pred.copy()
    crop_true[10:20] = (crop_pred[10:20] + 1) % 55
    disease_true[15:30] = (disease_pred[15:30] + 1) % 175
    npz.update(crop_id=crop_true, disease_id=disease_true, comp_id=np.zeros(n, dtype=int))
    npz = with_full_ids(npz)
    assert (npz["comp_id"] == crop_true * 175 + disease_true).all()
    expected = 100 * np.mean((crop_pred == crop_true) & (disease_pred == disease_true))
    assert 0 < expected < 100
    metrics = score_split_pair(Scorer("head"), npz, npz, full, 0.0)[0]["seen"]
    assert metrics["comp_top1"] == pytest.approx(expected)


def crafted_pair():
    cat = Catalogue(comp_id=np.array([0, 1]), crop_id=np.array([0, 0]), disease_id=np.array([0, 1]), is_seen=np.array([True, False]), names=["s", "u"])
    seen = {"crop_logits": np.zeros((1, 1)), "disease_logits": np.array([[2.0, 0.0]]), "crop_id": np.array([0]), "disease_id": np.array([0]),
            "comp_id": np.array([0])}
    unseen = {"crop_logits": np.zeros((1, 1)), "disease_logits": np.array([[1.0, 0.0]]), "crop_id": np.array([0]), "disease_id": np.array([1]),
              "comp_id": np.array([1])}
    return cat, seen, unseen


@pytest.mark.parametrize("gamma, a_s, a_u", [(0.0, 100.0, 0.0), (1.5, 100.0, 100.0), (3.0, 0.0, 100.0)])
def test_calibrated_metrics_use_the_given_gamma(gamma, a_s, a_u):
    cat, seen, unseen = crafted_pair()
    combined = score_split_pair(Scorer("head"), seen, unseen, cat, gamma)[0]["combined"]
    assert (combined["A_S"], combined["A_U"]) == (a_s, a_u)


def test_proto_scorer_uses_cosines_not_logits():
    cat, seen, unseen = crafted_pair()
    unit = np.eye(2, dtype=np.float32)
    banks = {"crop": unit, "disease": unit}
    contradicting = {"crop_logits": np.array([[0.0, 9.0]]), "disease_logits": np.array([[0.0, 9.0]]), "u_crop": unit[:1], "u_disease": unit[:1],
                     "crop_id": np.array([0]), "disease_id": np.array([0]), "comp_id": np.array([0])}
    metrics = score_split_pair(Scorer("proto-pooled", banks), contradicting, contradicting, cat, 0.0)[0]["seen"]
    assert metrics["crop_top1"] == 100 and metrics["disease_top1"] == 100 and metrics["comp_top1"] == 100
    head = score_split_pair(Scorer("head"), contradicting, contradicting, cat, 0.0)[0]["seen"]
    assert head["disease_top1"] == 0
    with pytest.raises(ValueError):
        Scorer("proto-pooled")


def sub_lock(world, tmp_path, name, head_gamma, proto_gamma):
    folder = tmp_path / name
    for sub in ("head_a", "proto_a"):
        shutil.copytree(world.lockdir / sub, folder / sub)
    make_lock(models_spec(folder), {"head_a": head_gamma, "proto_a": proto_gamma}, world.catalogue, folder / "lock.json")
    return verify_lock(folder / "lock.json")


def test_evaluate_model_uses_the_locked_gamma_and_catalogue_copy(world, tmp_path):
    lock = verify_lock(world.lock_path)
    res = final.evaluate_model(world.cfg, "head_a", lock, torch.device("cpu"), use_cache=False)
    assert set(res["metrics"]) == {"locked_raw", "locked_calibrated", "full_raw"}
    assert res["metrics"]["locked_calibrated"]["gamma"] == pytest.approx(world.gamma_head) and res["metrics"]["locked_raw"]["gamma"] == 0
    assert res["catalogue_sizes"] == {"locked": 10, "full": 12} and (res["n_seen"], res["n_unseen"]) == (6, 6)
    assert res["params"] > 1000 and res["seconds_per_image"] > 0
    big = sub_lock(world, tmp_path, "big", 1e6, 1e6)
    res_big = final.evaluate_model(world.cfg, "head_a", big, torch.device("cpu"), use_cache=False)
    assert res_big["metrics"]["locked_raw"] == res["metrics"]["locked_raw"]
    assert res_big["metrics"]["full_raw"] == res["metrics"]["full_raw"]
    assert res_big["metrics"]["locked_calibrated"]["seen"]["comp_top1"] == 0
    preds = res_big["predictions"]
    assert len(preds) == 36 and set(preds.split) == {"seen", "unseen"} and {f"top{k}_comp_id" for k in range(1, 6)} <= set(preds.columns)
    assert preds.groupby(["catalogue", "split"]).size().to_dict() == {(c, s): 6 for c in res["metrics"] for s in ("seen", "unseen")}
    unseen_ids = {comp(c, d) for c, d in UNSEEN + EXTRA_UNSEEN}
    assert set(preds[preds.catalogue == "locked_calibrated"].top1_comp_id) <= unseen_ids
    raw_seen = preds[(preds.catalogue == "locked_raw") & (preds.split == "seen")]
    assert (raw_seen.top1_comp_id == raw_seen.true_comp_id).mean() * 100 == pytest.approx(res["metrics"]["locked_raw"]["seen"]["comp_top1"])


def test_final_model_is_built_without_downloading_weights(world, monkeypatch):
    calls = []
    real = final.build_model
    monkeypatch.setattr(final, "build_model", lambda cfg, pretrained=None: calls.append(pretrained) or real(cfg, pretrained=False))
    final.evaluate_model(world.cfg, "head_a", verify_lock(world.lock_path), torch.device("cpu"), use_cache=False)
    assert calls == [False]


def test_proto_kind_builds_banks_from_all_training_rows(world, monkeypatch):
    captured = []
    real = final.score_split_pair

    def spy(scorer, *args, **kwargs):
        captured.append(scorer)
        return real(scorer, *args, **kwargs)

    monkeypatch.setattr(final, "score_split_pair", spy)
    final.evaluate_model(world.cfg, "proto_a", verify_lock(world.lock_path), torch.device("cpu"), use_cache=False)
    assert len(captured) == 3 and all(s.kind == "proto-pooled" for s in captured)
    assert captured[0].banks["crop"].shape == (3, 128) and captured[0].banks["disease"].shape == (4, 128)
    assert np.allclose(np.linalg.norm(captured[0].banks["crop"], axis=1), 1.0, atol=1e-5)


def test_confusion_is_323_by_333_on_the_real_catalogue(tmp_path):
    cfg = machine_cfg(tmp_path / "data")
    cat = load_catalogue(load_config()["paths"]["catalogue_final"])
    in_cat = set(cat.comp_id[cat.is_seen].tolist())
    seen_rows = pd.read_csv(REPO / "data" / "splits_v2" / "val_seen.csv")
    seen_rows = seen_rows[seen_rows.composition_id.isin(in_cat)].drop_duplicates("filename").head(10).reset_index(drop=True)
    assert len(seen_rows) == 10
    unseen_files = val_names("val_unseen")[:4]
    unseen_pairs = [(int(c), int(d), int(k)) for c, d, k in zip(cat.crop_id[~cat.is_seen], cat.disease_id[~cat.is_seen], cat.comp_id[~cat.is_seen])]
    train = seen_rows.iloc[:4][COLUMNS]
    test_seen = seen_rows.iloc[4:][COLUMNS]
    test_unseen = fake_frame(unseen_files, [unseen_pairs[i % len(unseen_pairs)] for i in range(4)])
    write_csvs(tmp_path / "data", train, test_seen, test_unseen)
    run_cfg = run_config(["bn.subset=4"])
    lockdir = tmp_path / "lock"
    save_fake_checkpoint(lockdir / "m" / "stage2_final.pt", run_cfg, train, 3)
    real_catalogue = load_config()["paths"]["catalogue_final"]
    make_lock([{"name": "m", "checkpoint": lockdir / "m" / "stage2_final.pt", "kind": "head"}], {"m": 0.2}, real_catalogue, lockdir / "lock.json")
    lock = verify_lock(lockdir / "lock.json")
    res = final.evaluate_model(cfg, "m", lock, torch.device("cpu"), use_cache=False)
    assert len(cat) == 333 and int(cat.is_seen.sum()) == 323
    for key in ("locked_raw", "locked_calibrated"):
        assert res["confusions"][key].shape == (323, 333)
        assert res["confusions"][key].sum() == 6
    assert res["catalogue_sizes"] == {"locked": 333, "full": 9625}


def argv(world, out_dir, *flags):
    return [str(world.lock_path), str(out_dir), *machine_overrides(world.data_root), *flags]


def test_final_test_refuses_without_the_confirm_flag(world, tmp_path):
    with pytest.raises(SystemExit):
        final_script.main(argv(world, tmp_path / "out", "--no-cache"))
    assert not (tmp_path / "out").exists()
    with pytest.raises(Refused):
        final.run_final_test(world.lock_path, tmp_path / "out2", world.cfg, confirm=False)
    assert not (tmp_path / "out2").exists()


def test_files_are_written_before_printing_then_a_second_run_is_refused(world, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def explode(table):
        raise RuntimeError("print failed")

    monkeypatch.setattr(final_script, "print_summary", explode)
    with pytest.raises(RuntimeError, match="print failed"):
        final_script.main(argv(world, out, "--confirm-final-test", "--no-cache"))
    tag = verify_lock(world.lock_path).sha256
    expected = ["final_metrics.csv", "final_metrics.json", "final_done.json"]
    for model in ("head_a", "proto_a"):
        expected.append(f"predictions_{model}.csv")
        for key in ("locked_raw", "locked_calibrated"):
            expected += [f"confusion_{model}_{key}_{tag[:12]}.npy", f"confusion_{model}_{key}.png"]
    assert all((out / name).is_file() for name in expected)
    table = pd.read_csv(out / "final_metrics.csv")
    assert list(table.columns) == ["model", "catalogue", "gamma", "split", "metric", "value", "lock_sha256"]
    assert set(table.lock_sha256) == {tag} and set(table.catalogue) == {"locked_raw", "locked_calibrated", "full_raw"}
    assert {"A_S", "A_U", "H", "gap", "comp_top1", "comp_top5", "crop_top1", "disease_top5"} <= set(table.metric)
    assert set(table.model) == {"head_a", "proto_a"} and set(table.split) == {"seen", "unseen", "combined"}
    for model in ("head_a", "proto_a"):
        calibrated = table[(table.model == model) & (table.catalogue == "locked_calibrated")]
        assert calibrated.gamma.iloc[0] == pytest.approx(verify_lock(world.lock_path)["models"][model]["gamma"])
    assert json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))["lock_sha256"] == tag
    assert json.loads((out / "final_done.json").read_text(encoding="utf-8"))["lock_sha256"] == tag
    assert set(pd.read_csv(out / "predictions_head_a.csv").lock_sha256) == {tag}
    matrix = np.load(out / f"confusion_head_a_locked_raw_{tag[:12]}.npy")
    assert matrix.shape == (6, 10) and matrix.sum() == 6
    assert tag in Image.open(out / "confusion_head_a_locked_raw.png").text["Description"]
    before = (out / "final_metrics.csv").read_bytes()
    with pytest.raises(Refused, match="already holds"):
        final_script.main(argv(world, out, "--confirm-final-test", "--no-cache"))
    assert (out / "final_metrics.csv").read_bytes() == before


def test_a_crashed_first_attempt_may_rerun_and_the_summary_prints_last(world, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    real = final.evaluate_model
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("out of memory")
        return real(*args, **kwargs)

    monkeypatch.setattr(final, "evaluate_model", flaky)
    with pytest.raises(RuntimeError, match="out of memory"):
        final_script.main(argv(world, out, "--confirm-final-test", "--no-cache"))
    assert not (out / "final_done.json").exists() and not (out / "final_metrics.csv").exists()
    capsys.readouterr()
    final_script.main(argv(world, out, "--confirm-final-test", "--no-cache"))
    assert (out / "final_done.json").is_file() and (out / "final_metrics.csv").is_file()
    printed = capsys.readouterr().out
    assert "locked_calibrated" in printed and "head_a" in printed and "A_S" in printed


def test_make_lock_script_builds_a_verifiable_lock_from_a_spec(world, tmp_path, monkeypatch):
    folder = tmp_path / "specs"
    for sub in ("head_a", "proto_a"):
        shutil.copytree(world.lockdir / sub, folder / sub)
    shutil.copytree(world.boards, folder / "boards")
    spec = {"settings": {"tol": 2.0}, "models": [
        {"name": "head_a", "checkpoint": "head_a/stage2_final.pt", "kind": "head", "boards": ["boards/s1.csv", "boards/s2.csv"]},
        {"name": "proto_a", "checkpoint": "proto_a/stage2_final.pt", "kind": "proto-pooled", "boards": ["boards/s1.csv", "boards/s2.csv"]}]}
    (folder / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(make_lock_script, "load_config", lambda overrides=(): {"paths": {"catalogue_final": str(world.catalogue)}})
    make_lock_script.main([str(folder / "spec.yaml"), str(folder / "lock.json")])
    lock = verify_lock(folder / "lock.json")
    assert lock["models"]["head_a"]["gamma"] == pytest.approx(0.6) and lock["models"]["proto_a"]["gamma"] == pytest.approx(0.4)
    assert lock["settings"]["tol"] == 2.0 and lock["catalogue"]["pairs"] == 10
