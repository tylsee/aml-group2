import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp import single_task as st

REAL_TRAIN = Path(r"D:\dev\uni\AML\data\DPD_Small\DPD_Small_train.csv")
REAL_SPLITS = Path(r"D:\dev\uni\AML\cdc-cp\data\splits_v2")
TINY = dict(expect_sizes=None, expect_classes=None)


def _rows(pairs, start):
    rows = []
    for k, (c, d) in enumerate(pairs):
        rows.append([f"img_{start + k}.jpg", c, d, c * 100 + d, f"crop{c}", f"dis{d}", f"crop{c}_dis{d}"])
    return pd.DataFrame(rows, columns=st.COLUMNS)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    images = root / "images"
    images.mkdir()
    fit_pairs = [(1, 10), (1, 11), (2, 11), (2, 12), (3, 10), (3, 11), (3, 12)] * 2
    seen_pairs = [(1, 10), (2, 11), (3, 12), (3, 10)]
    unseen_pairs = [(1, 12), (2, 10), (1, 12), (2, 10)]
    fit, seen, unseen = _rows(fit_pairs, 0), _rows(seen_pairs, 100), _rows(unseen_pairs, 200)
    rng = np.random.default_rng(0)
    for name in pd.concat([fit, seen, unseen]).filename:
        Image.fromarray(rng.integers(0, 255, (48, 40, 3), dtype=np.uint8)).save(images / name)
    splits = root / "splits"
    splits.mkdir()
    for name, df in (("train_fit", fit), ("val_seen", seen), ("val_unseen", unseen)):
        df.to_csv(splits / f"{name}.csv", index=False)
    train_csv = root / "train.csv"
    pd.concat([fit, seen, unseen]).to_csv(train_csv, index=False, header=False)
    return {"root": root, "images": images, "splits": splits, "train_csv": train_csv, "fit": fit, "seen": seen, "unseen": unseen}


@pytest.fixture(scope="module")
def stage1(tiny):
    out = tiny["root"] / "stage1"
    for arm in ("crop_b1", "disease_b1", "disease_zoom"):
        st.run_stage1(arm, tiny["splits"], tiny["images"], out, "cpu", epochs=3, workers=0, pretrained=False,
                      max_train_batches=1, **TINY)
    return out


def test_transforms():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 260, 3), dtype=np.uint8))
    for aug in st.AUGS:
        assert st.train_transform(aug)(image).shape == (3, 224, 224)
    assert st.eval_transform()(image).shape == (3, 224, 224)
    with pytest.raises(ValueError):
        st.train_transform("nope")


def test_transform_contents():
    b1 = st.train_transform("b1").transforms
    assert [type(t) for t in b1] == [T.RandomResizedCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomRotation,
                                     T.ColorJitter, T.ToTensor, T.Normalize]
    assert tuple(b1[0].scale) == (0.85, 1.0) and b1[1].p == 0.5 and b1[2].p == 0.5
    assert tuple(b1[3].degrees) == (-10.0, 10.0)
    assert tuple(b1[4].brightness) == pytest.approx((0.85, 1.15)) and tuple(b1[4].contrast) == pytest.approx((0.85, 1.15))
    zoom = st.train_transform("zoom").transforms
    assert tuple(zoom[0].scale) == (0.3, 1.0) and [type(t) for t in zoom] == [type(t) for t in b1]
    erase = st.train_transform("erase").transforms
    assert [type(t) for t in erase[:-1]] == [type(t) for t in b1] and isinstance(erase[-1], T.RandomErasing)
    assert erase[-1].p == 0.5 and tuple(erase[-1].scale) == (0.02, 0.2)


def test_build_model_and_optimizer():
    model = st.build_model("disease", 0.2, pretrained=False)
    assert model.classifier[-1][1].out_features == 175 and model.classifier[-1][0].p == 0.2
    assert st.build_model("crop", 0.3, pretrained=False).classifier[-1][1].out_features == 55
    opt = st.build_optimizer(model, 1e-5, 1e-3, 1e-3)
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-3]
    assert all(g["weight_decay"] == 1e-3 for g in opt.param_groups)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_harmonic():
    assert st.harmonic(0, 0) == 0.0
    assert st.harmonic(0.5, 1.0) == pytest.approx(2 / 3)


def test_composition_metrics_toy():
    crop_log = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    disease_log = np.log(np.array([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]))
    m = st.composition_metrics(crop_log, disease_log, np.array([0, 2]), np.array([1, 2]))
    assert m["crop_top1"] == 0.5 and m["crop_top5"] == 1.0
    assert m["disease_top1"] == 0.5 and m["disease_top5"] == 1.0
    assert m["composition_top1"] == 0.0 and m["composition_top5"] == 1.0
    exact = st.composition_metrics(crop_log, disease_log, np.array([0, 1]), np.array([0, 2]))
    assert exact["composition_top1"] == 1.0


def test_composition_top5_cutoff():
    crop_log = np.log(np.array([[0.5, 0.3, 0.2]]))
    disease_log = np.log(np.array([[0.5, 0.3, 0.2]]))
    m = st.composition_metrics(crop_log, disease_log, np.array([2]), np.array([2]))
    assert m["composition_top1"] == 0.0 and m["composition_top5"] == 0.0
    near = st.composition_metrics(crop_log, disease_log, np.array([0]), np.array([2]))
    assert near["composition_top5"] == 1.0 and near["composition_top1"] == 0.0


def test_composition_allowed_mask():
    crop_log = np.log(np.array([[0.7, 0.2, 0.1]]))
    disease_log = np.log(np.array([[0.6, 0.3, 0.1]]))
    allowed = np.ones(9, dtype=bool)
    allowed[0] = False
    masked = st.composition_metrics(crop_log, disease_log, np.array([0]), np.array([1]), allowed)
    assert masked["composition_top1"] == 1.0
    assert st.composition_metrics(crop_log, disease_log, np.array([0]), np.array([1]))["composition_top1"] == 0.0
    only_one = np.zeros(9, dtype=bool)
    only_one[4] = True
    never = st.composition_metrics(crop_log, disease_log, np.array([0]), np.array([1]), only_one)
    assert never["composition_top1"] == 0.0 and never["composition_top5"] == 0.0
    with pytest.raises(ValueError):
        st.composition_metrics(crop_log, disease_log, np.array([0]), np.array([1]), np.ones(8, dtype=bool))


def _board(**h):
    rows = [{"arm": a, "task": st.ARMS[a][0], "aug": st.ARMS[a][1], "individual_h": v} for a, v in h.items()]
    return pd.DataFrame(rows)


def test_pick_winner():
    assert st.pick_winner(_board(crop_b1=0.99, disease_b1=0.5, disease_zoom=0.6, disease_erase=0.55)) == "disease_zoom"
    assert st.pick_winner(_board(disease_b1=0.6, disease_zoom=0.6, disease_erase=0.6)) == "disease_b1"
    assert st.pick_winner(_board(disease_b1=0.5, disease_erase=0.7, disease_zoom=0.7)) == "disease_zoom"
    assert st.pick_winner(_board(crop_b1=0.9, disease_b1=0.1)) == "disease_b1"
    with pytest.raises(ValueError):
        st.pick_winner(_board(crop_b1=0.9))


def test_load_dev_rules(tiny, tmp_path):
    fit, seen, unseen = st.load_dev(tiny["splits"], None)
    assert (len(fit), len(seen), len(unseen)) == (14, 4, 4)
    with pytest.raises(ValueError, match="split sizes"):
        st.load_dev(tiny["splits"])
    bad = tmp_path / "bad"
    bad.mkdir()
    fit.to_csv(bad / "train_fit.csv", index=False)
    seen.to_csv(bad / "val_seen.csv", index=False)
    pd.concat([unseen.iloc[:3], _rows([(1, 10)], 300)]).to_csv(bad / "val_unseen.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        st.load_dev(bad, None)
    with pytest.raises(ValueError, match="class counts"):
        st.class_ids(fit, seen, unseen)
    assert st.class_ids(fit, seen, unseen, expect=None) == ([1, 2, 3], [10, 11, 12])


def test_read_labelled_csv_header_detection(tiny):
    headerless = st.read_labelled_csv(tiny["train_csv"])
    assert list(headerless.columns) == st.COLUMNS and len(headerless) == 22
    with_header = st.read_labelled_csv(tiny["splits"] / "val_seen.csv")
    assert len(with_header) == 4 and with_header.crop_id.dtype.kind == "i"


def test_stage1_outputs(tiny, stage1):
    folder = stage1 / "disease_b1"
    for name in ("config.json", "history.csv", "best.pt", "best_validation.npz", "done.json",
                 "predictions_epoch_01.npz", "predictions_epoch_02.npz", "predictions_epoch_03.npz"):
        assert (folder / name).exists(), name
    history = pd.read_csv(folder / "history.csv")
    assert list(history.columns) == st.HISTORY1 and history.epoch.tolist() == [1, 2, 3]
    best_h, best_epoch = -1.0, 0
    for row in history.itertuples():
        if row.individual_h > best_h + 1e-9:
            best_h, best_epoch = row.individual_h, row.epoch
    done = json.loads((folder / "done.json").read_text())
    assert done["best_epoch"] == best_epoch and done["best_h"] == pytest.approx(best_h)
    assert history.best_h.iloc[-1] == pytest.approx(best_h)
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == best_epoch and ckpt["task"] == "disease" and ckpt["aug"] == "b1"
    assert ckpt["disease_ids"] == [10, 11, 12] and ckpt["crop_ids"] == [1, 2, 3] and ckpt["dropout"] == 0.2
    best = np.load(folder / "best_validation.npz")
    epoch_file = np.load(folder / f"predictions_epoch_{best_epoch:02d}.npz")
    assert np.array_equal(best["seen"], epoch_file["seen"]) and best["seen"].shape == (4, 3)
    assert best["seen"].dtype == np.float32 and np.allclose(np.exp(best["seen"]).sum(1), 1, atol=1e-4)
    config = json.loads((folder / "config.json").read_text())
    assert config["b1"]["max_epochs"] == 30 and config["aug"] == "b1"


def test_stage1_refuses_overwrite(tiny, stage1):
    with pytest.raises(FileExistsError):
        st.run_stage1("disease_b1", tiny["splits"], tiny["images"], stage1, "cpu", epochs=1, workers=0, pretrained=False,
                      max_train_batches=1, **TINY)


def test_stage1_rejects_bad_epochs(tiny, tmp_path):
    with pytest.raises(ValueError):
        st.run_stage1("disease_b1", tiny["splits"], tiny["images"], tmp_path, "cpu", epochs=31, **TINY)
    with pytest.raises(ValueError):
        st.run_stage1("nope", tiny["splits"], tiny["images"], tmp_path, "cpu", **TINY)


def test_board_and_winner(stage1):
    board = st.stage1_board(stage1, ["crop_b1", "disease_b1", "disease_zoom"])
    assert board.arm.tolist() == ["crop_b1", "disease_b1", "disease_zoom"]
    for col in ("task", "aug", "best_epoch", "seen_acc", "unseen_acc", "individual_h", "seen_correct", "unseen_correct"):
        assert col in board.columns
    row = board[board.arm == "disease_b1"].iloc[0]
    assert row.seen_n == 4 and row.unseen_n == 4
    assert row.seen_correct == round(row.seen_acc * 4) and row.individual_h == pytest.approx(st.harmonic(row.seen_acc, row.unseen_acc))
    assert st.pick_winner(board) in ("disease_b1", "disease_zoom")


def test_dev_composition(tiny, stage1):
    result = st.dev_composition(stage1, "crop_b1", "disease_b1", tiny["splits"], expect_sizes=None, expect_classes=None)
    assert set(result) == {"seen", "unseen", "hm"}
    for part in ("seen", "unseen"):
        assert set(result[part]) == {"crop_top1", "crop_top5", "disease_top1", "disease_top5", "composition_top1",
                                     "composition_top5"}
        assert result[part]["composition_top5"] >= result[part]["composition_top1"]
    board = st.stage1_board(stage1, ["crop_b1", "disease_b1"])
    assert result["seen"]["crop_top1"] == pytest.approx(board.seen_acc[0])
    assert result["unseen"]["disease_top1"] == pytest.approx(board.unseen_acc[1])
    assert result["hm"] == pytest.approx(st.harmonic(result["seen"]["composition_top1"], result["unseen"]["composition_top1"]))
    with pytest.raises(ValueError):
        st.dev_composition(stage1, "disease_b1", "crop_b1", tiny["splits"], expect_sizes=None, expect_classes=None)


@pytest.fixture(scope="module")
def stage2(tiny, stage1):
    out = tiny["root"] / "stage2"
    st.run_stage2("disease_b1", stage1, tiny["train_csv"], tiny["images"], out, "cpu", epochs=3, workers=0,
                  max_train_batches=1, expect_rows=None)
    return out


def test_stage2_outputs(tiny, stage1, stage2):
    folder = stage2 / "disease_b1"
    for name in ("config.json", "history.csv", "best_train_acc.pt", "done.json"):
        assert (folder / name).exists(), name
    history = pd.read_csv(folder / "history.csv")
    assert list(history.columns) == st.HISTORY2 and history.epoch.tolist() == [1, 2, 3]
    best, best_epoch = -1.0, 0
    for row in history.itertuples():
        if row.train_acc > best:
            best, best_epoch = row.train_acc, row.epoch
    done = json.loads((folder / "done.json").read_text())
    assert done["best_epoch"] == best_epoch and done["best_train_acc"] == pytest.approx(best)
    ckpt = torch.load(folder / "best_train_acc.pt", map_location="cpu", weights_only=True)
    source = torch.load(stage1 / "disease_b1" / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["stage"] == 2 and ckpt["epoch"] == best_epoch and ckpt["source_epoch"] == source["epoch"]
    assert ckpt["disease_ids"] == [10, 11, 12] and ckpt["task"] == "disease"


def test_stage2_refuses_overwrite_and_checks_rows(tiny, stage1, stage2, tmp_path):
    with pytest.raises(FileExistsError):
        st.run_stage2("disease_b1", stage1, tiny["train_csv"], tiny["images"], stage2, "cpu", epochs=1, workers=0,
                      max_train_batches=1, expect_rows=None)
    with pytest.raises(ValueError, match="rows"):
        st.run_stage2("disease_b1", stage1, tiny["train_csv"], tiny["images"], tmp_path, "cpu", epochs=1, workers=0,
                      max_train_batches=1)
    short = tmp_path / "short.csv"
    pd.read_csv(tiny["train_csv"], header=None).iloc[:-1].to_csv(short, index=False, header=False)
    with pytest.raises(ValueError, match="filenames differ"):
        st.run_stage2("disease_b1", stage1, short, tiny["images"], tmp_path, "cpu", epochs=1, workers=0,
                      max_train_batches=1, expect_rows=None)


def test_load_trained_and_predict(tiny, stage1, stage2):
    path = stage2 / "disease_b1" / "best_train_acc.pt"
    model = st.load_trained(path, "cpu")
    assert not model.training
    ids = ([1, 2, 3], [10, 11, 12])
    out = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    assert out.shape == (4, 3) and out.dtype == np.float32
    assert np.allclose(np.exp(out).sum(1), 1, atol=1e-4)
    again = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    assert np.array_equal(out, again)


def test_lock_roundtrip(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    lock_path = tmp_path / "lock.json"
    lock = st.write_lock({"a": a, "b": b}, {"winner": "disease_b1"}, lock_path)
    assert lock["info"] == {"winner": "disease_b1"} and len(lock["files"]["a"]["sha256"]) == 64
    assert st.verify_lock(lock_path)["info"] == {"winner": "disease_b1"}
    with pytest.raises(FileExistsError):
        st.write_lock({"a": a}, {}, lock_path)
    a.write_bytes(b"alphb")
    with pytest.raises(ValueError, match="changed"):
        st.verify_lock(lock_path)
    a.write_bytes(b"alpha")
    b.unlink()
    with pytest.raises(ValueError, match="missing"):
        st.verify_lock(lock_path)


@pytest.mark.skipif(not (REAL_TRAIN.exists() and REAL_SPLITS.exists()), reason="real data not present")
def test_real_data_shapes():
    train_fit, val_seen, val_unseen = st.load_dev(REAL_SPLITS)
    assert (len(train_fit), len(val_seen), len(val_unseen)) == (6905, 695, 400)
    crops, diseases = st.class_ids(train_fit, val_seen, val_unseen)
    assert (len(crops), len(diseases)) == (55, 175)
    full = st.read_labelled_csv(REAL_TRAIN)
    assert len(full) == 8000
    assert set(full.filename) == set(train_fit.filename) | set(val_seen.filename) | set(val_unseen.filename)


def _max_drop_path(model):
    return max(m.p for m in model.modules() if type(m).__name__ == "StochasticDepth")


def test_b1_arms_are_pinned():
    assert st.B1 == dict(seed=42, batch_size=16, max_epochs=30, backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-3,
                         dropout={"crop": 0.3, "disease": 0.2}, stage2_epochs=10, stage2_backbone_lr=1e-6,
                         stage2_head_lr=1e-4, stage2_weight_decay=1e-3)
    assert st.RECIPES["b1"] == dict(label_smoothing=0.0, weight_decay=1e-3, stage2_weight_decay=1e-3,
                                    stochastic_depth={"convnext_tiny": None, "convnext_small": None})
    expected = {"crop_b1": ("crop", "b1"), "disease_b1": ("disease", "b1"), "disease_zoom": ("disease", "zoom"),
                "disease_erase": ("disease", "erase")}
    default_p = _max_drop_path(st.tvm.convnext_tiny(weights=None))
    assert default_p == 0.1
    for arm, (task, aug) in expected.items():
        spec = st.ARMS[arm]
        assert (spec.task, spec.aug, spec.backbone, spec.recipe) == (task, aug, "convnext_tiny", "b1")
        model = st.build_arm_model(arm, False, 3)
        assert _max_drop_path(model) == default_p
        assert model.classifier[-1][0].p == st.B1["dropout"][task]
        assert len(model.features[5]) == 9
    b1 = st.train_transform("b1").transforms
    assert [type(t) for t in b1] == [T.RandomResizedCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomRotation,
                                     T.ColorJitter, T.ToTensor, T.Normalize]
    assert tuple(b1[0].scale) == (0.85, 1.0)
    assert [type(t) for t in st.train_transform("erase").transforms][-1] is T.RandomErasing
    assert tuple(st.train_transform("zoom").transforms[0].scale) == (0.3, 1.0)


def test_strong_transform_contents():
    steps = st.train_transform("strong").transforms
    assert [type(t) for t in steps] == [T.RandomResizedCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip,
                                        T.TrivialAugmentWide, T.ToTensor, T.Normalize, T.RandomErasing]
    assert tuple(steps[0].scale) == (0.3, 1.0) and steps[1].p == 0.5 and steps[2].p == 0.5
    assert tuple(steps[5].mean) == st.MEAN and tuple(steps[5].std) == st.STD
    assert steps[6].p == 0.25 and tuple(steps[6].scale) == (0.02, 0.2)
    image = Image.fromarray(np.random.default_rng(2).integers(0, 255, (300, 260, 3), dtype=np.uint8))
    assert st.train_transform("strong")(image).shape == (3, 224, 224)
    assert [type(t) for t in st.eval_transform().transforms] == [T.Resize, T.CenterCrop, T.ToTensor, T.Normalize]


def test_arm_table():
    assert list(st.ARMS) == ["crop_b1", "crop_reg", "disease_b1", "disease_zoom", "disease_erase", "disease_reg",
                             "disease_small_reg", "disease_clip", "crop_clip"]
    assert st.ARMS["crop_reg"] == ("crop", "strong", "convnext_tiny", "reg")
    assert st.ARMS["disease_reg"] == ("disease", "strong", "convnext_tiny", "reg")
    assert st.ARMS["disease_small_reg"] == ("disease", "strong", "convnext_small", "reg")
    assert st.CONTROLS == {"crop": "crop_b1", "disease": "disease_b1"}


def test_reg_loss_uses_label_smoothing(monkeypatch):
    seen = []
    real = st.nn.CrossEntropyLoss

    def spy(*args, **kwargs):
        seen.append(kwargs.get("label_smoothing", 0.0))
        return real(*args, **kwargs)

    monkeypatch.setattr(st.nn, "CrossEntropyLoss", spy)
    model = st.build_arm_model("disease_b1", False, 3)
    opt = st.build_optimizer(model, 1e-5, 1e-3, 1e-3)
    loader = [(torch.zeros(2, 3, 64, 64), torch.tensor([0, 1]), torch.tensor([1, 2]))]
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    st.train_epoch(model, loader, "disease", opt, scaler, torch.device("cpu"), False, None,
                   st.RECIPES["b1"]["label_smoothing"])
    st.train_epoch(model, loader, "disease", opt, scaler, torch.device("cpu"), False, None,
                   st.RECIPES["reg"]["label_smoothing"])
    assert seen == [0.0, 0.1]


def test_evaluate_loss_is_unsmoothed():
    model = st.build_arm_model("disease_reg", False, 3)
    batch = (torch.zeros(2, 3, 64, 64), torch.tensor([0, 1]), torch.tensor([1, 2]))
    out = st.evaluate(model, [batch], "disease", torch.device("cpu"), False)
    log_probs = torch.from_numpy(out["log_probs"])
    assert out["loss"] == pytest.approx(torch.nn.functional.nll_loss(log_probs, batch[2]).item(), abs=1e-5)


def test_backbone_head_and_stochastic_depth():
    tiny_reg = st.build_arm_model("disease_reg", False, 3)
    small = st.build_arm_model("disease_small_reg", False, 3)
    assert _max_drop_path(tiny_reg) == 0.2
    assert _max_drop_path(small) == 0.4 == _max_drop_path(st.tvm.convnext_small(weights=None))
    assert small.classifier[-1][1].in_features == 768 and small.classifier[-1][1].out_features == 3
    assert small.classifier[-1][0].p == 0.2
    assert len(small.features[5]) == 27 and len(tiny_reg.features[5]) == 9
    assert st.build_arm_model("crop_reg", False, 3).classifier[-1][0].p == 0.3
    with pytest.raises(ValueError, match="backbone"):
        st.build_model("disease", 0.2, False, 3, "convnext_huge")


def _board_h(**h):
    rows = [{"arm": a, "task": st.ARMS[a].task, "aug": st.ARMS[a].aug, "individual_h": v} for a, v in h.items()]
    return pd.DataFrame(rows)


def test_pick_winner_per_task():
    board = _board_h(crop_b1=0.5, crop_reg=0.6, disease_b1=0.4, disease_reg=0.3, disease_small_reg=0.7)
    assert st.pick_winner(board) == "disease_small_reg"
    assert st.pick_winner(board, "disease") == "disease_small_reg"
    assert st.pick_winner(board, task="crop") == "crop_reg"
    assert st.pick_winner(_board_h(crop_b1=0.6, crop_reg=0.6, disease_b1=0.1), "crop") == "crop_b1"
    assert st.pick_winner(_board_h(crop_b1=0.5, crop_reg=0.6, disease_b1=0.1), "crop") == "crop_reg"
    assert st.pick_winner(_board_h(disease_b1=0.5, disease_reg=0.7, disease_small_reg=0.7, disease_zoom=0.7)) == "disease_zoom"
    assert st.pick_winner(_board_h(disease_b1=0.5, disease_small_reg=0.7, disease_reg=0.7)) == "disease_reg"
    assert st.pick_winner(_board_h(disease_b1=0.7, disease_reg=0.7, disease_small_reg=0.7)) == "disease_b1"
    assert st.pick_winner(_board_h(disease_reg=0.7, disease_small_reg=0.7, crop_reg=0.9)) == "disease_reg"
    with pytest.raises(ValueError):
        st.pick_winner(_board_h(disease_b1=0.5), "crop")
    with pytest.raises(ValueError):
        st.pick_winner(_board_h(crop_b1=0.5), "disease")
    with pytest.raises(ValueError, match="task"):
        st.pick_winner(_board_h(crop_b1=0.5), "weed")


def test_small_backbone_checkpoint_roundtrip(tmp_path):
    model = st.build_arm_model("disease_small_reg", False, 3)
    path = tmp_path / "small.pt"
    st._atomic_save(st._checkpoint(model, "disease_small_reg", "disease", "strong", 1, ([1, 2, 3], [10, 11, 12]), 0.5), path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["backbone"] == "convnext_small" and ckpt["recipe"] == "reg" and ckpt["arm"] == "disease_small_reg"
    loaded = st.load_trained(path, "cpu")
    assert not loaded.training
    assert len(loaded.features[5]) == 27 and loaded.classifier[-1][1].out_features == 3 and loaded.classifier[-1][0].p == 0.2
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    x = torch.zeros(1, 3, 32, 32)
    assert torch.allclose(model.eval()(x), loaded(x))


@pytest.fixture(scope="module")
def reg_runs(tiny):
    captured = []
    real = st.build_optimizer

    def spy(model, backbone_lr, head_lr, weight_decay):
        captured.append((backbone_lr, head_lr, weight_decay))
        return real(model, backbone_lr, head_lr, weight_decay)

    patch = pytest.MonkeyPatch()
    patch.setattr(st, "build_optimizer", spy)
    try:
        out1, out2 = tiny["root"] / "reg_stage1", tiny["root"] / "reg_stage2"
        for arm in ("crop_b1", "crop_reg", "disease_reg"):
            st.run_stage1(arm, tiny["splits"], tiny["images"], out1, "cpu", epochs=2, workers=0, pretrained=False,
                          max_train_batches=1, **TINY)
        st.run_stage2("disease_reg", out1, tiny["train_csv"], tiny["images"], out2, "cpu", epochs=2, workers=0,
                      max_train_batches=1, expect_rows=None)
    finally:
        patch.undo()
    return {"stage1": out1, "stage2": out2, "optimizers": captured}


def test_reg_optimizer_weight_decay_both_stages(reg_runs):
    assert reg_runs["optimizers"] == [(1e-5, 1e-3, 1e-3), (1e-5, 1e-3, 0.05), (1e-5, 1e-3, 0.05), (1e-6, 1e-4, 0.05)]


def test_reg_toy_run_outputs(tiny, reg_runs):
    folder = reg_runs["stage1"] / "disease_reg"
    config = json.loads((folder / "config.json").read_text())
    assert config["aug"] == "strong" and config["recipe"] == "reg" and config["backbone"] == "convnext_tiny"
    assert config["recipe_values"]["label_smoothing"] == 0.1
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["arm"] == "disease_reg" and ckpt["recipe"] == "reg" and ckpt["backbone"] == "convnext_tiny"
    stage2 = reg_runs["stage2"] / "disease_reg"
    s2 = torch.load(stage2 / "best_train_acc.pt", map_location="cpu", weights_only=True)
    assert s2["stage"] == 2 and s2["arm"] == "disease_reg" and s2["recipe"] == "reg"
    history = pd.read_csv(stage2 / "history.csv")
    assert list(history.columns) == st.HISTORY2 and history.epoch.tolist() == [1, 2]
    model = st.load_trained(stage2 / "best_train_acc.pt", "cpu")
    out = st.predict(model, tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0)
    assert out.shape == (4, 3)


def test_board_and_composition_with_reg_arms(tiny, reg_runs):
    board = st.stage1_board(reg_runs["stage1"], ["crop_b1", "crop_reg", "disease_reg"])
    assert board.backbone.tolist() == ["convnext_tiny"] * 3 and board.recipe.tolist() == ["b1", "reg", "reg"]
    assert st.pick_winner(board, "crop") in ("crop_b1", "crop_reg")
    assert st.pick_winner(board) == "disease_reg"
    result = st.dev_composition(reg_runs["stage1"], "crop_reg", "disease_reg", tiny["splits"], **TINY)
    assert set(result) == {"seen", "unseen", "hm"}
    with pytest.raises(ValueError):
        st.dev_composition(reg_runs["stage1"], "disease_reg", "crop_reg", tiny["splits"], **TINY)


B1_ARMS = ("crop_b1", "disease_b1", "disease_zoom", "disease_erase")
REG_ARMS = ("crop_reg", "disease_reg", "disease_small_reg")


@pytest.fixture(scope="module")
def spied_runs(tiny):
    events = {}
    context = {}

    def record(kind, value):
        events.setdefault((context["arm"], context["stage"]), []).append((kind, value))

    real_opt, real_loss, real_aug = st.build_optimizer, st.nn.CrossEntropyLoss, st.train_transform
    real_model, real_sched = st.build_model, torch.optim.lr_scheduler.CosineAnnealingLR

    def opt_spy(model, backbone_lr, head_lr, weight_decay):
        opt = real_opt(model, backbone_lr, head_lr, weight_decay)
        record("optimizer", [(g["lr"], g["weight_decay"]) for g in opt.param_groups])
        return opt

    def loss_spy(*args, **kwargs):
        record("label_smoothing", kwargs.get("label_smoothing", 0.0))
        return real_loss(*args, **kwargs)

    def aug_spy(aug):
        record("aug", aug)
        return real_aug(aug)

    def model_spy(*args, **kwargs):
        model = real_model(*args, **kwargs)
        record("drop_path", _max_drop_path(model))
        return model

    def sched_spy(optimizer, T_max, *args, **kwargs):
        record("t_max", T_max)
        return real_sched(optimizer, T_max, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(st, "build_optimizer", opt_spy)
    patch.setattr(st.nn, "CrossEntropyLoss", loss_spy)
    patch.setattr(st, "train_transform", aug_spy)
    patch.setattr(st, "build_model", model_spy)
    patch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", sched_spy)
    try:
        out1, out2 = tiny["root"] / "spy_stage1", tiny["root"] / "spy_stage2"
        for arm in B1_ARMS + REG_ARMS:
            context.update(arm=arm, stage=1)
            st.run_stage1(arm, tiny["splits"], tiny["images"], out1, "cpu", epochs=1, workers=0, pretrained=False,
                          max_train_batches=1, **TINY)
        for arm in ("crop_b1", "disease_zoom", "crop_reg", "disease_small_reg"):
            context.update(arm=arm, stage=2)
            st.run_stage2(arm, out1, tiny["train_csv"], tiny["images"], out2, "cpu", epochs=1, workers=0,
                          max_train_batches=1, expect_rows=None)
    finally:
        patch.undo()
    return events


def _of(events, arm, stage, kind):
    return [value for k, value in events[(arm, stage)] if k == kind]


@pytest.mark.parametrize("arm", B1_ARMS)
def test_b1_stage1_runs_use_baseline_1_settings(spied_runs, arm):
    assert _of(spied_runs, arm, 1, "optimizer") == [[(1e-5, 1e-3), (1e-3, 1e-3)]]
    assert _of(spied_runs, arm, 1, "label_smoothing") == [0.0]
    assert _of(spied_runs, arm, 1, "t_max") == [30]
    assert _of(spied_runs, arm, 1, "aug") == [st.ARMS[arm].aug]
    assert _of(spied_runs, arm, 1, "drop_path") == [0.1]


def test_b1_stage2_run_uses_baseline_1_settings(spied_runs):
    assert _of(spied_runs, "crop_b1", 2, "optimizer") == [[(1e-6, 1e-3), (1e-4, 1e-3)]]
    assert _of(spied_runs, "crop_b1", 2, "label_smoothing") == [0.0]
    assert _of(spied_runs, "crop_b1", 2, "t_max") == [10]
    assert _of(spied_runs, "crop_b1", 2, "aug") == ["b1"]
    assert _of(spied_runs, "crop_b1", 2, "drop_path") == [0.1]


def test_zoom_stage2_run_uses_zoom_aug_and_baseline_1_settings(spied_runs):
    assert _of(spied_runs, "disease_zoom", 2, "aug") == ["zoom"]
    assert _of(spied_runs, "disease_zoom", 2, "optimizer") == [[(1e-6, 1e-3), (1e-4, 1e-3)]]
    assert _of(spied_runs, "disease_zoom", 2, "label_smoothing") == [0.0]
    assert _of(spied_runs, "disease_zoom", 2, "t_max") == [10]
    assert _of(spied_runs, "disease_zoom", 2, "drop_path") == [0.1]


@pytest.mark.parametrize("arm,depth", [("crop_reg", 0.2), ("disease_reg", 0.2), ("disease_small_reg", 0.4)])
def test_reg_stage1_runs_use_reg_recipe(spied_runs, arm, depth):
    assert _of(spied_runs, arm, 1, "optimizer") == [[(1e-5, 0.05), (1e-3, 0.05)]]
    assert _of(spied_runs, arm, 1, "label_smoothing") == [0.1]
    assert _of(spied_runs, arm, 1, "t_max") == [30]
    assert _of(spied_runs, arm, 1, "aug") == ["strong"]
    assert _of(spied_runs, arm, 1, "drop_path") == [depth]


@pytest.mark.parametrize("arm,depth", [("crop_reg", 0.2), ("disease_small_reg", 0.4)])
def test_reg_stage2_runs_use_reg_recipe(spied_runs, arm, depth):
    assert _of(spied_runs, arm, 2, "optimizer") == [[(1e-6, 0.05), (1e-4, 0.05)]]
    assert _of(spied_runs, arm, 2, "label_smoothing") == [0.1]
    assert _of(spied_runs, arm, 2, "t_max") == [10]
    assert _of(spied_runs, arm, 2, "aug") == ["strong"]
    assert _of(spied_runs, arm, 2, "drop_path") == [depth]


CLIP_NAMES = ["dis10", "healthy", "Apple_scab"]


def _clip_model(n=3, init_scale=None):
    model = st.build_arm_model("disease_clip", False, n, CLIP_NAMES[:n])
    if init_scale is not None:
        model.logit_scale.data.fill_(float(np.log(init_scale)))
    return model


def _clip_batch(n=2):
    return [(torch.randn(n, 3, 224, 224), torch.zeros(n, dtype=torch.long), torch.tensor([0, 2][:n]))]


def _no_network(monkeypatch):
    import transformers

    def refuse(*args, **kwargs):
        raise AssertionError("from_pretrained must not be called")

    for name in ("CLIPVisionModelWithProjection", "CLIPTextModelWithProjection", "CLIPModel", "CLIPTokenizer"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", classmethod(refuse))


def test_clip_arm_spec_and_recipe():
    assert st.ARMS["disease_clip"] == ("disease", "b1", "clip_vit_b16", "clip")
    assert st.ARMS["crop_clip"] == ("crop", "b1", "clip_vit_b16", "clip")
    rc = st.RECIPES["clip"]
    assert rc["label_smoothing"] == 0.0 and rc["weight_decay"] == 0.1 and rc["stage2_weight_decay"] == 0.1
    assert st.stage_lrs("clip", 1) == (1e-5, 1e-3) and st.stage_lrs("clip", 2) == (1e-6, 1e-4)
    assert st.stage_lrs("b1", 1) == (1e-5, 1e-3) and st.stage_lrs("reg", 2) == (1e-6, 1e-4)


def test_clip_normalisation_constants_and_transforms():
    assert st.CLIP_MEAN == (0.48145466, 0.4578275, 0.40821073) and st.CLIP_STD == (0.26862954, 0.26130258, 0.27577711)
    norm = st.NORMS["clip_vit_b16"]
    train = st.train_transform("b1", norm=norm).transforms
    assert [type(t) for t in train] == [type(t) for t in st.train_transform("b1").transforms]
    assert tuple(train[0].scale) == (0.85, 1.0) and tuple(train[3].degrees) == (-10.0, 10.0)
    assert tuple(train[-1].mean) == st.CLIP_MEAN and tuple(train[-1].std) == st.CLIP_STD
    evaluate_steps = st.eval_transform(norm).transforms
    assert [type(t) for t in evaluate_steps] == [T.Resize, T.CenterCrop, T.ToTensor, T.Normalize]
    assert evaluate_steps[0].size == 256 and tuple(evaluate_steps[1].size) == (224, 224)
    assert tuple(evaluate_steps[-1].mean) == st.CLIP_MEAN and tuple(evaluate_steps[-1].std) == st.CLIP_STD
    assert tuple(st.train_transform("b1").transforms[-1].mean) == st.MEAN
    assert tuple(st.eval_transform().transforms[-1].std) == st.STD
    assert st._transform_kwargs("convnext_tiny") == {} and st._transform_kwargs("clip_vit_b16") == {"norm": norm}


HEALTHY_SET = ["a photo of a healthy plant leaf.", "a close-up photo of a healthy leaf.",
               "a healthy green leaf with no disease.", "a photo of a healthy plant."]


def test_clip_prompts():
    assert st.disease_prompts(["Apple_scab", "healthy", "Black rot", " Healthy "]) == [
        ["a photo of a plant leaf with Apple scab.", "a close-up photo of a leaf showing Apple scab.",
         "a leaf infected with Apple scab.", "a photo of Apple scab, a plant disease."],
        HEALTHY_SET,
        ["a photo of a plant leaf with Black rot.", "a close-up photo of a leaf showing Black rot.",
         "a leaf infected with Black rot.", "a photo of Black rot, a plant disease."],
        HEALTHY_SET]
    assert st.class_prompts("disease", ["healthy"]) == [HEALTHY_SET]


def test_clip_crop_prompts():
    assert st.class_prompts("crop", ["Bell_pepper", "healthy"]) == [
        ["a photo of a Bell pepper leaf.", "a close-up photo of a Bell pepper plant leaf.",
         "a leaf of a Bell pepper plant.", "a photo of a Bell pepper plant."],
        ["a photo of a healthy leaf.", "a close-up photo of a healthy plant leaf.", "a leaf of a healthy plant.",
         "a photo of a healthy plant."]]
    with pytest.raises(ValueError, match="unknown task"):
        st.class_prompts("composition", ["x"])


def test_clip_crop_names_follow_class_index(tiny):
    dfs = (tiny["fit"], tiny["seen"], tiny["unseen"])
    assert st.names_in_order("crop", dfs, [1, 2, 3], contiguous=False) == ["crop1", "crop2", "crop3"]
    with pytest.raises(ValueError, match="crop ids are not 0"):
        st.names_in_order("crop", dfs, [1, 2, 3], contiguous=True)
    with pytest.raises(ValueError, match="sorted"):
        st.names_in_order("crop", dfs, [2, 1, 3], contiguous=False)
    with pytest.raises(ValueError, match="cover"):
        st.names_in_order("crop", dfs, [1, 2], contiguous=False)
    renamed = tiny["seen"].copy()
    renamed["crop_name"] = "other"
    with pytest.raises(ValueError, match="more than one"):
        st.names_in_order("crop", (tiny["fit"], renamed), [1, 2, 3], contiguous=False)


def test_clip_disease_names_follow_class_index(tiny):
    dfs = (tiny["fit"], tiny["seen"], tiny["unseen"])
    assert st.disease_names_in_order(dfs, [10, 11, 12], contiguous=False) == ["dis10", "dis11", "dis12"]
    with pytest.raises(ValueError, match="not 0"):
        st.disease_names_in_order(dfs, [10, 11, 12], contiguous=True)
    with pytest.raises(ValueError, match="sorted"):
        st.disease_names_in_order(dfs, [11, 10, 12], contiguous=False)
    with pytest.raises(ValueError, match="cover"):
        st.disease_names_in_order(dfs, [10, 11], contiguous=False)
    renamed = tiny["seen"].copy()
    renamed["disease_name"] = "other"
    with pytest.raises(ValueError, match="more than one"):
        st.disease_names_in_order((tiny["fit"], renamed), [10, 11, 12], contiguous=False)
    zero_based = pd.DataFrame({"disease_id": [0, 1, 2], "disease_name": ["a", "b", "c"]})
    assert st.disease_names_in_order((zero_based,), [0, 1, 2], contiguous=True) == ["a", "b", "c"]


def test_clip_text_features_are_a_frozen_unit_norm_buffer():
    model = _clip_model()
    assert model.text_features.shape == (3, 512) and not model.text_features.requires_grad
    assert torch.allclose(model.text_features.norm(dim=1), torch.ones(3), atol=1e-5)
    assert "text_features" in dict(model.named_buffers()) and "text_features" in model.state_dict()
    assert all(p is not model.text_features for p in model.parameters())
    assert model.prompts == st.disease_prompts(CLIP_NAMES) and len(model.prompts[0]) == 4
    assert torch.equal(model.text_features, _clip_model().text_features)


def test_clip_text_features_stay_fixed_through_training():
    model = _clip_model()
    before_t, before_scale = model.text_features.clone(), model.logit_scale.detach().clone()
    opt = st.build_optimizer(model, 1e-3, 1e-1, 0.1)
    st.train_epoch(model, _clip_batch(), "disease", opt, torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
                   False)
    assert torch.equal(model.text_features, before_t)
    assert not torch.equal(model.logit_scale.detach(), before_scale)


def test_clip_head_is_scaled_cosine_to_text_features():
    model = _clip_model().eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        embeds = model.vision(pixel_values=images).image_embeds
        expected = model.scale() * torch.nn.functional.normalize(embeds, dim=-1) @ model.text_features.t()
        assert torch.allclose(model(images), expected, atol=1e-5)
    assert model(images).shape == (2, 3)
    assert model.scale().item() == pytest.approx(1 / 0.07, rel=1e-5)


def test_clip_logit_scale_is_clamped_at_100():
    model = _clip_model(init_scale=1000.0)
    assert model.scale().item() == pytest.approx(100.0, rel=1e-6)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).abs().max().item() <= 100.0 + 1e-3
    model.logit_scale.data.fill_(float(np.log(1000.0)))
    assert model.scale().item() == pytest.approx(100.0, rel=1e-6)
    model.clamp_logit_scale()
    assert model.logit_scale.item() == pytest.approx(float(np.log(100.0)), rel=1e-6)
    over = st.ClipDiseaseModel(model.vision, model.text_features, model.prompts, 500.0)
    assert over.scale().item() == pytest.approx(100.0, rel=1e-6) and over.logit_scale.item() <= float(np.log(100.0)) + 1e-6
    big_step = st.build_optimizer(model, 1e-3, 1e3, 0.1)
    st.train_epoch(model, _clip_batch(), "disease", big_step, torch.amp.GradScaler("cuda", enabled=False),
                   torch.device("cpu"), False)
    assert model.logit_scale.item() <= float(np.log(100.0)) + 1e-6


def test_clip_optimizer_groups():
    model = _clip_model()
    opt = st.build_optimizer(model, 1e-5, 1e-3, 0.1)
    assert [(g["lr"], g["weight_decay"]) for g in opt.param_groups] == [(1e-5, 0.1), (1e-3, 0.0)]
    assert opt.param_groups[1]["params"] == [model.logit_scale]
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_clip_pretrained_false_never_touches_the_network(monkeypatch, tmp_path):
    _no_network(monkeypatch)
    model = st.build_arm_model("disease_clip", False, 3, CLIP_NAMES)
    path = tmp_path / "clip.pt"
    st._atomic_save(st._checkpoint(model, "disease_clip", "disease", "b1", 1, ([1, 2, 3], [10, 11, 12]), 0.5), path)
    st.load_trained(path, "cpu")
    assert st.check_backbone("clip_vit_b16", pretrained=False) == (1, 512)
    with pytest.raises(AssertionError, match="from_pretrained"):
        st.check_backbone("clip_vit_b16", pretrained=True)


def test_clip_check_backbone_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="no backbone check"):
        st.check_backbone("convnext_tiny", pretrained=False)


def test_clip_pretrained_path_builds_text_anchored_head(monkeypatch):
    import math
    from types import SimpleNamespace

    import transformers
    tiny_text = transformers.CLIPTextConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                            num_attention_heads=2, max_position_embeddings=16, projection_dim=512,
                                            eos_token_id=2)
    text_model = transformers.CLIPTextModelWithProjection(tiny_text).eval()
    vision = transformers.CLIPVisionModelWithProjection(transformers.CLIPVisionConfig(**st.CLIP_TINY))
    seen_calls = []

    def serve(obj):
        def from_pretrained(name, *args, **kwargs):
            seen_calls.append(name)
            return obj
        return staticmethod(from_pretrained)

    flat = [text for group in st.disease_prompts(CLIP_NAMES) for text in group]
    ids = torch.randint(3, 100, (len(flat), 8))
    ids[:, -1] = 2

    def tokenizer(prompts, padding, return_tensors):
        assert prompts == flat and padding is True and return_tensors == "pt"
        return {"input_ids": ids}

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", serve(tokenizer))
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", serve(text_model))
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained",
                        serve(SimpleNamespace(logit_scale=torch.tensor(math.log(50.0)))))
    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection, "from_pretrained", serve(vision))
    model = st.build_arm_model("disease_clip", True, 3, CLIP_NAMES)
    with torch.no_grad():
        per_prompt = torch.nn.functional.normalize(text_model(input_ids=ids).text_embeds, dim=-1)
        expected = torch.nn.functional.normalize(per_prompt.view(3, 4, -1).mean(dim=1), dim=-1)
    assert torch.allclose(model.text_features, expected, atol=1e-6)
    assert model.scale().item() == pytest.approx(50.0, rel=1e-5) and model.vision is vision
    assert set(seen_calls) == {st.CLIP_CHECKPOINT} and len(seen_calls) == 4
    assert not any(isinstance(m, transformers.CLIPTextModelWithProjection) for m in model.modules())
    assert st.check_backbone("clip_vit_b16") == (1, 512)


def test_clip_checkpoint_roundtrip_gives_identical_predictions(tiny, tmp_path, monkeypatch):
    model = _clip_model().eval()
    path = tmp_path / "clip_roundtrip.pt"
    st._atomic_save(st._checkpoint(model, "disease_clip", "disease", "b1", 3, ([1, 2, 3], [10, 11, 12]), 0.5), path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["backbone"] == "clip_vit_b16" and ckpt["recipe"] == "clip" and ckpt["clip"]["prompts"] == model.prompts
    assert ckpt["clip"]["vision_config"]["projection_dim"] == 512 and "text_features" in ckpt["state_dict"]
    _no_network(monkeypatch)
    loaded = st.load_trained(path, "cpu")
    assert not loaded.training and isinstance(loaded, st.ClipDiseaseModel) and loaded.prompts == model.prompts
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    ids = ([1, 2, 3], [10, 11, 12])
    before = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    after = st.predict(loaded, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    assert before.shape == (4, 3) and np.array_equal(before, after)


def test_clip_predict_uses_clip_normalisation_and_others_keep_imagenet(tiny, monkeypatch):
    norms = []
    real = st.eval_transform

    def spy(norm=None):
        norms.append(norm)
        return real(norm)

    monkeypatch.setattr(st, "eval_transform", spy)
    ids = ([1, 2, 3], [10, 11, 12])
    st.predict(_clip_model().eval(), tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    st.predict(st.build_arm_model("disease_b1", False, 3).eval(), tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    assert norms == [st.NORMS["clip_vit_b16"], None]


@pytest.fixture(scope="module")
def clip_runs(tiny):
    events = {}
    context = {}

    def record(kind, value):
        events.setdefault(context["stage"], []).append((kind, value))

    real_opt, real_loss, real_train, real_eval = st.build_optimizer, st.nn.CrossEntropyLoss, st.train_transform, st.eval_transform
    real_sched = torch.optim.lr_scheduler.CosineAnnealingLR

    def opt_spy(model, backbone_lr, head_lr, weight_decay):
        opt = real_opt(model, backbone_lr, head_lr, weight_decay)
        record("optimizer", [(g["lr"], g["weight_decay"]) for g in opt.param_groups])
        return opt

    def loss_spy(*args, **kwargs):
        record("label_smoothing", kwargs.get("label_smoothing", 0.0))
        return real_loss(*args, **kwargs)

    def train_spy(aug, **kwargs):
        transform = real_train(aug, **kwargs)
        record("aug", aug)
        record("train_norm", (tuple(transform.transforms[-1].mean), tuple(transform.transforms[-1].std)))
        return transform

    def eval_spy(*args, **kwargs):
        transform = real_eval(*args, **kwargs)
        record("eval_norm", (tuple(transform.transforms[-1].mean), tuple(transform.transforms[-1].std)))
        return transform

    def sched_spy(optimizer, T_max, *args, **kwargs):
        record("t_max", T_max)
        return real_sched(optimizer, T_max, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(st, "build_optimizer", opt_spy)
    patch.setattr(st.nn, "CrossEntropyLoss", loss_spy)
    patch.setattr(st, "train_transform", train_spy)
    patch.setattr(st, "eval_transform", eval_spy)
    patch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", sched_spy)
    _no_network(patch)
    try:
        out1, out2 = tiny["root"] / "clip_stage1", tiny["root"] / "clip_stage2"
        context["stage"] = 1
        st.run_stage1("disease_clip", tiny["splits"], tiny["images"], out1, "cpu", epochs=2, workers=0, pretrained=False,
                      max_train_batches=1, **TINY)
        context["stage"] = 2
        st.run_stage2("disease_clip", out1, tiny["train_csv"], tiny["images"], out2, "cpu", epochs=2, workers=0,
                      max_train_batches=1, expect_rows=None)
    finally:
        patch.undo()
    return {"events": events, "stage1": out1, "stage2": out2}


def _clip_of(runs, stage, kind):
    return [value for k, value in runs["events"][stage] if k == kind]


CLIP_NORM_PAIR = (st.CLIP_MEAN, st.CLIP_STD)


def test_clip_stage1_run_uses_clip_recipe(clip_runs):
    assert _clip_of(clip_runs, 1, "optimizer") == [[(1e-5, 0.1), (1e-3, 0.0)]]
    assert _clip_of(clip_runs, 1, "label_smoothing") == [0.0, 0.0]
    assert _clip_of(clip_runs, 1, "t_max") == [30]
    assert _clip_of(clip_runs, 1, "aug") == ["b1"]
    assert _clip_of(clip_runs, 1, "train_norm") == [CLIP_NORM_PAIR]
    assert _clip_of(clip_runs, 1, "eval_norm") == [CLIP_NORM_PAIR, CLIP_NORM_PAIR]


def test_clip_stage2_run_uses_clip_recipe(clip_runs):
    assert _clip_of(clip_runs, 2, "optimizer") == [[(1e-6, 0.1), (1e-4, 0.0)]]
    assert _clip_of(clip_runs, 2, "label_smoothing") == [0.0, 0.0]
    assert _clip_of(clip_runs, 2, "t_max") == [10]
    assert _clip_of(clip_runs, 2, "aug") == ["b1"]
    assert _clip_of(clip_runs, 2, "train_norm") == [CLIP_NORM_PAIR]


def test_clip_run_outputs_and_frozen_text_features(tiny, clip_runs):
    config = json.loads((clip_runs["stage1"] / "disease_clip" / "config.json").read_text())
    assert config["backbone"] == "clip_vit_b16" and config["recipe"] == "clip" and config["pretrained"] is False
    assert config["prompts"] == st.disease_prompts(["dis10", "dis11", "dis12"])
    s1 = torch.load(clip_runs["stage1"] / "disease_clip" / "best.pt", map_location="cpu", weights_only=True)
    s2 = torch.load(clip_runs["stage2"] / "disease_clip" / "best_train_acc.pt", map_location="cpu", weights_only=True)
    assert s2["stage"] == 2 and s2["arm"] == "disease_clip" and s2["clip"]["prompts"] == config["prompts"]
    assert torch.equal(s1["state_dict"]["text_features"], s2["state_dict"]["text_features"])
    assert float(s2["state_dict"]["logit_scale"]) <= float(np.log(100.0)) + 1e-6
    history = pd.read_csv(clip_runs["stage2"] / "disease_clip" / "history.csv")
    assert list(history.columns) == st.HISTORY2 and history.epoch.tolist() == [1, 2]
    model = st.load_trained(clip_runs["stage2"] / "disease_clip" / "best_train_acc.pt", "cpu")
    out = st.predict(model, tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0)
    assert out.shape == (4, 3) and np.allclose(np.exp(out).sum(1), 1, atol=1e-4)


def test_clip_board_and_winner(clip_runs):
    board = st.stage1_board(clip_runs["stage1"], ["disease_clip"])
    assert board.backbone.tolist() == ["clip_vit_b16"] and board.recipe.tolist() == ["clip"]
    assert st.pick_winner(board) == "disease_clip"


class _NaNModel(torch.nn.Module):
    """Gives NaN outputs, as fp16 kernels did on the local GTX 1650 for eval batches of 15-17 images."""

    def __init__(self, n_out):
        super().__init__()
        self.lin = torch.nn.Linear(3, n_out)

    def forward(self, x):
        return self.lin(x.mean(dim=(2, 3))) * float("nan")


def test_evaluate_refuses_non_finite_outputs():
    batch = (torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="non-finite model outputs in evaluation"):
        st.evaluate(_NaNModel(3), [batch], "disease", torch.device("cpu"), False)


def test_predict_refuses_non_finite_outputs(tiny):
    with pytest.raises(ValueError, match="non-finite model outputs in predict"):
        st.predict(_NaNModel(3), tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0)


def test_train_epoch_refuses_non_finite_loss():
    model = _NaNModel(3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]), torch.tensor([1, 2]))]
    with pytest.raises(ValueError, match="training loss is not finite"):
        st.train_epoch(model, loader, "disease", opt, torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), False)


def test_job_cli_passes_the_amp_switch(monkeypatch):
    sys.path.insert(0, str(Path(st.__file__).resolve().parent.parent / "scripts"))
    import train_job
    calls = []
    monkeypatch.setattr(train_job, "run_stage1", lambda *a, **k: calls.append(("stage1", k["amp"])))
    monkeypatch.setattr(train_job, "run_stage2", lambda *a, **k: calls.append(("stage2", k["amp"])))
    base1 = ["stage1", "--arm", "crop_b1", "--images", "i", "--out", "o", "--splits", "s"]
    base2 = ["stage2", "--arm", "crop_b1", "--images", "i", "--out", "o", "--stage1", "s1", "--train-csv", "t"]
    train_job.main(base1)
    train_job.main(base1 + ["--no-amp"])
    train_job.main(base2)
    train_job.main(base2 + ["--no-amp"])
    assert calls == [("stage1", True), ("stage1", False), ("stage2", True), ("stage2", False)]


def _run_with_h_sequence(tiny, monkeypatch, out, h_values, **kwargs):
    values = iter(h_values)
    monkeypatch.setattr(st, "harmonic", lambda *a, **k: next(values))
    return st.run_stage1("crop_b1", tiny["splits"], tiny["images"], out, "cpu", epochs=len(h_values), workers=0,
                         pretrained=False, max_train_batches=1, **TINY, **kwargs)


def test_early_stop_halts_after_patience_and_keeps_best_epoch(tiny, tmp_path, monkeypatch, capsys):
    folder = _run_with_h_sequence(tiny, monkeypatch, tmp_path / "es", [0.1, 0.3, 0.2, 0.2, 0.2, 0.9, 0.9], patience=3)
    done = json.loads((folder / "done.json").read_text())
    assert done["best_epoch"] == 2 and done["last_epoch"] == 5 and done["stopped_early"] is True
    assert done["best_h"] == pytest.approx(0.3)
    history = pd.read_csv(folder / "history.csv")
    assert history.epoch.tolist() == [1, 2, 3, 4, 5]
    assert not (folder / "predictions_epoch_06.npz").exists()
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 2
    best, epoch_file = np.load(folder / "best_validation.npz"), np.load(folder / "predictions_epoch_02.npz")
    assert np.array_equal(best["seen"], epoch_file["seen"]) and np.array_equal(best["unseen"], epoch_file["unseen"])
    assert json.loads((folder / "config.json").read_text())["patience"] == 3
    assert "[crop_b1] early stop at epoch 5 (best epoch 2, patience 3)" in capsys.readouterr().out


def test_early_stop_does_not_fire_while_improving_or_at_the_last_epoch(tiny, tmp_path, monkeypatch, capsys):
    folder = _run_with_h_sequence(tiny, monkeypatch, tmp_path / "up", [0.1, 0.2, 0.3, 0.4], patience=2)
    done = json.loads((folder / "done.json").read_text())
    assert done["best_epoch"] == 4 and done["last_epoch"] == 4 and done["stopped_early"] is False
    assert "early stop" not in capsys.readouterr().out


def test_patience_none_runs_every_epoch(tiny, tmp_path, monkeypatch, capsys):
    folder = _run_with_h_sequence(tiny, monkeypatch, tmp_path / "none", [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    done = json.loads((folder / "done.json").read_text())
    assert done["best_epoch"] == 1 and done["last_epoch"] == 6 and done["stopped_early"] is False
    assert pd.read_csv(folder / "history.csv").epoch.tolist() == [1, 2, 3, 4, 5, 6]
    assert json.loads((folder / "config.json").read_text())["patience"] is None
    assert "early stop" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_invalid_patience_is_rejected(tiny, tmp_path, bad):
    with pytest.raises(ValueError, match="patience"):
        st.run_stage1("crop_b1", tiny["splits"], tiny["images"], tmp_path / "bad", "cpu", epochs=1, workers=0,
                      pretrained=False, max_train_batches=1, patience=bad, **TINY)
    assert not (tmp_path / "bad").exists()


def test_job_cli_passes_patience(monkeypatch):
    sys.path.insert(0, str(Path(st.__file__).resolve().parent.parent / "scripts"))
    import train_job
    calls = []
    monkeypatch.setattr(train_job, "run_stage1", lambda *a, **k: calls.append(k["patience"]))
    base = ["stage1", "--arm", "crop_b1", "--images", "i", "--out", "o", "--splits", "s"]
    train_job.main(base)
    train_job.main(base + ["--patience", "5"])
    assert calls == [None, 5]


def test_stage_amp_flag_reaches_training_and_eval(tiny, monkeypatch):
    seen = []
    real_train, real_eval = st.train_epoch, st.evaluate
    monkeypatch.setattr(st, "train_epoch", lambda *a, **k: (seen.append(("train", a[6])), real_train(*a, **k))[1])
    monkeypatch.setattr(st, "evaluate", lambda *a, **k: (seen.append(("eval", a[4])), real_eval(*a, **k))[1])
    st.run_stage1("crop_b1", tiny["splits"], tiny["images"], tiny["root"] / "amp_off", "cpu", epochs=1, workers=0,
                  pretrained=False, max_train_batches=1, amp=False, **TINY)
    assert seen and all(flag is False for _, flag in seen)


def _crop_clip_model(n=3):
    return st.build_arm_model("crop_clip", False, n, ["crop1", "crop2", "crop3"][:n])


def test_crop_clip_arm_builds_a_crop_text_anchored_head():
    model = _crop_clip_model()
    assert model.prompts == st.class_prompts("crop", ["crop1", "crop2", "crop3"])
    assert model.text_features.shape == (3, 512)
    with pytest.raises(ValueError, match="one crop name per class"):
        st.build_arm_model("crop_clip", False, 4, ["a", "b", "c"])


def test_clip_text_features_average_the_templates(monkeypatch):
    import transformers
    flat_count = 8
    embeds = torch.randn(flat_count, 512)

    class Text:
        def __call__(self, **kwargs):
            return type("Out", (), {"text_embeds": embeds})()

        def eval(self):
            return self

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", staticmethod(lambda name: lambda *a, **k: {}))
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", staticmethod(lambda name: Text()))
    features = st.clip_text_features([["a", "b", "c", "d"], ["e", "f", "g", "h"]], True)
    unit = torch.nn.functional.normalize(embeds, dim=-1)
    expected = torch.nn.functional.normalize(torch.stack([unit[:4].mean(0), unit[4:].mean(0)]), dim=-1)
    assert torch.allclose(features, expected, atol=1e-6)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_crop_clip_stage1_toy_run_writes_crop_prompts(tiny, tmp_path, monkeypatch):
    _no_network(monkeypatch)
    folder = st.run_stage1("crop_clip", tiny["splits"], tiny["images"], tmp_path / "crop_clip", "cpu", epochs=1, workers=0,
                           pretrained=False, max_train_batches=1, **TINY)
    config = json.loads((folder / "config.json").read_text())
    assert config["task"] == "crop" and config["prompts"] == st.class_prompts("crop", ["crop1", "crop2", "crop3"])
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["clip"]["prompts"] == config["prompts"] and ckpt["state_dict"]["text_features"].shape == (3, 512)
    model = st.load_trained(folder / "best.pt", "cpu")
    out = st.predict(model, tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0)
    assert out.shape == (4, 3)


def test_crop_clip_stage2_rebuild_and_flip_predict(tiny, tmp_path, monkeypatch):
    # The path taken on Kaggle when crop_clip wins on dev: stage 1, stage 2 from its best.pt, load, flip predict.
    _no_network(monkeypatch)
    st.run_stage1("crop_clip", tiny["splits"], tiny["images"], tmp_path / "s1", "cpu", epochs=1, workers=0,
                  pretrained=False, max_train_batches=1, **TINY)
    folder = st.run_stage2("crop_clip", tmp_path / "s1", tiny["train_csv"], tiny["images"], tmp_path / "s2", "cpu",
                           epochs=1, workers=0, max_train_batches=1, expect_rows=None)
    ckpt = torch.load(folder / "best_train_acc.pt", map_location="cpu", weights_only=True)
    assert ckpt["task"] == "crop" and ckpt["stage"] == 2
    assert ckpt["clip"]["prompts"] == st.class_prompts("crop", ["crop1", "crop2", "crop3"])
    model = st.load_trained(folder / "best_train_acc.pt", "cpu")
    plain = st.predict(model, tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0)
    flip = st.predict(model, tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0, flip=True)
    assert plain.shape == flip.shape == (4, 3)
    assert np.isfinite(flip).all() and np.allclose(np.exp(flip).sum(axis=1), 1.0, atol=1e-5)


def test_predict_flip_false_is_unchanged_and_flip_true_averages_probabilities(tiny):
    model = _clip_model().eval()
    ids = ([1, 2, 3], [10, 11, 12])
    plain = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0)
    default = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0, flip=False)
    assert np.array_equal(plain, default)
    flipped = st.predict(model, tiny["seen"], tiny["images"], *ids, "cpu", False, workers=0, flip=True)
    loader = st.make_loader(tiny["seen"], tiny["images"], *ids, st.eval_transform(model.NORM), st.EVAL_BATCH, False, 0)
    expected = []
    with torch.no_grad():
        for images, _, _ in loader:
            p = torch.softmax(model(images), dim=1)
            q = torch.softmax(model(images.flip(-1)), dim=1)
            expected.append(torch.log((p + q) / 2).numpy())
    expected = np.concatenate(expected)
    assert flipped.shape == plain.shape and np.allclose(flipped, expected, atol=1e-5)
    assert not np.allclose(flipped, plain, atol=1e-6)
    assert np.allclose(np.exp(flipped).sum(1), 1, atol=1e-4)


def test_predict_flip_on_a_symmetric_image_equals_no_flip(tmp_path):
    column = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1))
    symmetric = np.minimum(column, column[:, ::-1])
    image = np.stack([symmetric] * 3, axis=-1)
    assert np.array_equal(image, image[:, ::-1])
    Image.fromarray(image).save(tmp_path / "sym.png")
    df = pd.DataFrame([["sym.png", 1, 10, 110, "crop1", "dis10", "crop1_dis10"]], columns=st.COLUMNS)
    model = _clip_model().eval()
    ids = ([1, 2, 3], [10, 11, 12])
    plain = st.predict(model, df, tmp_path, *ids, "cpu", False, workers=0)
    flipped = st.predict(model, df, tmp_path, *ids, "cpu", False, workers=0, flip=True)
    assert np.allclose(plain, flipped, atol=1e-5)


def test_predict_flip_refuses_non_finite_outputs(tiny):
    with pytest.raises(ValueError, match="non-finite model outputs in predict"):
        st.predict(_NaNModel(3), tiny["seen"], tiny["images"], [1, 2, 3], [10, 11, 12], "cpu", False, workers=0, flip=True)


def test_clip_text_check_without_network(monkeypatch):
    _no_network(monkeypatch)
    assert st.clip_text_check("disease", CLIP_NAMES, pretrained=False) == (3, 512)
    assert st.clip_text_check("crop", ["a", "b"], pretrained=False) == (2, 512)
    with pytest.raises(AssertionError, match="from_pretrained"):
        st.clip_text_check("disease", CLIP_NAMES)


def test_clip_text_check_raises_on_wrong_shape(monkeypatch):
    import transformers

    class Text:
        def __call__(self, **kwargs):
            return type("Out", (), {"text_embeds": torch.randn(12, 256)})()

        def eval(self):
            return self

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", staticmethod(lambda name: lambda *a, **k: {}))
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", staticmethod(lambda name: Text()))
    with pytest.raises(ValueError, match="text features"):
        st.clip_text_check("disease", CLIP_NAMES)


def test_clip_text_check_raises_when_row_count_is_wrong(monkeypatch):
    import transformers

    class Text:
        def __call__(self, **kwargs):
            return type("Out", (), {"text_embeds": torch.randn(5, 512)})()

        def eval(self):
            return self

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", staticmethod(lambda name: lambda *a, **k: {}))
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", staticmethod(lambda name: Text()))
    with pytest.raises(ValueError, match="rows for 12 prompts"):
        st.clip_text_check("disease", CLIP_NAMES)


def test_clip_fp16_probe_runs_without_network_and_flags_non_finite(monkeypatch):
    _no_network(monkeypatch)
    assert st.clip_fp16_probe("cpu", batches=(1, 3), pretrained=False) == (1, 3)
    assert st.PROBE_BATCHES == (1, 8, 15, 16, 17, 32)

    class Broken(torch.nn.Module):
        def forward(self, pixel_values):
            return type("Out", (), {"image_embeds": torch.full((len(pixel_values), 512), float("nan"))})()

    monkeypatch.setattr(st, "_clip_vision", lambda pretrained: Broken())
    with pytest.raises(ValueError, match="set AMP = False"):
        st.clip_fp16_probe("cpu", batches=(2,), pretrained=False)


@pytest.mark.parametrize("task,names", [("disease", ["dis10", "dis11", "dis12"]), ("crop", ["crop1", "crop2", "crop3"])])
def test_clip_zero_shot_is_a_percentage_without_network(tiny, monkeypatch, task, names):
    _no_network(monkeypatch)
    score = st.clip_zero_shot(task, tiny["fit"], tiny["images"], names, "cpu", n=10, pretrained=False)
    assert 0.0 <= score <= 100.0
    assert (score * 10 / 100) == pytest.approx(round(score * 10 / 100))


def test_clip_zero_shot_rejects_unknown_names_and_duplicates(tiny):
    with pytest.raises(ValueError, match="not in the class names"):
        st.clip_zero_shot("disease", tiny["fit"], tiny["images"], ["dis10", "dis11"], "cpu", n=10, pretrained=False)
    with pytest.raises(ValueError, match="not unique"):
        st.clip_zero_shot("disease", tiny["fit"], tiny["images"], ["dis10", "dis10", "dis12"], "cpu", n=10, pretrained=False)
