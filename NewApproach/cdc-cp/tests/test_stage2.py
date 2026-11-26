"""Stage 2: continue a parent checkpoint on the official training split. Tiny CPU runs on real images with weights=None."""
import importlib.util
import itertools
import json
import math
import shutil

import pandas as pd
import pytest
import torch
import torch.nn as nn

import cdccp.stage2 as stage2
import cdccp.train as train
from cdccp.checkpoint import file_sha256, load_checkpoint
from cdccp.config import REPO, load_config
from cdccp.data import load_split
from cdccp.models import BACKBONES, build_model

CPU = torch.device("cpu")
TRAIN_ROWS = 12
STEPS = TRAIN_ROWS // 4
SMALL = ["loader.num_workers=0", "loader.eval_batch=8"]


def tiny_backbone(pretrained):
    return (nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.Dropout2d(0.5)), nn.AdaptiveAvgPool2d(1), nn.BatchNorm2d(8), 8)


def make_parent(root, config, extra=()):
    mp = pytest.MonkeyPatch()
    real = load_split
    mp.setitem(BACKBONES, "tiny_test", tiny_backbone)
    mp.setattr(train, "load_split", lambda cfg, name: real(cfg, name).sample(8 if name == "train_fit" else 4, random_state=0).reset_index(drop=True))
    try:
        cfg = load_config(REPO / "configs" / config, [
            "pretrained=False", "batch_size=4", "epochs=2", *SMALL, "bn.subset=8", "bn.batch=8", "amp=False",
            f"paths.runs_dir={root.as_posix()}", *extra,
        ])
        out_dir, _, _ = train.run(cfg, device=CPU, use_cache=False)
    finally:
        mp.undo()
    return sorted(out_dir.glob("epoch_*.pt"))[-1]


@pytest.fixture(scope="module")
def parent_p1(tmp_path_factory):
    return make_parent(tmp_path_factory.mktemp("p1"), "p1.yaml", ["max_skip_fraction=0.5"])


@pytest.fixture(scope="module")
def parent_p2(tmp_path_factory):
    return make_parent(tmp_path_factory.mktemp("p2"), "p2_plain.yaml", ["epochs=1"])


@pytest.fixture(scope="module")
def parent_spy(tmp_path_factory):
    return make_parent(tmp_path_factory.mktemp("spy"), "p2c_cross_spy.yaml", ["epochs=1", "model.backbone=tiny_test", "model.hidden=[8]"])


@pytest.fixture(scope="module")
def parent_frozen(tmp_path_factory):
    return make_parent(tmp_path_factory.mktemp("frozen"), "p1.yaml", ["model.backbone=tiny_test", "model.hidden=[8]", "model.freeze_backbone=True"])


@pytest.fixture
def splits(monkeypatch):
    """Stage 2 sees a small 'train' (12 rows) and records every split it loads."""
    real, loaded = load_split, []
    monkeypatch.setitem(BACKBONES, "tiny_test", tiny_backbone)

    def small(cfg, name):
        loaded.append(name)
        if name not in ("train", "train_fit"):
            raise AssertionError(f"Stage 2 loaded {name}")
        return real(cfg, name).sample(TRAIN_ROWS if name == "train" else 8, random_state=1).reset_index(drop=True)

    monkeypatch.setattr(stage2, "load_split", small)
    return loaded


def go(parent, out, *extra):
    return stage2.run_stage2(parent, out, [*SMALL, "bn.subset=8", *extra], device=CPU, use_cache=False)


def params(path):
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    return {k: v for k, v in state.items() if "running_" not in k and "num_batches" not in k}


def test_exact_updates_history_meta_and_gate_file(parent_p1, splits, tmp_path):
    out, sha = go(parent_p1, tmp_path)
    assert sha == file_sha256(out / "stage2_final.pt")
    history = pd.read_csv(out / "stage2_history.csv")
    assert list(history.epoch) == [1, 2] and list(history.applied) == [STEPS, 2 * STEPS] and list(history.skipped) == [0, 0]
    assert {"lr_backbone", "lr_heads", "train_ce_crop", "train_ce_disease", "train_sup_crop", "train_sup_disease", "train_loss", "seconds"} <= set(history.columns)
    meta = load_checkpoint(out / "stage2_final.pt")
    assert (meta["stage"], meta["applied"], meta["skipped"]) == (2, 2 * STEPS, 0)
    assert meta["parent_sha256"] == file_sha256(parent_p1) and meta["parent"] == str(parent_p1)
    assert meta["parent_epoch"] == load_checkpoint(parent_p1)["epoch"] == 2
    assert meta["stage2"] == {"epochs": 2, "lr_backbone": 1e-5, "lr_heads": 1e-4, "warmup_epochs": 0, "train_split": "train"}
    assert meta["config"]["stage2"] == meta["stage2"] and len(meta["training_files"]) == TRAIN_ROWS and meta["bn_policy"] == "recomputed_on_subset"
    gate = json.loads((out / "stage2_gate.json").read_text(encoding="utf-8"))
    assert gate["ok"] == stage2.stage2_gate(history)[0] and gate["train_loss_by_epoch"] == pytest.approx(list(history.train_loss))


def test_overrides_reach_the_settings(parent_p1, splits, tmp_path):
    out, _ = go(parent_p1, tmp_path, "stage2.epochs=1", "stage2.lr_heads=0.5")
    meta = load_checkpoint(out / "stage2_final.pt")
    assert meta["stage2"]["epochs"] == 1 and meta["stage2"]["lr_heads"] == 0.5 and meta["applied"] == STEPS
    with pytest.raises(ValueError, match="unknown Stage 2 setting"):
        go(parent_p1, tmp_path / "typo", "stage2.epoch=1")


def test_train_split_choice_is_used_and_recorded(parent_p1, splits, tmp_path):
    out, _ = go(parent_p1, tmp_path, "stage2.train_split=train_fit", "stage2.epochs=1")
    meta = load_checkpoint(out / "stage2_final.pt")
    assert splits == ["train_fit"] and meta["stage2"]["train_split"] == "train_fit"
    assert meta["applied"] == 8 // 4 * 1 and len(meta["training_files"]) == 8


def test_bad_train_split_raises(parent_p1, splits, tmp_path):
    for bad in ("val_seen", "val_unseen", "test", "''"):
        with pytest.raises(ValueError, match="train_split"):
            go(parent_p1, tmp_path / bad, f"stage2.train_split={bad}")
    assert splits == []


def test_skipped_updates_are_not_counted(parent_p1, splits, tmp_path, monkeypatch):
    real, calls = stage2.training_step, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        ok, parts = real(*args, **kwargs)
        return (False if calls["n"] == 2 else ok), parts

    monkeypatch.setattr(stage2, "training_step", flaky)
    out, _ = go(parent_p1, tmp_path)
    meta = load_checkpoint(out / "stage2_final.pt")
    assert (meta["applied"], meta["skipped"], calls["n"]) == (6, 1, 7)


def test_too_many_skips_abort(parent_p1, splits, tmp_path, monkeypatch):
    monkeypatch.setattr(stage2, "training_step", lambda *a, **k: (False, {"ce_crop": 1.0}))
    with pytest.raises(RuntimeError, match="skipped"):
        go(parent_p1, tmp_path)
    assert not (tmp_path / "stage2_final.pt").exists()


def test_parent_weights_are_the_start(parent_p1, splits, tmp_path):
    parent = params(parent_p1)
    zero, _ = go(parent_p1, tmp_path / "lr0", "stage2.lr_backbone=0.0", "stage2.lr_heads=0.0")
    after_zero = params(zero / "stage2_final.pt")
    assert all(torch.equal(after_zero[k], parent[k]) for k in parent)
    fresh = build_model(load_checkpoint(parent_p1)["config"], pretrained=False).state_dict()
    assert any(not torch.equal(fresh[k], parent[k]) for k in parent)
    moved, _ = go(parent_p1, tmp_path / "lr", "stage2.lr_backbone=1e-3", "stage2.lr_heads=1e-3")
    after = params(moved / "stage2_final.pt")
    assert any(not torch.equal(after[k], parent[k]) for k in parent if k.startswith("features"))
    assert any(not torch.equal(after[k], parent[k]) for k in parent if "classifier" in k)


def test_no_development_evaluation(parent_p1, splits, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("development evaluation ran in Stage 2")

    monkeypatch.setattr(train, "evaluate_epoch", boom)
    for target in ("cdccp.evaluate.split_metrics", "cdccp.evaluate.check_rows", "cdccp.extract.extract"):
        monkeypatch.setattr(target, boom)
    go(parent_p1, tmp_path)
    assert set(splits) == {"train"}


def test_lr_is_cosine_from_stage2_base_rates_without_warmup(parent_p1, splits, tmp_path, monkeypatch):
    real, seen = stage2.training_step, []

    def spy(model, optimiser, *args, **kwargs):
        seen.append([g["lr"] for g in optimiser.param_groups])
        return real(model, optimiser, *args, **kwargs)

    monkeypatch.setattr(stage2, "training_step", spy)
    out, _ = go(parent_p1, tmp_path, "stage2.lr_backbone=3e-5")
    total = 2 * STEPS
    assert len(seen) == total
    for i, (lr_b, lr_h) in enumerate(seen):
        f = 0.5 * (1 + math.cos(math.pi * i / total))
        assert lr_b == pytest.approx(3e-5 * f) and lr_h == pytest.approx(1e-4 * f)
    assert seen[0] == pytest.approx([3e-5, 1e-4])
    history = pd.read_csv(out / "stage2_history.csv")
    assert history.lr_backbone.iloc[-1] == pytest.approx(seen[-1][0]) and history.lr_heads.iloc[0] > history.lr_heads.iloc[-1]


def test_warmup_epochs_setting_ramps_the_lr(parent_p1, splits, tmp_path, monkeypatch):
    real, seen = stage2.training_step, []

    def spy(model, optimiser, *args, **kwargs):
        seen.append(optimiser.param_groups[1]["lr"])
        return real(model, optimiser, *args, **kwargs)

    monkeypatch.setattr(stage2, "training_step", spy)
    go(parent_p1, tmp_path, "stage2.warmup_epochs=1")
    assert seen[0] == pytest.approx(1e-4 / STEPS) and seen[STEPS - 1] == pytest.approx(1e-4)


def test_frozen_parent_stays_frozen(parent_frozen, splits, tmp_path, monkeypatch):
    real, trainable = stage2.training_step, []

    def spy(model, optimiser, *args, **kwargs):
        backbone = list(model.features.parameters()) + list(model.post.parameters()) + optimiser.param_groups[0]["params"]
        trainable.append(sum(p.numel() for p in backbone if p.requires_grad))
        return real(model, optimiser, *args, **kwargs)

    monkeypatch.setattr(stage2, "training_step", spy)
    out, _ = go(parent_frozen, tmp_path, "stage2.lr_heads=1e-2")
    assert len(trainable) == 2 * STEPS and set(trainable) == {0}
    before = torch.load(parent_frozen, weights_only=True)["model"]
    after = torch.load(out / "stage2_final.pt", weights_only=True)["model"]
    backbone = [k for k in before if k.startswith(("features", "post"))]
    assert backbone and all(torch.equal(before[k], after[k]) for k in backbone)
    assert any(not torch.equal(before[k], after[k]) for k in before if "classifier" in k)
    assert load_checkpoint(out / "stage2_final.pt")["bn_policy"] == "frozen_pretrained"


def test_p2_parent_runs_with_supcon_in_the_loss(parent_p2, splits, tmp_path):
    out, _ = go(parent_p2, tmp_path)
    h = pd.read_csv(out / "stage2_history.csv")
    assert h.train_sup_crop.gt(0).all() and h.train_sup_disease.gt(0).all()
    expected = h.train_ce_crop + h.train_ce_disease + 0.1 * h.train_sup_crop + 0.1 * h.train_sup_disease
    assert list(h.train_loss) == pytest.approx(list(expected))


def test_spy_parent_logs_spy_parts_and_weights_the_loss(parent_spy, splits, tmp_path):
    out, _ = go(parent_spy, tmp_path)
    h = pd.read_csv(out / "stage2_history.csv")
    assert {"train_spy_ce", "train_spy_acc"} <= set(h.columns)
    expected = h.train_ce_crop + h.train_ce_disease + 0.1 * h.train_sup_crop + 0.1 * h.train_sup_disease + 0.1 * h.train_spy_ce
    assert list(h.train_loss) == pytest.approx(list(expected))


def test_gate_cases():
    gate = lambda losses: stage2.stage2_gate(pd.DataFrame({"train_loss": losses}))
    assert gate([3.0, 2.0, 1.0])[0] and gate([2.0, 2.0])[0] and gate([1.0])[0]
    ok, reason = gate([1.0, 2.0])
    assert not ok and "rose" in reason
    for bad in ([1.0, float("nan")], [float("inf"), 1.0], [float("nan")]):
        ok, reason = gate(bad)
        assert not ok and "non-finite" in reason
    assert not gate([])[0]
    assert not gate([1.0, 0.5, 1.5])[0]


def test_failing_gate_is_written_not_raised(parent_p1, splits, tmp_path, monkeypatch):
    rising = itertools.count()
    monkeypatch.setattr(stage2, "_train_loss", lambda means, cfg: float(next(rising)))
    out, _ = go(parent_p1, tmp_path)
    gate = json.loads((out / "stage2_gate.json").read_text(encoding="utf-8"))
    assert gate["ok"] is False and "rose" in gate["reason"] and gate["train_loss_by_epoch"] == [0.0, 1.0]
    assert (out / "stage2_final.pt").exists()


def test_rerun_is_nothing_to_do(parent_p1, splits, tmp_path, monkeypatch, capsys):
    out, sha = go(parent_p1, tmp_path)
    mtime = (out / "stage2_final.pt").stat().st_mtime_ns

    def boom(*a, **k):
        raise AssertionError("trained again")

    monkeypatch.setattr(stage2, "training_step", boom)
    capsys.readouterr()
    out2, sha2 = go(parent_p1, tmp_path)
    assert "nothing to do" in capsys.readouterr().out
    assert (out2, sha2) == (out, sha) and (out / "stage2_final.pt").stat().st_mtime_ns == mtime


def test_partial_out_dir_restarts_from_the_parent(parent_p1, splits, tmp_path):
    pd.DataFrame({"epoch": [1, 2], "train_loss": [99.0, 99.0]}).to_csv(tmp_path / "stage2_history.csv", index=False)
    (tmp_path / "stage2_gate.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stage2_final.pt.tmp").write_bytes(b"junk")
    out, _ = go(parent_p1, tmp_path)
    history = pd.read_csv(out / "stage2_history.csv")
    assert list(history.epoch) == [1, 2] and (history.train_loss < 99).all()
    assert json.loads((out / "stage2_gate.json").read_text(encoding="utf-8")) != {}
    assert not (out / "stage2_final.pt.tmp").exists() and load_checkpoint(out / "stage2_final.pt")["applied"] == 2 * STEPS


def test_mismatching_complete_run_raises(parent_p1, splits, tmp_path):
    out, sha = go(parent_p1, tmp_path)
    with pytest.raises(ValueError, match="different parent or Stage 2 settings"):
        go(parent_p1, tmp_path, "stage2.epochs=1")
    other = tmp_path / "other_parent.pt"
    shutil.copy(parent_p1, other)
    with other.open("ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="different parent or Stage 2 settings"):
        go(other, tmp_path)
    assert file_sha256(out / "stage2_final.pt") == sha


def test_batchnorm_is_recomputed_once_after_the_last_update(parent_p1, splits, tmp_path, monkeypatch):
    events = []
    real_bn, real_step = stage2.recompute_bn, stage2.training_step

    def bn(model, loader, device):
        events.append(("bn", len(loader.dataset)))
        return real_bn(model, loader, device)

    def step(*args, **kwargs):
        events.append("step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(stage2, "recompute_bn", bn)
    monkeypatch.setattr(stage2, "training_step", step)
    go(parent_p1, tmp_path)
    assert events == ["step"] * (2 * STEPS) + [("bn", 8)]


def test_cli_prints_gate_and_exits_zero_on_failure(monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("stage2_cli", REPO / "scripts" / "stage2.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path / "stage2_gate.json").write_text(json.dumps({"ok": False, "reason": "rose", "train_loss_by_epoch": [1, 2]}), encoding="utf-8")
    seen = {}

    def fake(parent, out, overrides, use_cache):
        seen.update(p=parent, o=overrides, c=use_cache)
        return tmp_path, "abc"

    monkeypatch.setattr(cli, "run_stage2", fake)
    monkeypatch.setattr("sys.argv", ["stage2.py", "parent.pt", str(tmp_path), "stage2.epochs=1", "--no-cache"])
    cli.main()
    assert seen == {"p": "parent.pt", "o": ["stage2.epochs=1"], "c": False}
    assert "gate: FAIL (rose)" in capsys.readouterr().out
