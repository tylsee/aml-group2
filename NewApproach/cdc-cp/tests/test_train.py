"""A4: exact applied updates, no stop path, saved outputs. Tiny CPU runs on real images with weights=None."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cdccp.train as train
from cdccp.checkpoint import load_checkpoint
from cdccp.config import REPO, load_config
from cdccp.data import load_split
from cdccp.selection import budget_check, late_gain, moving_average, pick_epoch


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    def run_cfg(config="p1.yaml", extra=()):
        cfg = load_config(REPO / "configs" / config, [
            "pretrained=False", "batch_size=4", "epochs=2", "loader.num_workers=0", "loader.eval_batch=8", "bn.subset=8", "bn.batch=8",
            "amp=False", f"paths.runs_dir={tmp_path.as_posix()}", *extra,
        ])
        return cfg

    real = load_split

    def small(cfg, name):
        df = real(cfg, name)
        return df.sample(8 if name == "train_fit" else 4, random_state=0).reset_index(drop=True)

    monkeypatch.setattr(train, "load_split", small)
    return run_cfg


def test_run_has_exact_updates_and_saves_everything(tiny):
    cfg = tiny()
    out_dir, applied, skipped = train.run(cfg, device=torch.device("cpu"), use_cache=False)
    assert (applied, skipped) == (4, 0)
    history = pd.read_csv(out_dir / "history.csv")
    assert list(history.epoch) == [1, 2] and list(history.applied) == [2, 4]
    assert {"A_S", "A_U", "H", "train_ce_crop", "train_ce_disease", "val_ce_crop", "val_ce_disease"} <= set(history.columns)
    meta = load_checkpoint(out_dir / "epoch_02.pt")
    assert meta["applied"] == 4 and meta["stage"] == 1 and len(meta["training_files"]) == 8
    assert (out_dir / "val_02.npz").exists()


def test_skipped_steps_do_not_count_as_updates(tiny, monkeypatch):
    real_step = train.training_step
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        ok, parts = real_step(*args, **kwargs)
        return (False if calls["n"] == 2 else ok), parts

    monkeypatch.setattr(train, "training_step", flaky)
    cfg = tiny(extra=["max_skip_fraction=0.5"])
    _, applied, skipped = train.run(cfg, device=torch.device("cpu"), use_cache=False)
    assert (applied, skipped, calls["n"]) == (4, 1, 5)


def test_too_many_skips_abort(tiny, monkeypatch):
    monkeypatch.setattr(train, "training_step", lambda *a, **k: (False, {"ce_crop": 1.0}))
    with pytest.raises(RuntimeError, match="skipped"):
        train.run(tiny(), device=torch.device("cpu"), use_cache=False)


def test_two_view_run_and_supcon(tiny):
    cfg = tiny("p2_plain.yaml", ["epochs=1"])
    out_dir, applied, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False)
    assert applied == 2
    assert pd.read_csv(out_dir / "history.csv").train_sup_disease.notna().all()


def test_resume_continues_from_last_epoch(tiny):
    cfg = tiny()
    out_dir, applied, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    assert applied == 2
    _, applied, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False, resume=True)
    assert applied == 4
    assert list(pd.read_csv(out_dir / "history.csv").epoch) == [1, 2]


def test_cosine_reaches_zero():
    assert train.cosine_fraction(0, 100) == 1
    assert train.cosine_fraction(100, 100) == pytest.approx(0)


def test_warmup_rises_linearly_then_cosine_falls_to_zero():
    f = [train.lr_fraction(i, 100, 10) for i in range(101)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and f[10] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(f[:10], f[1:10]))
    assert all(a >= b for a, b in zip(f[10:], f[11:]))
    assert f[100] == pytest.approx(0)
    assert train.lr_fraction(5, 100, 0) == train.cosine_fraction(5, 100)


def test_p1_configs_differ_only_in_backbone_lr_and_name():
    e1, e2 = load_config(REPO / "configs" / "p1.yaml"), load_config(REPO / "configs" / "p1_lr1e-4.yaml")
    assert (e1["optim"]["lr_backbone"], e2["optim"]["lr_backbone"]) == (1e-5, 1e-4)
    for cfg in (e1, e2):
        cfg.pop("name")
        cfg["optim"].pop("lr_backbone")
    assert e1 == e2
    assert e1["optim"]["warmup_epochs"] == 1 and e1["epochs"] == 20 and e1["batch_size"] == 32


def test_rerun_without_resume_is_refused(tiny):
    cfg = tiny()
    train.run(cfg, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    with pytest.raises(FileExistsError):
        train.run(cfg, device=torch.device("cpu"), use_cache=False)


def test_resume_drops_history_rows_written_after_last_state(tiny):
    cfg = tiny()
    out_dir, _, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    history = pd.read_csv(out_dir / "history.csv")
    pd.concat([history, history.assign(epoch=2)]).to_csv(out_dir / "history.csv", index=False)  # a crash before last.pt
    train.run(cfg, device=torch.device("cpu"), use_cache=False, resume=True)
    assert list(pd.read_csv(out_dir / "history.csv").epoch) == [1, 2]


def test_resume_with_another_recipe_is_refused(tiny):
    train.run(tiny(), device=torch.device("cpu"), use_cache=False, max_epochs=1)
    with pytest.raises(ValueError, match="different config"):
        train.run(tiny(extra=["optim.lr_heads=0.5"]), device=torch.device("cpu"), use_cache=False, resume=True)


def test_p1_decide_runs_end_to_end(tiny, monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p1_decide", REPO / "scripts" / "p1_decide.py")
    decide = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decide)
    monkeypatch.setattr(decide, "load_split", train.load_split)  # the same small splits the runs used
    e1, _, _ = train.run(tiny("p1.yaml"), device=torch.device("cpu"), use_cache=False)
    e2, _, _ = train.run(tiny("p1_lr1e-4.yaml"), device=torch.device("cpu"), use_cache=False)
    out = tmp_path / "decision.md"
    monkeypatch.setattr("sys.argv", ["p1_decide.py", "--e1", str(e1), "--e2", str(e2), "--out", str(out), f"paths.runs_dir={tmp_path.as_posix()}"])
    decide.main()
    text = out.read_text(encoding="utf-8")
    assert "LR\\* = 1e-" in text and "a-priori value, E3 not run" in text and "seed=43" in text


def test_trainer_has_no_early_stopping_path():
    source = Path(train.__file__).read_text(encoding="utf-8").lower()
    code = "\n".join(line for line in source.splitlines() if "no early stopping" not in line)
    for word in ("early_stop", "earlystop", "patience"):
        assert word not in code


def test_selection_rule_handles_noise_and_late_rise():
    assert np.allclose(moving_average([1, 2, 3]), [1.5, 2, 2.5])
    one_epoch_spike = np.array([10, 30, 31, 60, 30, 29, 40, 41, 42])
    assert int(np.argmax(one_epoch_spike)) == 3
    assert pick_epoch(one_epoch_spike) == 8  # the last epoch averages two values: (41 + 42) / 2 beats the spike's 40.3
    rising = 5 + 40 * np.linspace(0, 1, 20)
    assert pick_epoch(rising) == 19
    assert late_gain(rising) > 1


def test_budget_check_fires_when_underfit_not_when_plateaued():
    rising = 5 + 40 * np.linspace(0, 1, 20)
    flat = 40 * (1 - np.exp(-np.arange(1, 21) / 2.0))
    assert budget_check(rising, rising + 0.2)
    assert not budget_check(flat, flat + 0.2)
    assert budget_check(rising)


def test_round2_configs_change_one_thing_each():
    load = lambda n: load_config(REPO / "configs" / n)
    base, aug, ls = load("p1b_base.yaml"), load("p1b_aug.yaml"), load("p1b_ls.yaml")
    assert base["optim"]["lr_backbone"] == 1e-4 and base["epochs"] == 12 and "aug" not in base
    assert ls["label_smoothing"] == 0.1 and aug["aug"]["crop_scale_min"] == 0.5
    for cfg, extra in ((aug, "aug"), (ls, "label_smoothing")):
        cfg = {k: v for k, v in cfg.items() if k not in ("name", extra)}
        assert cfg == {k: v for k, v in base.items() if k != "name"}


def test_aug_settings_reach_the_transform_and_reject_typos():
    from cdccp.data import train_tail
    rrc = train_tail({"aug": {"crop_scale_min": 0.5}}).transforms[0]
    assert rrc.scale == (0.5, 1.0)
    assert train_tail({}).transforms[0].scale == (0.85, 1.0)
    with pytest.raises(ValueError):
        train_tail({"aug": {"crop_scale": 0.5}})


def test_label_smoothing_changes_the_training_loss():
    out = {"crop_logits": torch.tensor([[5.0, 0.0]]), "disease_logits": torch.tensor([[0.0, 5.0]])}
    y = torch.tensor([0]), torch.tensor([1])
    plain, _ = train.p_loss(out, *y, 0.0, 0.0, 0.07)
    smooth, _ = train.p_loss(out, *y, 0.0, 0.0, 0.07, smoothing=0.1)
    assert smooth > plain


def test_p1_compare_keeps_a_clear_gain_and_refuses_unfinished(tiny, tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p1_compare", REPO / "scripts" / "p1_compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    root = tmp_path / "runs"
    for name, level in (("ctl", 50.0), ("good", 60.0)):
        for s in (42, 43):
            d = root / f"{name}_seed{s}"
            d.mkdir(parents=True)
            pd.DataFrame({"epoch": [1, 2, 3], "A_S": level + s % 2, "unseen_disease_top1": level / 2 + s % 2,
                          "unseen_crop_top1": 50.0, "A_U": 1.0, "H": 2.0}).to_csv(d / "history.csv", index=False)
            torch.save({"recipe": {"epochs": 3}}, d / "last.pt")
    out = tmp_path / "c.md"
    monkeypatch.setattr("sys.argv", ["p1_compare.py", str(root), "--control", "ctl", "--variants", "good", "--out", str(out)])
    compare.main()
    assert "**good: KEEP**" in out.read_text(encoding="utf-8")
    torch.save({"recipe": {"epochs": 5}}, root / "good_seed43" / "last.pt")
    with pytest.raises(SystemExit, match="unfinished"):
        compare.main()
