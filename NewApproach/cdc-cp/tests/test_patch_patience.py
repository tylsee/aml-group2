"""Patch-shuffle augmentation (teacher's tip) and the early-stopping wrapper around the unchanged trainer."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import cdccp.train as train
from cdccp.config import REPO, load_config
from cdccp.data import PatchShuffle, train_tail
from cdccp.patience import should_stop, train_with_patience
from test_train import tiny  # noqa: F401  (pytest fixture reused, not called directly)


def tiles(x, g):
    c, h, w = x.shape
    return x.reshape(c, g, h // g, g, w // g).permute(1, 3, 0, 2, 4).reshape(g * g, -1)


def test_patch_shuffle_moves_whole_tiles_and_keeps_every_pixel():
    torch.manual_seed(0)
    x = torch.arange(3 * 224 * 224, dtype=torch.float32).reshape(3, 224, 224)
    y = PatchShuffle(4, 1.0)(x)
    a, b = tiles(x, 4), tiles(y, 4)
    assert sorted(map(tuple, a[:, :3].tolist())) == sorted(map(tuple, b[:, :3].tolist()))
    assert all(any(torch.equal(t, s) for s in a) for t in b)
    assert not torch.equal(x, y)


def test_patch_shuffle_fires_about_p_of_the_time():
    x = torch.rand(3, 224, 224)
    torch.manual_seed(0)
    shuffle = PatchShuffle(2, 0.3)
    fired = sum(not torch.equal(shuffle(x), x) for _ in range(400))
    assert 70 < fired < 170


def test_patch_shuffle_rejects_bad_settings():
    for grid, p in ((3, 0.5), (1, 0.5), (4.0, 0.5), (True, 0.5), (2, 0.0), (2, 1.5)):
        with pytest.raises(ValueError):
            PatchShuffle(grid, p)


def test_patch_shuffle_is_off_by_default_and_on_when_configured():
    assert not any(isinstance(t, PatchShuffle) for t in train_tail({}).transforms)
    on = [t for t in train_tail({"aug": {"patch_grid": 2, "patch_p": 0.5}}).transforms if isinstance(t, PatchShuffle)]
    assert len(on) == 1 and (on[0].grid, on[0].p) == (2, 0.5)
    names = [type(t).__name__ for t in train_tail({"aug": {"patch_grid": 2, "patch_p": 0.5}}).transforms]
    assert names.index("PatchShuffle") == names.index("ToTensor") + 1


def test_patch_config_changes_only_the_patch_settings():
    base, patch = (load_config(REPO / "configs" / n) for n in ("p1b_base.yaml", "p1b_patch.yaml"))
    assert patch["aug"]["patch_grid"] >= 2 and 0 < patch["aug"]["patch_p"] < 1
    strip = lambda c: {k: v for k, v in c.items() if k not in ("name", "aug")}
    assert strip(patch) == strip(base) and "aug" not in base


@pytest.mark.parametrize("scores, patience, stop", [
    ([10, 12, 11, 11, 11], 3, True),
    ([10, 12, 11, 11], 3, False),
    ([10, 12, 11, 13, 12, 12], 3, False),
    ([10, 10, 10, 10], 3, True),
    ([5], 3, False),
])
def test_should_stop_after_patience_epochs_without_a_new_best(scores, patience, stop):
    assert should_stop(scores, patience) is stop


def test_wrapper_stops_early_and_records_why(tiny, monkeypatch):  # noqa: F811
    scores = iter([10.0, 12.0, 11.0, 11.0, 11.0, 99.0])
    real_eval = train.evaluate_epoch

    def fake_eval(*args, **kwargs):
        row, seen, unseen = real_eval(*args, **kwargs)
        return {**row, "unseen_disease_top1": next(scores)}, seen, unseen

    monkeypatch.setattr(train, "evaluate_epoch", fake_eval)
    cfg = tiny(extra=["epochs=6"])
    out_dir, best = train_with_patience(cfg, patience=3, device=torch.device("cpu"), use_cache=False)
    assert best == 2
    assert list(pd.read_csv(out_dir / "history.csv").epoch) == [1, 2, 3, 4, 5]
    record = json.loads((out_dir / "early_stop.json").read_text())
    assert {k: v for k, v in record.items() if k != "recipe"} == {
        "metric": "unseen_disease_top1", "patience": 3, "best_epoch": 2, "best": 12.0, "stopped_after_epoch": 5, "planned_epochs": 6}
    again, best_again = train_with_patience(cfg, patience=3, device=torch.device("cpu"), use_cache=False, resume=True)
    assert (again, best_again) == (out_dir, 2) and len(pd.read_csv(out_dir / "history.csv")) == 5
    with pytest.raises(ValueError, match="another metric, patience or config"):
        train_with_patience(tiny(extra=["epochs=7"]), patience=3, device=torch.device("cpu"), use_cache=False, resume=True)


def test_wrapper_runs_all_epochs_when_the_metric_keeps_rising(tiny):  # noqa: F811
    out_dir, best = train_with_patience(tiny(), patience=3, device=torch.device("cpu"), use_cache=False)
    assert list(pd.read_csv(out_dir / "history.csv").epoch) == [1, 2]
    assert json.loads((out_dir / "early_stop.json").read_text())["stopped_after_epoch"] == 2


def test_resume_after_a_crash_skips_finished_epochs(tiny, monkeypatch):  # noqa: F811
    cfg = tiny(extra=["epochs=3"])
    real_run = train.run

    def crash_at_3(*a, **k):
        if k["max_epochs"] == 3:
            raise RuntimeError("session killed")
        return real_run(*a, **k)

    monkeypatch.setattr(train, "run", crash_at_3)
    with pytest.raises(RuntimeError, match="session killed"):
        train_with_patience(cfg, patience=3, device=torch.device("cpu"), use_cache=False)
    calls = []
    monkeypatch.setattr(train, "run", lambda *a, **k: calls.append(k["max_epochs"]) or real_run(*a, **k))
    out_dir, _ = train_with_patience(cfg, patience=3, device=torch.device("cpu"), use_cache=False, resume=True)
    assert calls == [2, 3] and list(pd.read_csv(out_dir / "history.csv").epoch) == [1, 2, 3]


def test_resume_refuses_plain_trainer_runs_and_other_settings(tiny):  # noqa: F811
    cfg = tiny(extra=["epochs=3"])
    train.run(cfg, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    with pytest.raises(ValueError, match="did not start"):
        train_with_patience(cfg, patience=3, device=torch.device("cpu"), use_cache=False, resume=True)
    other = tiny(extra=["epochs=3", "seed=43"])
    train_with_patience(other, patience=1, device=torch.device("cpu"), use_cache=False)
    (Path(other["paths"]["runs_dir"]) / f"{other['name']}_seed43" / "early_stop.json").unlink()
    with pytest.raises(ValueError, match="another metric, patience or config"):
        train_with_patience(other, patience=3, device=torch.device("cpu"), use_cache=False, resume=True)


def test_a_refused_fresh_start_does_not_relabel_the_run(tiny):  # noqa: F811
    plain = tiny(extra=["epochs=3"])
    train.run(plain, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    wrapped = tiny(extra=["epochs=3", "seed=43"])
    train_with_patience(wrapped, patience=1, device=torch.device("cpu"), use_cache=False)
    (Path(wrapped["paths"]["runs_dir"]) / f"{wrapped['name']}_seed43" / "early_stop.json").unlink()
    for cfg, patience in ((plain, 3), (wrapped, 3)):
        with pytest.raises(FileExistsError, match="Pass --resume"):
            train_with_patience(cfg, patience=patience, device=torch.device("cpu"), use_cache=False)
        with pytest.raises(ValueError, match="did not start|another metric"):
            train_with_patience(cfg, patience=patience, device=torch.device("cpu"), use_cache=False, resume=True)


def test_memory_runs_and_nan_scores_are_refused(tiny):  # noqa: F811
    with pytest.raises(ValueError, match="pair memory"):
        train_with_patience(tiny(extra=["memory={'per_pair': 4}"]), device=torch.device("cpu"), use_cache=False)
    with pytest.raises(ValueError, match="NaN"):
        should_stop([float("nan"), 1, 2, 3, 4], 3)


def test_cli_takes_overrides_before_or_after_options():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_patience", REPO / "scripts" / "train_patience.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for argv in (["c.yaml", "--patience", "2", "--resume", "epochs=2", "seed=43"], ["c.yaml", "epochs=2", "--patience", "2", "seed=43", "--resume"]):
        args = cli.parse(argv)
        assert (args.config, args.overrides, args.patience, args.resume) == ("c.yaml", ["epochs=2", "seed=43"], 2, True)


def test_eval_transform_never_shuffles(cfg):
    from cdccp.data import DPDDataset, load_split
    df = load_split(cfg, "train_fit").head(1)
    patched = {**cfg, "aug": {"patch_grid": 4, "patch_p": 1.0}}
    assert torch.equal(DPDDataset(df, patched, "eval")[0][0], DPDDataset(df, cfg, "eval")[0][0])
