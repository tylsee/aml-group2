import math

import pandas as pd
import pytest
import torch
import torch.nn as nn

import cdccp.train as train
from cdccp.checkpoint import load_checkpoint
from cdccp.config import REPO, load_config
from cdccp.data import load_split
from cdccp.losses import GradReverse, grl_lambda, spy_loss, supcon
from cdccp.memory import PairMemory
from cdccp.models import build_model


def unit(rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


Z = unit([[1, 0.2, 0], [0.9, 0, 0.3], [0, 1, 0.1], [0.1, 0.8, 0.5], [0.4, 0.4, 0.9], [1, 0.3, 0.1]])
Y = [0, 0, 1, 1, 0, 0]
G = [0, 1, 0, 1, 2, 0]


def reference_cross(z, y, g, tau):
    """Row by row loop: positives share the label and differ in group, same-label same-group rows leave the denominator."""
    n, losses = len(z), []
    for i in range(n):
        kept = [j for j in range(n) if not (y[j] == y[i] and g[j] == g[i])]
        positives = [j for j in range(n) if y[j] == y[i] and g[j] != g[i]]
        if not positives:
            continue
        log_den = math.log(sum(math.exp(float(z[i] @ z[j]) / tau) for j in kept))
        losses.append(-sum(float(z[i] @ z[j]) / tau - log_den for j in positives) / len(positives))
    return sum(losses) / len(losses)


def test_cross_supcon_matches_hand_computation():
    got = supcon(Z, torch.tensor(Y), 0.5, group=torch.tensor(G))
    assert got.item() == pytest.approx(reference_cross(Z, Y, G, 0.5), abs=1e-5)


def test_same_label_same_group_rows_neither_pull_nor_push():
    z = unit([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    y, g = [0, 0, 1, 0], [0, 1, 0, 0]
    # Only anchor a is valid (n has no other-group positive). Memory row m shares a's label and group, so moving it
    # must not change the loss, which is then exactly -log softmax of a's single positive b against b and n.
    a, n, b = unit([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]])
    anchors, anchor_y, anchor_g = torch.stack([a, n]), torch.tensor([0, 1]), torch.tensor([0, 0])
    mem_y, mem_g = torch.tensor([0, 0]), torch.tensor([1, 0])
    loss_with = lambda m: supcon(anchors, anchor_y, 0.5, mem_z=torch.stack([b, m]), mem_y=mem_y, group=anchor_g, mem_group=mem_g).item()
    first, second = loss_with(unit([[0, 0, 1]])[0]), loss_with(unit([[-0.7, 0.2, 0.6]])[0])
    ab, an = (a @ b).item() / 0.5, (a @ n).item() / 0.5
    assert first == pytest.approx(second, abs=1e-6)
    assert first == pytest.approx(-(ab - math.log(math.exp(ab) + math.exp(an))), abs=1e-5)
    full = supcon(z, torch.tensor(y), 0.5, group=torch.tensor(g))
    assert full.item() == pytest.approx(reference_cross(z, y, g, 0.5), abs=1e-5)


def test_anchor_with_only_same_group_positives_is_skipped():
    z = unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]).requires_grad_()
    loss = supcon(z, torch.tensor([0, 0, 1]), 0.5, group=torch.tensor([0, 0, 1]))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(z.grad) == 0


def test_cross_supcon_with_memory_uses_memory_groups():
    mem_z = unit([[1, 0.1, 0]])
    z = unit([[1, 0, 0], [0, 1, 0]])
    loss = supcon(z, torch.tensor([0, 1]), 0.5, mem_z, torch.tensor([0]), group=torch.tensor([0, 0]), mem_group=torch.tensor([5]))
    to_memory, to_other = float(z[0] @ mem_z[0]) / 0.5, float(z[0] @ z[1]) / 0.5
    assert loss.item() == pytest.approx(-(to_memory - math.log(math.exp(to_memory) + math.exp(to_other))), abs=1e-5)


def test_plain_mode_matches_the_original_formula():
    tau, y = 0.5, torch.tensor(Y)
    sim, expected = Z @ Z.T / tau, []
    for i in range(len(Z)):
        others = [j for j in range(len(Z)) if j != i]
        log_den = torch.logsumexp(sim[i, others], 0)
        expected.append(-torch.stack([sim[i, j] - log_den for j in others if Y[j] == Y[i]]).mean())
    assert supcon(Z, y, tau).item() == pytest.approx(torch.stack(expected).mean().item(), abs=1e-5)


def test_memory_keeps_last_per_pair_with_aligned_labels():
    memory = PairMemory(2)
    assert memory.tensors() is None
    h = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)
    memory.update(h[:5], torch.tensor([0, 0, 0, 1, 1]), torch.tensor([7, 7, 7, 7, 8]))
    memory.update(h[5:6], torch.tensor([0]), torch.tensor([7]))
    stored, crop, disease = memory.tensors()
    assert stored.shape == (4, 3) and len(crop) == len(disease) == 4
    found = {(c, d): sorted(row[0] for row in stored[(crop == c) & (disease == d)].tolist()) for c, d in {(0, 7), (1, 7), (1, 8)}}
    assert found == {(0, 7): [2.0, 5.0], (1, 7): [3.0], (1, 8): [4.0]}


def test_memory_rejects_zero_capacity():
    with pytest.raises(ValueError):
        PairMemory(0)


def test_grad_reverse_is_identity_forward_and_negates_gradient():
    x = torch.randn(4, 3, requires_grad=True)
    y = GradReverse.apply(x, 0.7)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.7))


def test_grl_lambda_schedule():
    assert grl_lambda(0, 100) == 0.0
    assert grl_lambda(100, 100) == pytest.approx(2 / (1 + math.exp(-10)) - 1)


def test_spy_loss_uses_only_multi_crop_rows_and_is_zero_without_them():
    torch.manual_seed(0)
    spy, emb = nn.Linear(4, 3), torch.randn(5, 4, requires_grad=True)
    crop = torch.tensor([0, 1, 2, 0, 1])
    rows = torch.tensor([True, False, True, False, False])
    loss, acc = spy_loss(spy, emb, crop, rows, 0.5)
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(spy(emb)[rows], crop[rows]).item())
    assert 0.0 <= acc <= 1.0
    zero, acc = spy_loss(spy, emb, crop, torch.zeros(5, dtype=torch.bool), 0.5)
    assert zero.item() == 0.0 and acc == 0.0
    zero.backward()
    assert torch.count_nonzero(emb.grad) == 0


def test_spy_gradient_into_the_embedding_is_negated_by_lambda():
    torch.manual_seed(0)
    spy, crop, rows = nn.Linear(4, 3), torch.tensor([0, 1, 2]), torch.ones(3, dtype=torch.bool)
    base = torch.randn(3, 4)
    straight, flipped = base.clone().requires_grad_(), base.clone().requires_grad_()
    spy_loss(spy, straight, crop, rows, -1.0)[0].backward()
    spy_loss(spy, flipped, crop, rows, 1.0)[0].backward()
    assert torch.count_nonzero(straight.grad) > 0
    assert torch.allclose(straight.grad, -flipped.grad)


def test_multi_crop_diseases_from_frame():
    df = pd.DataFrame({"crop_id": [0, 1, 0, 2, 3], "disease_id": [5, 5, 6, 7, 7]})
    assert train.multi_crop_diseases(df).tolist() == [5, 7]


def test_spy_adds_parameters_only_when_requested():
    plain = build_model(load_config(REPO / "configs" / "p2c_cross.yaml", ["pretrained=False"]))
    spied = build_model(load_config(REPO / "configs" / "p2c_cross_spy.yaml", ["pretrained=False"]))
    assert not hasattr(plain, "crop_spy") and hasattr(spied, "crop_spy")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(spied) - count(plain) == 128 * 55 + 55
    heads = {id(p) for p in spied.param_groups(1e-4, 1e-3, 1e-4)[1]["params"]}
    assert all(id(p) in heads for p in spied.crop_spy.parameters())
    assert count(build_model(load_config(REPO / "configs" / "p1.yaml", ["pretrained=False"]))) == 4_758_882


def test_p2c_configs_differ_as_described():
    ctl, plain, cross, spy = (load_config(REPO / "configs" / f"p2c_{n}.yaml") for n in ("ctl", "plain", "cross", "cross_spy"))
    assert (ctl["alpha"], ctl["beta"]) == (0, 0) and (plain["alpha"], plain["beta"]) == (0.1, 0.1)
    assert cross["contrast"] == "cross" and cross["memory"] == {"per_pair": 2, "start_epoch": 1}
    assert "memory" not in plain and spy["spy"]["weight"] == 0.1
    for cfg in (ctl, plain, cross, spy):
        assert cfg["model"]["backbone"] == "convnext_tiny" and cfg["model"]["hidden"] == [256] and cfg["epochs"] == 10


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    def run_cfg(config, extra=()):
        return load_config(REPO / "configs" / config, [
            "pretrained=False", "batch_size=4", "epochs=2", "loader.num_workers=0", "loader.eval_batch=8", "bn.subset=8", "bn.batch=8",
            "amp=False", f"paths.runs_dir={tmp_path.as_posix()}", *extra,
        ])

    real = load_split

    def small(cfg, name):
        df = real(cfg, name)
        return df.sample(8 if name == "train_fit" else 4, random_state=0).reset_index(drop=True)

    monkeypatch.setattr(train, "load_split", small)
    return run_cfg


def test_cross_spy_run_has_exact_updates_and_logs_shortcut_metrics(tiny):
    out_dir, applied, skipped = train.run(tiny("p2c_cross_spy.yaml"), device=torch.device("cpu"), use_cache=False)
    assert (applied, skipped) == (4, 0)
    history = pd.read_csv(out_dir / "history.csv")
    assert {"train_spy_ce", "train_spy_acc", "seen_shortcut_rate", "unseen_shortcut_rate", "unseen_oracle_disease_top1"} <= set(history.columns)
    assert history.train_spy_ce.notna().all() and history.unseen_shortcut_rate.notna().all()


def test_cross_spy_checkpoint_loads_with_its_own_config(tiny):
    cfg = tiny("p2c_cross_spy.yaml", ["epochs=1"])
    out_dir, _, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False)
    meta = load_checkpoint(out_dir / "epoch_01.pt", build_model(meta_cfg(out_dir)))
    assert meta["config"]["spy"]["weight"] == 0.1


def meta_cfg(out_dir):
    return load_checkpoint(out_dir / "epoch_01.pt")["config"]


def test_resume_with_memory_rebuilds_it_empty(tiny, capsys):
    cfg = tiny("p2c_cross.yaml")
    train.run(cfg, device=torch.device("cpu"), use_cache=False, max_epochs=1)
    _, applied, _ = train.run(cfg, device=torch.device("cpu"), use_cache=False, resume=True)
    assert applied == 4
    assert "pair memory rebuilt empty" in capsys.readouterr().out


def test_memory_stores_one_view_per_image():
    cfg = load_config(REPO / "configs" / "p2c_cross.yaml", ["pretrained=False", "model.backbone=efficientnet_b0", "amp=False"])
    model = train.build_model(cfg)
    optimiser = torch.optim.AdamW(model.param_groups(1e-4, 1e-3, 1e-4))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    crop, disease = torch.tensor([0, 1, 0, 1]), torch.tensor([5, 6, 5, 6])  # 2 images x 2 views, stacked [view 1, view 2]
    memory = PairMemory(2)
    train.training_step(model, optimiser, scaler, torch.randn(4, 3, 64, 64), crop, disease, cfg, torch.device("cpu"), memory=memory)
    h, c, d = memory.tensors()
    assert len(h) == 2 and sorted(zip(c.tolist(), d.tolist())) == [(0, 5), (1, 6)]


def test_resume_of_an_older_history_adds_the_new_columns(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("epoch,H\n1,4.0\n", encoding="utf-8")
    fields = train._widen_history(path, ["epoch", "H"], {"epoch": 2, "H": 5.0, "unseen_shortcut_rate": 70.0})
    assert fields == ["epoch", "H", "unseen_shortcut_rate"]
    history = pd.read_csv(path)
    assert list(history.columns) == fields and len(history) == 1 and history.unseen_shortcut_rate.isna().all()


def test_bad_contrast_or_negative_spy_weight_is_refused():
    cfg = load_config(REPO / "configs" / "p2c_cross.yaml", ["pretrained=False", "model.backbone=efficientnet_b0", "amp=False"])
    model = train.build_model(cfg)
    args = (model, None, None, torch.randn(2, 3, 64, 64), torch.tensor([0, 1]), torch.tensor([5, 6]))
    for key, value in (("contrast", "crosss"), ("spy", {"weight": -0.1})):
        with pytest.raises(ValueError):
            train.training_step(*args, {**cfg, key: value}, torch.device("cpu"))
