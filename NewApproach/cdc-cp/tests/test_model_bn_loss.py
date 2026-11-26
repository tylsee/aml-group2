"""A2 model interface, A3 SupCon, and the BatchNorm rules."""
import math

import pytest
import torch
import torch.nn as nn

from cdccp.bn import assert_bn_frozen, freeze_bn, recompute_bn
from cdccp.losses import supcon
from cdccp.models import OUTPUTS, CDCCP


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CDCCP(pretrained=False)


def test_forward_returns_exactly_the_four_tensors(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
        h = model.forward_features(torch.randn(2, 3, 224, 224))
    assert set(out) == set(OUTPUTS)
    assert out["crop_logits"].shape == (2, 55) and out["disease_logits"].shape == (2, 175)
    assert out["crop_embedding"].shape == (2, 128) and out["disease_embedding"].shape == (2, 128)
    assert h.shape == (2, 1280)
    assert torch.allclose(out["crop_embedding"].norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out["disease_embedding"].norm(dim=1), torch.ones(2), atol=1e-5)


def test_parameter_count_and_groups(model):
    assert sum(p.numel() for p in model.parameters()) == 4_758_882
    backbone, heads = model.param_groups(1e-5, 1e-3, 1e-4)
    assert len(backbone["params"]) + len(heads["params"]) == len(list(model.parameters()))
    assert backbone["lr"] == 1e-5 and heads["lr"] == 1e-3
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == 49


def test_supcon_identical_vectors_give_log_31():
    z = torch.nn.functional.normalize(torch.ones(32, 8), dim=1)
    assert supcon(z, torch.zeros(32, dtype=torch.long)).item() == pytest.approx(math.log(31), abs=1e-3)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_supcon_orthogonal_clusters_give_log_k_minus_1(k):
    clusters = 32 // k
    z = torch.eye(clusters).repeat_interleave(k, dim=0)
    y = torch.arange(clusters).repeat_interleave(k)
    assert supcon(z, y).item() == pytest.approx(math.log(k - 1), abs=1e-3)


def test_supcon_no_positives_is_zero_and_differentiable():
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).requires_grad_()
    loss = supcon(z, torch.arange(8))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_supcon_finite_gradients_with_memory():
    z = torch.nn.functional.normalize(torch.randn(32, 128), dim=1).requires_grad_()
    y = torch.arange(16).repeat(2)
    memory = torch.nn.functional.normalize(torch.randn(50, 128), dim=1)
    loss = supcon(z, y, mem_z=memory, mem_y=torch.randint(0, 16, (50,)))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_supcon_skips_anchors_without_positives():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)
    y = torch.tensor([0, 0, 1, 2, 3])
    sim = z @ z.T / 0.07
    # Anchors 0 and 1 are positives of each other. The other three anchors must not enter the mean.
    expected = []
    for anchor, positive in ((0, 1), (1, 0)):
        others = [i for i in range(5) if i != anchor]
        expected.append(sim[anchor, positive] - torch.logsumexp(sim[anchor, others], dim=0))
    assert supcon(z, y, tau=0.07).item() == pytest.approx(-torch.stack(expected).mean().item(), abs=1e-5)


def _batches(n=4):
    torch.manual_seed(1)
    return [(torch.randn(8, 3, 64, 64),) for _ in range(n)]


def test_recompute_bn_is_deterministic_and_leaves_eval_mode():
    device = torch.device("cpu")
    model = CDCCP(pretrained=False)
    batches = _batches()
    momentum = [m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    recompute_bn(model, batches, device)
    first = {k: v.clone() for k, v in model.state_dict().items() if "running" in k}
    assert not model.training
    model.train()
    model(torch.randn(4, 3, 64, 64))
    changed = any((model.state_dict()[k] != v).any() for k, v in first.items())
    assert changed, "a train-mode forward must move BatchNorm statistics"
    recompute_bn(model, batches, device)
    for k, v in first.items():
        assert torch.equal(model.state_dict()[k], v)
    assert [m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm2d)] == momentum


def test_freeze_bn_holds_only_until_train_is_called_again():
    model = CDCCP(pretrained=False)
    model.train()
    freeze_bn(model)
    assert_bn_frozen(model)
    before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k or "num_batches" in k}
    model(torch.randn(4, 3, 64, 64))
    for k, v in before.items():
        assert torch.equal(model.state_dict()[k], v)
    model.train()
    with pytest.raises(AssertionError):
        assert_bn_frozen(model)


@pytest.mark.parametrize("kw, dim", [({"hidden": [512]}, 1280), ({"hidden": [512, 256]}, 1280), ({"backbone": "convnext_tiny"}, 768)])
def test_round3_models_keep_the_interface(kw, dim):
    m = CDCCP(pretrained=False, **kw).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
        h = m.forward_features(torch.randn(2, 3, 224, 224))
    assert h.shape == (2, dim) and out["crop_embedding"].shape == (2, 128) and out["disease_logits"].shape == (2, 175)
    backbone, heads = m.param_groups(1e-4, 1e-3, 1e-4)
    assert len(backbone["params"]) + len(heads["params"]) == len(list(m.parameters()))
    if "backbone" in kw:  # ConvNeXt's final LayerNorm trains at the backbone rate
        ids = {id(p) for p in backbone["params"]}
        assert all(id(p) in ids for p in m.post.parameters()) and len(list(m.post.parameters())) == 2


def test_efficientnet_parameter_names_are_unchanged():
    names = set(CDCCP(pretrained=False).state_dict())
    assert "features.0.0.weight" in names and not any(n.startswith("post.") for n in names)


@pytest.mark.parametrize("backbone, dim", [("efficientnet_b2", 1408), ("efficientnet_b3", 1536), ("resnet50", 2048), ("mobilenet_v3_large", 1280), ("convnext_small", 768)])
@pytest.mark.parametrize("hidden, act", [([], "relu"), ([256], "gelu")])
def test_sweep_backbones_and_heads_keep_the_interface(backbone, dim, hidden, act):
    m = CDCCP(pretrained=False, backbone=backbone, hidden=hidden, activation=act).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
        h = m.forward_features(torch.randn(2, 3, 224, 224))
    assert h.shape == (2, dim) and out["crop_embedding"].shape == (2, 128) and out["disease_logits"].shape == (2, 175)
    assert torch.allclose(out["disease_embedding"].norm(dim=1), torch.ones(2), atol=1e-5)
    backbone_g, heads_g = m.param_groups(1e-4, 1e-3, 1e-4)
    assert len(backbone_g["params"]) + len(heads_g["params"]) == len(list(m.parameters()))
    assert len(heads_g["params"]) == 2 * 2 * (len(hidden) + 1) + 4  # projections (weight+bias per Linear) + 2 classifiers
    if backbone.startswith("convnext"):  # ConvNeXt's final LayerNorm trains at the backbone rate
        ids = {id(p) for p in backbone_g["params"]}
        assert all(id(p) in ids for p in m.post.parameters()) and len(list(m.post.parameters())) == 2


def test_unknown_backbone_or_activation_is_refused():
    with pytest.raises(ValueError):
        CDCCP(pretrained=False, backbone="vit_b_16")
    with pytest.raises(ValueError):
        CDCCP(pretrained=False, activation="tanh")
