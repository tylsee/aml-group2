import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdccp.bn import recompute_bn
from cdccp.models import BACKBONES, build_model


@pytest.fixture
def tiny_cfg(monkeypatch):
    def backbone(pretrained):
        return (nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.Dropout2d(0.5)),
                nn.AdaptiveAvgPool2d(1), nn.BatchNorm2d(8), 8)

    monkeypatch.setitem(BACKBONES, "tiny_test", backbone)
    return {"pretrained": False, "model": {"crop_classes": 2, "disease_classes": 3,
            "dropout": 0.2, "backbone": "tiny_test", "hidden": [8], "freeze_backbone": True}}


def test_frozen_backbone_stays_inference_mode_while_heads_train(tiny_cfg):
    model = build_model(tiny_cfg)
    x = torch.randn(4, 3, 8, 8)
    for _ in range(2):
        assert model.train() is model
        assert model.training and model.crop_projection.training and model.disease_projection.training
        assert not any(m.training for block in (model.features, model.post) for m in block.modules())
        assert torch.equal(model.forward_features(x), model.forward_features(x))
        model.eval()
    assert all(not p.requires_grad for block in (model.features, model.post) for p in block.parameters())
    assert all(p.requires_grad for p in model.disease_projection.parameters())


def test_optimizer_updates_heads_without_changing_frozen_features(tiny_cfg):
    model = build_model(tiny_cfg).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.param_groups(1e-4, 1e-3, 1e-4))
    out = model(torch.randn(4, 3, 8, 8))
    loss = F.cross_entropy(out["crop_logits"], torch.tensor([0, 1, 0, 1]))
    loss += F.cross_entropy(out["disease_logits"], torch.tensor([0, 1, 2, 0]))
    loss.backward()
    opt.step()
    for key, value in model.state_dict().items():
        if key.startswith(("features.", "post.")):
            assert torch.equal(value, before[key]), key
    assert not torch.equal(model.disease_classifier.weight, before["disease_classifier.weight"])
    assert all(p.grad is None for block in (model.features, model.post) for p in block.parameters())


def test_recalibration_does_not_change_frozen_statistics(tiny_cfg):
    model = build_model(tiny_cfg).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    recompute_bn(model, [(torch.randn(4, 3, 8, 8),)], torch.device("cpu"))
    assert not model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_default_backbone_remains_trainable(tiny_cfg):
    tiny_cfg["model"].pop("freeze_backbone")
    model = build_model(tiny_cfg).train()
    assert model.features.training and model.post.training
    out = model(torch.randn(4, 3, 8, 8))
    out["disease_logits"].square().mean().backward()
    assert all(p.requires_grad and p.grad is not None for p in model.features.parameters())


def test_frozen_model_round_trip_keeps_checkpoint_keys(tiny_cfg):
    model = build_model(tiny_cfg)
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "config": tiny_cfg}, checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    restored = build_model(state["config"])
    restored.load_state_dict(state["model"], strict=True)
    restored.train()
    assert not restored.features.training
    tiny_cfg["model"]["freeze_backbone"] = False
    assert set(model.state_dict()) == set(build_model(tiny_cfg).state_dict())


def test_non_boolean_freezing_is_rejected(tiny_cfg):
    tiny_cfg["model"]["freeze_backbone"] = "false"
    with pytest.raises(ValueError, match="freeze_backbone"):
        build_model(tiny_cfg)
