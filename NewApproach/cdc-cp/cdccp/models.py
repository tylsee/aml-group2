import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, convnext_small, convnext_tiny, efficientnet_b0

OUTPUTS = ("crop_logits", "disease_logits", "crop_embedding", "disease_embedding")
EMBED = 128


# Each returns (trunk, pool, post, feature_dim). EfficientNet keeps the original attribute names,
# so round-1 and round-2 checkpoints still load.
def _efficientnet_b0(pretrained):
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    return net.features, net.avgpool, nn.Identity(), 1280


def _convnext_tiny(pretrained):
    net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    # classifier[0] is ConvNeXt's final LayerNorm, part of the pretrained feature extractor.
    return net.features, net.avgpool, net.classifier[0], 768


def _convnext_small(pretrained):
    net = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    return net.features, net.avgpool, net.classifier[0], 768


def _efficientnet(name, dim):
    """EfficientNet-B2/B3 at the shared 224 x 224 input (their native sizes are 288/300)."""
    def build(pretrained):
        net = getattr(tvm, name)(weights="DEFAULT" if pretrained else None)
        return net.features, net.avgpool, nn.Identity(), dim
    return build


def _resnet50(pretrained):
    net = tvm.resnet50(weights="DEFAULT" if pretrained else None)  # IMAGENET1K_V2
    trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
    return trunk, net.avgpool, nn.Identity(), 2048


def _mobilenet_v3_large(pretrained):
    net = tvm.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)  # IMAGENET1K_V2
    # classifier[0:2] (Linear 960 -> 1280, Hardswish) is pretrained and part of the feature path.
    return net.features, net.avgpool, nn.Sequential(nn.Flatten(1), net.classifier[0], net.classifier[1]), 1280


BACKBONES = {
    "efficientnet_b0": _efficientnet_b0,
    "efficientnet_b2": _efficientnet("efficientnet_b2", 1408),
    "efficientnet_b3": _efficientnet("efficientnet_b3", 1536),
    "resnet50": _resnet50,
    "mobilenet_v3_large": _mobilenet_v3_large,
    "convnext_tiny": _convnext_tiny,
    "convnext_small": _convnext_small,
}


ACTIVATIONS = {"relu": lambda: nn.ReLU(inplace=True), "gelu": nn.GELU, "silu": lambda: nn.SiLU(inplace=True)}


def _projection(in_dim, hidden, dropout, activation="relu"):
    """Linear -> activation -> Dropout for each hidden width, then Linear to the 128-d embedding.

    hidden=[256] gives the team plan's 1280 -> 256 -> 128.
    """
    layers, width = [], in_dim
    for h in hidden:
        layers += [nn.Linear(width, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, EMBED))


class CDCCP(nn.Module):
    """Shared backbone with separate crop and disease projections. Used by P1, P2 and P3."""

    def __init__(self, crop_classes=55, disease_classes=175, dropout=0.2, pretrained=True, backbone="efficientnet_b0", hidden=(256,), activation="relu", spy=False, freeze_backbone=False):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        if not isinstance(freeze_backbone, bool):
            raise ValueError("freeze_backbone must be a boolean")
        self.freeze_backbone = freeze_backbone
        self.features, self.avgpool, self.post, self.feature_dim = BACKBONES[backbone](pretrained)
        self.crop_projection = _projection(self.feature_dim, list(hidden), dropout, activation)
        self.disease_projection = _projection(self.feature_dim, list(hidden), dropout, activation)
        self.crop_classifier = nn.Linear(EMBED, crop_classes)
        self.disease_classifier = nn.Linear(EMBED, disease_classes)
        if spy:
            self.crop_spy = nn.Linear(EMBED, crop_classes)
        if self.freeze_backbone:
            self.features.requires_grad_(False)
            self.post.requires_grad_(False)
            self.features.eval()
            self.post.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
            self.post.eval()
        return self

    def forward_features(self, x):
        return self.post(self.avgpool(self.features(x))).flatten(1)

    def heads(self, h):
        crop = self.crop_projection(h)
        disease = self.disease_projection(h)
        return {
            "crop_logits": self.crop_classifier(crop),
            "disease_logits": self.disease_classifier(disease),
            "crop_embedding": F.normalize(crop, dim=1),
            "disease_embedding": F.normalize(disease, dim=1),
        }

    def forward(self, x):
        return self.heads(self.forward_features(x))

    def param_groups(self, lr_backbone, lr_heads, weight_decay):
        backbone = list(self.features.parameters()) + list(self.post.parameters())
        ids = {id(p) for p in backbone}
        heads = [p for p in self.parameters() if id(p) not in ids]
        return [
            {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay, "base_lr": lr_backbone},
            {"params": heads, "lr": lr_heads, "weight_decay": weight_decay, "base_lr": lr_heads},
        ]


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    return CDCCP(
        m["crop_classes"], m["disease_classes"], m["dropout"],
        cfg.get("pretrained", True) if pretrained is None else pretrained,
        backbone=m.get("backbone", "efficientnet_b0"), hidden=m.get("hidden", [256]), activation=m.get("activation", "relu"),
        spy=cfg.get("spy", {}).get("weight", 0.0) > 0,
        freeze_backbone=m.get("freeze_backbone", False),
    )
