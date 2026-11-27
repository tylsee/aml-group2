"""Baseline 2 (ConvNeXt-Tiny V1): shared visual backbone, two CE classifiers.

Input image -> pretrained ConvNeXt features + global pool + pretrained LayerNorm
            -> independent crop (55) and disease (175) logits.
Composition logits are calculated separately in evaluate.py with 9,625 pairs.
This module does not read datasets or access official-test files on import.
"""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class MultiTaskConvNeXtTiny(nn.Module):
    def __init__(self, crop_dropout=0.3, disease_dropout=0.2, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        network = convnext_tiny(weights=weights)
        if not isinstance(network.classifier, nn.Sequential):
            raise TypeError("Unexpected torchvision ConvNeXt classifier layout")
        original_linear = network.classifier[-1]
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("Expected an ImageNet Linear classifier")
        feature_dim = original_linear.in_features  # Verify from installed torchvision (Tiny: 768).
        network.classifier[-1] = nn.Identity()  # Retain pretrained final LayerNorm + flatten.
        self.backbone = network
        self.feature_dim = feature_dim
        self.crop_head = nn.Sequential(nn.Dropout(crop_dropout), nn.Linear(feature_dim, 55))
        self.disease_head = nn.Sequential(nn.Dropout(disease_dropout), nn.Linear(feature_dim, 175))

    def forward(self, images):
        shared_features = self.backbone(images)
        if shared_features.ndim != 2 or shared_features.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected pooled feature shape: {tuple(shared_features.shape)}")
        return {
            "crop_logits": self.crop_head(shared_features),
            "disease_logits": self.disease_head(shared_features),
        }

    def parameter_groups(self, backbone_lr=1e-5, head_lr=1e-3):
        return [
            {"params": list(self.backbone.parameters()), "lr": backbone_lr, "name": "backbone"},
            {"params": list(self.crop_head.parameters()) + list(self.disease_head.parameters()),
             "lr": head_lr, "name": "heads"},
        ]


def load_convnext_checkpoint(path, device="cpu"):
    """Restore a CN01/compatible checkpoint without re-downloading ImageNet weights."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = payload["config"]
    if config.get("backbone") != "convnext_tiny" or config.get("pretrained_weights") != "IMAGENET1K_V1":
        raise ValueError("Expected ConvNeXt-Tiny ImageNet-1K V1 checkpoint")
    model = MultiTaskConvNeXtTiny(
        crop_dropout=float(config["crop_dropout"]),
        disease_dropout=float(config["disease_dropout"]),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload
