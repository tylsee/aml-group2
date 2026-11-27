"""Baseline 2 ResNet-50: shared ImageNet V2 backbone and two classification heads.
Do not overwrite the original EfficientNet-B0 model.py.
"""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class MultiTaskResNet50(nn.Module):
    def __init__(self, crop_dropout=0.3, disease_dropout=0.2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = resnet50(weights=weights)
        feature_dim = network.fc.in_features  # 2048
        network.fc = nn.Identity()             # Preserve pooled feature extraction.
        self.backbone = network
        self.crop_head = nn.Sequential(nn.Dropout(crop_dropout), nn.Linear(feature_dim, 55))
        self.disease_head = nn.Sequential(nn.Dropout(disease_dropout), nn.Linear(feature_dim, 175))

    def forward(self, images):
        shared_features = self.backbone(images)
        return {
            "crop_logits": self.crop_head(shared_features),
            "disease_logits": self.disease_head(shared_features),
        }

    def parameter_groups(self, backbone_lr=1e-5, head_lr=1e-3):
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": list(self.crop_head.parameters()) + list(self.disease_head.parameters()),
             "lr": head_lr, "name": "heads"},
        ]


def load_resnet50_checkpoint(path, device="cpu"):
    """Load only ResNet-50 checkpoints saved by the companion notebook."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("backbone") != "resnet50" or config.get("pretrained_weights") != "IMAGENET1K_V2":
        raise ValueError("Expected ResNet-50 ImageNet V2 checkpoint, not an EfficientNet checkpoint")
    model = MultiTaskResNet50(
        crop_dropout=config["crop_dropout"],
        disease_dropout=config["disease_dropout"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
