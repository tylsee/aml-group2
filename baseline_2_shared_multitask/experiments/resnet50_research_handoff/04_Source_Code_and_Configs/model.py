"""Baseline 2 only: one shared EfficientNet-B0 and two independent heads."""
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, dropout=0.2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        network = efficientnet_b0(weights=weights)
        self.backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
        self.crop_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, 55))
        self.disease_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, 175))

    def forward(self, images):
        features = self.backbone(images)
        return {"crop_logits": self.crop_head(features),
                "disease_logits": self.disease_head(features)}

    def parameter_groups(self, backbone_lr=1e-5, head_lr=1e-3):
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": list(self.crop_head.parameters()) + list(self.disease_head.parameters()),
             "lr": head_lr, "name": "heads"},
        ]


def load_checkpoint(path, device="cpu"):
    """Restore our tensor/plain-data checkpoint without downloading pretrained weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MultiTaskEfficientNetB0(
        dropout=checkpoint["config"]["dropout"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
