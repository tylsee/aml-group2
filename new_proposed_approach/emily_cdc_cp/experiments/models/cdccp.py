from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ConceptProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1280,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128))
    def forward(self,x): return self.net(x)

class CDCCP(nn.Module):
    """Shared EfficientNet-B0 + independent crop/disease concept heads (P1 & P2)."""
    def __init__(self,num_crops=55,num_diseases=175,pretrained=True):
        super().__init__()
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone=efficientnet_b0(weights=weights)
        self.crop_projection=ConceptProjection()
        self.disease_projection=ConceptProjection()
        self.crop_classifier=nn.Linear(128,num_crops)
        self.disease_classifier=nn.Linear(128,num_diseases)

    def forward(self,x):
        x=self.backbone.features(x)
        x=self.backbone.avgpool(x).flatten(1)   # [B,1280]
        crop_raw=self.crop_projection(x)
        disease_raw=self.disease_projection(x)
        return {
            'crop_logits':self.crop_classifier(crop_raw),
            'disease_logits':self.disease_classifier(disease_raw),
            'crop_embedding':F.normalize(crop_raw.float(),dim=1),
            'disease_embedding':F.normalize(disease_raw.float(),dim=1),
        }
