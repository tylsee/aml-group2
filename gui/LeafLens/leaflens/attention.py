"""Grad-CAM saliency for a frozen EfficientNet-B0 crop/disease classifier.

This is *post-hoc class attribution*, not a lesion detector or bounding-box
model. Activation maps show influential image regions and can include shortcuts.
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from PIL import Image, ImageOps


def model_view(image: Image.Image, *, size: int = 320, resize_short: int = 366) -> Image.Image:
    """Match PIL Resize -> CenterCrop preprocessing used by the model adapter."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(resize_short),
        transforms.CenterCrop(size),
    ])
    return tf(ImageOps.exif_transpose(image).convert("RGB"))


def gradcam_for_prediction(model, image: Image.Image, *, crop_index: int, disease_index: int,
                           head: Literal['P2', 'P3'] = 'P2', prototypes=None,
                           explain: Literal['Disease', 'Crop'] = 'Disease',
                           size: int = 320, resize_short: int = 366) -> np.ndarray:
    """Return a 2-D [0, 1] Grad-CAM map aligned to the deterministic model input.

    Uses the final convolutional feature map and the predicted class score.
    If using P3, the objective is the corresponding prototype cosine score.
    Does not update model parameters or use ground-truth labels.
    """
    import torch
    import torch.nn.functional as F
    from .model import preprocess

    x = preprocess(image, size=size, resize_short=resize_short)
    x = x.requires_grad_(True)
    activation = {}

    def capture(_module, _inputs, output):
        activation['tensor'] = output

    hook = model.backbone.features[-1].register_forward_hook(capture)
    try:
        model.eval()
        with torch.enable_grad():
            crop_logits, disease_logits, crop_z, disease_z = model(x)
            feature_map = activation.get('tensor')
            if feature_map is None or feature_map.ndim != 4:
                raise RuntimeError('Could not capture EfficientNet spatial features.')
            if head == 'P3':
                if prototypes is None:
                    raise ValueError('P3 explanations require the P3 prototype bank.')
                crop_p, disease_p = prototypes
                if explain == 'Crop':
                    p = torch.as_tensor(crop_p[crop_index], dtype=crop_z.dtype, device=crop_z.device)
                    target = torch.dot(crop_z[0], p)
                else:
                    p = torch.as_tensor(disease_p[disease_index], dtype=disease_z.dtype, device=disease_z.device)
                    target = torch.dot(disease_z[0], p)
            else:
                target = crop_logits[0, crop_index] if explain == 'Crop' else disease_logits[0, disease_index]
            gradients = torch.autograd.grad(target, feature_map, allow_unused=False)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            weighted_map = (weights * feature_map).sum(dim=1, keepdim=True)
            cam = F.relu(weighted_map)
            # Very occasionally the selected logit has no *positive* Grad-CAM
            # contribution. Fall back to magnitude attribution (influence, not
            # positive evidence) instead of displaying a fabricated attention box.
            if float(cam.amax().detach()) <= 1e-12:
                cam = weighted_map.abs()
            cam = F.interpolate(cam, size=(size, size), mode='bilinear', align_corners=False)
            result = cam[0, 0].detach().cpu().float().numpy()
    finally:
        hook.remove()

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(result))
    if maximum <= 1e-10:
        return np.zeros((size, size), dtype=np.float32)
    result = result / maximum
    return result.astype(np.float32)

