"""Inference for the group's crop_b1 / disease_clip Stage-2 checkpoints.
Architecture and transforms follow cdccp/single_task.py in aml-group2.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import ImageOps

CLIP_NORM = ((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
IMAGENET_NORM = ((.485,.456,.406),(.229,.224,.225))

class ClipDiseaseModel(nn.Module):
    def __init__(self, info):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        self.vision = CLIPVisionModelWithProjection(CLIPVisionConfig.from_dict(info['vision_config']))
        self.prompts = info['prompts']
        self.register_buffer('text_features', torch.zeros(len(self.prompts),self.vision.config.projection_dim))
        self.logit_scale = nn.Parameter(torch.zeros(()))
    def forward(self, images):
        embeds = self.vision(pixel_values=images).image_embeds
        return self.logit_scale.clamp(max=math.log(100.)).exp() * F.normalize(embeds.float(),dim=-1) @ self.text_features.t()

def load_pair(crop_path, disease_path):
    c = torch.load(crop_path,map_location='cpu',weights_only=True)
    d = torch.load(disease_path,map_location='cpu',weights_only=True)
    for raw,task,backbone,count,key in [(c,'crop','convnext_tiny',55,'crop_ids'),(d,'disease','clip_vit_b16',175,'disease_ids')]:
        if raw.get('task')!=task or raw.get('backbone')!=backbone or list(raw.get(key,[]))!=list(range(count)):
            raise ValueError(f'Checkpoint does not match expected {task} {backbone} with canonical {count} class IDs.')
    crop = models.convnext_tiny(weights=None)
    crop.classifier[-1] = nn.Sequential(nn.Dropout(float(c['dropout'])),nn.Linear(crop.classifier[-1].in_features,55))
    crop.load_state_dict(c['state_dict'],strict=True)
    disease = ClipDiseaseModel(d['clip'])
    disease.load_state_dict(d['state_dict'],strict=True)
    return crop.eval().requires_grad_(False), disease.eval().requires_grad_(False)

def input_tensor(image,norm):
    return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(*norm)])(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)

def log_probs(model,x):
    return torch.logsumexp(torch.stack([F.log_softmax(model(x).float(),dim=1),F.log_softmax(model(x.flip(-1)).float(),dim=1)]),dim=0)-math.log(2.)

def predict_pair(pair,image,crops,diseases):
    crop,disease = pair
    xc,xd = input_tensor(image,IMAGENET_NORM),input_tensor(image,CLIP_NORM)
    with torch.inference_mode():
        cl,dl=log_probs(crop,xc),log_probs(disease,xd)
        if not torch.isfinite(cl).all() or not torch.isfinite(dl).all():
            raise ValueError('Non-finite model output')
        ci,di=int(cl.argmax()),int(dl.argmax())
        crop_score=float(cl[0,ci].exp())
        # Rank the same TTA-averaged disease distribution used for top-1.
        # Stable sorting preserves argmax's first-index tie handling.
        ranked=torch.argsort(dl[0],descending=True,stable=True)[:3].tolist()
        disease_top3=[{'label':diseases[i], 'score':float(dl[0,i].exp())} for i in ranked]
    # Input-gradient magnitude for the actual TTA disease prediction; not Grad-CAM.
    with torch.enable_grad():
        xd=xd.requires_grad_(True)
        target=log_probs(disease,xd)[0,di]
        grad=torch.autograd.grad(target,xd)[0]
        saliency=grad.abs().amax(dim=1,keepdim=True)
        saliency=F.avg_pool2d(saliency,kernel_size=9,stride=1,padding=4)[0,0]
        saliency=saliency.detach().numpy()
    saliency=np.nan_to_num(saliency,nan=0.,posinf=0.,neginf=0.)
    maxval=float(saliency.max())
    saliency=saliency/maxval if maxval>1e-12 else np.zeros_like(saliency)
    return {'crop':crops[ci], 'disease':diseases[di], 'cam':saliency,
            'crop_score':crop_score, 'disease_score':float(disease_top3[0]['score']),
            'disease_top3':disease_top3}

# Norman's final Stage-2 P3 ensemble, as specified by cdccp/single_task.py
# and cdccp/ensemble.py. This is NOT the separate CDC-CP extension.
from pathlib import Path

FINAL_ARMS = {
    'crop_b1': ('crop', 'convnext_tiny', 55, 111507819),
    'crop_clip': ('crop', 'clip_vit_b16', 55, 344964043),
    'disease_clip': ('disease', 'clip_vit_b16', 175, 345238603),
    'disease_small_reg': ('disease', 'convnext_small', 175, 198466235),
}


def checkpoint_paths_ready(paths):
    """Check presence and expected release byte sizes, not checkpoint validity."""
    try:
        return all(Path(paths[a]).is_file() and Path(paths[a]).stat().st_size == spec[3]
                   for a, spec in FINAL_ARMS.items())
    except (OSError, KeyError):
        return False


def load_ensemble(paths):
    """Load only the four expected, trusted final Stage-2 release checkpoints.

    Reconstruct from saved architecture metadata; no pretrained download or
    optimisation is required. Strict loading prevents accidental partial models.
    """
    loaded = {}
    for arm, (task, backbone, count, expected_size) in FINAL_ARMS.items():
        path = Path(paths[arm])
        if not path.is_file():
            raise FileNotFoundError(f'Missing final checkpoint for {arm}: {path}')
        if path.stat().st_size != expected_size:
            raise ValueError(f'{path.name}: expected {expected_size:,} bytes; got {path.stat().st_size:,}. Check download completion and release version.')
        raw = torch.load(path, map_location='cpu', weights_only=True)
        if (raw.get('stage') != 2 or raw.get('arm') != arm or raw.get('task') != task
                or raw.get('backbone') != backbone):
            raise ValueError(f'{path.name}: not the expected Stage-2 {arm} checkpoint.')
        for key, n in [('crop_ids',55), ('disease_ids',175)]:
            if list(raw.get(key, [])) != list(range(n)):
                raise ValueError(f'{path.name}: unexpected {key} mapping.')
        if backbone == 'clip_vit_b16':
            if len(raw['clip']['prompts']) != count:
                raise ValueError(f'{path.name}: unexpected number of CLIP prompt groups.')
            model = ClipDiseaseModel(raw['clip'])
        else:
            constructor = models.convnext_tiny if backbone == 'convnext_tiny' else models.convnext_small
            model = constructor(weights=None)
            model.classifier[-1] = nn.Sequential(nn.Dropout(float(raw['dropout'])),
                                                nn.Linear(model.classifier[-1].in_features, count))
        model.load_state_dict(raw['state_dict'], strict=True)
        loaded[arm] = model.eval().requires_grad_(False)
        del raw
    return loaded


def image_pixels(image):
    """Shared unnormalised pixels; both model families use the same centre crop."""
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor()])(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)


def normalise_pixels(x, norm):
    mean, std = norm
    return (x - x.new_tensor(mean).view(1,3,1,1)) / x.new_tensor(std).view(1,3,1,1)


def member_log_probs(model, pixels, arm):
    norm = CLIP_NORM if FINAL_ARMS[arm][1] == 'clip_vit_b16' else IMAGENET_NORM
    return log_probs(model, normalise_pixels(pixels, norm))


def ensemble_heads(ensemble, pixels):
    """Exactly the official unrestricted ensemble rule: mean member log-probs."""
    c = .5 * (member_log_probs(ensemble['crop_b1'], pixels, 'crop_b1') +
              member_log_probs(ensemble['crop_clip'], pixels, 'crop_clip'))
    d = .5 * (member_log_probs(ensemble['disease_clip'], pixels, 'disease_clip') +
              member_log_probs(ensemble['disease_small_reg'], pixels, 'disease_small_reg'))
    if not torch.isfinite(c).all() or not torch.isfinite(d).all():
        raise ValueError('Non-finite ensemble scores.')
    return c, d


def disease_attribution(ensemble, pixels, disease_index):
    """Gradient magnitude of the fused raw disease score, in shared pixel space.

    Each gradient is computed sequentially, so only one disease model's
    activations are retained at a time. Both horizontal views contribute.
    This is input saliency, not lesion localisation or Grad-CAM.
    """
    grads = []
    with torch.enable_grad():
        for arm in ('disease_clip', 'disease_small_reg'):
            x = pixels.detach().clone().requires_grad_(True)
            target = member_log_probs(ensemble[arm], x, arm)[0, disease_index]
            grads.append(torch.autograd.grad(target, x)[0].detach())
        grad = .5 * (grads[0] + grads[1])
        saliency = grad.abs().amax(dim=1, keepdim=True)
        saliency = F.avg_pool2d(saliency, kernel_size=9, stride=1, padding=4)[0,0]
        saliency = saliency.cpu().numpy()
    saliency = np.nan_to_num(saliency, nan=0., posinf=0., neginf=0.)
    maximum = float(saliency.max())
    return saliency / maximum if maximum > 1e-12 else np.zeros_like(saliency)


def predict_ensemble(ensemble, image, crops, diseases):
    if (len(crops), len(diseases)) != (55, 175):
        raise ValueError('Expected 55 crop labels and 175 disease labels.')
    pixels = image_pixels(image)
    with torch.inference_mode():
        c, d = ensemble_heads(ensemble, pixels)
        if c.shape != (1,55) or d.shape != (1,175):
            raise ValueError('Unexpected ensemble output shapes.')
        # Adding c[:,None] + d[None,:] ranks every one of the 9,625 pairs.
        # For this unrestricted additive rule, independent argmaxes give the
        # same top-1 pair as explicitly ranking the full Cartesian product.
        ci, di = int(c.argmax()), int(d.argmax())
        # Normalise only the UI scores; this cannot change official raw ranking.
        cp, dp = c.softmax(dim=1), d.softmax(dim=1)
        ranked = torch.argsort(d[0], descending=True, stable=True)[:3].tolist()
        top3 = [{'label': diseases[i], 'score': float(dp[0,i])} for i in ranked]
        crop_score, disease_score = float(cp[0,ci]), float(dp[0,di])
    cam = disease_attribution(ensemble, pixels, di)
    return {'crop': crops[ci], 'disease': diseases[di], 'cam': cam,
            'crop_score': crop_score, 'disease_score': disease_score,
            'disease_top3': top3}
