"""Final Stage-2 four-arm raw ensemble. No training or candidate masking."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms as T
from PIL import ImageOps, Image, ImageFilter

BASE = Path(__file__).resolve().parent
ASSETS = json.loads((BASE / "checkpoints.json").read_text())
CATALOGUE = json.loads((BASE / "metadata.json").read_text())
CROPS, DISEASES = CATALOGUE["crops"], CATALOGUE["diseases"]
ARMS = {"crop_b1": ("crop", "convnext_tiny", 55),
        "crop_clip": ("crop", "clip_vit_b16", 55),
        "disease_clip": ("disease", "clip_vit_b16", 175),
        "disease_small_reg": ("disease", "convnext_small", 175)}
CLIP_NORM = ((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
IMAGENET_NORM = ((.485,.456,.406),(.229,.224,.225))

def paths():
    return {arm: BASE / "models" / info["filename"] for arm, info in ASSETS.items()}

def check_files(verify_hash=True):
    for arm, p in paths().items():
        info = ASSETS[arm]
        if not p.is_file() or p.stat().st_size != info["bytes"]:
            raise ValueError("Missing or incomplete checkpoint: " + str(p))
        if verify_hash:
            with p.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != info["sha256"]:
                raise ValueError("SHA-256 mismatch: " + p.name)

class ClipModel(nn.Module):
    def __init__(self, info):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        self.vision = CLIPVisionModelWithProjection(CLIPVisionConfig.from_dict(info["vision_config"]))
        self.prompts = info["prompts"]
        self.register_buffer("text_features", torch.zeros(len(self.prompts), self.vision.config.projection_dim))
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.vision(pixel_values=x).image_embeds
        return self.logit_scale.clamp(max=math.log(100.)).exp() * F.normalize(e.float(), dim=-1) @ self.text_features.t()

def load_ensemble():
    check_files()
    loaded = {}
    for arm, p in paths().items():
        task, backbone, count = ARMS[arm]
        raw = torch.load(p, map_location="cpu", weights_only=True)
        if (raw.get("stage"),raw.get("arm"),raw.get("task"),raw.get("backbone")) != (2,arm,task,backbone):
            raise ValueError("Unexpected checkpoint metadata: " + p.name)
        if list(raw["crop_ids"]) != list(range(55)) or list(raw["disease_ids"]) != list(range(175)):
            raise ValueError("Noncanonical label IDs: " + p.name)
        if backbone == "clip_vit_b16":
            model = ClipModel(raw["clip"])
            if len(model.prompts) != count:
                raise ValueError("Wrong prompt count")
        else:
            constructor = models.convnext_tiny if backbone == "convnext_tiny" else models.convnext_small
            model = constructor(weights=None)
            model.classifier[-1] = nn.Sequential(nn.Dropout(float(raw["dropout"])),
                        nn.Linear(model.classifier[-1].in_features, count))
        model.load_state_dict(raw["state_dict"], strict=True)
        loaded[arm] = model.eval().requires_grad_(False)
        del raw
    return loaded

def model_view(image):
    return T.Compose([T.Resize(256),T.CenterCrop(224)])(ImageOps.exif_transpose(image).convert("RGB"))

def log_probs(model, x):
    return torch.logsumexp(torch.stack([F.log_softmax(model(x).float(),1),
                 F.log_softmax(model(x.flip(-1)).float(),1)]),0)-math.log(2.)

def member(ensemble, arm, pixels):
    norm = CLIP_NORM if ARMS[arm][1] == "clip_vit_b16" else IMAGENET_NORM
    mean,std = (pixels.new_tensor(v).view(1,3,1,1) for v in norm)
    return log_probs(ensemble[arm], (pixels-mean)/std)

def fused(ensemble, pixels):
    c = .5*(member(ensemble,"crop_b1",pixels)+member(ensemble,"crop_clip",pixels))
    d = .5*(member(ensemble,"disease_clip",pixels)+member(ensemble,"disease_small_reg",pixels))
    if c.shape!=(1,55) or d.shape!=(1,175) or not torch.isfinite(c).all() or not torch.isfinite(d).all():
        raise ValueError("Unexpected or nonfinite model scores")
    return c,d

def attribution(ensemble,pixels,di):
    gradients=[]
    with torch.enable_grad():
        for arm in ("disease_clip","disease_small_reg"):
            x=pixels.detach().clone().requires_grad_(True)
            gradients.append(torch.autograd.grad(member(ensemble,arm,x)[0,di],x)[0].detach())
    saliency=(.5*(gradients[0]+gradients[1])).abs().amax(1,keepdim=True)
    a=F.avg_pool2d(saliency,9,stride=1,padding=4)[0,0].numpy()
    a=np.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)
    return a/a.max() if a.max()>1e-12 else np.zeros_like(a)

def predict(ensemble,image):
    view=model_view(image)
    pixels=T.ToTensor()(view).unsqueeze(0)
    with torch.inference_mode():
        c,d=fused(ensemble,pixels)
        ci,di=int(c.argmax()),int(d.argmax())
        cp,dp=c.softmax(1),d.softmax(1)
        ranks=torch.argsort(d[0],descending=True,stable=True)[:3].tolist()
        result=dict(crop=CROPS[ci], disease=DISEASES[di], crop_score=float(cp[0,ci]),
                    disease_score=float(dp[0,di]),top3=[(DISEASES[i],float(dp[0,i])) for i in ranks],
                    seen=di in CATALOGUE["seen_disease_ids_by_crop"][ci])
    result["cam"]=attribution(ensemble,pixels,di)
    return result

def overlay(image,cam):
    base=model_view(image)
    a=np.clip(np.nan_to_num(cam),0,1)
    if a.max()<1e-8:
        return base,False
    strength=np.clip((a-.10)/.82,0,1)
    opacity=strength**.78*.83
    t=np.clip((strength-.40)/.60,0,1)
    colour=np.stack((np.full_like(a,255),np.where(strength<.40,232-72*strength/.40,160-107*t),
                     np.where(strength<.40,55-10*strength/.40,45-10*t)),axis=-1)
    mix=np.asarray(base,dtype=np.float32)*(1-opacity[...,None])+colour*opacity[...,None]
    high=Image.fromarray(np.uint8(a>=.64)*255)
    edge=(np.asarray(high.filter(ImageFilter.MaxFilter(7)),dtype=np.int16)>
          np.asarray(high.filter(ImageFilter.MinFilter(5)),dtype=np.int16))&(a>=.25)
    mix[edge]=.2*mix[edge]+.8*np.array([255,253,233],dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(mix,0,255))),True
