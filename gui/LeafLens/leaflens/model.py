"""PyTorch adapter for the stage-2 EfficientNet-B0 Deeper P2 checkpoint.

The adapter maps tensors by verified names, sizes and backbone key structure,
so common notebook renamings don't require retraining. Unexpected structures
fail explicitly rather than silently performing bogus inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Prediction:
    crop: str
    disease: str
    top_pairs: list[dict]
    crop_embedding: np.ndarray
    disease_embedding: np.ndarray
    inference_head: str


def _get_components():
    import torch
    from torch import nn
    from torchvision import models, transforms
    return torch, nn, models, transforms


def _extract_state(raw):
    """Extract tensor state dict from a trusted local .pt checkpoint."""
    from collections.abc import Mapping
    if not isinstance(raw, Mapping):
        raise ValueError('Expected checkpoint mapping/state_dict, not a serialized full model object.')
    if all(hasattr(v,'shape') for v in raw.values()) and raw:
        state = raw
    else:
        state = None
        for key in ('model_state_dict','model_state','state_dict','model','net'):
            value = raw.get(key)
            if isinstance(value, Mapping) and value and any(hasattr(v,'shape') for v in value.values()):
                state = value
                break
        if state is None:
            raise ValueError('No model tensor state found in checkpoint. Supply the model_state_dict or adapt leaflens/model.py.')
    result = {}
    for key,value in state.items():
        if not hasattr(value, 'shape'):
            continue
        k = str(key)
        for pref in ('module.','_orig_mod.','model.'):
            if k.startswith(pref):
                k = k[len(pref):]
        result[k] = value
    return result


def _find_param(state, shape, concept=None, *, exclude=()):
    matches = []
    for key,tensor in state.items():
        if key.endswith('.weight') and tuple(tensor.shape)==tuple(shape):
            low=key.lower()
            if concept and concept not in low:
                continue
            if any(e in low for e in exclude):
                continue
            matches.append(key)
    if len(matches)!=1:
        raise ValueError(f'Expected one {concept or ""} weight with shape {shape}; found {matches}. Use the exact training architecture in leaflens/model.py.')
    return matches[0]


def _assign_linear(state, linear, shape, concept):
    key = _find_param(state,shape,concept)
    bias_key=key[:-7]+'.bias'
    if bias_key not in state or tuple(state[bias_key].shape)!=(shape[0],):
        raise ValueError(f'Missing bias or unexpected bias shape for {key}')
    with _get_components()[0].no_grad():
        linear.weight.copy_(state[key]); linear.bias.copy_(state[bias_key])
    return key


def load_p2_checkpoint(path: str | Path, n_crop=55, n_disease=175):
    torch, nn, models, _ = _get_components()
    path=Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {path}')
    raw=torch.load(path, map_location='cpu', weights_only=True)
    state=_extract_state(raw)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=models.efficientnet_b0(weights=None)
            self.backbone.classifier=nn.Identity()
            def proj():
                return nn.Sequential(nn.Linear(1280,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128))
            self.crop_projection=proj()
            self.disease_projection=proj()
            self.crop_classifier=nn.Linear(128,n_crop)
            self.disease_classifier=nn.Linear(128,n_disease)
        def forward(self,x):
            import torch.nn.functional as F
            pooled=self.backbone(x)
            zc=self.crop_projection(pooled)
            zd=self.disease_projection(pooled)
            return self.crop_classifier(zc), self.disease_classifier(zd), F.normalize(zc,p=2,dim=-1),F.normalize(zd,p=2,dim=-1)
    model=Model()
    base_state=model.backbone.state_dict()
    mapped={}
    for key,tensor in state.items():
        # Match torchvision backbone through the unique '.features.' prefix.
        pos=key.find('features.')
        if pos<0:
            pos=key.find('avgpool.')
        if pos>=0:
            nk=key[pos:]
            if nk in base_state and tuple(tensor.shape)==tuple(base_state[nk].shape):
                mapped[nk]=tensor
    # Most of EfficientNet-B0's parameters must be present; missing backbone weights = invalid inference.
    coverage=len(mapped)/max(len(base_state),1)
    if coverage < .97:
        raise ValueError(f'Backbone state dict matched only {coverage:.1%} of expected tensors. Check that this is the correct Deeper P2 EfficientNet-B0 checkpoint.')
    missing,unexpected=model.backbone.load_state_dict(mapped,strict=False)
    if missing:
        raise ValueError(f'Backbone keys missing: {missing[:10]}')
    for concept,projection in (('crop',model.crop_projection),('disease',model.disease_projection)):
        for layer,shape in ((0,(512,1280)),(3,(256,512)),(6,(128,256))):
            _assign_linear(state,projection[layer],shape,concept)
    _assign_linear(state,model.crop_classifier,(n_crop,128),'crop')
    _assign_linear(state,model.disease_classifier,(n_disease,128),'disease')
    model.eval()
    return model


def load_prototypes(path:str | Path,n_crop=55,n_disease=175):
    path=Path(path).expanduser()
    with np.load(path,allow_pickle=False) as z:
        arrays={key:np.asarray(z[key]) for key in z.files}
    def find(concept,shape):
        candidates=[(key,v) for key,v in arrays.items() if v.shape==shape and concept in key.lower()]
        if len(candidates)!=1:
            raise ValueError(f'Cannot identify unique {concept} prototype matrix of shape {shape}. Found {[k for k,_ in candidates]}.')
        return candidates[0][1].astype(np.float32)
    crop=find('crop',(n_crop,128)); disease=find('disease',(n_disease,128))
    norm=lambda a:a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    return norm(crop),norm(disease)


def preprocess(image, size=320, resize_short=366):
    _,_,_,transforms=_get_components()
    from PIL import ImageOps
    image=ImageOps.exif_transpose(image).convert('RGB')
    tfm=transforms.Compose([transforms.Resize(resize_short),transforms.CenterCrop(size),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return tfm(image).unsqueeze(0)


def predict_p2(model,image,crop_classes:list[str],disease_classes:list[str],*,
               head='P2',prototypes=None,size=320,resize_short=366,top_k=5):
    torch,_,_,_=_get_components()
    x=preprocess(image,size=size,resize_short=resize_short)
    with torch.inference_mode():
        c_logits,d_logits,zc,zd=model(x)
        if head=='P3':
            if prototypes is None:
                raise ValueError('P3 selected but prototype bank is not connected.')
            cp,dp=prototypes
            c=(cp@zc.cpu().numpy().reshape(-1)).reshape(-1)
            d=(dp@zd.cpu().numpy().reshape(-1)).reshape(-1)
            pair_score=.5*c[:,None]+.5*d[None,:]
        else:
            import torch.nn.functional as F
            c=F.log_softmax(c_logits,dim=1).cpu().numpy().reshape(-1)
            d=F.log_softmax(d_logits,dim=1).cpu().numpy().reshape(-1)
            pair_score=c[:,None]+d[None,:]
    if pair_score.shape!=(len(crop_classes),len(disease_classes)):
        raise ValueError('Model output dimensions differ from canonical class labels.')
    flat=pair_score.ravel()
    top=np.argsort(flat)[-top_k:][::-1]
    predictions=[]
    for rank,k in enumerate(top,1):
        ci,di=np.unravel_index(k,pair_score.shape)
        predictions.append({'rank':rank,'crop':crop_classes[ci],'disease':disease_classes[di], 'score':float(flat[k])})
    return Prediction(predictions[0]['crop'],predictions[0]['disease'],predictions,zc.cpu().numpy().reshape(-1),zd.cpu().numpy().reshape(-1),head)
