"""Synthetic unit tests of ranking, fusion, and true gradient attribution.
These are algorithm checks, not a trained-model evaluation.
"""
import math
import numpy as np
import torch
from torch import nn
from PIL import Image
from leaflens.dual_model import (ensemble_heads, member_log_probs, predict_ensemble,
    disease_attribution, checkpoint_paths_ready, FINAL_ARMS, image_pixels, log_probs)

class ToyModel(nn.Module):
    def __init__(self, n, winner, sign=1):
        super().__init__()
        self.n=n; self.winner=winner; self.sign=sign
    def forward(self, x):
        base=x[:,0,:5,:5].mean(dim=(1,2))*self.sign
        winner=torch.nn.functional.one_hot(torch.tensor(self.winner),self.n).to(x)
        return winner.unsqueeze(0)*(10+base.unsqueeze(1))


def toy_ensemble():
    return {a:ToyModel(spec[2], 2 if spec[0]=='crop' else 5).eval()
            for a,spec in FINAL_ARMS.items()}


def test_flip_probability_average():
    model=ToyModel(55,2)
    x=torch.rand(1,3,224,224)
    expected=(model(x).softmax(1)+model(x.flip(-1)).softmax(1))/2
    assert torch.allclose(log_probs(model,x).exp(),expected,atol=1e-6)


def test_fusion_and_pair_rule():
    model=toy_ensemble(); x=torch.rand(1,3,224,224)
    c,d=ensemble_heads(model,x)
    for got,arms in [(c,('crop_b1','crop_clip')),(d,('disease_clip','disease_small_reg'))]:
        want=torch.stack([member_log_probs(model[a],x,a) for a in arms]).mean(0)
        assert torch.allclose(got,want)
    assert int((c[:,:,None]+d[:,None,:]).flatten().argmax()) == int(c.argmax())*175+int(d.argmax())


def test_prediction_and_attribution():
    result=predict_ensemble(toy_ensemble(),Image.new('RGB',(300,260),(80,130,65)),
                            [str(i) for i in range(55)], [str(i) for i in range(175)])
    assert result['crop']=='2' and result['disease']=='5'
    assert result['disease_top3'][0]['label']=='5'
    assert result['disease_score']==result['disease_top3'][0]['score']
    assert all(0<=r['score']<=1 for r in result['disease_top3'])
    assert result['cam'].shape==(224,224) and np.isfinite(result['cam']).all()
    assert np.isclose(result['cam'].max(),1)


def test_incomplete_weights_not_ready(tmp_path):
    paths={a:tmp_path/(a+'.pt') for a in FINAL_ARMS}
    assert not checkpoint_paths_ready(paths)
    for p in paths.values(): p.write_bytes(b'partial')
    assert not checkpoint_paths_ready(paths)
