import math
import torch
from torch import nn
from PIL import Image
from ensemble import CATALOGUE, ARMS, fused, member, log_probs, predict

class Toy(nn.Module):
    def __init__(self,n,winner):
        super().__init__()
        self.n=n
        self.winner=winner
    def forward(self,x):
        a=x[:,0,:5,:5].mean((1,2))
        w=torch.nn.functional.one_hot(torch.tensor(self.winner),self.n).to(x)
        return (10+a[:,None])*w[None,:]

def test_metadata():
    assert len(CATALOGUE["crops"])==55 and len(CATALOGUE["diseases"])==175
    assert sum(map(len,CATALOGUE["seen_disease_ids_by_crop"]))==323
    assert 14 in CATALOGUE["seen_disease_ids_by_crop"][0]
    assert 55 not in CATALOGUE["seen_disease_ids_by_crop"][51]

def test_fusion():
    models={a:Toy(n,2 if task=="crop" else 5) for a,(task,b,n) in ARMS.items()}
    x=torch.rand(1,3,224,224)
    c,d=fused(models,x)
    assert torch.allclose(c,(member(models,"crop_b1",x)+member(models,"crop_clip",x))/2)
    assert int((c[:,:,None]+d[:,None,:]).flatten().argmax())==int(c.argmax())*175+int(d.argmax())
    m=models["crop_b1"]
    assert torch.allclose(log_probs(m,x).exp(),(m(x).softmax(1)+m(x.flip(-1)).softmax(1))/2,atol=1e-6)
    result=predict(models,Image.new("RGB",(300,260),(80,130,65)))
    assert result["top3"][0][0]==result["disease"]
    assert result["cam"].shape==(224,224)
