import torch
import torch.nn.functional as F

def supervised_contrastive_loss(embeddings,labels,temperature=0.07):
    """2D [N,128] embeddings, [N] labels. Exclude self-pairs and ignore anchors without positives."""
    if temperature<=0: raise ValueError('temperature must be > 0')
    if embeddings.ndim!=2 or labels.ndim!=1 or embeddings.size(0)!=labels.numel():
        raise ValueError('Expected embeddings [N,D], labels [N]')
    z=F.normalize(embeddings.float(),dim=1)
    labels=labels.reshape(-1)
    n=labels.numel()
    eye=torch.eye(n,device=z.device,dtype=torch.bool)
    positives=labels[:,None].eq(labels[None,:]) & ~eye
    valid=positives.sum(1)>0
    if not valid.any(): return z.sum()*0.0
    logits=z@z.T/temperature
    logits=logits-logits.max(dim=1,keepdim=True).values.detach()
    logden=torch.logsumexp(logits.masked_fill(eye,float('-inf')),dim=1,keepdim=True)
    logprob=logits-logden
    mean_log_pos=(logprob.masked_fill(~positives,0.0).sum(1)/positives.sum(1).clamp_min(1))
    return -mean_log_pos[valid].mean()
