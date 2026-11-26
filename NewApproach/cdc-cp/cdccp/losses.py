import math

import torch
import torch.nn.functional as F


def supcon(z, y, tau=0.07, mem_z=None, mem_y=None, group=None, mem_group=None):
    """Supervised contrastive loss on unit embeddings, computed in fp32.

    Self is excluded, anchors with no positive are skipped, and there is no base-temperature
    rescaling. Returns 0 when no anchor has a positive.

    With `group` (cross mode) a positive shares the label but not the group. Contrast entries that share
    both label and group with the anchor (self included) leave the denominator, so they are neither
    pulled nor pushed.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        contrast, labels = z, y
        has_memory = mem_z is not None and len(mem_z)
        if has_memory:
            contrast = torch.cat([z, mem_z.float()])
            labels = torch.cat([y, mem_y])
        similarity = z @ contrast.T / tau
        if group is not None:
            groups = torch.cat([group, mem_group]) if has_memory else group
            same_label = y[:, None] == labels[None, :]
            same_group = group[:, None] == groups[None, :]
            removed = same_label & same_group
            positive = same_label & ~same_group
            log_prob = similarity - torch.logsumexp(similarity.masked_fill(removed, float("-inf")), dim=1, keepdim=True)
            log_prob = log_prob.masked_fill(~positive, 0.0)
        else:
            is_self = torch.zeros_like(similarity, dtype=torch.bool)
            is_self[torch.arange(len(z)), torch.arange(len(z))] = True
            log_prob = similarity - torch.logsumexp(similarity.masked_fill(is_self, float("-inf")), dim=1, keepdim=True)
            positive = (y[:, None] == labels[None, :]) & ~is_self
        count = positive.sum(1)
        valid = count > 0
        if not valid.any():
            return z.sum() * 0.0
        return -((log_prob * positive).sum(1)[valid] / count[valid]).mean()


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass, gradient multiplied by -lam in the backward pass."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grl_lambda(applied, total):
    """DANN schedule: 0 at the start, towards 1 as training progresses."""
    return 2.0 / (1.0 + math.exp(-10.0 * applied / total)) - 1.0


def spy_loss(spy, embedding, crop, multi_rows, lam):
    """Cross-entropy of the crop spy on reversed disease embeddings, over rows whose disease occurs on 2+ crops.

    Returns (loss, accuracy). The loss is zero, with a graph to the spy and embedding, when no row qualifies.
    """
    logits = spy(GradReverse.apply(embedding, lam))
    if not multi_rows.any():
        return logits.sum() * 0.0, 0.0
    selected, target = logits[multi_rows], crop[multi_rows]
    return F.cross_entropy(selected, target), (selected.argmax(1) == target).float().mean().item()
