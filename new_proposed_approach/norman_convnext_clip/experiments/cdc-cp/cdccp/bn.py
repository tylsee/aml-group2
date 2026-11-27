import torch
import torch.nn as nn


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recompute_bn(model, loader, device):
    """Replace BatchNorm statistics with a cumulative average over `loader`.

    Everything else stays in eval mode, so dropout and stochastic depth are off. The model is
    left in eval mode. A frozen backbone keeps its existing statistics.
    """
    model.eval()
    if getattr(model, "freeze_backbone", False):
        return
    layers = _bn_layers(model)
    saved = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for batch in loader:
        model.forward_features(batch[0].to(device, non_blocking=True))
    for m, momentum in zip(layers, saved):
        m.momentum = momentum
        m.eval()


def freeze_bn(model):
    """Call after every model.train(), which silently re-enables BatchNorm updates."""
    for m in _bn_layers(model):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def assert_bn_frozen(model):
    for m in _bn_layers(model):
        if m.training or any(p.requires_grad for p in m.parameters()):
            raise AssertionError("BatchNorm is not frozen")
