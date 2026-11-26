import numpy as np
import torch

from .data import eval_loader


@torch.no_grad()
def extract(model, df, cfg, device, use_cache=True):
    """Deterministic outputs for every row of df. The caller sets the BatchNorm state first."""
    model.eval()
    out = {k: [] for k in ("crop_logits", "disease_logits", "u_crop", "u_disease", "h")}
    amp = cfg.get("amp", False) and device.type == "cuda"
    for batch in eval_loader(df, cfg, use_cache=use_cache):
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            h = model.forward_features(x)
            heads = model.heads(h)
        if amp and not all(torch.isfinite(t).all() for t in (h, heads["crop_logits"], heads["disease_logits"])):
            # fp16 overflowed (seen with MobileNetV3 on some GPUs): redo this batch in fp32.
            h = model.forward_features(x)
            heads = model.heads(h)
        out["crop_logits"].append(heads["crop_logits"].float().cpu().numpy())
        out["disease_logits"].append(heads["disease_logits"].float().cpu().numpy())
        out["u_crop"].append(heads["crop_embedding"].float().cpu().numpy())
        out["u_disease"].append(heads["disease_embedding"].float().cpu().numpy())
        out["h"].append(h.float().cpu().numpy())
    result = {k: np.concatenate(v) for k, v in out.items()}
    result.update(
        fn=df.filename.values.astype(str),
        crop_id=df.crop_id.values.astype(np.int64),
        disease_id=df.disease_id.values.astype(np.int64),
        comp_id=df.composition_id.values.astype(np.int64),
    )
    return result
