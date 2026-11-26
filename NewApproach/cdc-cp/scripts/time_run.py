"""G0: time training steps of one config with real images and the real loader, then extrapolate T to its full update budget.

Usage: python scripts/time_run.py [--config configs/p1.yaml] [--workers N] [--no-cache] [key=value ...]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.catalogue import load_catalogue
from cdccp.checkpoint import training_meta
from cdccp.config import REPO, load_config
from cdccp.data import batch_stream, load_split
from cdccp.models import build_model
from cdccp.train import evaluate_epoch, training_step

WARMUP = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--config", default=str(REPO / "configs" / "p2_plain.yaml"))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, [f"loader.num_workers={args.workers}", "pretrained=False", *args.overrides])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = not args.no_cache
    model = build_model(cfg).to(device)
    optimiser = torch.optim.AdamW(model.param_groups(cfg["optim"]["lr_backbone"], cfg["optim"]["lr_heads"], cfg["optim"]["weight_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    train_df = load_split(cfg, "train_fit")
    updates, evaluations = len(train_df) // cfg["batch_size"] * cfg["epochs"], cfg["epochs"]
    stream = batch_stream(train_df, cfg, cfg["batch_size"], cfg["views"], cfg["seed"], use_cache)

    model.train()
    loader_s, step_s, skipped = [], [], 0
    for i in range(args.steps + WARMUP):
        t = time.perf_counter()
        x, crop, disease = next(stream)
        t1 = time.perf_counter()
        ok, _ = training_step(model, optimiser, scaler, x, crop, disease, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        skipped += not ok
        if i >= WARMUP:
            loader_s.append(t1 - t)
            step_s.append(t2 - t1)
    peak = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0

    cat = load_catalogue(cfg["paths"]["catalogue_dev"])
    bn_df = train_df.sample(min(cfg["bn"]["subset"], len(train_df)), random_state=cfg["seed"])
    val_seen, val_unseen = load_split(cfg, "val_seen"), load_split(cfg, "val_unseen")
    t = time.perf_counter()
    evaluate_epoch(model, cfg, device, bn_df, val_seen, val_unseen, cat, training_meta(train_df), use_cache)
    eval_s = time.perf_counter() - t

    per_step = float(np.mean(loader_s) + np.mean(step_s))
    total_min = (per_step * updates + eval_s * evaluations) / 60
    scope = "full" if total_min <= 90 else "default" if total_min <= 180 else "move to Colab or reduce views"
    print(f"device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} workers {args.workers} cache {use_cache}")
    print(f"steps {args.steps} loader wait {np.mean(loader_s):.3f} s/step, gpu {np.mean(step_s):.3f} s/step, skipped {skipped}, peak {peak:.0f} MiB")
    print(f"one evaluation pass {eval_s:.1f} s")
    print(f"{cfg['name']}: T for one {updates:,}-update run with {evaluations} evaluations: {total_min:.0f} min -> scope: {scope}")


if __name__ == "__main__":
    main()
