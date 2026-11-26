"""Stage 1 training. Runs until exactly U applied updates. There is deliberately no early stopping."""
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .bn import recompute_bn
from .catalogue import load_catalogue
from .checkpoint import git_state, load_checkpoint, save_checkpoint, training_meta
from .data import batch_stream, eval_loader, load_split
from .evaluate import check_rows, combined_metrics, shortcut_metrics, split_metrics
from .extract import extract
from .losses import grl_lambda, spy_loss, supcon
from .memory import PairMemory
from .models import build_model
from .scoring import head_score


def cosine_fraction(applied, total):
    return 0.5 * (1 + math.cos(math.pi * applied / total))


def lr_fraction(applied, total, warmup=0):
    """Linear warm-up over `warmup` updates, then cosine from 1 to 0 over the rest. Same factor for every group."""
    if applied < warmup:
        return (applied + 1) / warmup
    return cosine_fraction(applied - warmup, total - warmup)


def _atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _recipe(cfg):
    """The config minus machine-specific keys (paths, loader workers), for checking a resume is the same run."""
    return json.loads(json.dumps({k: v for k, v in cfg.items() if k not in ("paths", "loader")}))


def _trim_history(path, last_epoch):
    """Drop rows written after the state we resume from, so a crash between the two writes cannot duplicate an epoch."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if int(r["epoch"]) <= last_epoch]
    if not fields:
        path.unlink()
        return None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return fields


def _widen_history(path, fields, row):
    """Rewrite history.csv with the columns of `row` that an older header lacks. Old rows get blanks there."""
    fields = fields + [k for k in row if k not in fields]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return fields


def set_lr(optimiser, fraction):
    for group in optimiser.param_groups:
        group["lr"] = group["base_lr"] * fraction


def p_loss(out, crop, disease, alpha, beta, tau, smoothing=0.0, cross=False, memory=None):
    """`memory` is None or {"crop": (z, y, group), "disease": (z, y, group)} re-projected through the current heads."""
    ce_crop = F.cross_entropy(out["crop_logits"], crop, label_smoothing=smoothing)
    ce_disease = F.cross_entropy(out["disease_logits"], disease, label_smoothing=smoothing)

    def contrast(embedding, y, group, key):
        mem_z, mem_y, mem_group = memory[key] if memory else (None, None, None)
        if not cross:
            return supcon(embedding, y, tau, mem_z, mem_y)
        return supcon(embedding, y, tau, mem_z, mem_y, group=group, mem_group=mem_group)

    sup_crop = contrast(out["crop_embedding"], crop, disease, "crop") if alpha else torch.zeros((), device=crop.device)
    sup_disease = contrast(out["disease_embedding"], disease, crop, "disease") if beta else torch.zeros((), device=crop.device)
    total = ce_crop + ce_disease + alpha * sup_crop + beta * sup_disease
    return total, {"ce_crop": ce_crop.item(), "ce_disease": ce_disease.item(), "sup_crop": sup_crop.item(), "sup_disease": sup_disease.item()}


def multi_crop_diseases(train_df):
    """Disease ids that occur on 2+ crops in the training frame."""
    crops_per_disease = train_df.groupby("disease_id")["crop_id"].nunique()
    return crops_per_disease[crops_per_disease >= 2].index.to_numpy()


@torch.no_grad()
def memory_targets(model, memory):
    """Stored backbone features re-projected through the current projections (dropout off), unit length."""
    stored = memory.tensors()
    if stored is None:
        return None
    h, crop, disease = stored
    projections = (model.crop_projection, model.disease_projection)
    was_training = [p.training for p in projections]
    for p in projections:
        p.eval()
    try:
        crop_z = F.normalize(model.crop_projection(h).float(), dim=1)
        disease_z = F.normalize(model.disease_projection(h).float(), dim=1)
    finally:
        for p, flag in zip(projections, was_training):
            p.train(flag)
    return {"crop": (crop_z, crop, disease), "disease": (disease_z, disease, crop)}


def training_step(model, optimiser, scaler, x, crop, disease, cfg, device, memory=None, use_memory=False, multi_diseases=None, progress=0.0):
    """One forward and backward pass. Returns (applied, parts): a skipped GradScaler step is not an update.

    `memory` (a PairMemory) is fed the detached features after every step. `use_memory` switches its
    entries into the contrastive loss. `progress` is applied / total updates, for the spy schedule.
    """
    crop, disease = crop.to(device), disease.to(device)
    spy_weight = cfg.get("spy", {}).get("weight", 0.0)
    if spy_weight < 0:
        raise ValueError(f"spy.weight must be >= 0, got {spy_weight}")
    if cfg.get("contrast", "plain") not in ("plain", "cross"):
        raise ValueError(f"contrast must be plain or cross, got {cfg['contrast']!r}")
    if spy_weight and multi_diseases is None:
        raise ValueError("a spy run needs multi_diseases (disease ids on 2+ training crops)")
    with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
        h = model.forward_features(x.to(device, non_blocking=True))
        out = model.heads(h)
        mem = memory_targets(model, memory) if memory is not None and use_memory else None
        total, parts = p_loss(out, crop, disease, cfg["alpha"], cfg["beta"], cfg["tau"], cfg.get("label_smoothing", 0.0), cfg.get("contrast", "plain") == "cross", mem)
        if spy_weight:
            lam = grl_lambda(progress, 1.0)
            multi_rows = torch.isin(disease, multi_diseases)
            spy_ce, spy_acc = spy_loss(model.crop_spy, out["disease_embedding"], crop, multi_rows, lam)
            total = total + spy_weight * spy_ce
            parts.update({"spy_ce": spy_ce.item(), "spy_acc": spy_acc})
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite loss: {parts}")
    optimiser.zero_grad(set_to_none=True)
    before = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.step(optimiser)
    scaler.update()
    if memory is not None:
        first_view = len(h) // cfg["views"]
        memory.update(h[:first_view], crop[:first_view], disease[:first_view])
    return scaler.get_scale() >= before, parts


def _val_ce(npz):
    ce = lambda logits, labels: F.cross_entropy(torch.from_numpy(npz[logits]), torch.from_numpy(npz[labels])).item()
    return ce("crop_logits", "crop_id"), ce("disease_logits", "disease_id")


def evaluate_epoch(model, cfg, device, bn_df, val_seen, val_unseen, cat, meta, use_cache=True):
    """Recompute BatchNorm, then score both validation sets with the model's own head score."""
    check_rows(val_seen, "seen", meta)
    check_rows(val_unseen, "unseen", meta)
    recompute_bn(model, eval_loader(bn_df, cfg, cfg["bn"]["batch"], use_cache=use_cache), device)
    seen = extract(model, val_seen, cfg, device, use_cache)
    unseen = extract(model, val_unseen, cfg, device, use_cache)
    m_s, m_u = split_metrics(head_score(seen, cat), seen, cat), split_metrics(head_score(unseen, cat), unseen, cat)
    row = combined_metrics(m_s, m_u)
    for prefix, split in (("seen", seen), ("unseen", unseen)):
        scores = head_score(split, cat)
        row.update({f"{prefix}_{k}": v for k, v in shortcut_metrics(scores, split["disease_logits"], split, cat).items()})
    row.update({f"seen_{k}": v for k, v in m_s.items()})
    row.update({f"unseen_{k}": v for k, v in m_u.items()})
    row["val_ce_crop"], row["val_ce_disease"] = _val_ce(seen)
    return row, seen, unseen


def run(cfg, device=None, use_cache=True, max_epochs=None, resume=False):
    """Train one run. Epochs are measured in applied updates, so U is exact even when AMP skips steps."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(cfg["paths"]["runs_dir"]) / f"{cfg['name']}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(cfg, cfg["train_csv"])
    val_seen, val_unseen = load_split(cfg, "val_seen"), load_split(cfg, "val_unseen")
    cat = load_catalogue(cfg["paths"]["catalogue_dev"])
    bn_df = train_df.sample(min(cfg["bn"]["subset"], len(train_df)), random_state=seed).sort_values("filename")

    steps = len(train_df) // cfg["batch_size"]
    total_updates = steps * cfg["epochs"]
    warmup = round(steps * cfg["optim"].get("warmup_epochs", 0))
    model = build_model(cfg).to(device)
    optimiser = torch.optim.AdamW(model.param_groups(cfg["optim"]["lr_backbone"], cfg["optim"]["lr_heads"], cfg["optim"]["weight_decay"]))
    trainable = [sum(p.numel() for p in group["params"] if p.requires_grad) for group in optimiser.param_groups]
    print(f"trainable parameters: backbone={trainable[0]} heads={trainable[1]}", flush=True)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    meta = {
        **training_meta(train_df),
        "stage": 1, "parent": None, "config": json.loads(json.dumps(cfg)), "seed": seed,
        "git": git_state(), "updates_total": total_updates, "warmup_updates": warmup,
        "bn_policy": "frozen_pretrained" if model.freeze_backbone else "recomputed_on_subset",
    }

    applied = skipped = 0
    generator_state = None
    history_path = out_dir / "history.csv"
    last_path = out_dir / "last.pt"
    if not resume and (history_path.exists() or last_path.exists()):
        raise FileExistsError(f"{out_dir} already holds a run. Pass --resume to continue it, or use another name or seed.")
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        applied, skipped, generator_state = state["applied"], state["skipped"], state["generator"].cpu()
        if state.get("recipe") not in (None, _recipe(cfg)):
            raise ValueError(f"{last_path} was trained with a different config; refusing to mix runs")
    fields = _trim_history(history_path, applied // steps)

    target_epochs = min(cfg["epochs"], max_epochs) if max_epochs else cfg["epochs"]
    if applied >= steps * target_epochs:
        print(f"nothing to do: {out_dir} already has {applied} applied updates", flush=True)
        return out_dir, applied, skipped
    per_pair = cfg.get("memory", {}).get("per_pair", 0)
    memory = PairMemory(per_pair) if per_pair else None
    memory_start = cfg.get("memory", {}).get("start_epoch", 1) * steps
    if memory is not None and resume and applied:
        print(f"resume: pair memory rebuilt empty at applied {applied}; it refills from the next updates", flush=True)
    multi_diseases = torch.as_tensor(multi_crop_diseases(train_df), device=device)
    stream = batch_stream(train_df, cfg, cfg["batch_size"], cfg["views"], seed, use_cache, generator_state)
    running, count, started = {}, 0, time.time()
    while applied < steps * target_epochs:
        model.train()
        set_lr(optimiser, lr_fraction(applied, total_updates, warmup))
        x, crop, disease = next(stream)
        ok, parts = training_step(
            model, optimiser, scaler, x, crop, disease, cfg, device,
            memory=memory, use_memory=applied >= memory_start, multi_diseases=multi_diseases, progress=applied / total_updates,
        )
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        count += 1
        if not ok:
            skipped += 1
            if skipped > cfg["max_skip_fraction"] * total_updates:
                raise RuntimeError(f"{skipped} skipped updates exceeds the allowed fraction")
            continue
        applied += 1
        if applied % steps:
            continue

        epoch = applied // steps
        row, seen, unseen = evaluate_epoch(model, cfg, device, bn_df, val_seen, val_unseen, cat, meta, use_cache)
        row = {
            "epoch": epoch, "applied": applied, "skipped": skipped,
            "lr_backbone": optimiser.param_groups[0]["lr"], "lr_heads": optimiser.param_groups[1]["lr"],
            **{f"train_{k}": v / count for k, v in running.items()}, **row,
            "seconds": time.time() - started,
        }
        write_header = not history_path.exists()
        fields = fields or list(row)
        if set(row) - set(fields):
            fields = _widen_history(history_path, fields, row)
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        save_checkpoint(out_dir / f"epoch_{epoch:02d}.pt", model, {**meta, "epoch": epoch, "applied": applied, "skipped": skipped})
        np.savez_compressed(out_dir / f"val_{epoch:02d}.npz", **{f"seen_{k}": v for k, v in seen.items() if k != "h"}, **{f"unseen_{k}": v for k, v in unseen.items() if k != "h"})
        _atomic_save({
            "model": model.state_dict(), "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            # A resumed run reshuffles from this state rather than finishing the interrupted pass.
            "numpy_rng": np.random.get_state(), "applied": applied, "skipped": skipped, "generator": torch.get_rng_state(),
            "recipe": _recipe(cfg),
        }, last_path)
        print(f"epoch {epoch}/{cfg['epochs']} applied {applied} skipped {skipped} A_S {row['A_S']:.2f} A_U {row['A_U']:.2f} H {row['H']:.2f} {row['seconds']:.0f}s", flush=True)
        running, count, started = {}, 0, time.time()
    return out_dir, applied, skipped
