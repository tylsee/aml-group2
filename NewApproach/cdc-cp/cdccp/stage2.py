"""Stage 2: continue the chosen development checkpoint on all official training images, then save it for the one-shot test.

No development evaluation happens here: val_seen and val_unseen are training images by now. The only check is a
gate on the training loss, fixed before the run and computed from no test data.
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bn import recompute_bn
from .checkpoint import file_sha256, git_state, load_checkpoint, save_checkpoint, training_meta
from .config import _parse, load_config
from .data import batch_stream, eval_loader, load_split
from .memory import PairMemory
from .models import build_model
from .train import lr_fraction, multi_crop_diseases, set_lr, training_step

STAGE2_DEFAULTS = {"epochs": 2, "lr_backbone": 1.0e-5, "lr_heads": 1.0e-4, "warmup_epochs": 0, "train_split": "train"}
HISTORY, GATE, FINAL = "stage2_history.csv", "stage2_gate.json", "stage2_final.pt"
STAGE2_FILES = (HISTORY, GATE, FINAL, FINAL + ".tmp")
LOSS_PARTS = ("ce_crop", "ce_disease", "sup_crop", "sup_disease")


def split_overrides(overrides):
    """Returns (stage2 settings, the remaining overrides). `stage2.key=value` edits the Stage 2 block."""
    settings, rest = dict(STAGE2_DEFAULTS), []
    for item in overrides:
        key, _ = item.split("=", 1)
        if not key.startswith("stage2."):
            rest.append(item)
            continue
        name = key[len("stage2."):]
        if name not in STAGE2_DEFAULTS:
            raise ValueError(f"unknown Stage 2 setting {name!r}; choose from {sorted(STAGE2_DEFAULTS)}")
        settings[name] = _parse(item.split("=", 1)[1])
    epochs = settings["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"stage2.epochs must be a positive integer, got {epochs!r}")
    for name in ("lr_backbone", "lr_heads", "warmup_epochs"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"stage2.{name} must be a number >= 0, got {value!r}")
    if settings["train_split"] not in ("train", "train_fit"):
        raise ValueError(f"stage2.train_split must be 'train' or 'train_fit', got {settings['train_split']!r}")
    if settings["warmup_epochs"] >= epochs:
        raise ValueError("stage2.warmup_epochs must be smaller than stage2.epochs")
    return settings, rest


def stage2_gate(history):
    """Pre-declared check. Passes when every epoch's train_loss is finite and the last is no higher than the first."""
    losses = [float(v) for v in history["train_loss"]]
    if not losses:
        return False, "no epochs in the history"
    if not all(math.isfinite(v) for v in losses):
        return False, f"non-finite train_loss: {losses}"
    if losses[-1] > losses[0]:
        return False, f"train_loss rose from {losses[0]:.4f} to {losses[-1]:.4f}"
    return True, f"train_loss went from {losses[0]:.4f} to {losses[-1]:.4f}"


def _train_loss(means, cfg):
    total = means["ce_crop"] + means["ce_disease"] + cfg["alpha"] * means["sup_crop"] + cfg["beta"] * means["sup_disease"]
    spy_weight = cfg.get("spy", {}).get("weight", 0.0)
    if spy_weight:
        total += spy_weight * means["spy_ce"]
    return total


def _already_done(out_dir, parent_sha, settings):
    """True when out_dir holds the finished run for this parent and these settings. Partial Stage 2 files are removed."""
    final = out_dir / FINAL
    if final.exists():
        meta = load_checkpoint(final)
        if meta.get("stage") != 2 or meta.get("parent_sha256") != parent_sha or meta.get("stage2") != settings:
            raise ValueError(f"{final} is a finished run with a different parent or Stage 2 settings; use another out_dir")
        return True
    for name in STAGE2_FILES:
        (out_dir / name).unlink(missing_ok=True)
    return False


def run_stage2(parent_path, out_dir, overrides=(), device=None, use_cache=True):
    """Continue the parent checkpoint for a fixed number of epochs on every official training image. Returns (out_dir, sha256)."""
    parent_path, out_dir = Path(parent_path), Path(out_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings, load_overrides = split_overrides(overrides)
    parent_sha = file_sha256(parent_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, parent_sha, settings):
        print(f"nothing to do: {out_dir / FINAL} already holds this Stage 2 run", flush=True)
        return out_dir, file_sha256(out_dir / FINAL)

    parent_meta = load_checkpoint(parent_path)
    here = load_config(overrides=load_overrides)
    cfg = {**parent_meta["config"], "paths": here["paths"], "loader": here["loader"], "stage2": settings}
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = load_split(cfg, settings["train_split"])
    bn_df = train_df.sample(min(cfg["bn"]["subset"], len(train_df)), random_state=seed).sort_values("filename")
    steps = len(train_df) // cfg["batch_size"]
    if steps < 1:
        raise ValueError(f"{len(train_df)} training images is fewer than one batch of {cfg['batch_size']}")
    total_updates = steps * settings["epochs"]
    warmup = round(steps * settings["warmup_epochs"])

    model = build_model(cfg, pretrained=False)
    load_checkpoint(parent_path, model)
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.param_groups(settings["lr_backbone"], settings["lr_heads"], cfg["optim"]["weight_decay"]))
    trainable = [sum(p.numel() for p in group["params"] if p.requires_grad) for group in optimiser.param_groups]
    print(f"trainable parameters: backbone={trainable[0]} heads={trainable[1]}", flush=True)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")

    per_pair = cfg.get("memory", {}).get("per_pair", 0)
    memory = PairMemory(per_pair) if per_pair else None
    memory_start = cfg.get("memory", {}).get("start_epoch", 1) * steps
    multi_diseases = torch.as_tensor(multi_crop_diseases(train_df), device=device)
    stream = batch_stream(train_df, cfg, cfg["batch_size"], cfg["views"], seed, use_cache)

    history_path = out_dir / HISTORY
    applied = skipped = 0
    running, count, started = {}, 0, time.time()
    while applied < total_updates:
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
        means = {k: v / count for k, v in running.items()}
        row = {
            "epoch": epoch, "applied": applied, "skipped": skipped,
            "lr_backbone": optimiser.param_groups[0]["lr"], "lr_heads": optimiser.param_groups[1]["lr"],
            **{f"train_{k}": v for k, v in means.items()}, "train_loss": _train_loss(means, cfg),
            "seconds": time.time() - started,
        }
        write_header = not history_path.exists()
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"stage 2 epoch {epoch}/{settings['epochs']} applied {applied} skipped {skipped} train_loss {row['train_loss']:.4f} {row['seconds']:.0f}s", flush=True)
        running, count, started = {}, 0, time.time()

    recompute_bn(model, eval_loader(bn_df, cfg, cfg["bn"]["batch"], use_cache=use_cache), device)

    history = pd.read_csv(history_path)
    ok, reason = stage2_gate(history)
    (out_dir / GATE).write_text(json.dumps({"ok": ok, "reason": reason, "train_loss_by_epoch": [float(v) for v in history["train_loss"]]}, indent=2), encoding="utf-8")
    meta = {
        **training_meta(train_df),
        "stage": 2, "parent": str(parent_path), "parent_sha256": parent_sha, "parent_epoch": parent_meta["epoch"],
        "config": json.loads(json.dumps(cfg)), "seed": seed, "stage2": settings,
        "applied": applied, "skipped": skipped, "git": git_state(),
        "bn_policy": "frozen_pretrained" if model.freeze_backbone else "recomputed_on_subset",
    }
    sha = save_checkpoint(out_dir / FINAL, model, meta)
    print(f"stage 2 gate {'PASS' if ok else 'FAIL'}: {reason}", flush=True)
    return out_dir, sha
