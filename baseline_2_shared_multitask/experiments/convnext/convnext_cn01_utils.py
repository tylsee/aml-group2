"""Development-only helpers for the ConvNeXt-Tiny Baseline 2 CN01 experiment.

Imports only training split functions from the existing evaluate.py; never reads,
searches for, or evaluates official seen/unseen test CSV files. Does not start
training upon import. Config/paths are checked before any output is created.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
import random
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from evaluate import (DPDDataset, MEAN, STD, composition_confusion,
                      evaluate_loader, generalisation_metrics, image_paths,
                      make_candidates, require_candidate_coverage,
                      synthetic_self_test, validate_development_splits)
from model_convnext import MultiTaskConvNeXtTiny, load_convnext_checkpoint

# Exact SHA256 from the successfully executed ResNet-50 Exp09 development record.
EXPECTED_SPLIT_HASHES = {
    "train_fit.csv": "fe02d6099c002b10e5a428c2260d5de50f3e677d3b7454216b2a340e55c2a377",
    "val_seen.csv": "ab8d18a13b28fd5a7b9ea531b8ffbdcf4174fd6920337ee9f1a726129390daf5",
    "val_unseen.csv": "b9eae2a57d8331d97636f4e8a8e32bc2e81704d7180fdac22ae923273aadec67",
    "heldout_compositions.csv": "437f57924eef3e5788ed0993caddd969e08f9f86add64791d944425f4bddfe7f",
    "split_manifest.json": "c120d065677e37db6d00880104684c0cd70ccd1c0774ec4aac2eb4b4b0889df8",
}


def default_config():
    """Exp09-aligned except Tiny backbone, safe microbatch, worker count and accumulation."""
    return {
        "experiment_id": "convnext_cn01", "backbone": "convnext_tiny",
        "pretrained_weights": "IMAGENET1K_V1", "crop_classes": 55,
        "disease_classes": 175, "image_size": 224,
        "batch_size": 8, "gradient_accumulation_steps": 2,
        "effective_batch_size": 16,  # 8 x 2, close to Exp09's physical batch 16
        "max_epochs": 30, "patience": 15, "min_delta": 1e-6,
        "backbone_lr": 1e-5, "head_lr": 1e-3,
        "crop_dropout": 0.3, "disease_dropout": 0.2,
        "weight_decay": 1e-3, "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR", "crop_loss_weight": 1.0,
        "disease_loss_weight": 1.0, "seed": 42,
        "num_workers": 0, "amp": True,
        "crop_scale": [0.85, 1.0],
        "flip_probability": 0.5, "vertical_flip_probability": 0.5,
        "rotation_degrees": 10, "brightness": 0.15, "contrast": 0.15,
        "interpolation": "bilinear_resize_nearest_rotation",
        "eval_resize": 256, "eval_crop": 224,
        "mean": list(MEAN), "std": list(STD),
        "candidate_mode": "all_pairs", "candidate_catalog": None,
        "training_sampling": "natural_shuffle",
        "checkpoint_selection": "development_composition_harmonic_mean",
        "stage": 1,
    }


def validate_config(config):
    expected = default_config()
    if not isinstance(config, dict):
        raise TypeError("CONFIG must be a dict")
    if config.get("backbone") != "convnext_tiny" or config.get("pretrained_weights") != "IMAGENET1K_V1":
        raise ValueError("CN01 only supports ConvNeXt-Tiny V1 pretrained on ImageNet-1K V1")
    for key in ("crop_classes", "disease_classes", "image_size", "seed", "crop_loss_weight", "disease_loss_weight", "candidate_mode"):
        if config.get(key) != expected[key]:
            raise ValueError(f"CN01 locked configuration differs at {key}: {config.get(key)!r}")
    if config["training_sampling"] != "natural_shuffle" or config["candidate_catalog"] is not None:
        raise ValueError("CN01 must use natural shuffle and all 9,625 unfiltered candidate pairs")
    if config["interpolation"] != "bilinear_resize_nearest_rotation" or config["eval_resize"] != 256 or config["eval_crop"] != 224:
        raise ValueError("CN01 must retain Exp09 validation/training interpolation settings")
    batch = config["batch_size"]
    accum = config["gradient_accumulation_steps"]
    if type(batch) is not int or type(accum) is not int or batch < 1 or accum < 1:
        raise ValueError("batch_size and accumulation_steps must be positive integers")
    if config["effective_batch_size"] != batch * accum:
        raise ValueError("Effective batch size must equal batch_size * gradient_accumulation_steps")
    if config["max_epochs"] < 1 or config["patience"] < 1 or config["num_workers"] < 0:
        raise ValueError("Invalid epoch, patience, or worker setting")
    if config["experiment_id"] != "convnext_cn01":
        raise ValueError("Use CN01's fixed output ID; do not overwrite prior runs with changed configs")
    return True


def locate_project(start=None):
    start = Path(os.environ.get("B2_PROJECT_ROOT", start or Path.cwd())).expanduser().resolve()
    options = [start, *start.parents]
    for root in options:
        if (root / "evaluate.py").is_file() and (root / "splits_v2").is_dir():
            return root
    raise FileNotFoundError(
        "Could not locate Baseline2. Open notebook from Baseline2/notebooks/convnext, "
        "or set B2_PROJECT_ROOT to C:\\AML\\ML_Project\\Baseline2. "
        "The directory must contain evaluate.py and splits_v2/."
    )


def find_training_images(project_dir):
    project_dir = Path(project_dir)
    default = project_dir.parent / "DPD_Small_dataset"
    if default.is_dir():
        return default
    second = project_dir / "DPD_Small_dataset"
    if second.is_dir():
        return second
    raise FileNotFoundError(
        f"Training image directory not found at {default} or {second}; "
        "set IMAGE_ROOT explicitly to the folder containing TRAINING photos."
    )


def training_transform(config):
    return T.Compose([
        T.RandomResizedCrop(config["image_size"], scale=tuple(config["crop_scale"]),
                            interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=config["flip_probability"]),
        T.RandomVerticalFlip(p=config["vertical_flip_probability"]),
        T.RandomRotation(config["rotation_degrees"], interpolation=InterpolationMode.NEAREST),
        T.ColorJitter(brightness=config["brightness"], contrast=config["contrast"]),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def evaluation_transform(config):
    # Intentionally NOT evaluate.validation_transform(): it uses bicubic,
    # whereas Exp09/15 executed bilinear Resize(256), CenterCrop(224).
    return T.Compose([
        T.Resize(config["eval_resize"], interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(config["eval_crop"]),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_preflight(project_dir, image_root, verify_decode=False):
    """Validate fixed metadata, file hashes, path containment, and optionally every image decode."""
    frames, report = validate_development_splits(Path(project_dir) / "splits_v2")
    if report["sha256"] != EXPECTED_SPLIT_HASHES:
        raise ValueError("Development split hashes no longer match executed Exp09. STOP; verify files.")
    candidates = make_candidates("all_pairs")
    require_candidate_coverage(candidates, list(frames.values()))
    if tuple(candidates.shape) != (9625, 2) or torch.unique(candidates, dim=0).shape[0] != 9625:
        raise AssertionError("Expected exactly 9,625 unique crop-disease candidates")
    if candidates[0].tolist() != [0, 0] or candidates[-1].tolist() != [54, 174]:
        raise AssertionError("Pair ordering differs from the ResNet baseline")
    total_paths = 0
    for frame in frames.values():
        paths = image_paths(frame, image_root)  # Only CSV-listed TRAIN images; never glob for official data.
        total_paths += len(paths)
        if verify_decode:
            for path in paths:
                with Image.open(path) as im:
                    im.convert("RGB").load()
    report["image_paths_checked"] = True
    report["image_decoding_checked"] = verify_decode
    report["training_image_root"] = str(Path(image_root).resolve())
    assert total_paths == 8000
    return frames, report, candidates


def smoke_test_model():
    """Pure synthetic CPU tests. No dataset, no pretrained download, no output folder."""
    synthetic_self_test()
    seed_everything(42)
    model = MultiTaskConvNeXtTiny(pretrained=False)
    assert model.feature_dim == 768, f"Unexpected Tiny feature dimension {model.feature_dim}"
    groups = model.parameter_groups()
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())), "Optimiser groups must cover all parameters once"
    x = torch.randn(2, 3, 64, 64)
    crop_ids = torch.tensor([0, 54])
    disease_ids = torch.tensor([0, 174])
    model.train()
    out = model(x)
    assert out["crop_logits"].shape == (2, 55) and out["disease_logits"].shape == (2, 175)
    crop_loss = nn.functional.cross_entropy(out["crop_logits"], crop_ids)
    disease_loss = nn.functional.cross_entropy(out["disease_logits"], disease_ids)
    (crop_loss + disease_loss).backward()
    for name, parameter in [
        ("backbone", next(model.backbone.parameters())),
        ("crop head", model.crop_head[-1].weight),
        ("disease head", model.disease_head[-1].weight),
    ]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"No finite gradient: {name}"
        assert parameter.grad.abs().sum() > 0, f"All-zero gradient: {name}"
    from evaluate import composition_scores
    pair_logits = composition_scores(out["crop_logits"].detach(), out["disease_logits"].detach(), make_candidates())
    assert pair_logits.shape == (2, 9625)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.pt"
        state = {"config": default_config(), "model_state": model.state_dict(), "stage": 1}
        torch.save(state, path)
        restored, payload = load_convnext_checkpoint(path)
        assert len(payload["model_state"]) == len(model.state_dict())
        assert torch.equal(restored.crop_head[-1].weight, model.crop_head[-1].weight)
    del model, restored, out
    gc.collect()
    return "Synthetic CPU PASS: heads, shared gradients, parameters, pairs, checkpoint save/load"


def cuda_memory_preflight(config, batch_size=None):
    """Try one training-shaped synthetic minibatch on CUDA before real training."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "note": "No GPU here; do not attempt full training."}
    device = torch.device("cuda")
    b = int(batch_size or config["batch_size"])
    model = MultiTaskConvNeXtTiny(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(config["backbone_lr"], config["head_lr"]),
                                  weight_decay=config["weight_decay"])
    amp = bool(config["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(b, 3, 224, 224, device=device)
        crops = torch.randint(0, 55, (b,), device=device)
        diseases = torch.randint(0, 175, (b,), device=device)
        with torch.autocast(device_type="cuda", enabled=amp):
            output = model(x)
            loss = nn.functional.cross_entropy(output["crop_logits"], crops) + nn.functional.cross_entropy(output["disease_logits"], diseases)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        return {"cuda_available": True, "batch_tested": b, "amp": amp, "peak_allocated_GiB": round(peak, 2), "passed": True}
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Synthetic CUDA preflight OOM at batch {b}. Restart kernel, set batch_size=4 "
            "and gradient_accumulation_steps=4 (effective=16), and rerun preflight BEFORE training."
        ) from exc
    finally:
        del model, optimizer, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def make_loaders(frames, image_root, config, device):
    datasets = {
        name: DPDDataset(frame, image_root,
                         training_transform(config) if name == "train_fit" else evaluation_transform(config))
        for name, frame in frames.items()
    }
    loaders = {}
    for name, dataset in datasets.items():
        workers = int(config["num_workers"])
        options = dict(batch_size=config["batch_size"],
                       shuffle=(name == "train_fit"), num_workers=workers,
                       drop_last=False, pin_memory=device.type == "cuda",
                       generator=torch.Generator().manual_seed(config["seed"]))
        if workers > 0:
            options.update(prefetch_factor=2, persistent_workers=True)
        loaders[name] = DataLoader(dataset, **options)
    return loaders


def mappings_from_frames(frames):
    joined = pd.concat(frames.values(), ignore_index=True)
    class_mappings = {
        task: {str(int(row[0])): str(row[1]) for row in joined[[f"{task}_id", f"{task}_name"]].drop_duplicates().itertuples(index=False, name=None)}
        for task in ("crop", "disease", "composition")
    }
    composition_mapping = (joined[["composition_id", "crop_id", "disease_id", "composition_name"]]
                           .drop_duplicates().sort_values("composition_id").to_dict("records"))
    return class_mappings, composition_mapping


def environment_details(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "pandas": pd.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda}


def atomic_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _numpy_state_plain():
    state = np.random.get_state()
    return {"bit_generator": state[0], "keys": state[1].tolist(), "pos": int(state[2]),
            "has_gauss": int(state[3]), "cached_gaussian": float(state[4])}


def _restore_numpy_state(saved):
    np.random.set_state((saved["bit_generator"], np.array(saved["keys"], dtype=np.uint32),
                         saved["pos"], saved["has_gauss"], saved["cached_gaussian"]))


def _capture_random_state(train_loader):
    return {"torch": torch.get_rng_state(), "numpy": _numpy_state_plain(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_loader": train_loader.generator.get_state()}


def _restore_random_state(saved, train_loader):
    torch.set_rng_state(saved["torch"].cpu())
    _restore_numpy_state(saved["numpy"])
    random.setstate(saved["python"])
    if torch.cuda.is_available() and saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])
    train_loader.generator.set_state(saved["train_loader"].cpu())


def train_one_epoch(model, loader, optimizer, scaler, device, config):
    """Sample-weighted microbatch accumulation; last incomplete effective batch normalized correctly."""
    model.train()
    total = np.zeros(3, dtype=float)
    n_images = len(loader.dataset)
    if n_images == 0:
        raise ValueError("Empty train loader")
    micro = config["batch_size"]
    accum = config["gradient_accumulation_steps"]
    effective = micro * accum
    n_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        crop = batch["crop"].to(device, non_blocking=True)
        disease = batch["disease"].to(device, non_blocking=True)
        block = (step // accum) * effective
        group_size = min(effective, n_images - block)  # handles incomplete last accumulation group
        with torch.autocast(device_type=device.type, enabled=config["amp"] and device.type == "cuda"):
            out = model(image)
            lc = nn.functional.cross_entropy(out["crop_logits"], crop)
            ld = nn.functional.cross_entropy(out["disease_logits"], disease)
            loss = config["crop_loss_weight"] * lc + config["disease_loss_weight"] * ld
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss. Stop; do not continue with corrupt gradients.")
        scaler.scale(loss * (len(image) / group_size)).backward()
        total += len(image) * np.array([float(loss.item()), float(lc.item()), float(ld.item())])
        if (step + 1) % accum == 0 or step + 1 == n_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return {name: float(value / n_images) for name, value in
            zip(("total_loss", "crop_loss", "disease_loss"), total)}


def plot_training_history(history, best_epoch, path, config):
    if not history:
        return
    table = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    axes[0, 0].plot(table.epoch, table.train_total_loss, label="Training")
    for split, label in (("seen", "Seen val"), ("unseen", "Unseen val")):
        weighted = config["crop_loss_weight"] * table[f"{split}_crop_loss"] + config["disease_loss_weight"] * table[f"{split}_disease_loss"]
        axes[0, 0].plot(table.epoch, weighted, label=label)
    axes[0, 0].set(title="Weighted CE loss", ylabel="Loss")
    axes[0, 1].plot(table.epoch, table.train_crop_loss, label="Crop train CE")
    axes[0, 1].plot(table.epoch, table.train_disease_loss, label="Disease train CE")
    axes[0, 1].set(title="Task losses", ylabel="Loss")
    axes[1, 0].plot(table.epoch, table.seen_composition_top1 * 100, label="Seen")
    axes[1, 0].plot(table.epoch, table.unseen_composition_top1 * 100, label="Unseen")
    axes[1, 0].set(title="Composition Top-1", ylabel="Accuracy (%)")
    axes[1, 1].plot(table.epoch, table.composition_harmonic_mean * 100, label="HM")
    axes[1, 1].set(title="Checkpoint criterion", ylabel="HM (%)")
    for ax in axes.flat:
        ax.axvline(best_epoch, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_stage1(project_dir, image_root, config, frames, split_report, candidates, *, resume=False):
    """Stage 1 only; raises instead of overwriting. Saves complete per-epoch resume state."""
    validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full CN01 training requires a CUDA GPU. Run this notebook on the Windows RTX 4050 laptop.")
    output_dir = Path(project_dir) / "outputs" / config["experiment_id"]
    if not resume and output_dir.exists():
        raise FileExistsError(f"CN01 output already exists: {output_dir}. Review it; do not overwrite.")
    if resume:
        if not (output_dir / "last_training.pt").is_file():
            raise FileNotFoundError("Cannot resume: no last_training.pt. Use resume=False for first run.")
        if (output_dir / "run_status.json").exists():
            status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                raise FileExistsError("CN01 already completed. Do not repeat it or overwrite the evidence.")
    seed_everything(config["seed"])
    loaders = make_loaders(frames, image_root, config, device)
    # A failed ImageNet download is an error; NEVER silently replace with random weights.
    model = MultiTaskConvNeXtTiny(crop_dropout=config["crop_dropout"],
                                  disease_dropout=config["disease_dropout"], pretrained=not resume).to(device)
    # Pretrained download succeeds before creating output directory on the first run.
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameter_groups(config["backbone_lr"], config["head_lr"]),
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    mappings, composition_mapping = mappings_from_frames(frames)
    env = environment_details(device)
    config_info = {"config": copy.deepcopy(config), "environment": env,
                   "development_split": split_report, "class_mappings": mappings,
                   "composition_mapping": composition_mapping,
                   "preprocessing_note": "Bilinear Resize(256)->CenterCrop(224), matched to Exp09 notebook, not bicubic evaluate.validation_transform",
                   "comparison_note": "ConvNeXt microbatch/grad accumulation differs from Exp09 physical batch 16; workers=0 vs Exp09 4.",
                   "accuracy_units": "fractions in [0,1]"}
    start_epoch, best, best_epoch, bad_epochs, history = 1, -math.inf, 0, 0, []
    if resume:
        saved_config = json.loads((output_dir / "configuration.json").read_text(encoding="utf-8"))
        if saved_config["config"] != config or saved_config["development_split"]["sha256"] != split_report["sha256"]:
            raise ValueError("Resume configuration/splits mismatch. STOP; do not corrupt the run.")
        state = torch.load(output_dir / "last_training.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best, best_epoch, bad_epochs = float(state["best"]), int(state["best_epoch"]), int(state["bad_epochs"])
        history = state["history"]
        if len(history) != start_epoch - 1:
            raise ValueError("Resume history and checkpoint epoch disagree")
        # Recover CSV even if a previous session crashed between checkpoint and CSV writes.
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        _restore_random_state(state["rng_state"], loaders["train_fit"])
        print(f"Resuming the same CN01 run from epoch {start_epoch} (best={best_epoch}).")
    else:
        atomic_json(config_info, output_dir / "configuration.json")
        atomic_json({"status": "training", "last_epoch": 0}, output_dir / "run_status.json")
    start_time = time.perf_counter()
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        lr = [group["lr"] for group in optimizer.param_groups]
        try:
            train = train_one_epoch(model, loaders["train_fit"], optimizer, scaler, device, config)
            seen, _ = evaluate_loader(model, loaders["val_seen"], device, candidates, amp=config["amp"])
            unseen, _ = evaluate_loader(model, loaders["val_unseen"], device, candidates, amp=config["amp"])
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA OOM: stop and review partial CN01 outputs. Do NOT change batch size within this run. "
                               "For a new run use batch=4/accumulation=4 and a distinct experiment identity.") from exc
        joint = generalisation_metrics(seen, unseen)
        metric = joint["composition_harmonic_mean"]
        if not math.isfinite(metric):
            raise RuntimeError("Nonfinite validation HM")
        row = {"epoch": epoch, "backbone_lr": lr[0], "head_lr": lr[1],
               **{f"train_{k}": v for k, v in train.items()},
               **{f"seen_{k}": v for k, v in seen.items()},
               **{f"unseen_{k}": v for k, v in unseen.items()}, **joint,
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        improved = metric > best + config["min_delta"]
        if improved:
            best, best_epoch, bad_epochs = metric, epoch, 0
        else:
            bad_epochs += 1
        scheduler.step()  # Same once-per-epoch schedule as Exp09.
        if improved:
            best_metrics = {"epoch": epoch, "seen": seen, "unseen": unseen, **joint}
            atomic_save({"stage": 1, "epoch": epoch, "config": copy.deepcopy(config),
                         "model_state": model.state_dict(), "candidate_pairs": candidates.tolist(),
                         "class_mappings": mappings, "composition_mapping": composition_mapping,
                         "development_split": split_report, "environment": env,
                         "best_metrics": best_metrics,
                         "parameter_count": sum(p.numel() for p in model.parameters())},
                        output_dir / "best_development.pt")
        atomic_save({"stage": 1, "epoch": epoch, "history": history, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "rng_state": _capture_random_state(loaders["train_fit"]),
                     "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs},
                    output_dir / "last_training.pt")
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        atomic_json({"status": "training", "last_epoch": epoch, "best_epoch": best_epoch}, output_dir / "run_status.json")
        print(f"Epoch {epoch:02d} | train loss={train['total_loss']:.4f} | "
              f"seen={seen['composition_top1']:.2%} | unseen={unseen['composition_top1']:.2%} | "
              f"HM={metric:.2%}" + ("  [BEST]" if improved else ""), flush=True)
        if bad_epochs >= config["patience"]:
            print(f"Early stopping after {bad_epochs} non-improving epochs.")
            break
    if best_epoch == 0:
        raise RuntimeError("No best checkpoint recorded; inspect training logs")
    best_ckpt = torch.load(output_dir / "best_development.pt", map_location="cpu", weights_only=True)
    summary = {**best_ckpt["best_metrics"], "epochs_run": len(history),
               "session_seconds": time.perf_counter() - start_time,
               "accuracy_units": "fractions in [0,1]"}
    atomic_json(summary, output_dir / "validation_metrics.json")
    plot_training_history(history, best_epoch, output_dir / "training_curves.png", config)
    atomic_json({"status": "complete", "last_epoch": len(history), "best_epoch": best_epoch},
                output_dir / "run_status.json")
    print(f"CN01 Stage 1 complete. Best epoch {best_epoch}; saved in {output_dir}")
    return output_dir


def _error_table(records, split):
    entries = []
    for r in records:
        crop_ok = r["crop_id"] == r["predicted_crop_id"]
        disease_ok = r["disease_id"] == r["predicted_disease_id"]
        pair_ok = r["crop_id"] == r["predicted_pair_crop_id"] and r["disease_id"] == r["predicted_pair_disease_id"]
        entries.append({**r, "split": split, "crop_correct": crop_ok,
                        "disease_correct": disease_ok, "pair_correct": pair_ok,
                        "error_type": ("both_correct" if crop_ok and disease_ok else
                                       "crop_only_correct" if crop_ok else
                                       "disease_only_correct" if disease_ok else "neither_correct")})
    return pd.DataFrame(entries)


def _per_class_table(df, frames, task, split):
    truth_col = f"{task}_id"
    if task not in ("crop", "disease", "composition"):
        raise ValueError("Invalid task")
    if task == "composition":
        grouped = df.groupby("composition_id", sort=True).agg(
            support=("composition_id", "size"), correct=("pair_correct", "sum"))
        mapping = dict(zip(frames[split].composition_id, frames[split].composition_name))
    else:
        grouped = df.groupby(truth_col, sort=True).agg(
            support=(truth_col, "size"), correct=("crop_correct" if task == "crop" else "disease_correct", "sum"))
        mapping = dict(zip(frames[split][truth_col], frames[split][f"{task}_name"]))
    grouped["accuracy"] = grouped.correct / grouped.support
    grouped["label_name"] = [str(mapping[i]) for i in grouped.index]
    grouped.index.name = "class_id"
    return grouped.reset_index()


def _save_confusion(df, task, out_dir, split):
    if task == "composition":
        matrix = composition_confusion(df.to_dict("records"))
        matrix.to_csv(out_dir / f"{split}_{task}_confusion_full.csv", index_label="true / predicted")
        # Exact-pair plot can be enormous; leave full matrix as CSV.
        return
    n_classes = 55 if task == "crop" else 175
    true_col, predicted = f"{task}_id", f"predicted_{task}_id"
    matrix = pd.crosstab(df[true_col], df[predicted]).reindex(
        index=range(n_classes), columns=range(n_classes), fill_value=0)
    matrix.to_csv(out_dir / f"{split}_{task}_confusion_full.csv", index_label="true / predicted")
    # Keep figures readable: plot top 20 frequent true labels + Other.
    prominent = df[true_col].value_counts().head(20).index.tolist()
    true_plot = df[true_col].apply(lambda x: f"{x}" if x in prominent else "Other")
    pred_plot = df[predicted].apply(lambda x: f"{x}" if x in prominent else "Other")
    names = [str(x) for x in prominent] + ["Other"]
    display_matrix = pd.crosstab(true_plot, pred_plot).reindex(index=names, columns=names, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(display_matrix, cmap="Blues")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           title=f"{split.title()} {task.title()} Confusion: Top 20 + Other",
           xlabel="Predicted class ID", ylabel="True class ID")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(im, ax=ax, label="Images")
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_{task}_confusion_top20.png", dpi=150)
    plt.close(fig)


def _save_mistake_gallery(df, image_root, out_dir, split, mappings):
    errors = df[~df.pair_correct]
    if errors.empty:
        return
    selected = pd.concat([errors[errors.error_type == "crop_only_correct"].head(4),
                          errors[errors.error_type == "neither_correct"].head(4),
                          errors[errors.error_type == "disease_only_correct"].head(4)], ignore_index=True)
    if selected.empty:
        return
    selected = selected.drop_duplicates("filename").head(12)
    cols = 3
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3.2), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, item in zip(axes.flat, selected.to_dict("records")):
        with Image.open(Path(image_root) / str(item["filename"]).replace("\\", "/")) as im:
            axis.imshow(ImageOps.exif_transpose(im.convert("RGB")))
        crop_names, disease_names = mappings["crop"], mappings["disease"]
        ac = crop_names.get(str(item["crop_id"]), f"c{item['crop_id']}")[:18]
        ad = disease_names.get(str(item["disease_id"]), f"d{item['disease_id']}")[:21]
        pc = crop_names.get(str(item["predicted_pair_crop_id"]), f"c{item['predicted_pair_crop_id']}")[:18]
        pdisease = disease_names.get(str(item["predicted_pair_disease_id"]), f"d{item['predicted_pair_disease_id']}")[:21]
        axis.set_title(f"{str(item['filename'])[:24]}\nTrue: {ac} / {ad}\nPred: {pc} / {pdisease}\n{item['error_type']}", fontsize=7.5)
    fig.suptitle(f"Actual {split} misclassified development images (no test images)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_error_examples.png", dpi=140)
    plt.close(fig)


def historical_comparison(cn01_metrics):
    """ResNet values transcribed from user-provided executed development notebooks; NOT new runs."""
    result = [
        {"run": "ResNet-50 Exp09", "source": "historical executed notebook", "backbone": "ResNet-50 ImageNet V2",
         "physical_batch": 16, "accumulation_steps": 1, "disease_weight": 1.0,
         "seen_composition_top1": 507/695, "unseen_composition_top1": 16/400, "composition_hm": 2*(507/695)*(16/400)/((507/695)+(16/400))},
        {"run": "ResNet-50 Exp15", "source": "historical executed notebook", "backbone": "ResNet-50 ImageNet V2",
         "physical_batch": 16, "accumulation_steps": 1, "disease_weight": 1.5,
         "seen_composition_top1": 504/695, "unseen_composition_top1": 17/400, "composition_hm": 2*(504/695)*(17/400)/((504/695)+(17/400))},
        {"run": "ConvNeXt CN01", "source": "new development evaluation", "backbone": "ConvNeXt-Tiny ImageNet V1",
         "physical_batch": 8, "accumulation_steps": 2, "disease_weight": 1.0,
         "seen_composition_top1": cn01_metrics["seen_composition_top1"],
         "unseen_composition_top1": cn01_metrics["unseen_composition_top1"],
         "composition_hm": cn01_metrics["composition_harmonic_mean"]},
    ]
    return pd.DataFrame(result)


def audit_development(project_dir, image_root, config):
    """Evaluate best saved checkpoint on fixed dev seen/unseen; never official test."""
    validate_config(config)
    out_dir = Path(project_dir) / "outputs" / config["experiment_id"]
    checkpoint_path = out_dir / "best_development.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("CN01 best_development.pt not found. Complete Stage 1 first.")
    frames, report, candidates = split_preflight(project_dir, image_root, verify_decode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_convnext_checkpoint(checkpoint_path, device)
    if checkpoint["config"] != config or checkpoint["development_split"]["sha256"] != report["sha256"]:
        raise ValueError("Loaded checkpoint config/split mismatch. Stop without evaluation.")
    if not torch.equal(torch.as_tensor(checkpoint["candidate_pairs"]), candidates):
        raise ValueError("Candidate ordering differs from training checkpoint")
    loaders = make_loaders(frames, image_root, config, device)
    results = {}
    combined_errors = []
    pair_train = set(map(tuple, frames["train_fit"][["crop_id", "disease_id"]].to_numpy().tolist()))
    for split, key in (("seen", "val_seen"), ("unseen", "val_unseen")):
        scores, records = evaluate_loader(model, loaders[key], device, candidates,
                                          amp=config["amp"] and device.type == "cuda", collect_predictions=True)
        df = _error_table(records, key)
        df["predicted_pair_seen_in_training"] = [
            (int(x), int(y)) in pair_train for x, y in zip(df.predicted_pair_crop_id, df.predicted_pair_disease_id)]
        # Save JSON list-columns as strings suitable for CSV round-tripping.
        csv_df = df.copy()
        for col in ("ranked_crop_ids", "ranked_disease_ids", "ranked_composition_pairs"):
            csv_df[col] = csv_df[col].map(json.dumps)
        csv_df.to_csv(out_dir / f"{key}_predictions.csv", index=False)
        combined_errors.append(csv_df)
        class_tables = {}
        for task in ("crop", "disease", "composition"):
            t = _per_class_table(df, frames, task, key)
            t.to_csv(out_dir / f"{key}_{task}_per_class.csv", index=False)
            class_tables[task] = t
            _save_confusion(df, task, out_dir, key)
        _save_mistake_gallery(df, image_root, out_dir, key, checkpoint["class_mappings"])
        err_counts = df.error_type.value_counts().to_dict()
        results[split] = {
            **scores,
            "crop_macro_accuracy": float(class_tables["crop"].accuracy.mean()),
            "disease_macro_accuracy": float(class_tables["disease"].accuracy.mean()),
            "composition_macro_accuracy": float(class_tables["composition"].accuracy.mean()),
            "error_breakdown": {k: int(err_counts.get(k, 0)) for k in (
                "crop_only_correct", "disease_only_correct", "both_correct", "neither_correct")},
            "fraction_predictions_in_training_compositions": float(df.predicted_pair_seen_in_training.mean()),
        }
    pd.concat(combined_errors, ignore_index=True).to_csv(out_dir / "task_error_breakdown.csv", index=False)
    merged = {"checkpoint_epoch": int(checkpoint["epoch"]), "seen": results["seen"],
              "unseen": results["unseen"], **generalisation_metrics(results["seen"], results["unseen"]),
              "source": "fixed development validation only", "accuracy_units": "fractions in [0,1]"}
    atomic_json(merged, out_dir / "development_audit.json")
    comparison = historical_comparison(merged)
    comparison.to_csv(out_dir / "development_comparison.csv", index=False)
    return merged, comparison
