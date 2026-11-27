"""Independent single-task models (crop or disease): ConvNeXt and CLIP arms, Baseline 1 recipe plus regularised "reg" arms."""
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm
from torchvision import transforms as T
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights

COLUMNS = ["filename", "crop_id", "disease_id", "composition_id", "crop_name", "disease_name", "composition_name"]
B1 = dict(seed=42, batch_size=16, max_epochs=30, backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-3,
          dropout={"crop": 0.3, "disease": 0.2}, stage2_epochs=10, stage2_backbone_lr=1e-6,
          stage2_head_lr=1e-4, stage2_weight_decay=1e-3)
AUGS = ("b1", "zoom", "erase", "strong")
Arm = namedtuple("Arm", "task aug backbone recipe")
ARMS = {"crop_b1": Arm("crop", "b1", "convnext_tiny", "b1"),
        "crop_reg": Arm("crop", "strong", "convnext_tiny", "reg"),
        "disease_b1": Arm("disease", "b1", "convnext_tiny", "b1"),
        "disease_zoom": Arm("disease", "zoom", "convnext_tiny", "b1"),
        "disease_erase": Arm("disease", "erase", "convnext_tiny", "b1"),
        "disease_reg": Arm("disease", "strong", "convnext_tiny", "reg"),
        "disease_small_reg": Arm("disease", "strong", "convnext_small", "reg"),
        "disease_clip": Arm("disease", "b1", "clip_vit_b16", "clip"),
        "crop_clip": Arm("crop", "b1", "clip_vit_b16", "clip")}
CONTROLS = {"crop": "crop_b1", "disease": "disease_b1"}
BACKBONES = {"convnext_tiny": (tvm.convnext_tiny, ConvNeXt_Tiny_Weights),
             "convnext_small": (tvm.convnext_small, ConvNeXt_Small_Weights)}
RECIPES = {
    "b1": dict(label_smoothing=0.0, weight_decay=B1["weight_decay"], stage2_weight_decay=B1["stage2_weight_decay"],
               stochastic_depth={"convnext_tiny": None, "convnext_small": None}),
    "reg": dict(label_smoothing=0.1, weight_decay=0.05, stage2_weight_decay=0.05,
                stochastic_depth={"convnext_tiny": 0.2, "convnext_small": 0.4}),
    "clip": dict(label_smoothing=0.0, weight_decay=0.1, stage2_weight_decay=0.1, backbone_lr=1e-5, head_lr=1e-3,
                 stage2_backbone_lr=1e-6, stage2_head_lr=1e-4, stochastic_depth={"clip_vit_b16": None}),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
NORMS = {"clip_vit_b16": (CLIP_MEAN, CLIP_STD)}
CLIP_CHECKPOINT = "openai/clip-vit-base-patch16"
CLIP_PROJECTION_DIM = 512
CLIP_MAX_SCALE = 100.0
CLIP_TINY_INIT_SCALE = 1 / 0.07
PROMPT_TEMPLATES = {
    "disease": ("a photo of a plant leaf with {n}.", "a close-up photo of a leaf showing {n}.", "a leaf infected with {n}.",
                "a photo of {n}, a plant disease."),
    "crop": ("a photo of a {n} leaf.", "a close-up photo of a {n} plant leaf.", "a leaf of a {n} plant.", "a photo of a {n} plant."),
}
HEALTHY_PROMPTS = ("a photo of a healthy plant leaf.", "a close-up photo of a healthy leaf.",
                   "a healthy green leaf with no disease.", "a photo of a healthy plant.")
PROBE_BATCHES = (1, 8, 15, 16, 17, 32)
CLIP_TINY = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, patch_size=16,
                 image_size=224, projection_dim=CLIP_PROJECTION_DIM)
DEV_SIZES = (6905, 695, 400)
CLASS_COUNTS = (55, 175)
EVAL_BATCH = 32
HISTORY1 = ["epoch", "train_loss", "train_acc", "seen_loss", "seen_acc", "unseen_loss", "unseen_acc", "individual_h", "best_h", "seconds"]
HISTORY2 = ["epoch", "train_loss", "train_acc", "best_train_acc", "best_epoch", "seconds"]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _normalise(norm=None):
    return T.Normalize(*(norm or (MEAN, STD)))


def _transform_kwargs(backbone):
    return {"norm": NORMS[backbone]} if backbone in NORMS else {}


def train_transform(aug, norm=None):
    if aug not in AUGS:
        raise ValueError(f"unknown aug {aug!r}; choose from {AUGS}")
    if aug == "strong":
        _require(norm is None, "the strong aug has no custom normalisation")
        return _strong_transform()
    scale = (0.3, 1.0) if aug == "zoom" else (0.85, 1.0)
    steps = [
        T.RandomResizedCrop(224, scale=scale),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        _normalise(norm),
    ]
    if aug == "erase":
        steps.append(T.RandomErasing(p=0.5, scale=(0.02, 0.2)))
    return T.Compose(steps)


def _strong_transform():
    return T.Compose([T.RandomResizedCrop(224, scale=(0.3, 1.0)), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5),
                      T.TrivialAugmentWide(), T.ToTensor(), _normalise(), T.RandomErasing(p=0.25, scale=(0.02, 0.2))])


def eval_transform(norm=None):
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _normalise(norm)])


def read_labelled_csv(path):
    """Header row if the first cell is 'filename', else the official headerless 7 columns."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline().split(",")[0].strip().lower()
    df = pd.read_csv(path, header=0 if first == "filename" else None, names=None if first == "filename" else COLUMNS,
                     encoding="utf-8-sig")
    for col in ("crop_id", "disease_id", "composition_id"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    _require(df.filename.is_unique, f"duplicate filenames in {path}")
    return df


def _pairs(df):
    return set(zip(df.crop_id.tolist(), df.disease_id.tolist()))


def load_dev(splits_dir, expect_sizes=DEV_SIZES):
    d = Path(splits_dir)
    train_fit, val_seen, val_unseen = (read_labelled_csv(d / f"{n}.csv") for n in ("train_fit", "val_seen", "val_unseen"))
    if expect_sizes is not None:
        got = (len(train_fit), len(val_seen), len(val_unseen))
        _require(got == tuple(expect_sizes), f"split sizes {got} != {tuple(expect_sizes)}")
    names = [set(x.filename) for x in (train_fit, val_seen, val_unseen)]
    _require(not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]), "splits share filenames")
    fit_pairs, seen_pairs, unseen_pairs = _pairs(train_fit), _pairs(val_seen), _pairs(val_unseen)
    _require(seen_pairs <= fit_pairs, "val_seen has pairs that train_fit lacks")
    _require(not (unseen_pairs & (fit_pairs | seen_pairs)), "val_unseen pairs overlap train_fit or val_seen")
    return train_fit, val_seen, val_unseen


def class_ids(*dfs, expect=CLASS_COUNTS):
    crops = sorted(set().union(*(set(df.crop_id) for df in dfs)))
    diseases = sorted(set().union(*(set(df.disease_id) for df in dfs)))
    if expect is not None:
        _require((len(crops), len(diseases)) == tuple(expect), f"class counts {(len(crops), len(diseases))} != {tuple(expect)}")
    return [int(c) for c in crops], [int(d) for d in diseases]


class PlantDataset(Dataset):
    def __init__(self, df, image_dir, crop_ids, disease_ids, transform):
        self.rows = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.crop_index = {int(c): i for i, c in enumerate(crop_ids)}
        self.disease_index = {int(d): i for i, d in enumerate(disease_ids)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.image_dir / str(row.filename)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.crop_index[int(row.crop_id)], self.disease_index[int(row.disease_id)]


def make_loader(df, image_dir, crop_ids, disease_ids, transform, batch_size, shuffle, workers):
    return DataLoader(PlantDataset(df, image_dir, crop_ids, disease_ids, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(task, dropout, pretrained=True, n_classes=None, backbone="convnext_tiny", stochastic_depth_prob=None):
    if task not in ("crop", "disease"):
        raise ValueError(f"unknown task {task!r}")
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    constructor, weights = BACKBONES[backbone]
    extra = {} if stochastic_depth_prob is None else {"stochastic_depth_prob": stochastic_depth_prob}
    model = constructor(weights=weights.IMAGENET1K_V1 if pretrained else None, **extra)
    n = n_classes or CLASS_COUNTS[0 if task == "crop" else 1]
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[-1].in_features, n))
    return model


def class_prompts(task, names):
    """Per class, the list of prompt templates filled with the class name (underscores become spaces)."""
    _require(task in PROMPT_TEMPLATES, f"unknown task {task!r}")
    prompts = []
    for name in names:
        name = str(name).strip()
        if task == "disease" and name.lower() == "healthy":
            prompts.append(list(HEALTHY_PROMPTS))
        else:
            prompts.append([t.format(n=name.replace("_", " ")) for t in PROMPT_TEMPLATES[task]])
    return prompts


def disease_prompts(names):
    return class_prompts("disease", names)


def names_in_order(task, dfs, ids, contiguous):
    """Class names in class-index order. Class index j is ids[j]; with contiguous the ids must be 0..n-1."""
    _require(task in PROMPT_TEMPLATES, f"unknown task {task!r}")
    id_col, name_col = f"{task}_id", f"{task}_name"
    _require(list(ids) == sorted(set(ids)), f"{task} ids must be sorted and unique")
    if contiguous:
        _require(list(ids) == list(range(len(ids))), f"{task} ids are not 0..{len(ids) - 1}")
    pairs = pd.concat([df[[id_col, name_col]] for df in dfs]).drop_duplicates()
    _require(pairs[id_col].is_unique, f"a {task} id has more than one {name_col}")
    by_id = dict(zip(pairs[id_col].astype(int), pairs[name_col]))
    _require(set(by_id) == set(ids), f"{task} names do not cover the {task} ids")
    return [by_id[int(i)] for i in ids]


def disease_names_in_order(dfs, disease_ids, contiguous):
    return names_in_order("disease", dfs, disease_ids, contiguous)


class ClipDiseaseModel(nn.Module):
    """CLIP vision encoder plus a head fixed to frozen text embeddings: logits = s * cos(image, T). Serves both tasks."""

    NORM = NORMS["clip_vit_b16"]

    def __init__(self, vision, text_features, prompts, init_scale):
        super().__init__()
        _require(text_features.shape == (len(prompts), vision.config.projection_dim), "text features do not match prompts")
        self.vision = vision
        self.prompts = [list(p) if isinstance(p, (list, tuple)) else p for p in prompts]
        self.register_buffer("text_features", F.normalize(text_features.detach().float(), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(min(init_scale, CLIP_MAX_SCALE))))

    def scale(self):
        return self.logit_scale.clamp(max=math.log(CLIP_MAX_SCALE)).exp()

    @torch.no_grad()
    def clamp_logit_scale(self):
        self.logit_scale.clamp_(max=math.log(CLIP_MAX_SCALE))

    def forward(self, images):
        embeds = self.vision(pixel_values=images).image_embeds
        with torch.autocast(images.device.type, enabled=False):
            return self.scale() * F.normalize(embeds.float(), dim=-1) @ self.text_features.t()


def clip_text_features(prompts, pretrained):
    """One unit-norm row per class: each template encoded and L2-normalised, averaged per class, L2-normalised again."""
    if not pretrained:
        generator = torch.Generator().manual_seed(B1["seed"])
        return F.normalize(torch.randn(len(prompts), CLIP_PROJECTION_DIM, generator=generator), dim=-1)
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_CHECKPOINT)
    text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_CHECKPOINT).eval()
    flat = [text for group in prompts for text in group]
    with torch.no_grad():
        embeds = F.normalize(text_model(**tokenizer(flat, padding=True, return_tensors="pt")).text_embeds.float(), dim=-1)
    del text_model
    _require(embeds.shape[0] == len(flat), f"text model returned {embeds.shape[0]} rows for {len(flat)} prompts")
    rows, start = [], 0
    for group in prompts:
        rows.append(embeds[start:start + len(group)].mean(dim=0))
        start += len(group)
    return F.normalize(torch.stack(rows), dim=-1)


def build_clip_model(names, pretrained, task="disease"):
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    prompts = class_prompts(task, names)
    text_features = clip_text_features(prompts, pretrained)
    if pretrained:
        from transformers import CLIPModel
        init_scale = CLIPModel.from_pretrained(CLIP_CHECKPOINT).logit_scale.exp().item()
        vision = CLIPVisionModelWithProjection.from_pretrained(CLIP_CHECKPOINT)
    else:
        init_scale = CLIP_TINY_INIT_SCALE
        vision = CLIPVisionModelWithProjection(CLIPVisionConfig(**CLIP_TINY))
    return ClipDiseaseModel(vision, text_features, prompts, init_scale)


def rebuild_clip_model(ckpt):
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    info = ckpt["clip"]
    vision = CLIPVisionModelWithProjection(CLIPVisionConfig.from_dict(info["vision_config"]))
    model = ClipDiseaseModel(vision, torch.zeros(len(info["prompts"]), vision.config.projection_dim), info["prompts"], 1.0)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def check_backbone(backbone, pretrained=True):
    """Minute-1 check: the backbone loads and one zero image gives a [1, 512] embedding."""
    _require(backbone == "clip_vit_b16", f"no backbone check for {backbone!r}")
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    vision = (CLIPVisionModelWithProjection.from_pretrained(CLIP_CHECKPOINT) if pretrained
              else CLIPVisionModelWithProjection(CLIPVisionConfig(**CLIP_TINY))).eval()
    with torch.no_grad():
        shape = tuple(vision(pixel_values=torch.zeros(1, 3, 224, 224)).image_embeds.shape)
    _require(shape == (1, CLIP_PROJECTION_DIM), f"{backbone} embedding shape {shape} != (1, {CLIP_PROJECTION_DIM})")
    return shape


def _clip_vision(pretrained):
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    if pretrained:
        return CLIPVisionModelWithProjection.from_pretrained(CLIP_CHECKPOINT)
    return CLIPVisionModelWithProjection(CLIPVisionConfig(**CLIP_TINY))


def clip_text_check(task, names, pretrained=True):
    """Minute-1 check: the tokenizer and text model load and give one finite (n_classes, 512) feature matrix."""
    features = clip_text_features(class_prompts(task, names), pretrained)
    shape = tuple(features.shape)
    _require(shape == (len(names), CLIP_PROJECTION_DIM), f"{task} text features {shape} != ({len(names)}, {CLIP_PROJECTION_DIM})")
    _require(bool(torch.isfinite(features).all()), f"{task} text features are not finite")
    return shape


@torch.no_grad()
def clip_fp16_probe(device, batches=PROBE_BATCHES, pretrained=True):
    """Minute-1 check: the CLIP vision tower gives finite outputs under fp16 autocast for each batch size."""
    device = _device(device)
    vision = _clip_vision(pretrained).to(device).eval()
    generator = torch.Generator().manual_seed(B1["seed"])
    for batch in batches:
        images = torch.randn(batch, 3, 224, 224, generator=generator).to(device)
        with torch.autocast(device.type, dtype=torch.float16):
            embeds = vision(pixel_values=images).image_embeds
        _require(bool(torch.isfinite(embeds.float()).all()), f"fp16 CLIP output is not finite at batch size {batch}: set AMP = False")
    return tuple(batches)


@torch.no_grad()
def clip_zero_shot(task, df, image_dir, names, device, n=256, pretrained=True):
    """Zero-shot Top-1 (%) of the untrained text-anchored model on the first n rows. Labels come from the name column."""
    _require(task in PROMPT_TEMPLATES, f"unknown task {task!r}")
    names = [str(x) for x in names]
    _require(len(set(names)) == len(names), "class names are not unique")
    name_col = f"{task}_name"
    rows = df.head(n).reset_index(drop=True)
    _require(len(rows) > 0, "no rows to score")
    index = {name: i for i, name in enumerate(names)}
    unknown = sorted(set(rows[name_col].astype(str)) - set(index))
    _require(not unknown, f"{name_col} values not in the class names: {unknown[:5]}")
    labels = torch.tensor([index[str(x)] for x in rows[name_col]])
    device = _device(device)
    model = build_clip_model(names, pretrained, task).to(device).eval()
    transform = eval_transform(model.NORM)
    correct = 0
    for start in range(0, len(rows), EVAL_BATCH):
        chunk = rows.iloc[start:start + EVAL_BATCH]
        tensors = []
        for filename in chunk.filename:
            with Image.open(Path(image_dir) / str(filename)) as image:
                tensors.append(transform(image.convert("RGB")))
        logits = model(torch.stack(tensors).to(device))
        _require(bool(torch.isfinite(logits).all()), "non-finite zero-shot logits")
        correct += int((logits.argmax(dim=1).cpu() == labels[start:start + len(chunk)]).sum())
    return 100.0 * correct / len(rows)


def build_arm_model(arm, pretrained, n_classes, names=None):
    spec = ARMS[arm]
    if spec.backbone == "clip_vit_b16":
        _require(names is not None and len(names) == n_classes, f"clip arm needs one {spec.task} name per class")
        return build_clip_model(names, pretrained, spec.task)
    return build_model(spec.task, B1["dropout"][spec.task], pretrained, n_classes, spec.backbone,
                       RECIPES[spec.recipe]["stochastic_depth"][spec.backbone])


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    if isinstance(model, ClipDiseaseModel):
        head, backbone, head_decay = [model.logit_scale], list(model.vision.parameters()), 0.0
    else:
        head = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
        backbone = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
        head_decay = weight_decay
    _require(head and backbone and len(head) + len(backbone) == len(list(model.parameters())), "bad parameter split")
    return torch.optim.AdamW([{"params": backbone, "lr": backbone_lr, "weight_decay": weight_decay},
                              {"params": head, "lr": head_lr, "weight_decay": head_decay}])


def stage_lrs(recipe, stage):
    rc = RECIPES[recipe]
    if stage == 1:
        return rc.get("backbone_lr", B1["backbone_lr"]), rc.get("head_lr", B1["head_lr"])
    return rc.get("stage2_backbone_lr", B1["stage2_backbone_lr"]), rc.get("stage2_head_lr", B1["stage2_head_lr"])


def harmonic(a, b):
    return 0.0 if a + b == 0 else 2.0 * a * b / (a + b)


def _labels(task, crops, diseases, device):
    return (crops if task == "crop" else diseases).to(device, non_blocking=True)


def train_epoch(model, loader, task, optimizer, scaler, device, amp, max_batches=None, label_smoothing=0.0):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total, loss_sum, correct = 0, 0.0, 0
    for i, (images, crops, diseases) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = _labels(task, crops, diseases, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        _require(bool(torch.isfinite(loss)), f"training loss is not finite (amp={amp}); stop, do not keep this run")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if isinstance(model, ClipDiseaseModel):
            model.clamp_logit_scale()
        n = labels.numel()
        total += n
        loss_sum += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
    _require(total > 0, "train_epoch saw no batches")
    return {"loss": loss_sum / total, "top1": correct / total}


@torch.no_grad()
def evaluate(model, loader, task, device, amp):
    model.eval()
    total, loss_sum, correct, outputs = 0, 0.0, 0, []
    for images, crops, diseases in loader:
        images = images.to(device, non_blocking=True)
        labels = _labels(task, crops, diseases, device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        log_probs = F.log_softmax(logits.float(), dim=1)
        _require(bool(torch.isfinite(log_probs).all()), f"non-finite model outputs in evaluation (amp={amp}); stop, do not keep this run")
        loss_sum += F.nll_loss(log_probs, labels, reduction="sum").item()
        total += labels.numel()
        correct += (log_probs.argmax(1) == labels).sum().item()
        outputs.append(log_probs.cpu().numpy())
    _require(total > 0, "evaluate saw no samples")
    return {"loss": loss_sum / total, "top1": correct / total, "log_probs": np.concatenate(outputs, axis=0).astype(np.float32)}


def _device(device):
    return torch.device(device)


def _fresh_dir(out_dir, arm):
    folder = Path(out_dir) / arm
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already holds files; refusing to overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _checkpoint(model, arm, task, aug, epoch, ids, individual_h):
    crop_ids, disease_ids = ids
    ckpt = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "arm": arm, "task": task,
            "aug": aug, "backbone": ARMS[arm].backbone, "recipe": ARMS[arm].recipe, "epoch": epoch, "dropout": B1["dropout"][task], "crop_ids": list(crop_ids),
            "disease_ids": list(disease_ids), "individual_h": float(individual_h)}
    if isinstance(model, ClipDiseaseModel):
        ckpt["clip"] = {"vision_config": model.vision.config.to_dict(), "prompts": list(model.prompts)}
    return ckpt


def _eta(seconds_each, done, total):
    left = max(total - done, 0) * seconds_each
    return f"{int(left // 60)}m{int(left % 60):02d}s"


def run_stage1(arm, splits_dir, image_dir, out_dir, device, epochs=30, workers=2, pretrained=True, max_train_batches=None,
               expect_sizes=DEV_SIZES, expect_classes=CLASS_COUNTS, amp=True, patience=None):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    _require(1 <= epochs <= B1["max_epochs"], f"epochs must be 1..{B1['max_epochs']}")
    _require(patience is None or (isinstance(patience, int) and not isinstance(patience, bool) and patience >= 1),
             "patience must be None or an int >= 1")
    task, aug, backbone, recipe = ARMS[arm]
    rc = RECIPES[recipe]
    device = _device(device)
    amp = bool(amp) and device.type == "cuda"
    folder = _fresh_dir(out_dir, arm)
    train_fit, val_seen, val_unseen = load_dev(splits_dir, expect_sizes)
    ids = class_ids(train_fit, val_seen, val_unseen, expect=expect_classes)
    n_classes = len(ids[0] if task == "crop" else ids[1])
    names = None
    if backbone == "clip_vit_b16":
        names = names_in_order(task, (train_fit, val_seen, val_unseen), ids[0] if task == "crop" else ids[1],
                               expect_classes is not None)
    tf_kwargs = _transform_kwargs(backbone)
    _write_json({"arm": arm, "task": task, "aug": aug, "backbone": backbone, "recipe": recipe, "recipe_values": rc,
                 "epochs": epochs, "patience": patience, "b1": B1, "splits_dir": str(splits_dir),
                 "image_dir": str(image_dir), "pretrained": pretrained, "max_train_batches": max_train_batches,
                 "expect_sizes": None if expect_sizes is None else list(expect_sizes),
                 "expect_classes": None if expect_classes is None else list(expect_classes), "device": str(device),
                 "torch": torch.__version__, "prompts": None if names is None else class_prompts(task, names)},
                folder / "config.json")
    train_loader = make_loader(train_fit, image_dir, *ids, train_transform(aug, **tf_kwargs), B1["batch_size"], True, workers)
    seen_loader = make_loader(val_seen, image_dir, *ids, eval_transform(**tf_kwargs), EVAL_BATCH, False, workers)
    unseen_loader = make_loader(val_unseen, image_dir, *ids, eval_transform(**tf_kwargs), EVAL_BATCH, False, workers)

    seed_all(B1["seed"])
    model = build_arm_model(arm, pretrained, n_classes, names).to(device)
    optimizer = build_optimizer(model, *stage_lrs(recipe, 1), rc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=B1["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best_h, best_epoch, started = [], -1.0, 0, time.time()
    last_epoch, stopped_early = 0, False
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train = train_epoch(model, train_loader, task, optimizer, scaler, device, amp, max_train_batches,
                            rc["label_smoothing"])
        seen = evaluate(model, seen_loader, task, device, amp)
        unseen = evaluate(model, unseen_loader, task, device, amp)
        current_h = harmonic(seen["top1"], unseen["top1"])
        pred_path = folder / f"predictions_epoch_{epoch:02d}.npz"
        np.savez_compressed(pred_path, seen=seen["log_probs"], unseen=unseen["log_probs"])
        if current_h > best_h + 1e-9:
            best_h, best_epoch = current_h, epoch
            _atomic_save(_checkpoint(model, arm, task, aug, epoch, ids, current_h), folder / "best.pt")
            shutil.copy2(pred_path, folder / "best_validation.npz")
        seconds = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_acc": train["top1"], "seen_loss": seen["loss"],
                        "seen_acc": seen["top1"], "unseen_loss": unseen["loss"], "unseen_acc": unseen["top1"],
                        "individual_h": current_h, "best_h": best_h, "seconds": seconds})
        pd.DataFrame(history, columns=HISTORY1).to_csv(folder / "history.csv", index=False)
        scheduler.step()
        elapsed = time.time() - started
        print(f"[{arm}] epoch {epoch}/{epochs} train={train['top1']:.4f} seen={seen['top1']:.4f} unseen={unseen['top1']:.4f} "
              f"h={current_h:.4f} best_h={best_h:.4f}@{best_epoch} {seconds:.0f}s eta {_eta(elapsed / epoch, epoch, epochs)}", flush=True)
        last_epoch = epoch
        if patience is not None and epoch - best_epoch >= patience:
            stopped_early = True
            print(f"[{arm}] early stop at epoch {epoch} (best epoch {best_epoch}, patience {patience})", flush=True)
            break
    _write_json({"best_epoch": best_epoch, "best_h": best_h, "seconds": time.time() - started,
                 "last_epoch": last_epoch, "stopped_early": stopped_early}, folder / "done.json")
    return folder


def stage1_board(stage1_dir, arms):
    rows = []
    for arm in arms:
        folder = Path(stage1_dir) / arm
        done = json.loads((folder / "done.json").read_text())
        history = pd.read_csv(folder / "history.csv")
        best = history[history.epoch == done["best_epoch"]].iloc[0]
        npz = np.load(folder / "best_validation.npz")
        n_seen, n_unseen = npz["seen"].shape[0], npz["unseen"].shape[0]
        task, aug, backbone, recipe = ARMS[arm]
        rows.append({"arm": arm, "task": task, "aug": aug, "backbone": backbone, "recipe": recipe, "best_epoch": int(done["best_epoch"]),
                     "seen_acc": float(best.seen_acc), "unseen_acc": float(best.unseen_acc),
                     "individual_h": float(best.individual_h),
                     "seen_correct": int(round(best.seen_acc * n_seen)), "seen_n": int(n_seen),
                     "unseen_correct": int(round(best.unseen_acc * n_unseen)), "unseen_n": int(n_unseen)})
    return pd.DataFrame(rows)


def pick_winner(board, task="disease"):
    _require(task in CONTROLS, f"unknown task {task!r}")
    rows = board[board.task == task]
    _require(len(rows) > 0, f"no {task} arms on the board")
    scores = {r.arm: r.individual_h for r in rows.itertuples()}
    top = max(scores.values())
    if scores.get(CONTROLS[task]) == top:
        return CONTROLS[task]
    for arm in ARMS:
        if scores.get(arm) == top:
            return arm
    raise ValueError("no winner found; unknown arm on the board")


def composition_metrics(crop_log, disease_log, true_crop, true_disease, allowed=None):
    crop_log, disease_log = np.asarray(crop_log, dtype=np.float64), np.asarray(disease_log, dtype=np.float64)
    true_crop, true_disease = np.asarray(true_crop), np.asarray(true_disease)
    n_crop, n_disease = crop_log.shape[1], disease_log.shape[1]
    scores = (crop_log[:, :, None] + disease_log[:, None, :]).reshape(len(crop_log), n_crop * n_disease)
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        _require(allowed.shape == (n_crop * n_disease,), f"allowed must have shape ({n_crop * n_disease},)")
        scores[:, ~allowed] = -np.inf
    true_pair = true_crop * n_disease + true_disease

    def top(matrix, truth, k):
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :min(k, matrix.shape[1])]
        hit = (order == truth[:, None]) & np.isfinite(np.take_along_axis(matrix, order, axis=1))
        return float(hit.any(axis=1).mean())

    return {"crop_top1": top(crop_log, true_crop, 1), "crop_top5": top(crop_log, true_crop, 5),
            "disease_top1": top(disease_log, true_disease, 1), "disease_top5": top(disease_log, true_disease, 5),
            "composition_top1": top(scores, true_pair, 1), "composition_top5": top(scores, true_pair, 5)}


def dev_composition(stage1_dir, crop_arm, disease_arm, splits_dir, expect_sizes=DEV_SIZES, expect_classes=CLASS_COUNTS):
    _require(crop_arm in ARMS and disease_arm in ARMS and ARMS[crop_arm].task == "crop" and ARMS[disease_arm].task == "disease",
             "need one crop arm and one disease arm")
    train_fit, val_seen, val_unseen = load_dev(splits_dir, expect_sizes)
    crop_ids, disease_ids = class_ids(train_fit, val_seen, val_unseen, expect=expect_classes)
    crop_index = {c: i for i, c in enumerate(crop_ids)}
    disease_index = {d: i for i, d in enumerate(disease_ids)}
    crop_npz = np.load(Path(stage1_dir) / crop_arm / "best_validation.npz")
    disease_npz = np.load(Path(stage1_dir) / disease_arm / "best_validation.npz")
    out = {}
    for name, df in (("seen", val_seen), ("unseen", val_unseen)):
        out[name] = composition_metrics(crop_npz[name], disease_npz[name], df.crop_id.map(crop_index).to_numpy(),
                                        df.disease_id.map(disease_index).to_numpy())
    out["hm"] = harmonic(out["seen"]["composition_top1"], out["unseen"]["composition_top1"])
    return out


def run_stage2(arm, stage1_dir, train_csv, image_dir, out_dir, device, epochs=10, workers=2, max_train_batches=None,
               expect_rows=8000, amp=True):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    task, aug, backbone, recipe = ARMS[arm]
    rc = RECIPES[recipe]
    device = _device(device)
    amp = bool(amp) and device.type == "cuda"
    folder = _fresh_dir(out_dir, arm)
    src = Path(stage1_dir) / arm
    cfg1 = json.loads((src / "config.json").read_text())
    source = torch.load(src / "best.pt", map_location="cpu", weights_only=True)
    _require(source["arm"] == arm and source["task"] == task and source["aug"] == aug
             and source.get("backbone", "convnext_tiny") == backbone and source.get("recipe", "b1") == recipe, "stage 1 checkpoint is for another arm")
    ids = (source["crop_ids"], source["disease_ids"])
    full = read_labelled_csv(train_csv)
    if expect_rows is not None:
        _require(len(full) == expect_rows, f"train csv has {len(full)} rows, expected {expect_rows}")
    sizes = cfg1["expect_sizes"]
    parts = load_dev(cfg1["splits_dir"], sizes)
    _require(set(full.filename) == set().union(*(set(p.filename) for p in parts)),
             "train csv filenames differ from train_fit + val_seen + val_unseen")
    _require(set(full.crop_id) <= set(ids[0]) and set(full.disease_id) <= set(ids[1]), "train csv has ids unknown to stage 1")
    _write_json({"arm": arm, "task": task, "aug": aug, "backbone": backbone, "recipe": recipe, "recipe_values": rc, "stage": 2, "epochs": epochs, "source_epoch": source["epoch"],
                 "b1": B1, "train_csv": str(train_csv), "image_dir": str(image_dir), "stage1_dir": str(stage1_dir),
                 "max_train_batches": max_train_batches, "expect_rows": expect_rows, "device": str(device),
                 "torch": torch.__version__}, folder / "config.json")
    loader = make_loader(full, image_dir, *ids, train_transform(aug, **_transform_kwargs(backbone)), B1["batch_size"], True, workers)

    seed_all(B1["seed"])
    if backbone == "clip_vit_b16":
        model = rebuild_clip_model(source)
    else:
        model = build_arm_model(arm, False, len(ids[0] if task == "crop" else ids[1]))
        model.load_state_dict(source["state_dict"], strict=True)
    model.to(device)
    optimizer = build_optimizer(model, *stage_lrs(recipe, 2), rc["stage2_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=B1["stage2_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best_acc, best_epoch, started = [], -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train = train_epoch(model, loader, task, optimizer, scaler, device, amp, max_train_batches, rc["label_smoothing"])
        if train["top1"] > best_acc:
            best_acc, best_epoch = train["top1"], epoch
            ckpt = _checkpoint(model, arm, task, aug, epoch, ids, source["individual_h"])
            ckpt.update({"stage": 2, "source_epoch": int(source["epoch"]), "train_acc": float(train["top1"])})
            _atomic_save(ckpt, folder / "best_train_acc.pt")
        scheduler.step()
        seconds = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_acc": train["top1"], "best_train_acc": best_acc,
                        "best_epoch": best_epoch, "seconds": seconds})
        pd.DataFrame(history, columns=HISTORY2).to_csv(folder / "history.csv", index=False)
        elapsed = time.time() - started
        print(f"[{arm}] stage2 epoch {epoch}/{epochs} train={train['top1']:.4f} best={best_acc:.4f}@{best_epoch} "
              f"{seconds:.0f}s eta {_eta(elapsed / epoch, epoch, epochs)}", flush=True)
    _write_json({"best_epoch": best_epoch, "best_train_acc": best_acc, "source_epoch": int(source["epoch"]),
                 "seconds": time.time() - started}, folder / "done.json")
    return folder


def load_trained(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    task = ckpt["task"]
    if ckpt.get("backbone") == "clip_vit_b16":
        model = rebuild_clip_model(ckpt)
    else:
        model = build_model(task, ckpt["dropout"], False, len(ckpt["crop_ids"] if task == "crop" else ckpt["disease_ids"]),
                            ckpt.get("backbone", "convnext_tiny"))
        model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(_device(device)).eval()


@torch.no_grad()
def predict(model, df, image_dir, crop_ids, disease_ids, device, amp, workers=2, flip=False):
    device = _device(device)
    loader = make_loader(df, image_dir, crop_ids, disease_ids, eval_transform(getattr(model, "NORM", None)), EVAL_BATCH, False, workers)
    model.eval()
    outputs = []
    for images, _, _ in loader:
        with torch.autocast(device.type, enabled=amp):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            flipped = model(images.flip(-1)) if flip else None
        log_probs = F.log_softmax(logits.float(), dim=1)
        if flip:
            both = torch.stack([log_probs, F.log_softmax(flipped.float(), dim=1)])
            log_probs = torch.logsumexp(both, dim=0) - math.log(2.0)
        _require(bool(torch.isfinite(log_probs).all()), f"non-finite model outputs in predict (amp={amp})")
        outputs.append(log_probs.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_lock(files, info, out_path):
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite")
    lock = {"files": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in files.items()}, "info": info}
    out.write_text(json.dumps(lock, indent=2, sort_keys=True))
    return lock


def verify_lock(lock_path):
    lock = json.loads(Path(lock_path).read_text())
    for name, entry in lock["files"].items():
        _require(Path(entry["path"]).exists(), f"locked file {name} is missing: {entry['path']}")
        _require(_sha256(entry["path"]) == entry["sha256"], f"locked file {name} changed: {entry['path']}")
    return lock
