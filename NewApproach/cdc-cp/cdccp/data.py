from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

from .config import image_dir

COLUMNS = ["filename", "crop_id", "disease_id", "composition_id", "crop_name", "disease_name", "composition_name"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TRAIN_SHORT_SIDE = 288
EVAL_SHORT_SIDE = 256
CROP = 224
MODES = ("train1", "train2v", "eval")

_normalise = T.Normalize(MEAN, STD)
EVAL_RESIZE = T.Resize(EVAL_SHORT_SIDE, IM.BICUBIC, antialias=True)
EVAL_TAIL = T.Compose([T.CenterCrop(CROP), T.ToTensor(), _normalise])
# Defaults match Baseline 1. A run config may override them under `aug:`.
AUG_DEFAULTS = {"crop_scale_min": 0.85, "rotation": 10, "brightness": 0.15, "contrast": 0.15, "saturation": 0.0, "hue": 0.0,
                "patch_grid": 0, "patch_p": 0.0}


class PatchShuffle:
    """With probability p, cut the image into grid x grid tiles and put them back in random order.

    The leaf outline breaks up while each disease spot stays whole inside its tile, so the model has to
    use the spots rather than the leaf shape. A coarse grid keeps the spots large enough to read.
    """

    def __init__(self, grid, p):
        if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2 or CROP % grid:
            raise ValueError(f"patch_grid must be >= 2 and divide {CROP}, got {grid}")
        if not 0 < p <= 1:
            raise ValueError(f"patch_p must be in (0, 1], got {p}")
        self.grid, self.p = grid, p

    def __call__(self, x):
        if torch.rand(()) >= self.p:
            return x
        c, h, w = x.shape
        g = self.grid
        tiles = x.reshape(c, g, h // g, g, w // g).permute(1, 3, 0, 2, 4).reshape(g * g, c, h // g, w // g)
        tiles = tiles[torch.randperm(g * g)]
        return tiles.reshape(g, g, c, h // g, w // g).permute(2, 0, 3, 1, 4).reshape(c, h, w)

    def __repr__(self):
        return f"PatchShuffle(grid={self.grid}, p={self.p})"


def train_tail(cfg=None):
    a = {**AUG_DEFAULTS, **((cfg or {}).get("aug") or {})}
    unknown = set(a) - set(AUG_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown aug keys: {sorted(unknown)}")
    patch = [PatchShuffle(a["patch_grid"], a["patch_p"])] if a["patch_grid"] or a["patch_p"] else []
    return T.Compose([
        T.RandomResizedCrop(CROP, scale=(a["crop_scale_min"], 1.0), interpolation=IM.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(a["rotation"], interpolation=IM.BILINEAR),
        T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"], saturation=a["saturation"], hue=a["hue"]),
        T.ToTensor(),
        *patch,
        _normalise,
    ])


TRAIN_TAIL = train_tail()


def load_meta(path, header=False):
    """Official CSVs have no header row; the split CSVs do."""
    if header:
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=COLUMNS)


def official_csv(cfg, name):
    return Path(cfg["paths"]["data_root"]) / f"DPD_Small_{name}.csv"


def load_split(cfg, name):
    """name is train, train_fit, val_seen or val_unseen. Official test CSVs load only through the lock path."""
    if name == "train":
        return load_meta(official_csv(cfg, "train"))
    if name in ("train_fit", "val_seen", "val_unseen"):
        return load_meta(Path(cfg["paths"]["splits_dir"]) / f"{name}.csv", header=True)
    raise ValueError(f"unknown split {name!r}")


ORIENTATION = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
               5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}


def upright(image):
    """The pixels of ImageOps.exif_transpose, without rewriting the EXIF block (which crashes on some official test images)."""
    method = ORIENTATION.get(image.getexif().get(0x0112))
    return image if method is None else image.transpose(method)  # FLIP_LEFT_RIGHT is 0, so test for None


def canonical_eval(path):
    """Decode by content, apply EXIF orientation, convert to RGB. No resize."""
    with Image.open(path) as image:
        return upright(image).convert("RGB")


def canonical_train(path):
    image = canonical_eval(path)
    short = min(image.size)
    if short > TRAIN_SHORT_SIDE:
        scale = TRAIN_SHORT_SIDE / short
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    return image


def cache_path(cache_dir, kind, filename):
    return Path(cache_dir) / kind / f"{filename}.npy"


class DPDDataset(Dataset):
    """Items are (image, crop_id, disease_id, composition_id, index). In train2v, image is [2,3,224,224]."""

    def __init__(self, df, cfg, mode, use_cache=False):
        if mode not in MODES:
            raise ValueError(mode)
        self.df = df.reset_index(drop=True)
        self.images = image_dir(cfg)
        self.mode = mode
        self.cache = Path(cfg["paths"]["cache_dir"]) if use_cache else None
        self.tail = train_tail(cfg)

    def __len__(self):
        return len(self.df)

    def _cached(self, kind, filename):
        if self.cache is None:
            return None
        path = cache_path(self.cache, kind, filename)
        return Image.fromarray(np.load(path)) if path.exists() else None

    def _load(self, filename):
        if self.mode == "eval":
            return self._cached("eval256", filename) or EVAL_RESIZE(canonical_eval(self.images / filename))
        return self._cached("train288", filename) or canonical_train(self.images / filename)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = self._load(row.filename)
        if self.mode == "eval":
            x = EVAL_TAIL(image)
        elif self.mode == "train1":
            x = self.tail(image)
        else:
            x = torch.stack([self.tail(image), self.tail(image)])
        return x, int(row.crop_id), int(row.disease_id), int(row.composition_id), i


def eval_loader(df, cfg, batch_size=None, use_cache=True):
    workers = cfg["loader"]["num_workers"]
    return DataLoader(
        DPDDataset(df, cfg, "eval", use_cache=use_cache),
        batch_size=batch_size or cfg["loader"]["eval_batch"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def batch_stream(df, cfg, batch_size, views, seed, use_cache=True, generator_state=None):
    """Endless shuffled batches with drop_last. Yields ([B*views,3,224,224], crop, disease)."""
    generator = torch.Generator().manual_seed(seed)
    if generator_state is not None:
        generator.set_state(generator_state)
    dataset = DPDDataset(df, cfg, "train2v" if views == 2 else "train1", use_cache=use_cache)
    workers = cfg["loader"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    while True:
        for x, crop, disease, _, _ in loader:
            if views == 2:
                x = x.transpose(0, 1).flatten(0, 1)
                crop, disease = crop.repeat(2), disease.repeat(2)
            yield x, crop, disease
