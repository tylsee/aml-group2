"""Cache train images as raw uint8 arrays. Official test images are never cached.

Usage: python scripts/build_cache.py [--verify] [--workers N] [key=value ...]   e.g. paths.cache_dir=/content/cache
Writes are atomic, so an interrupted build never leaves a half-written file. --verify loads every file,
deletes any that is unreadable or the wrong shape, and rebuilds it.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import image_dir, load_config
from cdccp.data import EVAL_RESIZE, EVAL_SHORT_SIDE, TRAIN_SHORT_SIDE, cache_path, canonical_eval, canonical_train, load_split


def _save(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _valid(path, kind):
    try:
        a = np.load(path)
    except Exception:
        return False
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        return False
    short = min(a.shape[:2])
    return short == EVAL_SHORT_SIDE if kind == "eval256" else short <= TRAIN_SHORT_SIDE


def build(filename, images, cache, verify):
    rebuilt = 0
    for kind, make in (
        ("train288", lambda: np.asarray(canonical_train(images / filename))),
        ("eval256", lambda: np.asarray(EVAL_RESIZE(canonical_eval(images / filename)))),
    ):
        path = cache_path(cache, kind, filename)
        if path.exists() and verify and not _valid(path, kind):
            path.unlink()
        if not path.exists():
            _save(path, make())
            rebuilt += 1
    return rebuilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(overrides=args.overrides)
    images, cache = image_dir(cfg), Path(cfg["paths"]["cache_dir"])
    for kind in ("train288", "eval256"):
        (cache / kind).mkdir(parents=True, exist_ok=True)
        for leftover in (cache / kind).glob("*.tmp"):
            leftover.unlink()
    names = list(load_split(cfg, "train").filename)
    with ThreadPoolExecutor(args.workers) as pool:
        rebuilt = sum(pool.map(lambda n: build(n, images, cache, args.verify), names))
    print(f"{len(names)} images checked{' and verified' if args.verify else ''}; {rebuilt} files written to {cache}")


if __name__ == "__main__":
    main()
