import hashlib
import os
import subprocess
from pathlib import Path

import torch

from .config import REPO


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"
    return commit + ("+dirty" if dirty else "")


def training_meta(df):
    return {"training_files": sorted(df.filename), "training_pairs": sorted(int(c) for c in set(df.composition_id))}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def load_checkpoint(path, model=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    if model is not None:
        model.load_state_dict(state["model"])
    return state["meta"]
