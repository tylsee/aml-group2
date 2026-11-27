"""The lock and the final test. Official test rows are read only through load_official, which needs a verified lock.

A lock folder holds lock.json, catalogue.json and the checkpoints, all addressed by paths relative to lock.json, so the
folder can move (for example from /kaggle/working to /kaggle/input) and still verify.
"""
import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import torch

from .bn import recompute_bn
from .catalogue import Catalogue, load_catalogue
from .checkpoint import file_sha256, load_checkpoint
from .data import eval_loader, load_meta, load_split, official_csv
from .evaluate import Refused, check_rows, combined_metrics, confusion, seen_rows, split_metrics
from .extract import extract
from .models import build_model
from .prototypes import build_banks
from .scoring import head_score, proto_cosines, proto_score, rank

LOCK_FORMAT = "cdccp-lock-v1"
LOCK_NAME = "lock.json"
CATALOGUE_NAME = "catalogue.json"
KINDS = ("head", "proto-pooled", "proto-balanced")
OFFICIAL = {"test_seen": "official_seen", "test_unseen": "official_unseen"}
DEFAULT_SETTINGS = {"r": 1.0, "tol": 3.0, "bn_policy": "recomputed on a seeded subset of the official training rows"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class VerifiedLock(dict):
    """A lock dict that verify_lock accepted. dir is the folder holding lock.json, sha256 the hash of its bytes."""

    def __init__(self, data, directory, sha256):
        super().__init__(data)
        self.dir = Path(directory)
        self.sha256 = sha256

    def path(self, relative):
        return _resolve(self.dir, relative)


def _require_verified(lock):
    if not isinstance(lock, VerifiedLock):
        raise Refused("official test rows need a lock from verify_lock")


def _safe_relative(relative, what):
    if not isinstance(relative, str) or not relative:
        raise Refused(f"{what}: path must be a non-empty string")
    windows = PureWindowsPath(relative)
    if PurePosixPath(relative).is_absolute() or windows.is_absolute() or windows.drive or relative.startswith(("/", "\\")):
        raise Refused(f"{what}: absolute path {relative!r} is not allowed")
    if ".." in re.split(r"[\\/]+", relative):
        raise Refused(f"{what}: path {relative!r} leaves the lock folder")
    return relative


def _resolve(directory, relative):
    _safe_relative(relative, "lock path")
    base = Path(directory).resolve()
    target = (base / relative).resolve()
    if os.path.commonpath([str(base), str(target)]) != str(base):
        raise Refused(f"lock path {relative!r} resolves outside the lock folder")
    return target


def full_catalogue(n_crop=55, n_disease=175, seen_pairs=()):
    """Every (crop, disease) pair, comp_id = crop * n_disease + disease. The rule Baseline 1 ranks under."""
    seen = {(int(c), int(d)) for c, d in seen_pairs}
    bad = [p for p in seen if not (0 <= p[0] < n_crop and 0 <= p[1] < n_disease)]
    if bad:
        raise ValueError(f"seen pairs outside the grid: {sorted(bad)[:5]}")
    crop = np.repeat(np.arange(n_crop), n_disease)
    disease = np.tile(np.arange(n_disease), n_crop)
    return Catalogue(
        comp_id=crop * n_disease + disease,
        crop_id=crop,
        disease_id=disease,
        is_seen=np.array([(int(c), int(d)) in seen for c, d in zip(crop, disease)]),
        names=[f"crop{c}_disease{d}" for c, d in zip(crop, disease)],
    )


def with_full_ids(npz, n_disease=175):
    """The same rows with comp_id replaced by the synthetic full-catalogue id."""
    return {**npz, "comp_id": np.asarray(npz["crop_id"], dtype=np.int64) * n_disease + np.asarray(npz["disease_id"], dtype=np.int64)}


def seen_pair_set(cat):
    return {(int(c), int(d)) for c, d in zip(cat.crop_id[cat.is_seen], cat.disease_id[cat.is_seen])}


def gammas_from_boards(board_csv_paths, scorer):
    """Mean of `gamma keep` for one scorer across dev boards (one per seed)."""
    paths = list(board_csv_paths)
    if not paths:
        raise Refused(f"no dev boards given for scorer {scorer!r}")
    values = []
    for path in paths:
        board = pd.read_csv(path)
        rows = board[board["scorer"] == scorer]
        if len(rows) != 1:
            raise Refused(f"{path}: expected one row for scorer {scorer!r}, found {len(rows)}")
        value = float(rows["gamma keep"].iloc[0])
        if not math.isfinite(value):
            raise Refused(f"{path}: gamma keep for {scorer!r} is not a finite number")
        values.append(value)
    return float(np.mean(values))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_lock(models, gammas, catalogue_path, out_path, settings=None):
    """Write lock.json before any test run. models: [{"name", "checkpoint", "kind"}]. gammas: {name: gamma}.

    Checkpoints must sit inside the lock folder. The catalogue is copied next to the lock as catalogue.json.
    """
    out_path = Path(out_path)
    base = out_path.parent.resolve()
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; a lock is never overwritten")
    if not models:
        raise ValueError("no models to lock")
    entries, seen_names = {}, set()
    for spec in models:
        name, kind = spec["name"], spec["kind"]
        if not _NAME.match(name) or name in seen_names:
            raise ValueError(f"model name {name!r} is invalid or repeated")
        seen_names.add(name)
        if kind not in KINDS:
            raise ValueError(f"model {name}: kind {kind!r} not in {KINDS}")
        if name not in gammas:
            raise ValueError(f"model {name}: no gamma given")
        gamma = float(gammas[name])
        if not math.isfinite(gamma):
            raise ValueError(f"model {name}: gamma is not finite")
        checkpoint = Path(spec["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        try:
            relative = checkpoint.relative_to(base).as_posix()
        except ValueError:
            raise ValueError(f"model {name}: checkpoint {checkpoint} is not inside the lock folder {base}") from None
        meta = load_checkpoint(checkpoint)
        for key in ("config", "training_files", "training_pairs"):
            if key not in meta:
                raise ValueError(f"model {name}: checkpoint meta lacks {key!r}")
        entries[name] = {
            "checkpoint": relative, "sha256": file_sha256(checkpoint), "kind": kind, "gamma": gamma,
            "bank_source": relative, "stage": meta.get("stage"), "run": meta["config"].get("name"), "seed": meta["config"].get("seed"),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalogue_copy = base / CATALOGUE_NAME
    if not (catalogue_copy.exists() and os.path.samefile(catalogue_path, catalogue_copy)):
        shutil.copyfile(catalogue_path, catalogue_copy)
    cat = load_catalogue(catalogue_copy)
    lock = {
        "format": LOCK_FORMAT,
        "test_run": False,
        "created": _now(),
        "catalogue": {"file": CATALOGUE_NAME, "sha256": file_sha256(catalogue_copy), "pairs": len(cat), "seen": int(cat.is_seen.sum()),
                      "copied_from": str(catalogue_path)},
        "settings": {**DEFAULT_SETTINGS, **(settings or {})},
        "models": entries,
    }
    out_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def verify_lock(lock_path):
    """Recompute every hash. Raises Refused on any mismatch, missing file or a file make_lock did not produce."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise Refused(f"lock file {lock_path} is missing")
    raw = lock_path.read_bytes()
    try:
        lock = json.loads(raw)
    except ValueError as error:
        raise Refused(f"lock file is not JSON: {error}") from None
    if not isinstance(lock, dict) or lock.get("format") != LOCK_FORMAT or lock.get("test_run") is not False:
        raise Refused("not a lock produced by make_lock, or already marked as run")
    for key in ("created", "catalogue", "settings", "models"):
        if key not in lock:
            raise Refused(f"lock lacks {key!r}")
    if not isinstance(lock["models"], dict) or not lock["models"]:
        raise Refused("lock has no models")
    directory = lock_path.resolve().parent
    catalogue = lock["catalogue"]
    if not isinstance(catalogue, dict):
        raise Refused("lock catalogue entry is malformed")
    _check_file(directory, catalogue.get("file"), catalogue.get("sha256"), "catalogue")
    for name, entry in lock["models"].items():
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            raise Refused(f"model {name}: malformed entry")
        gamma = entry.get("gamma")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma):
            raise Refused(f"model {name}: gamma is not a finite number")
        if entry.get("bank_source") != entry.get("checkpoint"):
            raise Refused(f"model {name}: prototype banks must come from the same checkpoint")
        _check_file(directory, entry.get("checkpoint"), entry.get("sha256"), f"checkpoint of {name}")
    return VerifiedLock(lock, directory, hashlib.sha256(raw).hexdigest())


def _check_file(directory, relative, expected, what):
    path = _resolve(directory, relative)
    if not path.is_file():
        raise Refused(f"{what}: file {relative} is missing")
    if not isinstance(expected, str) or file_sha256(path) != expected:
        raise Refused(f"{what}: sha256 mismatch for {relative}")


def load_official(cfg, name, lock):
    """The only reader of the official test CSVs."""
    _require_verified(lock)
    if name not in OFFICIAL:
        raise ValueError(f"name must be one of {sorted(OFFICIAL)}, got {name!r}")
    return load_meta(official_csv(cfg, name))


class Scorer:
    """Pair scores and the crop and disease scores that go with them, for one locked model."""

    def __init__(self, kind, banks=None, r=1.0):
        if kind not in KINDS or (kind != "head" and banks is None):
            raise ValueError(f"scorer {kind!r} needs banks for prototype kinds")
        self.kind, self.banks, self.r = kind, banks, r

    def scores(self, npz, cat, g=0.0):
        if self.kind == "head":
            return head_score(npz, cat, self.r, g)
        return proto_score(npz, self.banks, cat, self.r, g)

    def heads(self, npz):
        """(crop_scores, disease_scores) for split_metrics. None means the logits."""
        if self.kind == "head":
            return None, None
        return proto_cosines(npz, self.banks)


def score_split_pair(scorer, seen, unseen, cat, gamma):
    """Metrics of both splits under one catalogue and gamma, plus the scores."""
    out = {}
    for label, npz in (("seen", seen), ("unseen", unseen)):
        scores = scorer.scores(npz, cat, gamma)
        crop, disease = scorer.heads(npz)
        out[label] = (scores, split_metrics(scores, npz, cat, crop, disease))
    seen_m, unseen_m = out["seen"][1], out["unseen"][1]
    return {"seen": seen_m, "unseen": unseen_m, "combined": combined_metrics(seen_m, unseen_m)}, out["seen"][0], out["unseen"][0]


def prediction_rows(scores, npz, cat, split, lock_sha256):
    top = rank(scores)[:, :5]
    frame = pd.DataFrame({
        "lock_sha256": lock_sha256, "filename": npz["fn"], "split": split,
        "true_crop_id": npz["crop_id"], "true_disease_id": npz["disease_id"], "true_comp_id": npz["comp_id"],
    })
    for k in range(5):
        frame[f"top{k + 1}_comp_id"] = cat.comp_id[top[:, k]]
    frame["top1_name"] = [cat.names[i] for i in top[:, 0]]
    frame["top1_crop_id"] = cat.crop_id[top[:, 0]]
    frame["top1_disease_id"] = cat.disease_id[top[:, 0]]
    return frame


def evaluate_model(cfg, name, lock, device=None, use_cache=True):
    """Score one locked model on the official test sets. cfg supplies this machine's paths and loader settings."""
    _require_verified(lock)
    entry = lock["models"][name]
    checkpoint = lock.path(entry["checkpoint"])
    if file_sha256(checkpoint) != entry["sha256"]:
        raise Refused(f"checkpoint of {name} changed since the lock was verified")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_checkpoint(checkpoint)
    run_cfg = {**meta["config"], "paths": cfg["paths"], "loader": cfg["loader"]}
    r = float(lock["settings"]["r"])
    n_crop, n_disease = run_cfg["model"]["crop_classes"], run_cfg["model"]["disease_classes"]

    model = build_model(run_cfg, pretrained=False).to(device)  # weights come from the checkpoint; no download
    load_checkpoint(checkpoint, model)
    cat = load_catalogue(lock.path(lock["catalogue"]["file"]))
    full = full_catalogue(n_crop, n_disease, seen_pair_set(cat))

    train_df = load_split(run_cfg, "train")
    seen_df, unseen_df = load_official(run_cfg, "test_seen", lock), load_official(run_cfg, "test_unseen", lock)
    check_rows(seen_df, "official_seen", meta, lock)
    check_rows(unseen_df, "official_unseen", meta, lock)

    bn_df = train_df.sample(min(run_cfg["bn"]["subset"], len(train_df)), random_state=run_cfg["seed"]).sort_values("filename")
    recompute_bn(model, eval_loader(bn_df, run_cfg, run_cfg["bn"]["batch"], use_cache=use_cache), device)
    start = time.perf_counter()
    seen = extract(model, seen_df, run_cfg, device, use_cache)
    unseen = extract(model, unseen_df, run_cfg, device, use_cache)
    seconds_per_image = (time.perf_counter() - start) / (len(seen_df) + len(unseen_df))

    banks = None
    if entry["kind"] != "head":
        train = extract(model, train_df, run_cfg, device, use_cache)
        banks = build_banks(train, entry["kind"].split("-", 1)[1], n_crop, n_disease)
    scorer = Scorer(entry["kind"], banks, r)

    metrics, confusions = {}, {}
    raw, raw_s, raw_u = score_split_pair(scorer, seen, unseen, cat, 0.0)
    cal, cal_s, cal_u = score_split_pair(scorer, seen, unseen, cat, entry["gamma"])
    metrics["locked_raw"] = {"gamma": 0.0, **raw}
    metrics["locked_calibrated"] = {"gamma": entry["gamma"], **cal}
    confusions["locked_raw"] = seen_rows(confusion(raw_s, seen, cat), cat)
    confusions["locked_calibrated"] = seen_rows(confusion(cal_s, seen, cat), cat)
    full_seen, full_unseen = with_full_ids(seen, n_disease), with_full_ids(unseen, n_disease)
    full_m, full_s, full_u = score_split_pair(scorer, full_seen, full_unseen, full, 0.0)
    metrics["full_raw"] = {"gamma": 0.0, **full_m}

    parts = []
    for label, catalogue, s_scores, u_scores, s_npz, u_npz in (
            ("locked_raw", cat, raw_s, raw_u, seen, unseen), ("locked_calibrated", cat, cal_s, cal_u, seen, unseen),
            ("full_raw", full, full_s, full_u, full_seen, full_unseen)):
        for split, scores, npz in (("seen", s_scores, s_npz), ("unseen", u_scores, u_npz)):
            parts.append(prediction_rows(scores, npz, catalogue, split, lock.sha256).assign(catalogue=label))
    predictions = pd.concat(parts, ignore_index=True)
    return {
        "name": name, "kind": entry["kind"], "gamma": entry["gamma"],
        "params": int(sum(p.numel() for p in model.parameters())),
        "seconds_per_image": seconds_per_image,
        "seconds_per_image_note": "forward pass plus image loading, on the device this run used",
        "n_seen": len(seen_df), "n_unseen": len(unseen_df),
        "catalogue_sizes": {"locked": len(cat), "full": len(full)},
        "metrics": metrics, "confusions": confusions, "predictions": predictions,
    }


def metrics_table(results, lock_sha256):
    rows = []
    for res in results:
        for catalogue, block in res["metrics"].items():
            for split in ("seen", "unseen", "combined"):
                for metric, value in block[split].items():
                    rows.append({"model": res["name"], "catalogue": catalogue, "gamma": block["gamma"], "split": split,
                                 "metric": metric, "value": float(value), "lock_sha256": lock_sha256})
    return pd.DataFrame(rows)


def _heatmap(matrix, path, title, lock_sha256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9.5))
    image = ax.imshow(np.log1p(matrix), aspect="auto", cmap="viridis")
    ax.set_xlabel("predicted catalogue pair")
    ax.set_ylabel("true seen pair")
    ax.set_title(f"{title}\nlock {lock_sha256[:12]}")
    fig.colorbar(image, ax=ax, label="log(1 + count)")
    fig.savefig(path, dpi=120, metadata={"Description": f"lock_sha256={lock_sha256}"})
    plt.close(fig)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(out_dir, lock, results):
    """Every file except final_done.json. final_metrics.csv goes last, so its presence means all other files exist."""
    out_dir = Path(out_dir)
    tag = lock.sha256[:12]
    files = []
    for res in results:
        name = res["name"]
        res["predictions"].to_csv(out_dir / f"predictions_{name}.csv", index=False)
        files.append(f"predictions_{name}.csv")
        for key, matrix in res["confusions"].items():
            np.save(out_dir / f"confusion_{name}_{key}_{tag}.npy", matrix)
            _heatmap(matrix, out_dir / f"confusion_{name}_{key}.png", f"{name} {key}: seen pairs x {matrix.shape[1]} catalogue pairs", lock.sha256)
            files += [f"confusion_{name}_{key}_{tag}.npy", f"confusion_{name}_{key}.png"]
    summary = {
        "lock_sha256": lock.sha256, "lock_created": lock["created"], "settings": lock["settings"], "catalogue": lock["catalogue"],
        "models": {res["name"]: {k: v for k, v in res.items() if k not in ("confusions", "predictions", "name")} for res in results},
        "files": {f: file_sha256(out_dir / f) for f in files},
    }
    _write_json(out_dir / "final_metrics.json", summary)
    table = metrics_table(results, lock.sha256)
    table.to_csv(out_dir / "final_metrics.csv", index=False)
    return table, files + ["final_metrics.json", "final_metrics.csv"]


def already_finished(out_dir, lock_sha256):
    out_dir = Path(out_dir)
    if (out_dir / "final_done.json").exists():
        return True
    csv = out_dir / "final_metrics.csv"
    return csv.exists() and lock_sha256 in set(pd.read_csv(csv, usecols=["lock_sha256"]).lock_sha256)


def run_final_test(lock_path, out_dir, cfg, confirm=False, use_cache=True, device=None):
    """Verify the lock, evaluate every model once, write all files, then mark the run finished."""
    if not confirm:
        raise Refused("the final test needs the explicit confirm flag")
    lock = verify_lock(lock_path)
    out_dir = Path(out_dir)
    if already_finished(out_dir, lock.sha256):
        raise Refused(f"{out_dir} already holds a finished final test; it is run once per lock")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_model(cfg, name, lock, device, use_cache) for name in sorted(lock["models"])]
    table, files = write_outputs(out_dir, lock, results)
    _write_json(out_dir / "final_done.json", {
        "lock_sha256": lock.sha256, "finished": _now(), "models": sorted(lock["models"]),
        "files": {f: file_sha256(out_dir / f) for f in files},
    })
    return table
