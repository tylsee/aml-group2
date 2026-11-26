"""Early stopping around the unchanged trainer: train one epoch at a time and stop after `patience` epochs without a new best.

The LR schedule still assumes the configured number of epochs. Each epoch is a resume, and the trainer reshuffles the
data on every resume, so a stopped run is not bit-identical to the start of one uninterrupted run. Compare runs made
by this wrapper only with other runs made by this wrapper.
"""
import json
import math
from pathlib import Path

import pandas as pd

from . import train


def should_stop(scores, patience):
    """True when the best score is `patience` or more epochs old. Higher is better; ties do not count as a new best."""
    scores = [float(s) for s in scores]
    if any(math.isnan(s) for s in scores):
        raise ValueError(f"NaN in the early-stopping metric: {scores}")
    best = max(range(len(scores)), key=lambda i: (scores[i], -i)) if scores else 0
    return len(scores) - 1 - best >= patience


def train_with_patience(cfg, patience=3, metric="unseen_disease_top1", device=None, use_cache=True, resume=False):
    """Returns (run folder, best epoch). Writes early_stop.json when it stops or reaches the last epoch.

    `metric` is a history.csv column where higher is better. patience_start.json (metric, patience, recipe) is written
    before the first epoch, so a resume refuses a run folder that the plain trainer or other settings made.
    """
    if cfg.get("memory", {}).get("per_pair", 0):
        raise ValueError("the pair memory restarts empty on every resume, so per-epoch resumes would change a memory run")
    out_dir = Path(cfg["paths"]["runs_dir"]) / f"{cfg['name']}_seed{cfg['seed']}"
    marker = out_dir / "early_stop.json"
    recipe = train._recipe(cfg)
    if resume and marker.exists():
        done = json.loads(marker.read_text(encoding="utf-8"))
        if (done["metric"], done["patience"], done["recipe"]) != (metric, patience, recipe):
            raise ValueError(f"{marker} was written by another metric, patience or config; refusing to mix runs")
        print(f"nothing to do: {out_dir} stopped after epoch {done['stopped_after_epoch']}", flush=True)
        return out_dir, done["best_epoch"]
    history_path = out_dir / "history.csv"
    start = out_dir / "patience_start.json"
    identity = json.loads(json.dumps({"metric": metric, "patience": patience, "recipe": recipe}))
    first = 1
    occupied = history_path.exists() or (out_dir / "last.pt").exists()
    if occupied and not resume:
        raise FileExistsError(f"{out_dir} already holds a run. Pass --resume to continue it, or use another name or seed.")
    if occupied:
        if not start.exists():
            raise ValueError(f"{out_dir} holds epochs that train_with_patience did not start; refusing to mix runs")
        if json.loads(start.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"{start} was written by another metric, patience or config; refusing to mix runs")
        first = max(1, len(pd.read_csv(history_path)) if history_path.exists() else 0)
    if first > cfg["epochs"]:
        raise ValueError(f"{history_path} has {first} epochs, more than the planned {cfg['epochs']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    start.write_text(json.dumps(identity), encoding="utf-8")
    history = None
    for target in range(first, cfg["epochs"] + 1):
        out_dir, _, _ = train.run(cfg, device=device, use_cache=use_cache, max_epochs=target, resume=resume or target > 1)
        history = pd.read_csv(history_path)
        if should_stop(history[metric], patience):
            break
    scores = list(history[metric])
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    record = {"metric": metric, "patience": patience, "best_epoch": int(history.epoch.iloc[best]), "best": float(scores[best]),
              "stopped_after_epoch": int(history.epoch.iloc[-1]), "planned_epochs": cfg["epochs"], "recipe": recipe}
    marker.write_text(json.dumps(record), encoding="utf-8")
    print(f"early stop: best {metric} {record['best']:.2f} at epoch {record['best_epoch']}, stopped after epoch {record['stopped_after_epoch']}", flush=True)
    return out_dir, record["best_epoch"]
