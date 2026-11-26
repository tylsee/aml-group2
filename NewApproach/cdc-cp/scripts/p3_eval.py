"""P3 prototype scoring: classifier head against crop-pooled and crop-balanced prototypes, with shortcut metrics.

For one epoch of one run: load epoch_NN.pt, preserve frozen-backbone statistics or recompute BatchNorm on train_fit, extract train_fit, val_seen and
val_unseen, build prototype banks from train_fit, and score development validation with each scorer:
  head      log P(crop) + log P(disease)
  proto-X   cosine to the crop bank + cosine to the disease bank, for each form X (pooled, balanced)
Per scorer: raw (gamma 0) A_S / A_U / H, keep-seen (largest A_U with A_S within --tol points of raw A_S), best-H, AUC,
and unseen shortcut metrics at gamma 0. Official test files are never opened.

Writes RUN_DIR/p3_board_epNN.md, p3_board_epNN.csv and p3_extract_epNN.npz.

Usage: python scripts/p3_eval.py RUN_DIR --epoch N [--forms pooled balanced] [--tol 3] [--no-cache] [key=value ...]
key=value overrides paths and loader settings for this machine, e.g. paths.cache_dir=/tmp/cache loader.num_workers=2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.bn import recompute_bn  # noqa: E402
from cdccp.catalogue import load_catalogue  # noqa: E402
from cdccp.checkpoint import load_checkpoint  # noqa: E402
from cdccp.config import load_config  # noqa: E402
from cdccp.data import eval_loader, load_split  # noqa: E402
from cdccp.evaluate import check_rows, curve_auc, gamma_grid, shortcut_metrics, sweep  # noqa: E402
from cdccp.extract import extract  # noqa: E402
from cdccp.models import build_model  # noqa: E402
from cdccp.prototypes import FORMS, build_banks  # noqa: E402
from cdccp.scoring import head_score, proto_cosines, proto_score  # noqa: E402


def local_cfg(meta_cfg, overrides):
    """The run's own config, with this machine's paths and loader settings."""
    here = load_config(overrides=overrides)
    return {**meta_cfg, "paths": here["paths"], "loader": here["loader"]}


def score_row(name, ss, su, du, seen, unseen, cat, tol):
    """du: this scorer's disease scores on the unseen split, for its shortcut metrics."""
    rows = sweep(ss, seen, su, unseen, cat, gamma_grid(ss, su, cat))
    raw = rows[0]
    keep = rows[rows[:, 1] >= raw[1] - tol]
    k = keep[np.lexsort((-keep[:, 1], -keep[:, 2]))[0]]
    b = rows[np.lexsort((-rows[:, 1], -rows[:, 3]))[0]]
    short = shortcut_metrics(su, du, unseen, cat)
    return {
        "scorer": name,
        "A_S raw": raw[1], "A_U raw": raw[2], "H raw": raw[3],
        "gamma keep": k[0], "A_S keep": k[1], "A_U keep": k[2], "H keep": k[3],
        "gamma bestH": b[0], "A_S bestH": b[1], "A_U bestH": b[2], "H bestH": b[3],
        "AUC": curve_auc(rows),
        "unseen disease": 100 * float((np.argmax(du, axis=1) == unseen["disease_id"]).mean()),
        "unseen shortcut rate": short["shortcut_rate"],
        "shortcut chance": short["shortcut_chance"],
        "unseen oracle disease": short["oracle_disease_top1"],
    }


def board_text(table, run_dir, epoch, tol, forms):
    f = lambda v: f"{v:.1f}"
    lines = [f"# P3 prototype board: {run_dir.name}, epoch {epoch}", "",
             f"Development validation only. Banks come from all of train_fit ({', '.join(forms)}). "
             f"Keep-seen = best unseen with seen within {tol:g} points of raw. Gamma is tuned on the images it is scored on, so compare rows, "
             "do not read these as test estimates. Shortcut rate = share of wrong unseen disease guesses that form a seen pair with the true crop "
             "(chance = the share if the wrong guess were uniform).", "",
             "| Scorer | Unseen disease Top-1 | Raw S / U / H | Keep-seen S / U / H (gamma) | Best-H S / U / H | AUC | Unseen shortcut rate (chance) | Unseen oracle disease |",
             "|---|---|---|---|---|---|---|---|"]
    for _, d in table.iterrows():
        lines.append(f"| {d['scorer']} | {f(d['unseen disease'])} | {f(d['A_S raw'])} / {f(d['A_U raw'])} / {f(d['H raw'])} | "
                     f"{f(d['A_S keep'])} / {f(d['A_U keep'])} / **{f(d['H keep'])}** ({d['gamma keep']:.2f}) | "
                     f"{f(d['A_S bestH'])} / {f(d['A_U bestH'])} / {f(d['H bestH'])} | {d['AUC']:.3f} | "
                     f"{f(d['unseen shortcut rate'])} ({f(d['shortcut chance'])}) | {f(d['unseen oracle disease'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--forms", nargs="+", default=list(FORMS), choices=FORMS)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_intermixed_args()
    run_dir, epoch, use_cache = args.run_dir, args.epoch, not args.no_cache

    ckpt = run_dir / f"epoch_{epoch:02d}.pt"
    meta = load_checkpoint(ckpt)
    cfg = local_cfg(meta["config"], args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)

    train_fit = load_split(cfg, "train_fit")
    val_seen, val_unseen = load_split(cfg, "val_seen"), load_split(cfg, "val_unseen")
    check_rows(val_seen, "seen", meta)
    check_rows(val_unseen, "unseen", meta)
    cat = load_catalogue(cfg["paths"]["catalogue_dev"])

    recompute_bn(model, eval_loader(train_fit, cfg, cfg["bn"]["batch"], use_cache=use_cache), device)
    train = extract(model, train_fit, cfg, device, use_cache)
    seen = extract(model, val_seen, cfg, device, use_cache)
    unseen = extract(model, val_unseen, cfg, device, use_cache)

    n_crop, n_disease = cfg["model"]["crop_classes"], cfg["model"]["disease_classes"]
    scorers = {"head": (head_score(seen, cat), head_score(unseen, cat), unseen["disease_logits"])}
    for form in args.forms:
        banks = build_banks(train, form, n_crop, n_disease)
        scorers[f"proto-{form}"] = (proto_score(seen, banks, cat), proto_score(unseen, banks, cat), proto_cosines(unseen, banks)[1])

    table = pd.DataFrame([score_row(name, ss, su, du, seen, unseen, cat, args.tol) for name, (ss, su, du) in scorers.items()])
    table.to_csv(run_dir / f"p3_board_ep{epoch:02d}.csv", index=False)
    text = board_text(table, run_dir, epoch, args.tol, args.forms)
    (run_dir / f"p3_board_ep{epoch:02d}.md").write_text(text + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / f"p3_extract_ep{epoch:02d}.npz",
                        **{f"{name}_{k}": v for name, z in (("train", train), ("seen", seen), ("unseen", unseen)) for k, v in z.items() if k != "h"})
    print(text)


if __name__ == "__main__":
    main()
