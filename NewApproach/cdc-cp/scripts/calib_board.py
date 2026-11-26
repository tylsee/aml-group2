"""Calibrated scoreboard: rank finished runs on development validation, with and without the seen-pair bias.

For each run, each of its last 3 epochs is scored with S(c,d) = log P(c) + r * log P(d) - gamma * [pair seen],
and the numbers are averaged over those epochs. Columns:
  raw       gamma = 0 (the plain row every model reports)
  keep-seen the largest unseen accuracy whose seen accuracy stays within --tol points of raw (default 3).
            This is the ranking column: Dr. Lee asked for unseen gains that do not cost much seen accuracy.
  best-H    the gamma with the highest harmonic mean, shown for the record (it usually costs more seen accuracy)
  AUC       area under the seen-vs-unseen curve over the whole gamma sweep (0-1 scale)
Gamma is picked on the same validation images it is scored on, so these numbers compare configurations;
they do not estimate the official test.

Runs sharing a name apart from the _seedNN suffix are also averaged into one config row.
Writes calib_board.md, calib_board.csv and gamma_sensitivity.csv (the calibration ablation) into RUNS_DIR.

Usage: python scripts/calib_board.py RUNS_DIR [--match GLOB ...] [--tol 3] [--r 1.0]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.catalogue import load_catalogue  # noqa: E402
from cdccp.config import load_config  # noqa: E402
from cdccp.evaluate import curve_auc, gamma_grid, harmonic, sweep  # noqa: E402
from cdccp.scoring import head_score  # noqa: E402

SENS_GAMMAS = (0, 2, 4, 6, 8, 10, 12, 14, 16)
SENS_R = (0.5, 1.0, 2.0)


def finished(run_dir):
    """Epoch numbers of a finished run, or None if it stopped early or has no saved validation outputs."""
    hist = run_dir / "history.csv"
    if not (hist.is_file() and (run_dir / "last.pt").is_file()):
        return None
    epochs = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["recipe"]["epochs"]
    done = list(pd.read_csv(hist).epoch)
    if done != list(range(1, epochs + 1)) or not all((run_dir / f"val_{e:02d}.npz").is_file() for e in done[-3:]):
        return None
    return done


def split(z, prefix):
    return {k[len(prefix) + 1:]: z[k] for k in z.files if k.startswith(prefix + "_")}


def top1(logits, y):
    return 100 * (logits.argmax(1) == y).mean()


def score_epoch(path, cat, r, tol):
    z = np.load(path)
    s, u = split(z, "seen"), split(z, "unseen")
    ss, su = head_score(s, cat, r=r), head_score(u, cat, r=r)
    rows = sweep(ss, s, su, u, cat, gamma_grid(ss, su, cat))
    raw = rows[0]
    keep = rows[rows[:, 1] >= raw[1] - tol]
    k = keep[np.lexsort((-keep[:, 1], -keep[:, 2]))[0]]
    b = rows[np.lexsort((-rows[:, 1], -rows[:, 3]))[0]]
    sens = [(rr, g, *sweep(head_score(s, cat, r=rr), s, head_score(u, cat, r=rr), u, cat, [g])[0, 1:3])
            for rr in SENS_R for g in SENS_GAMMAS]
    out = {
        "A_S raw": raw[1], "A_U raw": raw[2], "H raw": raw[3],
        "gamma keep": k[0], "A_S keep": k[1], "A_U keep": k[2], "H keep": k[3],
        "gamma bestH": b[0], "A_S bestH": b[1], "A_U bestH": b[2], "H bestH": b[3],
        "AUC": curve_auc(rows),
        "unseen crop": top1(u["crop_logits"], u["crop_id"]),
        "unseen disease": top1(u["disease_logits"], u["disease_id"]),
        "seen disease": top1(s["disease_logits"], s["disease_id"]),
    }
    return out, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--match", nargs="+", default=["*"], help="glob patterns for run folder names")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--r", type=float, default=1.0)
    args = ap.parse_args()
    cat = load_catalogue(load_config()["paths"]["catalogue_dev"])

    dirs = sorted({d for pat in args.match for d in args.runs_dir.glob(pat) if d.is_dir()})
    per_run, sens_rows, skipped = [], [], []
    for d in dirs:
        done = finished(d)
        if done is None:
            skipped.append(d.name)
            continue
        scored = [score_epoch(d / f"val_{e:02d}.npz", cat, args.r, args.tol) for e in done[-3:]]
        row = pd.DataFrame([m for m, _ in scored]).mean().to_dict()
        recipe = torch.load(d / "last.pt", map_location="cpu", weights_only=False)["recipe"]
        model = recipe.get("model", {}) if isinstance(recipe, dict) else {}
        per_run.append({"run": d.name, "config": re.sub(r"_seed\d+$", "", d.name),
                        "backbone": model.get("backbone", "efficientnet_b0"), "hidden": str(model.get("hidden", [256])),
                        "activation": model.get("activation", "relu"), **row})
        sens = pd.DataFrame([x for _, ss in scored for x in ss], columns=["r", "gamma", "A_S", "A_U"])
        sens = sens.groupby(["r", "gamma"], as_index=False).mean()
        sens.insert(0, "run", d.name)
        sens_rows.append(sens)
    if not per_run:
        raise SystemExit(f"no finished runs under {args.runs_dir} matching {args.match}; skipped {skipped}")

    runs = pd.DataFrame(per_run)
    num = [c for c in runs.columns if c not in ("run", "config", "backbone", "hidden", "activation")]
    cfg = runs.groupby("config").agg({**{c: "mean" for c in num}, "run": "count", "backbone": "first", "hidden": "first", "activation": "first"})
    cfg = cfg.rename(columns={"run": "seeds"})
    cfg["H keep spread"] = runs.groupby("config")["H keep"].agg(lambda x: x.max() - x.min())
    cfg = cfg.sort_values("H keep", ascending=False).reset_index()
    runs.to_csv(args.runs_dir / "calib_board.csv", index=False)
    pd.concat(sens_rows).to_csv(args.runs_dir / "gamma_sensitivity.csv", index=False)

    f = lambda v: f"{v:.1f}"
    lines = ["# Calibrated scoreboard (development validation, mean of each run's last 3 epochs)", "",
             f"Score: log P(crop) + {args.r} x log P(disease) - gamma x [seen]. 'keep-seen' = best unseen with seen within {args.tol:g} points of raw. "
             "Gamma is tuned on the images it is scored on: compare rows, do not read these as test estimates.", "",
             "| Rank | Config | Backbone | Head | Seeds | Raw S / U / H | Keep-seen S / U / H (gamma) | Best-H S / U / H | AUC | Unseen crop / disease |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in cfg.iterrows():
        spread = f" +/-{c['H keep spread'] / 2:.1f}" if c["seeds"] > 1 else ""
        lines.append(f"| {i + 1} | {c['config']} | {c['backbone']} | {c['hidden']} {c['activation']} | {c['seeds']} | "
                     f"{f(c['A_S raw'])} / {f(c['A_U raw'])} / {f(c['H raw'])} | "
                     f"{f(c['A_S keep'])} / {f(c['A_U keep'])} / **{f(c['H keep'])}**{spread} ({c['gamma keep']:.1f}) | "
                     f"{f(c['A_S bestH'])} / {f(c['A_U bestH'])} / {f(c['H bestH'])} | {c['AUC']:.3f} | "
                     f"{f(c['unseen crop'])} / {f(c['unseen disease'])} |")
    top = cfg.iloc[0]["config"]
    sens = pd.concat(sens_rows)
    sens = sens[sens.run.str.replace(r"_seed\d+$", "", regex=True) == top].groupby(["r", "gamma"], as_index=False).mean(numeric_only=True)
    lines += ["", f"## Calibration ablation for the top config ({top}): seen / unseen at each r and gamma", "",
              "| r \\ gamma | " + " | ".join(str(g) for g in SENS_GAMMAS) + " |", "|---|" + "---|" * len(SENS_GAMMAS)]
    for rr in SENS_R:
        cells = sens[sens.r == rr].set_index("gamma")
        lines.append(f"| {rr:g} | " + " | ".join(f"{cells.loc[g, 'A_S']:.0f} / {cells.loc[g, 'A_U']:.0f}" for g in SENS_GAMMAS) + " |")
    if skipped:
        lines += ["", f"Skipped (unfinished or no saved validation outputs): {', '.join(skipped)}"]
    text = "\n".join(lines)
    (args.runs_dir / "calib_board.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
