"""P1 round 2: compare variants against the p1b_base control on the steadier scoreboard.

Unseen pair accuracy rests on 5-13 correct images out of 400, so it is reported but not used to decide.
The decision uses seen pair accuracy (A_S) and unseen disease Top-1, each averaged over a run's last 3 epochs.

A variant is kept when, for both metrics:
  - each of its seeds is above the control's two-seed mean, and
  - its two-seed mean beats the control's by at least the control's seed difference (minimum 0.5 points).

Usage (each argument is a runs folder holding <name>_seed42 and <name>_seed43):
  python scripts/p1_compare.py RUNS_DIR [--control p1b_base] [--variants p1b_aug p1b_ls] [--out FILE]
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

METRICS = {"A_S": "seen pairs", "unseen_disease_top1": "unseen disease"}
REPORT = {**METRICS, "unseen_crop_top1": "unseen crop", "A_U": "unseen pairs", "H": "H"}
SEEDS = (42, 43)


def load(run_dir):
    """Last-3-epoch means of a finished run. Refuses a run that stopped early."""
    history = pd.read_csv(run_dir / "history.csv")
    epochs = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["recipe"]["epochs"]
    if list(history.epoch) != list(range(1, epochs + 1)):
        raise SystemExit(f"{run_dir.name} is unfinished: has epochs {list(history.epoch)}, expected 1..{epochs}")
    return history.tail(3)[list(REPORT)].mean()


def runs_of(root, name):
    found = {s: root / f"{name}_seed{s}" for s in SEEDS}
    missing = [p.name for p in found.values() if not (p / "history.csv").is_file()]
    return found, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--control", default="p1b_base")
    ap.add_argument("--variants", nargs="+", default=["p1b_aug", "p1b_ls"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    control, missing = runs_of(args.runs_dir, args.control)
    if missing:
        raise SystemExit(f"control runs missing: {missing}")
    c = {s: load(p) for s, p in control.items()}
    c_mean = pd.concat(c.values(), axis=1).mean(axis=1)
    c_noise = {m: max(0.5, abs(c[42][m] - c[43][m])) for m in METRICS}

    lines = ["# P1 round 2 comparison (development data only, mean of each run's last 3 epochs)", "",
             "| Run | " + " | ".join(REPORT.values()) + " |", "|---|" + "---|" * len(REPORT)]
    row = lambda label, v: lines.append(f"| {label} | " + " | ".join(f"{v[m]:.2f}" for m in REPORT) + " |")
    for s in SEEDS:
        row(f"{args.control} seed {s}", c[s])
    row(f"**{args.control} mean**", c_mean)

    verdicts = []
    for name in args.variants:
        runs, missing = runs_of(args.runs_dir, name)
        if missing:
            verdicts.append(f"- {name}: not compared, missing {missing}")
            continue
        v = {s: load(p) for s, p in runs.items()}
        v_mean = pd.concat(v.values(), axis=1).mean(axis=1)
        for s in SEEDS:
            row(f"{name} seed {s}", v[s])
        row(f"**{name} mean**", v_mean)
        checks = []
        for m, label in METRICS.items():
            gain = v_mean[m] - c_mean[m]
            ok = gain >= c_noise[m] and all(v[s][m] > c_mean[m] for s in SEEDS)
            checks.append(ok)
            verdicts.append(f"- {name}, {label}: {gain:+.2f} points (needs >= {c_noise[m]:.2f} and both seeds above control mean) -> {'pass' if ok else 'fail'}")
        verdicts.append(f"- **{name}: {'KEEP' if all(checks) else 'DROP'}**")

    lines += ["", "## Verdicts", "", *verdicts, "",
              "Unseen pair accuracy and H are shown for the record only; they rest on too few correct images to decide on.", "",
              "Signed by: ____________   Date: ____________"]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
