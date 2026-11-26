"""Apply the written P1 rules to finished runs and write p1_decision.md. Development data only; no test files are read.

Usage:
  python scripts/p1_decide.py --e1 RUNS/p1_lr1e-5_seed42 --e2 RUNS/p1_lr1e-4_seed42 [--e3 RUNS/p1_lr..._seed43] [--out FILE] [key=value ...]

Rules (plan section 4, with the 12 Nov run order E1 -> E2 -> E3):
  * Each run's epoch: highest 3-epoch moving average of development H (head score, gamma 0, dev catalogue).
  * delta: seed noise from E3 and the seed-42 run at the same LR; 3 points a priori if E3 has not run.
  * Gate E2 (1e-4) against E1 (1e-5) at their picked epochs: (a) delta H >= delta, (b) 10th percentile of the
    pair bootstrap > 0, (c) no subgroup's unseen accuracy falls by more than delta. LR* = 1e-4 only if all hold.
  * Budget check on the LR* runs: late gain >= delta on every seed means underfit, so double U once.
A person reads the file and signs it. The script never changes a config.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.catalogue import load_catalogue
from cdccp.config import load_config
from cdccp.data import load_split
from cdccp.evaluate import topk_hits
from cdccp.gate import accept, subgroup_masks
from cdccp.scoring import head_score
from cdccp.selection import budget_check, late_gain, moving_average, pick_epoch, seed_delta


def load_run(run_dir):
    run_dir = Path(run_dir)
    history = pd.read_csv(run_dir / "history.csv")
    picked = pick_epoch(history.H)
    return {
        "dir": run_dir, "name": run_dir.name, "lr_name": run_dir.name.rsplit("_seed", 1)[0],
        "history": history, "H": history.H.to_numpy(), "picked_index": picked,
        "epoch": int(history.epoch.iloc[picked]), "peak_avg": float(moving_average(history.H)[picked]),
    }


def gate_record(run, cat, val_seen, val_unseen, train_fit):
    """Per-image Top-1 hits at the run's picked epoch, in the shape gate.accept expects."""
    z = np.load(run["dir"] / f"val_{run['epoch']:02d}.npz")
    out = {}
    for split, df in (("seen", val_seen), ("unseen", val_unseen)):
        npz = {k[len(split) + 1:]: z[k] for k in z.files if k.startswith(split + "_")}
        if list(npz["fn"]) != list(df.filename):
            raise ValueError(f"{run['name']}: {split} rows are not in split-file order")
        out[f"{split}_hits"] = topk_hits(head_score(npz, cat), cat.index_of(npz["comp_id"]), 1)
        out[f"{split}_pair"] = cat.index_of(npz["comp_id"])
    out["subgroups"] = subgroup_masks(val_unseen, train_fit)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", required=True)
    parser.add_argument("--e2", required=True)
    parser.add_argument("--e3")
    parser.add_argument("--out")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(overrides=args.overrides)
    cat = load_catalogue(cfg["paths"]["catalogue_dev"])
    train_fit, val_seen, val_unseen = (load_split(cfg, n) for n in ("train_fit", "val_seen", "val_unseen"))

    e1, e2 = load_run(args.e1), load_run(args.e2)
    e3 = load_run(args.e3) if args.e3 else None
    twin = None
    if e3:
        twin = next((r for r in (e1, e2) if r["lr_name"] == e3["lr_name"]), None)
        if twin is None:
            raise ValueError(f"E3 ({e3['name']}) must repeat E1 or E2 with another seed")
    delta = seed_delta(twin["H"], e3["H"]) if e3 else 3.0

    verdict = accept(gate_record(e2, cat, val_seen, val_unseen, train_fit), gate_record(e1, cat, val_seen, val_unseen, train_fit), delta)
    winner = e2 if verdict["accept"] else e1
    seeds = [winner] + ([e3] if e3 and e3["lr_name"] == winner["lr_name"] else [])
    underfit = budget_check(seeds[0]["H"], seeds[1]["H"] if len(seeds) > 1 else None)

    lines = [
        "# P1 decision (development data only)", "",
        "| Run | Picked epoch | Peak 3-epoch average H | A_S | A_U | H | Late gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in filter(None, (e1, e2, e3)):
        row = r["history"].iloc[r["picked_index"]]
        lines.append(f"| {r['name']} | {r['epoch']} | {r['peak_avg']:.2f} | {row.A_S:.2f} | {row.A_U:.2f} | {row.H:.2f} | {late_gain(r['H']):+.2f} |")
    falls = ", ".join(f"{k} {v:+.2f}" for k, v in verdict["subgroup_falls"].items())
    lines += [
        "",
        f"delta = {delta:.2f} points ({'seed noise from ' + twin['name'] + ' and ' + e3['name'] if e3 else 'a-priori value, E3 not run'})",
        "",
        f"Gate, {e2['name']} against {e1['name']}: delta H {verdict['delta_H']:+.2f}, bootstrap 10th percentile {verdict['p10']:+.2f}, "
        f"unseen falls by subgroup: {falls}",
        f"Clauses: (a) {verdict['clauses']['a']}, (b) {verdict['clauses']['b']}, (c) {verdict['clauses']['c']} -> "
        f"{'ACCEPT' if verdict['accept'] else 'REJECT'}",
        "",
        f"**LR\\* = {'1e-4' if verdict['accept'] else '1e-5'}** (from {winner['name']}, epoch {winner['epoch']}). "
        f"This LR applies to P2 and should be offered to the B2 owner.",
        f"Budget check on {', '.join(r['name'] for r in seeds)}: "
        f"{'UNDERFIT -> double U once (40 epochs) and rerun P1 at LR*' if underfit else 'not underfit -> keep 20 epochs'}",
    ]
    if not e3:
        better = max((e1, e2), key=lambda r: r["peak_avg"])
        lines += ["", f"E3 not run yet. Next: rerun {better['lr_name']} with seed=43 (higher peak average H)."]
    if max(r["history"].A_U.max() for r in (e1, e2)) < 5:
        lines += ["", "A_U stays under 5% in both runs, so H at gamma 0 is degenerate. Say so in the report."]
    lines += ["", "Signed by: ____________   Date: ____________"]

    out = Path(args.out) if args.out else Path(args.e1).parent / "p1_decision.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
