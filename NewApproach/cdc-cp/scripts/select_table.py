"""Write selection.md for a run. A person reads it and signs the pick. Usage: select_table.py runs/p1_seed42"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.selection import moving_average, pick_epoch


def main():
    run_dir = Path(sys.argv[1])
    history = pd.read_csv(run_dir / "history.csv")
    average = moving_average(history.H)
    picked = pick_epoch(history.H)
    lines = [
        f"# Selection: {run_dir.name}", "",
        "Rule: highest 3-epoch moving average of composition Top-1 H, at r = 1 and gamma = 0.", "",
        "| Epoch | A_S | A_U | H | H average | Val CE crop | Val CE disease | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in history.iterrows():
        mark = "rule pick" if i == picked else ""
        lines.append(f"| {int(row.epoch)} | {row.A_S:.2f} | {row.A_U:.2f} | {row.H:.2f} | {average[i]:.2f} | {row.val_ce_crop:.3f} | {row.val_ce_disease:.3f} | {mark} |")
    if history.A_U.max() < 5:
        lines += ["", "A_U is under 5% in every epoch, so H is degenerate: use swept-gamma H and say so."]
    lines += ["", "Signed by: ____________   Epoch picked: ____", "", "If this differs from the rule pick, the reason is: ____________"]
    (run_dir / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rule pick: epoch {int(history.epoch[picked])}; wrote {run_dir / 'selection.md'}")


if __name__ == "__main__":
    main()
