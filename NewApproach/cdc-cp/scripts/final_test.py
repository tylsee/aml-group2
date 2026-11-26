"""The final test: official seen and unseen test sets, run once on the models in a lock.

Usage: python scripts/final_test.py LOCK_JSON OUT_DIR --confirm-final-test [--no-cache] [key=value ...]
key=value overrides paths and loader settings for this machine, e.g. paths.data_root=/kaggle/input/dpd loader.num_workers=2

Writes final_metrics.csv/json, predictions_NAME.csv, confusion_NAME_*.npy/png, then final_done.json. Prints the summary last.
A second run for the same OUT_DIR is refused once final_done.json or final_metrics.csv exists.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import load_config  # noqa: E402
from cdccp.final import run_final_test  # noqa: E402


def print_summary(table):
    wide = table[table.split == "combined"].pivot_table(index=["model", "catalogue", "gamma"], columns="metric", values="value")
    print(wide[["A_S", "A_U", "H", "gap"]].round(2).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lock", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--confirm-final-test", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_intermixed_args(argv)
    if not args.confirm_final_test:
        ap.error("the final test runs once; pass --confirm-final-test to run it")
    table = run_final_test(args.lock, args.out_dir, load_config(overrides=args.overrides), confirm=True, use_cache=not args.no_cache)
    print_summary(table)


if __name__ == "__main__":
    main()
