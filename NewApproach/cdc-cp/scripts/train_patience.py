"""Usage: python scripts/train_patience.py configs/p1b_patch.yaml [key=value ...] [--patience 3] [--metric unseen_disease_top1] [--resume]

Options and key=value overrides may come in any order.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import load_config
from cdccp.patience import train_with_patience


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--metric", default="unseen_disease_top1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_intermixed_args(argv)


def main():
    args = parse()
    cfg = load_config(args.config, args.overrides)
    out_dir, best = train_with_patience(cfg, args.patience, args.metric, use_cache=not args.no_cache, resume=args.resume)
    print(f"done: {out_dir} best epoch {best}")


if __name__ == "__main__":
    main()
