"""Usage: python scripts/train.py configs/p1.yaml [--resume] [--max-epochs N] [key=value ...]   e.g. seed=43"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import load_config
from cdccp.train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-epochs", type=int, help="stop after this many epochs (smoke tests); the LR schedule still assumes the full budget")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    out_dir, applied, skipped = run(cfg, use_cache=not args.no_cache, resume=args.resume, max_epochs=args.max_epochs)
    print(f"done: {out_dir} applied={applied} skipped={skipped}")


if __name__ == "__main__":
    main()
