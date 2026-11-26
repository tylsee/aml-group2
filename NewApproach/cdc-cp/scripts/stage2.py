"""Usage: python scripts/stage2.py PARENT_PT OUT_DIR [key=value ...] [--no-cache]

Continues the parent checkpoint on all official training images (Stage 2) and prints the gate result.
key=value sets stage2.epochs, stage2.lr_backbone, stage2.lr_heads, stage2.warmup_epochs, stage2.train_split,
or this machine's paths and loader settings, e.g. paths.cache_dir=/tmp/cache loader.num_workers=2.
The exit code is 0 even when the gate fails: read OUT_DIR/stage2_gate.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.stage2 import run_stage2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("out_dir")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_intermixed_args()
    out_dir, sha = run_stage2(args.parent, args.out_dir, args.overrides, use_cache=not args.no_cache)
    gate = json.loads((out_dir / "stage2_gate.json").read_text(encoding="utf-8"))
    print(f"done: {out_dir / 'stage2_final.pt'} sha256={sha}")
    print(f"gate: {'PASS' if gate['ok'] else 'FAIL'} ({gate['reason']})")


if __name__ == "__main__":
    main()
