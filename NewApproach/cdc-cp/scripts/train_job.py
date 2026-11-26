import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.single_task import ARMS, run_stage1, run_stage2


def main(argv=None):
    p = argparse.ArgumentParser(description="Single-task training job (stage 1 or stage 2)")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("stage1", "stage2"):
        s = sub.add_parser(name)
        s.add_argument("--arm", required=True, choices=sorted(ARMS))
        s.add_argument("--images", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--epochs", type=int)
        s.add_argument("--workers", type=int, default=2)
        s.add_argument("--device", default="cuda")
        s.add_argument("--max-train-batches", type=int)
        s.add_argument("--no-amp", action="store_true", help="train and evaluate in fp32 (no mixed precision)")
    s1, s2 = sub.choices["stage1"], sub.choices["stage2"]
    s1.add_argument("--splits", required=True)
    s1.add_argument("--no-pretrained", action="store_true")
    s1.add_argument("--patience", type=int, default=None, help="stop stage 1 after this many epochs without a better dev HM")
    s2.add_argument("--stage1", required=True)
    s2.add_argument("--train-csv", required=True)
    a = p.parse_args(argv)
    if a.command == "stage1":
        run_stage1(a.arm, a.splits, a.images, a.out, a.device, 30 if a.epochs is None else a.epochs, a.workers,
                   not a.no_pretrained, a.max_train_batches, amp=not a.no_amp,
                   patience=a.patience)
    else:
        run_stage2(a.arm, a.stage1, a.train_csv, a.images, a.out, a.device, 10 if a.epochs is None else a.epochs, a.workers,
                   a.max_train_batches, amp=not a.no_amp)


if __name__ == "__main__":
    main()
