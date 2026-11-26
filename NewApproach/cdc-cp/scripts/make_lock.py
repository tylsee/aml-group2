"""Write lock.json for the final test. Reads checkpoints, dev boards and the catalogue. Never reads official test CSVs.

Usage: python scripts/make_lock.py SPEC LOCK_JSON [key=value ...]

SPEC is JSON or YAML:
  settings: {tol: 3.0}                 # optional, recorded in the lock
  models:
    - name: stage2_head
      checkpoint: stage2_head/stage2_final.pt     # must sit inside the folder of LOCK_JSON
      kind: head                                   # head, proto-pooled or proto-balanced
      boards: [runs/a/p3_board_ep10.csv, runs/b/p3_board_ep10.csv]   # one dev board per seed
Relative paths in SPEC resolve against the SPEC file's folder. The catalogue (paths.catalogue_final) is copied next to LOCK_JSON as
catalogue.json. Gamma is the mean `gamma keep` of the scorer (head, proto-pooled, proto-balanced) over the boards.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import load_config  # noqa: E402
from cdccp.final import gammas_from_boards, make_lock  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("lock", type=Path)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    here = args.spec.resolve().parent
    where = lambda p: Path(p) if Path(p).is_absolute() else here / p
    cfg = load_config(overrides=args.overrides)
    models, gammas = [], {}
    for item in spec["models"]:
        models.append({"name": item["name"], "checkpoint": where(item["checkpoint"]), "kind": item["kind"]})
        gammas[item["name"]] = gammas_from_boards([where(b) for b in item["boards"]], item["kind"])
    lock = make_lock(models, gammas, cfg["paths"]["catalogue_final"], args.lock, spec.get("settings"))
    for name, entry in lock["models"].items():
        print(f"{name}: {entry['kind']} gamma={entry['gamma']:.4f} sha256={entry['sha256'][:12]}")
    print(f"lock written to {args.lock}")


if __name__ == "__main__":
    main()
