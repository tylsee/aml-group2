"""Write the development (K=323) and final (K=333) catalogues. Official test metadata supplies pair names only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.catalogue import build_catalogue
from cdccp.checkpoint import file_sha256
from cdccp.config import load_config
from cdccp.data import load_meta, load_split, official_csv


def main():
    cfg = load_config()
    train_csv, unseen_csv = official_csv(cfg, "train"), official_csv(cfg, "test_unseen")
    fit_csv = Path(cfg["paths"]["splits_dir"]) / "train_fit.csv"
    val_csv = Path(cfg["paths"]["splits_dir"]) / "val_unseen.csv"
    dev = build_catalogue(load_split(cfg, "train_fit"), load_split(cfg, "val_unseen"), {"seen": file_sha256(fit_csv), "unseen": file_sha256(val_csv)})
    final = build_catalogue(load_split(cfg, "train"), load_meta(unseen_csv), {"seen": file_sha256(train_csv), "unseen": file_sha256(unseen_csv)})
    for key, catalogue in (("catalogue_dev", dev), ("catalogue_final", final)):
        path = Path(cfg["paths"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalogue, indent=1), encoding="utf-8")
        seen = sum(p["is_seen"] for p in catalogue["pairs"])
        print(f"{path.name}: K={len(catalogue['pairs'])} seen={seen} unseen={len(catalogue['pairs']) - seen}")


if __name__ == "__main__":
    main()
