"""Zip the code, configs, split files and catalogues for Colab. Leaves out runs/, caches and checkpoints.

Usage: python scripts/pack_for_colab.py [OUT.zip]     (default: ../cdc-cp_code.zip). Upload the zip to Google Drive.
"""
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INCLUDE = ("cdccp", "configs", "data", "scripts", "tests", "notebooks", "requirements.txt")
SKIP_PARTS = {"__pycache__", ".pytest_cache", "runs", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".pyc", ".tmp"}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "cdc-cp_code.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for top in INCLUDE:
            root = REPO / top
            for path in [root] if root.is_file() else sorted(root.rglob("*")):
                rel = path.relative_to(REPO)
                if path.is_dir() or SKIP_PARTS & set(rel.parts) or path.suffix in SKIP_SUFFIXES:
                    continue
                z.write(path, Path("cdc-cp") / rel)
                count += 1
    print(f"wrote {out} ({count} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
