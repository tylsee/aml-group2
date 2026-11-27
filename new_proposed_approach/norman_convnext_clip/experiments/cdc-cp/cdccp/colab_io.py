"""Copy files between Google Drive and the Colab disk. Training reads images only from the local disk (Drive is slow)."""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

TEMP_SUFFIXES = (".tmp", ".part")


def _same_size(src, dst):
    return dst.is_file() and dst.stat().st_size == src.stat().st_size


def _copy_once_or_retry(copy, src, dst, retries, wait, printer):
    """Copy src to dst through a .part file and a rename, so dst never holds half a file. Drive mounts give
    occasional I/O errors, so an OSError is retried `retries` times; the last error is raised."""
    part = dst.with_name(dst.name + ".part")
    for attempt in range(retries + 1):
        try:
            copy(src, part)
            os.replace(part, dst)
            return
        except OSError as exc:
            if attempt == retries:
                raise
            printer(f"retry {attempt + 1}/{retries} for {src}: {exc}")
            time.sleep(wait)


def copy_tree(src, dst, threads=32, retries=3, wait=5.0, printer=print):
    """Copy every file under src to dst with several threads. Files already in dst with the same size are kept.

    After the copy, every file in dst must have the size of its source, or this raises. Returns the number of files in src.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a folder")
    files = [p for p in src.rglob("*") if p.is_file()]
    todo = [p for p in files if not _same_size(p, dst / p.relative_to(src))]

    def one(path):
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_once_or_retry(shutil.copyfile, path, target, retries, wait, printer)

    started = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in pool.map(one, todo):
            pass
    wrong = [p for p in files if not _same_size(p, dst / p.relative_to(src))]
    if wrong:
        raise OSError(f"{len(wrong)} copied files differ in size from the source, e.g. {wrong[:3]}")
    printer(f"copied {len(todo)} of {len(files)} files from {src} to {dst} in {time.time() - started:.0f} s (sizes checked)")
    return len(files)


def copy_file(src, dst, retries=3, wait=5.0, printer=print):
    """Copy one file (with its time) through a .part file and a rename, retrying passing Drive I/O errors."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_once_or_retry(shutil.copy2, src, dst, retries, wait, printer)


def check_images(folder, threads=16):
    """Decode every file in folder as an image (all compressed data is read, so a cut-off file fails here).

    JPEG files are decoded at 1/8 scale (draft mode): every byte is still read, but it is much faster than full size.
    Raises with the bad file names. Returns the number checked.
    """
    files = sorted(p for p in Path(folder).iterdir() if p.is_file())

    def bad(path):
        try:
            with Image.open(path) as image:
                image.draft(image.mode, (max(1, image.width // 8), max(1, image.height // 8)))
                image.load()
            return None
        except Exception as exc:   # PIL raises many types for a damaged file; each one is reported below, not hidden
            return f"{path.name}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=threads) as pool:
        problems = [p for p in pool.map(bad, files) if p]
    if problems:
        raise OSError(f"{len(problems)} image files do not open, e.g. {problems[:3]}")
    return len(files)


def save_results(out_dir, drive_dir, model_folders=(), retries=3, wait=5.0, printer=print):
    """Copy the run folder to Drive: every file except model checkpoints (*.pt) and temporary files (*.tmp, *.part).

    Checkpoints are copied only when their path starts with a folder named in model_folders (for example 'stage2').
    Files already on Drive with the same size and time are skipped. Returns the number of files copied.
    """
    out_dir, drive_dir = Path(out_dir), Path(drive_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"{out_dir} is not a folder")
    copied = 0
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(out_dir)
        if path.name.endswith(TEMP_SUFFIXES):
            continue
        if path.suffix == ".pt" and rel.parts[0] not in model_folders:
            continue
        target = drive_dir / rel
        if target.is_file() and target.stat().st_size == path.stat().st_size and target.stat().st_mtime >= path.stat().st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_once_or_retry(shutil.copy2, path, target, retries, wait, printer)
        copied += 1
    printer(f"saved {copied} new or changed files to {drive_dir}")
    return copied
