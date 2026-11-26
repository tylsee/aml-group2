import os
from pathlib import Path

import pytest

from cdccp.colab_io import check_images, copy_tree, save_results


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_copy_tree_copies_every_file_and_keeps_same_size_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for i in range(50):
        _write(src / f"img_{i:03d}.jpg", bytes([i]) * (i + 1))
    _write(src / "sub" / "a.txt", b"abc")
    _write(dst / "img_000.jpg", b"Z")              # same size as the source: kept as it is
    _write(dst / "img_001.jpg", b"short")          # different size: replaced
    lines = []
    assert copy_tree(src, dst, threads=4, printer=lines.append) == 51
    assert (dst / "img_000.jpg").read_bytes() == b"Z"
    assert (dst / "img_001.jpg").read_bytes() == bytes([1]) * 2
    for i in range(2, 50):
        assert (dst / f"img_{i:03d}.jpg").read_bytes() == bytes([i]) * (i + 1)
    assert (dst / "sub" / "a.txt").read_bytes() == b"abc"
    assert not list(dst.rglob("*.part"))
    assert "copied 50 of 51 files" in lines[0]
    lines.clear()
    copy_tree(src, dst, threads=4, printer=lines.append)
    assert "copied 0 of 51 files" in lines[0]


def test_copy_tree_missing_source_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_tree(tmp_path / "nope", tmp_path / "dst", printer=lambda *_: None)


def test_copy_tree_error_in_one_file_is_raised(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for i in range(5):
        _write(src / f"{i}.jpg", b"x")
    import cdccp.colab_io as cio

    real = cio.shutil.copyfile

    def broken(a, b):
        if str(a).endswith("3.jpg"):
            raise OSError("drive read failed")
        return real(a, b)

    monkeypatch.setattr(cio.shutil, "copyfile", broken)
    lines = []
    with pytest.raises(OSError, match="drive read failed"):
        copy_tree(src, dst, threads=2, retries=2, wait=0, printer=lines.append)
    assert not (dst / "3.jpg").exists()
    assert sum("retry" in line and "3.jpg" in line for line in lines) == 2, lines


def test_copy_tree_retries_a_passing_drive_error(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for i in range(3):
        _write(src / f"{i}.jpg", b"xyz")
    import cdccp.colab_io as cio

    real, failed = cio.shutil.copyfile, []

    def flaky(a, b):
        if str(a).endswith("1.jpg") and not failed:
            failed.append(a)
            Path(b).write_bytes(b"x")          # a short half-written .part file, then the error
            raise OSError("Input/output error")
        return real(a, b)

    monkeypatch.setattr(cio.shutil, "copyfile", flaky)
    lines = []
    assert copy_tree(src, dst, threads=1, retries=3, wait=0, printer=lines.append) == 3
    assert (dst / "1.jpg").read_bytes() == b"xyz" and not list(dst.rglob("*.part"))
    assert any(line.startswith("retry 1/3") for line in lines), lines


def test_copy_tree_raises_when_a_copied_file_has_the_wrong_size(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src / "a.jpg", b"abcdef")
    import cdccp.colab_io as cio

    monkeypatch.setattr(cio.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"abc"))
    with pytest.raises(OSError, match="differ in size"):
        copy_tree(src, dst, threads=1, printer=lambda *_: None)


def test_check_images_names_a_damaged_file(tmp_path):
    from PIL import Image as PILImage

    for i in range(3):
        PILImage.new("RGB", (8, 8), (i, 0, 0)).save(tmp_path / f"{i}.jpg")
    assert check_images(tmp_path) == 3
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(OSError, match="bad.jpg"):
        check_images(tmp_path)


def test_check_images_finds_a_cut_off_jpeg_that_verify_misses(tmp_path):
    import io

    import numpy as np
    from PIL import Image as PILImage

    pixels = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels).save(buf, "JPEG", quality=95)
    data = buf.getvalue()
    (tmp_path / "cut.jpg").write_bytes(data[: len(data) // 2])
    with PILImage.open(tmp_path / "cut.jpg") as image:
        image.verify()                                   # the header-only check passes: that is why it is not enough
    with pytest.raises(OSError, match="cut.jpg"):
        check_images(tmp_path)
    (tmp_path / "cut.jpg").write_bytes(data)
    assert check_images(tmp_path) == 1


def test_a_copy_that_always_fails_half_way_leaves_the_old_target(tmp_path, monkeypatch):
    import cdccp.colab_io as cio
    from cdccp.colab_io import copy_file

    src, dst = tmp_path / "new.csv", tmp_path / "drive" / "new.csv"
    _write(src, b"new contents")
    _write(dst, b"old")

    def half(a, b):
        Path(b).write_bytes(b"new co")
        raise OSError("drive went away")

    monkeypatch.setattr(cio.shutil, "copy2", half)
    with pytest.raises(OSError, match="drive went away"):
        copy_file(src, dst, retries=1, wait=0, printer=lambda *_: None)
    assert dst.read_bytes() == b"old"


def test_save_results_retries_a_passing_drive_error(tmp_path, monkeypatch):
    import cdccp.colab_io as cio

    out, drive = tmp_path / "p5", tmp_path / "drive"
    _write(out / "results" / "t.csv", b"abc")
    real, failed = cio.shutil.copy2, []

    def flaky(a, b):
        if not failed:
            failed.append(a)
            raise OSError("Input/output error")
        return real(a, b)

    monkeypatch.setattr(cio.shutil, "copy2", flaky)
    lines = []
    assert save_results(out, drive, wait=0, printer=lines.append) == 1
    assert (drive / "results" / "t.csv").read_bytes() == b"abc"
    assert any(line.startswith("retry 1/3") for line in lines), lines


def test_save_results_skips_checkpoints_except_named_folders(tmp_path):
    out, drive = tmp_path / "p5", tmp_path / "drive" / "p5_results"
    _write(out / "results" / "table.csv", b"a,b")
    _write(out / "lock.json", b"{}")
    _write(out / "stage1" / "crop_b1" / "best.pt", b"model1")
    _write(out / "stage1" / "crop_b1" / "history.csv", b"h")
    _write(out / "stage2" / "crop_b1" / "best_train_acc.pt", b"model2")
    _write(out / "stage1" / "crop_b1" / "best.pt.tmp", b"half")       # left by an interrupted save: never copied
    _write(out / "results" / "x.csv.part", b"half")
    assert save_results(out, drive, printer=lambda *_: None) == 3
    assert not list(drive.rglob("*.tmp")) and not list(drive.rglob("*.part"))
    assert (drive / "results" / "table.csv").read_bytes() == b"a,b"
    assert (drive / "stage1" / "crop_b1" / "history.csv").is_file()
    assert not (drive / "stage1" / "crop_b1" / "best.pt").exists()
    assert not (drive / "stage2" / "crop_b1" / "best_train_acc.pt").exists()
    assert save_results(out, drive, printer=lambda *_: None) == 0
    assert save_results(out, drive, model_folders=("stage2",), printer=lambda *_: None) == 1
    assert (drive / "stage2" / "crop_b1" / "best_train_acc.pt").read_bytes() == b"model2"
    assert not (drive / "stage1" / "crop_b1" / "best.pt").exists()


def test_save_results_copies_a_changed_file_again(tmp_path):
    out, drive = tmp_path / "p5", tmp_path / "drive"
    _write(out / "results" / "log.txt", b"one")
    save_results(out, drive, printer=lambda *_: None)
    _write(out / "results" / "log.txt", b"one two")
    later = (drive / "results" / "log.txt").stat().st_mtime + 10
    os.utime(out / "results" / "log.txt", (later, later))
    assert save_results(out, drive, printer=lambda *_: None) == 1
    assert (drive / "results" / "log.txt").read_bytes() == b"one two"


def test_save_results_uses_time_as_well_as_size(tmp_path):
    out, drive = tmp_path / "p5", tmp_path / "drive"
    _write(out / "marker.json", b"AAAA")
    save_results(out, drive, printer=lambda *_: None)
    assert save_results(out, drive, printer=lambda *_: None) == 0          # same size, same time: skipped
    _write(out / "marker.json", b"BBBB")                                    # same size, newer content
    later = (drive / "marker.json").stat().st_mtime + 10
    os.utime(out / "marker.json", (later, later))
    assert save_results(out, drive, printer=lambda *_: None) == 1
    assert (drive / "marker.json").read_bytes() == b"BBBB"
