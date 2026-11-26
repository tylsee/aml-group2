"""A0: counts, hash, filenames, decoding, EXIF. Official test images are never opened."""
import io
import os

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

from cdccp.catalogue import load_catalogue
from cdccp.checkpoint import file_sha256
from cdccp.config import image_dir
from cdccp.data import EVAL_RESIZE, EVAL_TAIL, MEAN, STD, DPDDataset, canonical_eval, canonical_train, load_meta, load_split, official_csv


def test_official_counts_and_no_header(cfg):
    assert len(load_meta(official_csv(cfg, "train"))) == 8000
    assert len(load_meta(official_csv(cfg, "test_seen"))) == 2000
    assert len(load_meta(official_csv(cfg, "test_unseen"))) == 1448


def test_train_hash_matches_manifest(cfg):
    assert file_sha256(official_csv(cfg, "train")).startswith("9f1b729a")


def test_split_is_disjoint_and_unions_to_train(cfg):
    fit, seen, unseen = (load_split(cfg, n) for n in ("train_fit", "val_seen", "val_unseen"))
    assert (len(fit), len(seen), len(unseen)) == (6905, 695, 400)
    names = [set(fit.filename), set(seen.filename), set(unseen.filename)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
    train = load_split(cfg, "train").set_index("filename")
    assert names[0] | names[1] | names[2] == set(train.index)
    for frame in (fit, seen, unseen):
        assert (train.loc[frame.filename, "composition_id"].values == frame.composition_id.values).all()


def test_exact_case_filenames_resolve(cfg):
    on_disk = set(os.listdir(image_dir(cfg)))
    for name in ("train", "test_seen", "test_unseen"):
        assert set(load_meta(official_csv(cfg, name)).filename) <= on_disk


def test_all_train_images_decode_to_rgb(cfg):
    images = image_dir(cfg)
    for name in load_split(cfg, "train").filename:
        assert canonical_eval(images / name).mode == "RGB"


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20), "red").save(path, exif=exif)
    assert canonical_eval(path).size == (20, 40)


def test_train_step_downscales_but_never_upscales(tmp_path):
    large, small = tmp_path / "l.png", tmp_path / "s.png"
    Image.new("RGB", (600, 300)).save(large)
    Image.new("RGB", (100, 50)).save(small)
    assert min(canonical_train(large).size) == 288
    assert canonical_train(small).size == (100, 50)


def test_eval_path_matches_team_plan_transform(cfg):
    fixed = T.Compose([T.Resize(256, IM.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    images = image_dir(cfg)
    for name in load_split(cfg, "train_fit").sample(20, random_state=0).filename:
        image = canonical_eval(images / name)
        assert (fixed(image) == EVAL_TAIL(EVAL_RESIZE(image))).all()


def test_dataset_modes_shapes(cfg):
    df = load_split(cfg, "train_fit").head(2)
    assert DPDDataset(df, cfg, "eval")[0][0].shape == (3, 224, 224)
    assert DPDDataset(df, cfg, "train1")[0][0].shape == (3, 224, 224)
    assert DPDDataset(df, cfg, "train2v")[0][0].shape == (2, 3, 224, 224)
    item = DPDDataset(df, cfg, "eval")[1]
    assert item[1:4] == (df.crop_id.iloc[1], df.disease_id.iloc[1], df.composition_id.iloc[1])


def test_catalogues(cfg):
    dev, final = load_catalogue(cfg["paths"]["catalogue_dev"]), load_catalogue(cfg["paths"]["catalogue_final"])
    assert (len(dev), dev.is_seen.sum(), len(final), final.is_seen.sum()) == (323, 313, 333, 323)
    official = set(load_meta(official_csv(cfg, "test_unseen")).composition_id)
    assert not official & set(dev.comp_id)
    assert official == set(final.comp_id[~final.is_seen])
    assert set(load_split(cfg, "val_unseen").composition_id) == set(dev.comp_id[~dev.is_seen])
    pairs = set(zip(final.crop_id, final.disease_id))
    assert len(pairs) == 333
    assert (final.comp_id == np.arange(333)).all()
    with pytest.raises(KeyError):
        dev.index_of([next(iter(official))])


@pytest.mark.parametrize("orientation", range(1, 9))
def test_upright_matches_exif_transpose(orientation):
    from PIL import ImageOps
    from cdccp.data import upright
    image = Image.fromarray(np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3))
    exif = image.getexif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif.tobytes(), quality=100)
    with Image.open(io.BytesIO(buffer.getvalue())) as a, Image.open(io.BytesIO(buffer.getvalue())) as b:
        assert np.array_equal(np.asarray(upright(a).convert("RGB")), np.asarray(ImageOps.exif_transpose(b).convert("RGB")))


def test_upright_never_rewrites_exif(monkeypatch, tmp_path):
    """Official test images with bad EXIF values crash Exif.tobytes inside exif_transpose; upright must not call it."""
    from cdccp.data import canonical_eval
    image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(tmp_path / "x.jpg", exif=exif.tobytes())

    def boom(*args, **kwargs):
        raise TypeError("bad operand type for abs(): 'str'")

    monkeypatch.setattr(Image.Exif, "tobytes", boom)
    assert canonical_eval(tmp_path / "x.jpg").size == (4, 6)
