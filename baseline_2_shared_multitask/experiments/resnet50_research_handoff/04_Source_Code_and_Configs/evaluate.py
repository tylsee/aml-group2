"""Shared data/metric helpers. Importing this module never reads any dataset.

Default CLI: synthetic checks only. Official evaluation requires explicit
--official --approve-official-testing flags AND user-supplied paths.

The official evaluation offers TWO PREDECLARED MODES:
- baseline: original B2 composition ranking (gamma=0).
- both: baseline plus optional Exp06 fixed gamma=4.0 calibration, from ONE
  inference pass per official split. Requires final_eval_lock.json and the
  original *training* splits to verify the 8,000-image seen-pair mask.
Never select gamma/model/hyperparameters using official test results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

COLUMNS = ["filename", "crop_id", "disease_id", "composition_id",
           "crop_name", "disease_name", "composition_name"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_metadata(path):
    """Accept the supplied headered splits and the original 7-column format."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        first = next(csv.reader(handle), [])
    header = first == COLUMNS
    frame = pd.read_csv(path, header=0 if header else None,
                        names=None if header else COLUMNS, encoding="utf-8-sig")
    if list(frame.columns) != COLUMNS or frame.empty or frame.isna().any().any():
        raise ValueError(f"Expected nonempty seven-column metadata without missing values: {path}")
    for col, limit in [("crop_id", 55), ("disease_id", 175), ("composition_id", None)]:
        values = pd.to_numeric(frame[col], errors="raise")
        if (values != np.floor(values)).any() or (values < 0).any():
            raise ValueError(f"Invalid canonical integer IDs: {col}")
        if limit and (values >= limit).any():
            raise ValueError(f"{col} must be in [0, {limit - 1}]")
        frame[col] = values.astype("int64")
    if frame.filename.duplicated().any():
        raise ValueError("Duplicate image filenames")
    return frame


def validate_development_splits(split_dir):
    """Read only the five named, supplied development files; never discover test files."""
    split_dir = Path(split_dir)
    frames = {name: read_metadata(split_dir / f"{name}.csv")
              for name in ("train_fit", "val_seen", "val_unseen")}
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    expected = {"train_fit": 6905, "val_seen": 695, "val_unseen": 400}
    for name, frame in frames.items():
        if len(frame) != expected[name] or manifest["split_counts"][name] != expected[name]:
            raise ValueError(f"Unexpected split count: {name}")
    if manifest["split_counts"]["original_training"] != 8000:
        raise ValueError("Original training count must be 8,000")
    combined = pd.concat(frames.values(), ignore_index=True)
    if combined.filename.duplicated().any():
        raise ValueError("Images overlap across development partitions")
    for label in ("crop", "disease", "composition"):
        if combined.groupby(f"{label}_id")[f"{label}_name"].nunique().max() != 1:
            raise ValueError(f"Inconsistent {label} ID/name mapping")
    catalog = combined[["composition_id", "crop_id", "disease_id"]].drop_duplicates()
    if catalog.composition_id.duplicated().any() or catalog.duplicated(["crop_id", "disease_id"]).any():
        raise ValueError("Composition ID and crop/disease pair must map one-to-one")
    train, seen, unseen = (frames[n] for n in expected)
    if not set(seen.composition_id) <= set(train.composition_id):
        raise ValueError("Seen validation contains a composition absent from training")
    if set(unseen.composition_id) & set(train.composition_id):
        raise ValueError("Held-out compositions occur in training")
    heldout = pd.read_csv(split_dir / "heldout_compositions.csv")
    held_ids = set(heldout.composition_id)
    if (len(heldout) != 10 or held_ids != set(unseen.composition_id)
            or held_ids != set(manifest["pseudo_unseen_composition_ids"])):
        raise ValueError("Held-out composition lists disagree")
    for row in heldout.itertuples(index=False):
        matching = unseen[unseen.composition_id == row.composition_id]
        if len(matching) != row.images:
            raise ValueError("Held-out image counts disagree")
        for col in COLUMNS[1:]:
            if not (matching[col] == getattr(row, col)).all():
                raise ValueError(f"Held-out metadata disagrees: {col}")
    for col, n in (("crop_id", 55), ("disease_id", 175)):
        if set(train[col]) != set(range(n)):
            raise ValueError(f"Training must cover every canonical {col}")
        if not set(unseen[col]) <= set(train[col]):
            raise ValueError(f"Unseen validation contains unseen individual concepts: {col}")
    hashes = {name: sha256_file(split_dir / name) for name in
              ["train_fit.csv", "val_seen.csv", "val_unseen.csv",
               "heldout_compositions.csv", "split_manifest.json"]}
    report = {"counts": {k: len(v) for k, v in frames.items()},
              "coverage": {k: {c: int(v[c].nunique()) for c in
                           ("crop_id", "disease_id", "composition_id")} for k, v in frames.items()},
              "sha256": hashes, "image_paths_checked": False}
    return frames, report


def image_paths(frame, image_root):
    """Resolve only CSV-listed paths under the user-selected TRAINING image folder.

    No recursive directory scan or fallback to another dataset directory.
    """
    root = Path(image_root).expanduser().resolve()
    paths = []
    for filename in frame.filename:
        path = (root / str(filename).replace("\\", "/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes configured root: {filename}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV-listed image: {path}. Set IMAGE_ROOT to its containing folder.")
        paths.append(path)
    return paths


class DPDDataset(Dataset):
    def __init__(self, frame, image_root, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.paths = image_paths(self.frame, image_root)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "crop": int(row.crop_id), "disease": int(row.disease_id),
                "composition": int(row.composition_id), "filename": str(row.filename)}


def validation_transform():
    return T.Compose([T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                      T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def make_candidates(mode="all_pairs", catalog_path=None):
    """One global catalogue, never restricted using an individual image's labels."""
    if mode == "all_pairs":
        return torch.cartesian_prod(torch.arange(55), torch.arange(175))
    if mode != "catalog" or catalog_path is None:
        raise ValueError("Use all_pairs or supply a group-approved catalogue CSV")
    frame = pd.read_csv(catalog_path)
    pairs = frame[["crop_id", "disease_id"]].to_numpy()
    if (len(pairs) < 5 or not np.isfinite(pairs).all() or (pairs != np.floor(pairs)).any()
            or (pairs < 0).any() or (pairs[:, 0] >= 55).any() or (pairs[:, 1] >= 175).any()
            or len(np.unique(pairs, axis=0)) != len(pairs)):
        raise ValueError("Invalid/duplicate candidate pairs; need at least five")
    return torch.as_tensor(pairs, dtype=torch.long)


def require_candidate_coverage(candidates, frames):
    allowed = set(map(tuple, candidates.cpu().tolist()))
    required = set()
    for frame in frames:
        required.update(map(tuple, frame[["crop_id", "disease_id"]].to_numpy().tolist()))
    missing = required - allowed
    if missing:
        raise ValueError(f"Candidate catalogue excludes ground-truth pairs: {sorted(missing)}")


def composition_scores(crop_logits, disease_logits, candidates):
    candidates = candidates.to(crop_logits.device)
    return (crop_logits.float().log_softmax(1)[:, candidates[:, 0]]
            + disease_logits.float().log_softmax(1)[:, candidates[:, 1]])


def topk_accuracy(logits, labels, k=1):
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits) or not len(labels):
        raise ValueError("Expected nonempty logits [N,C] and labels [N]")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError("k exceeds the number of classes")
    return float((logits.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean())


def harmonic_mean(seen, unseen):
    if seen < 0 or unseen < 0 or not np.isfinite([seen, unseen]).all():
        raise ValueError("Accuracies must be finite and nonnegative")
    return 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)


def generalisation_metrics(seen, unseen):
    a, b = seen["composition_top1"], unseen["composition_top1"]
    return {"seen_composition_top1": a, "unseen_composition_top1": b,
            "composition_harmonic_mean": harmonic_mean(a, b), "seen_unseen_gap": a - b}


def score_with_calibration(raw_pair_scores, seen_pair_mask, gamma):
    """Exact Exp06 formula: raw log-probability pair scores minus gamma for train-seen pairs.

    All 9,625 pairs remain eligible; no ground-truth-dependent filtering.
    """
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Calibration gamma must be finite and nonnegative")
    mask = torch.as_tensor(seen_pair_mask, dtype=torch.bool, device=raw_pair_scores.device)
    if raw_pair_scores.ndim != 2 or mask.ndim != 1 or raw_pair_scores.shape[1] != mask.numel():
        raise ValueError("Scores/candidate mask shape mismatch")
    if gamma == 0:
        return raw_pair_scores
    return raw_pair_scores - gamma * mask.to(raw_pair_scores.dtype)[None, :]


def load_locked_final_calibration(lock_path, training_split_dir, checkpoint_path, checkpoint, candidates):
    """Verify *training-only* full-data mask and fixed Exp06 gamma; no test read.

    Refuses if lock, checkpoint, split hashes, or candidate ordering disagree.
    """
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Lock checkpoint hash differs from supplied final model")
    if (lock.get("checkpoint") != Path(checkpoint_path).name or
            lock.get("candidate_mode") != "all_pairs" or
            lock.get("total_candidates") != 9625 or
            lock.get("interpolation") != "bicubic" or
            lock.get("resize") != 256 or lock.get("center_crop") != 224 or
            lock.get("tta") is not False or
            lock.get("official_test_completed") is not False):
        raise ValueError("Locked final evaluation settings do not match original B2 plan")
    if lock.get("gamma_source") != "Exp06 development validation":
        raise ValueError("Gamma must come from the pre-existing Exp06 development experiment")
    if float(lock.get("gamma", -1)) != 4.0:
        raise ValueError("Expected preselected Exp06 gamma=4.0; do not retune on official test")
    if checkpoint.get("stage") != 2 or checkpoint.get("full_training_count") != 8000:
        raise ValueError("Expected Stage 2B full-data checkpoint")
    if checkpoint["config"].get("candidate_mode") != "all_pairs":
        raise ValueError("Unexpected candidate policy")
    if not torch.equal(candidates.cpu(), make_candidates("all_pairs")):
        raise ValueError("Candidate ordering differs from canonical 9,625 all_pairs")

    frames, report = validate_development_splits(training_split_dir)
    if report["sha256"] != checkpoint["development_split"]["sha256"]:
        raise ValueError("Original training split SHA256 hashes do not match final checkpoint")
    full = pd.concat([frames[k] for k in ("train_fit", "val_seen", "val_unseen")], ignore_index=True)
    if len(full) != 8000 or full.filename.duplicated().any():
        raise ValueError("Invalid 8,000-image original training union")
    actual_pairs = set(map(tuple, full[["crop_id", "disease_id"]].to_numpy(dtype=int).tolist()))
    checkpoint_pairs = {(int(r["crop_id"]), int(r["disease_id"])) for r in checkpoint["composition_mapping"]}
    if actual_pairs != checkpoint_pairs:
        raise ValueError("Full-training pair set does not match checkpoint composition mapping")
    mask = torch.tensor([tuple(p) in actual_pairs for p in candidates.tolist()], dtype=torch.bool)
    indices = mask.nonzero().view(-1).tolist()
    if (len(indices) != 323 or lock.get("seen_pair_count") != len(indices) or
            lock.get("seen_pair_indices") != indices or
            lock.get("mask_source") != "all 8000 original training images"):
        raise ValueError("Saved calibration mask differs from reconstructed 8,000-image mask")
    return mask, 4.0


@torch.inference_mode()
def evaluate_loader_modes(model, loader, device, candidates, amp=False,
                          collect_predictions=False, seen_pair_mask=None, gamma=0.0):
    """Evaluate baseline (+ optional fixed calibration) in a single model pass.

    Returns {mode: (metrics, records)}. The crop/disease outputs/losses are
    identical across modes; only composition ranking is adjusted.
    """
    model.eval()
    candidates = candidates.to(device)
    with_calibration = seen_pair_mask is not None
    modes = ("baseline", "calibrated") if with_calibration else ("baseline",)
    if with_calibration:
        mask = torch.as_tensor(seen_pair_mask, dtype=torch.bool, device=device)
        if mask.numel() != len(candidates) or mask.ndim != 1:
            raise ValueError("Invalid candidate mask")
        if gamma != 4.0:
            raise ValueError("Fixed predeclared Exp06 gamma=4.0 required for official calibrated mode")
    count_keys = [f"{task}_top{k}" for task in ("crop", "disease", "composition") for k in (1, 5)]
    counts = {mode: {key: 0 for key in count_keys} for mode in modes}
    records = {mode: [] for mode in modes}
    losses = np.zeros(3, dtype=float)
    n = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        crops = batch["crop"].to(device)
        diseases = batch["disease"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            crop_loss = criterion(output["crop_logits"], crops)
            disease_loss = criterion(output["disease_logits"], diseases)
        size = len(images)
        losses += size * np.array([float(crop_loss + disease_loss), float(crop_loss), float(disease_loss)])
        n += size
        crop_ranks = output["crop_logits"].float().topk(5, dim=1).indices
        disease_ranks = output["disease_logits"].float().topk(5, dim=1).indices
        for mode in modes:
            for k in (1, 5):
                counts[mode][f"crop_top{k}"] += int((crop_ranks[:, :k] == crops[:, None]).any(1).sum())
                counts[mode][f"disease_top{k}"] += int((disease_ranks[:, :k] == diseases[:, None]).any(1).sum())
        raw_scores = composition_scores(output["crop_logits"], output["disease_logits"], candidates)
        pair_targets = torch.stack((crops, diseases), dim=1)
        for mode in modes:
            scores = score_with_calibration(raw_scores, mask, gamma) if mode == "calibrated" else raw_scores
            pair_ranks = candidates[scores.topk(5, dim=1).indices]
            for k in (1, 5):
                counts[mode][f"composition_top{k}"] += int(
                    (pair_ranks[:, :k] == pair_targets[:, None]).all(2).any(1).sum())
            if collect_predictions:
                cp, dp, pp = crop_ranks.cpu().tolist(), disease_ranks.cpu().tolist(), pair_ranks.cpu().tolist()
                for i, filename in enumerate(batch["filename"]):
                    records[mode].append({
                        "filename": str(filename),
                        "crop_id": int(batch["crop"][i]),
                        "disease_id": int(batch["disease"][i]),
                        "composition_id": int(batch["composition"][i]),
                        "predicted_crop_id": cp[i][0],
                        "predicted_disease_id": dp[i][0],
                        "predicted_pair_crop_id": pp[i][0][0],
                        "predicted_pair_disease_id": pp[i][0][1],
                        "ranked_crop_ids": cp[i],
                        "ranked_disease_ids": dp[i],
                        "ranked_composition_pairs": pp[i]})
    if not n:
        raise ValueError("Cannot evaluate an empty loader")
    results = {}
    for mode in modes:
        result = {key: value / n for key, value in counts[mode].items()}
        result.update(dict(zip(("total_loss", "crop_loss", "disease_loss"), (losses / n).tolist())))
        result["n_images"] = n
        results[mode] = (result, records[mode])
    return results


@torch.inference_mode()
def evaluate_loader(model, loader, device, candidates, amp=False, collect_predictions=False):
    """Backward-compatible original uncalibrated B2 evaluator."""
    return evaluate_loader_modes(model, loader, device, candidates, amp, collect_predictions)["baseline"]


def composition_confusion(records):
    """Exact pair confusion, including predictions outside the true seen catalogue.

    True rows and predicted columns use their union, so no off-catalogue errors
    disappear. Labels retain canonical composition IDs when known; unseen predicted
    pairs have explicit crop/disease labels instead of invented composition IDs.
    """
    truth = [(r["crop_id"], r["disease_id"]) for r in records]
    predicted = [(r["predicted_pair_crop_id"], r["predicted_pair_disease_id"]) for r in records]
    labels = sorted(set(truth) | set(predicted))
    positions = {pair: i for i, pair in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(truth, predicted):
        matrix[positions[a], positions[b]] += 1
    known = {(r["crop_id"], r["disease_id"]): r["composition_id"] for r in records}
    names = [f"{known[p]}: c{p[0]} d{p[1]}" if p in known else f"c{p[0]} d{p[1]} (outside seen)" for p in labels]
    return pd.DataFrame(matrix, index=names, columns=names)


def synthetic_self_test():
    c = torch.tensor([[9., 1., 0., -1., -2., -3.], [0., 9., 1., -1., -2., -3.]])
    d = torch.tensor([[0., 9., 1., -1., -2., -3.], [0., 9., 1., -1., -2., -3.]])
    pairs = torch.cartesian_prod(torch.arange(6), torch.arange(6))
    ranks = pairs[composition_scores(c, d, pairs).topk(5, 1).indices]
    assert ranks[:, 0].tolist() == [[0, 1], [1, 1]]
    assert topk_accuracy(c, torch.tensor([0, 2]), 1) == 0.5
    assert topk_accuracy(c, torch.tensor([0, 2]), 5) == 1.0
    assert harmonic_mean(0, 0) == 0 and harmonic_mean(0, 1) == 0
    assert abs(harmonic_mean(0.8, 0.2) - 0.32) < 1e-7
    subset = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]])
    expected = c.log_softmax(1)[:, subset[:, 0]] + d.log_softmax(1)[:, subset[:, 1]]
    assert torch.allclose(composition_scores(c, d, subset), expected)
    rec = [{"crop_id": 0, "disease_id": 0, "composition_id": 7,
            "predicted_pair_crop_id": 1, "predicted_pair_disease_id": 1}]
    cm = composition_confusion(rec)
    assert cm.to_numpy().sum() == 1 and cm.to_numpy().trace() == 0
    assert generalisation_metrics({"composition_top1": 0.2}, {"composition_top1": 0.8})["seen_unseen_gap"] < 0
    fake = torch.tensor([[0.9, 0.8, 0.1]], dtype=torch.float32)
    mask = torch.tensor([True, False, False])
    assert torch.equal(score_with_calibration(fake, mask, 0.0), fake)
    assert score_with_calibration(fake, mask, 0.2).argmax(1).item() == 1
    assert score_with_calibration(fake, mask, 0.0).argmax(1).item() == 0
    print("Synthetic metric + Exp06 calibration checks passed; no dataset accessed.")


def _save_predictions(output_dir, split, records):
    """Write full audit CSVs and a size-safe confusion plot for each split.

    The full exact-pair confusion CSV includes every true/predicted pair.
    The PNG groups pairs outside the top 25 (by total frequency) into 'Other',
    so the figure cannot accidentally allocate a gigapixel image.
    """
    prepared = []
    for row in records:
        rec = dict(row)
        for key in ("ranked_crop_ids", "ranked_disease_ids", "ranked_composition_pairs"):
            rec[key] = json.dumps(rec[key])
        prepared.append(rec)
    pd.DataFrame(prepared).to_csv(output_dir / f"{split}_predictions.csv", index=False)

    cm = composition_confusion(records)
    cm.to_csv(output_dir / f"{split}_confusion.csv", index_label="true / predicted")
    errors = pd.DataFrame(records)
    errors = errors[(errors.crop_id != errors.predicted_pair_crop_id) |
                    (errors.disease_id != errors.predicted_pair_disease_id)]
    errors.to_csv(output_dir / f"{split}_errors.csv", index=False)

    # Visual summary only; unlike the full CSV, it groups the long tail.
    from collections import Counter
    from matplotlib import pyplot as plt
    truth = [f"c{r['crop_id']} d{r['disease_id']}" for r in records]
    predicted = [f"c{r['predicted_pair_crop_id']} d{r['predicted_pair_disease_id']}" for r in records]
    top25 = [pair for pair, _ in (Counter(truth) + Counter(predicted)).most_common(25)]
    labels = top25 + ["Other"]
    positions = {pair: i for i, pair in enumerate(top25)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    other = len(top25)
    for t, p in zip(truth, predicted):
        matrix[positions.get(t, other), positions.get(p, other)] += 1
    assert int(matrix.sum()) == len(records)
    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted Crop-Disease Pair", ylabel="True Crop-Disease Pair",
           title=f"{split.title()} Composition Confusion — Top 25 + Other (full CSV saved)")
    ax.tick_params(axis="both", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(image, ax=ax, label="Image count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_confusion.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="Check final checkpoint + lock + training-only mask; no official data")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--approve-official-testing", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "both"), default="baseline",
                        help="baseline: original gamma=0; both: baseline and fixed Exp06 gamma=4")
    parser.add_argument("--lock-json", type=Path,
                        help="Required for mode=both: original final_eval_lock.json saved before official test")
    parser.add_argument("--training-splits", type=Path,
                        help="Required for mode=both: original splits_v2 folder; never official test splits")
    for name in ("checkpoint", "seen-csv", "unseen-csv", "seen-image-root", "unseen-image-root", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if not args.official:
        if args.approve_official_testing:
            parser.error("--approve-official-testing requires --official")
        if args.preflight and args.self_test:
            parser.error("Use either --preflight or --self-test, not both")
        synthetic_self_test()
        if args.preflight:
            if args.checkpoint is None or args.lock_json is None or args.training_splits is None:
                parser.error("--preflight requires --checkpoint, --lock-json, and --training-splits")
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            pairs = torch.as_tensor(ck["candidate_pairs"], dtype=torch.long)
            mask, gamma = load_locked_final_calibration(
                args.lock_json, args.training_splits, args.checkpoint, ck, pairs)
            print("PASS: Stage 2B checkpoint + Exp06 locked protocol verified")
            print(f"Candidates: {len(pairs)} | train-seen pairs: {int(mask.sum())} | fixed gamma: {gamma}")
            print("Official test data not accessed. Training not started.")
        return
    if args.self_test or args.preflight:
        parser.error("Do not combine --self-test/--preflight with --official")
    if not args.approve_official_testing:
        parser.error("Official evaluation is disabled without explicit user approval")
    required = ("checkpoint", "seen_csv", "unseen_csv", "seen_image_root", "unseen_image_root", "output")
    if any(getattr(args, key) is None for key in required):
        parser.error("Official evaluation requires checkpoint, both CSVs, both image roots, and output folder")
    if args.mode == "both" and (args.lock_json is None or args.training_splits is None):
        parser.error("--mode both requires --lock-json and --training-splits")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite any existing official output: {args.output}")

    # No official test data is read before all approval/configuration guards above.
    from model import load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    pairs = torch.as_tensor(checkpoint["candidate_pairs"], dtype=torch.long)
    if (checkpoint.get("stage") != 2 or checkpoint.get("full_training_count") != 8000 or
            checkpoint["config"].get("candidate_mode") != "all_pairs" or
            len(pairs) != 9625 or not torch.equal(pairs, make_candidates("all_pairs"))):
        raise ValueError("Final full-data Stage 2B checkpoint/candidate mode mismatch")

    seen_pair_mask, gamma = None, 0.0
    if args.mode == "both":
        seen_pair_mask, gamma = load_locked_final_calibration(
            args.lock_json, args.training_splits, args.checkpoint, checkpoint, pairs)

    # Only now read the user-supplied official test metadata/images.
    frames = [read_metadata(args.seen_csv), read_metadata(args.unseen_csv)]
    require_candidate_coverage(pairs, frames)
    if len(frames[0]) != 2000 or len(frames[1]) != 1448:
        raise ValueError("Official split counts must be 2,000 seen and 1,448 unseen")
    if set(frames[0].filename) & set(frames[1].filename):
        raise ValueError("Official split image overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ("baseline", "calibrated") if args.mode == "both" else ("baseline",)
    for mode in modes:
        (args.output / mode).mkdir(exist_ok=False)
    metrics = {mode: {} for mode in modes}
    for split, frame, root in zip(("seen", "unseen"), frames,
                                  (args.seen_image_root, args.unseen_image_root)):
        ds = DPDDataset(frame, root, validation_transform())
        loader = DataLoader(ds, batch_size=checkpoint["config"]["batch_size"], num_workers=0)
        outputs = evaluate_loader_modes(model, loader, device, pairs, amp=True,
                                        collect_predictions=True,
                                        seen_pair_mask=seen_pair_mask, gamma=gamma)
        for mode in modes:
            metrics[mode][split], records = outputs[mode]
            _save_predictions(args.output / mode, split, records)
    for mode in modes:
        metrics[mode].update(generalisation_metrics(metrics[mode]["seen"], metrics[mode]["unseen"]))
        metrics[mode]["checkpoint_sha256"] = sha256_file(args.checkpoint)
        metrics[mode]["candidate_mode"] = "all_pairs"
        metrics[mode]["gamma"] = 0.0 if mode == "baseline" else gamma
        metrics[mode]["accuracy_units"] = "fractions in [0,1]"
        (args.output / mode / "metrics.json").write_text(json.dumps(metrics[mode], indent=2), encoding="utf-8")
    audit = {
        "model": "b2_final_exp04", "checkpoint_sha256": sha256_file(args.checkpoint),
        "mode": args.mode, "official_test_completed": True, "settings_selected_before_test": True,
        "baseline_gamma": 0.0,
        "optional_calibration_gamma": gamma if args.mode == "both" else None,
        "optional_calibration_seen_pairs": int(seen_pair_mask.sum()) if args.mode == "both" else None,
        "calibration_note": "Separate post-processing extension; do not select it based on official test results",
        "preprocess": "Resize(256,bicubic), CenterCrop(224), ImageNet normalize; no TTA",
        "metrics": {mode: {
            "seen_top1": metrics[mode]["seen_composition_top1"],
            "unseen_top1": metrics[mode]["unseen_composition_top1"],
            "harmonic_mean": metrics[mode]["composition_harmonic_mean"]
        } for mode in modes},
    }
    (args.output / "comparison.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
