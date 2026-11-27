"""Training-only manifest loading and honest reference retrieval.

Never use validation or official-test images for the 'training matches' gallery.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import numpy as np


@dataclass(frozen=True)
class TrainRecord:
    filename: str
    crop_id: int
    disease_id: int
    crop: str
    disease: str
    image_path: Path


def load_training_manifest(csv_path: str | Path, image_root: str | Path, *, require_images: bool = True) -> list[TrainRecord]:
    """Read the project's canonical seven-column, headerless train.csv.

    0 filename, 1 crop id, 2 disease id, 3 composition id, 4 crop name,
    5 disease name, 6 composition name. Headers are tolerated.
    """
    csv_path = Path(csv_path).expanduser()
    root = Path(image_root).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if require_images and not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 7:
                raise ValueError('Training CSV must have at least 7 columns: filename, crop_id, disease_id, composition_id, crop_name, disease_name, composition_name.')
            try:
                crop_id, disease_id = int(row[1]), int(row[2])
            except ValueError:
                if not rows and row[0].lower().strip() in ('filename', 'image', 'image_path', 'path'):
                    continue
                raise ValueError(f'Non-integer crop/disease ID in training CSV row: {row[:4]}')
            name = row[0].strip()
            # Never resolve paths outside the designated training image root.
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Unsafe training image path (outside image_root): {name}')
            rows.append(TrainRecord(name, crop_id, disease_id, row[4].strip(), row[5].strip(), path))
    if not rows:
        raise ValueError('Training manifest is empty.')
    seen = set()
    for r in rows:
        if r.filename in seen:
            raise ValueError(f'Duplicate training image filename in manifest: {r.filename}')
        seen.add(r.filename)
    return rows


def class_catalogue(records: list[TrainRecord]) -> tuple[list[str], list[str]]:
    """Build ID-ordered class labels, refusing holes or ambiguous label mappings."""
    crop_map, disease_map = {}, {}
    for r in records:
        if r.crop_id in crop_map and crop_map[r.crop_id] != r.crop:
            raise ValueError(f'Conflicting crop names for ID {r.crop_id}.')
        if r.disease_id in disease_map and disease_map[r.disease_id] != r.disease:
            raise ValueError(f'Conflicting disease names for ID {r.disease_id}.')
        crop_map[r.crop_id] = r.crop
        disease_map[r.disease_id] = r.disease
    def ordered(mapping, kind):
        ids = sorted(mapping)
        if ids != list(range(len(ids))):
            raise ValueError(f'{kind} IDs are not contiguous 0..N-1; supply an explicit canonical mapping before using the model.')
        return [mapping[i] for i in ids]
    return ordered(crop_map, 'Crop'), ordered(disease_map, 'Disease')


@dataclass
class Match:
    record: TrainRecord
    reason: str
    similarity: Optional[float] = None


def load_embedding_index(path: str | Path) -> dict:
    """Expect filename, crop_embeddings, disease_embeddings saved from TRAIN ONLY."""
    with np.load(path, allow_pickle=False) as z:
        for field in ('filenames', 'crop_embeddings', 'disease_embeddings'):
            if field not in z:
                raise ValueError(f'Training embedding index missing: {field}')
        names = [str(s) for s in z['filenames'].tolist()]
        crop = np.asarray(z['crop_embeddings'], dtype=np.float32)
        disease = np.asarray(z['disease_embeddings'], dtype=np.float32)
    if crop.shape != disease.shape or len(names) != len(crop) or crop.ndim != 2:
        raise ValueError('Training index shapes do not match.')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate filenames in training embedding index.')
    norm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return {'filenames': names, 'crop': norm(crop), 'disease': norm(disease), 'positions': {name:i for i,name in enumerate(names)}}


def find_matches(records: list[TrainRecord], crop: str, disease: str, *,
                 query_crop_embedding=None, query_disease_embedding=None,
                 index: Optional[dict]=None, n: int=6) -> list[Match]:
    """Prefer exact crop+disease examples, fall back to explicitly labeled related examples.

    If an embedding index exists, use averaged crop+disease cosine similarity.
    Otherwise show unranked class-relevant examples, NEVER invented similarities.
    """
    exact = [r for r in records if r.crop.casefold() == crop.casefold() and r.disease.casefold() == disease.casefold() and r.image_path.is_file()]
    same_disease = [r for r in records if r.crop.casefold()!=crop.casefold() and r.disease.casefold()==disease.casefold() and r.image_path.is_file()]
    same_crop = [r for r in records if r.crop.casefold()==crop.casefold() and r.disease.casefold()!=disease.casefold() and r.image_path.is_file()]

    scores = {}
    if index is not None and query_crop_embedding is not None and query_disease_embedding is not None:
        qc = np.asarray(query_crop_embedding, dtype=np.float32).reshape(-1)
        qd = np.asarray(query_disease_embedding, dtype=np.float32).reshape(-1)
        if qc.shape[0] != index['crop'].shape[1] or qd.shape[0] != index['disease'].shape[1]:
            raise ValueError('Query embedding dimensions differ from training index.')
        qc = qc / max(float(np.linalg.norm(qc)), 1e-12)
        qd = qd / max(float(np.linalg.norm(qd)), 1e-12)
        sim = 0.5*(index['crop']@qc + index['disease']@qd)
        scores = {name:float(sim[pos]) for name,pos in index['positions'].items()}
    def sort_bucket(bucket):
        if scores:
            return sorted(bucket, key=lambda r:scores.get(r.filename,-2.0), reverse=True)
        return sorted(bucket, key=lambda r:r.filename)
    out = []
    for bucket,reason in ((exact,'Exact composition'),(same_disease,'Same disease · different crop'),(same_crop,'Same crop · different disease')):
        for r in sort_bucket(bucket):
            if len(out) >= n:
                break
            out.append(Match(r, reason, scores.get(r.filename)))
        if len(out)>=n:
            break
    return out
