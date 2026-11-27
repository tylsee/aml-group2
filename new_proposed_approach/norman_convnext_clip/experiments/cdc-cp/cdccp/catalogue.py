import json
from dataclasses import dataclass

import numpy as np


def build_catalogue(seen_df, unseen_df, sources):
    """Fixed, ordered pair list. `sources` records which files produced it."""
    pairs = []
    for df, is_seen in ((seen_df, True), (unseen_df, False)):
        unique = df[["composition_id", "crop_id", "disease_id", "composition_name"]].drop_duplicates()
        for row in unique.itertuples():
            pairs.append({
                "comp_id": int(row.composition_id),
                "crop_id": int(row.crop_id),
                "disease_id": int(row.disease_id),
                "name": row.composition_name,
                "is_seen": is_seen,
            })
    pairs.sort(key=lambda p: p["comp_id"])
    if len({p["comp_id"] for p in pairs}) != len(pairs):
        raise ValueError("a composition appears as both seen and unseen")
    return {"version": 1, "sources": sources, "pairs": pairs}


@dataclass
class Catalogue:
    comp_id: np.ndarray
    crop_id: np.ndarray
    disease_id: np.ndarray
    is_seen: np.ndarray
    names: list

    def __post_init__(self):
        self._index = {int(c): i for i, c in enumerate(self.comp_id)}

    def __len__(self):
        return len(self.comp_id)

    def index_of(self, comp_ids):
        """Catalogue index of each composition ID. Raises KeyError if one is not in the catalogue."""
        return np.array([self._index[int(c)] for c in comp_ids], dtype=np.int64)


def load_catalogue(path):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    return Catalogue(
        comp_id=np.array([p["comp_id"] for p in pairs]),
        crop_id=np.array([p["crop_id"] for p in pairs]),
        disease_id=np.array([p["disease_id"] for p in pairs]),
        is_seen=np.array([p["is_seen"] for p in pairs]),
        names=[p["name"] for p in pairs],
    )
