from collections import deque

import torch


class PairMemory:
    """Ring buffer of the last `per_pair` backbone features for each (crop, disease) pair."""

    def __init__(self, per_pair):
        if per_pair < 1:
            raise ValueError(f"per_pair must be at least 1, got {per_pair}")
        self.per_pair = per_pair
        self.rows = {}

    def update(self, h, crop, disease):
        h = h.detach().float()
        for feature, c, d in zip(h, crop.tolist(), disease.tolist()):
            self.rows.setdefault((c, d), deque(maxlen=self.per_pair)).append(feature.clone())

    def tensors(self):
        """(h [M, D], crop [M], disease [M]), or None while empty."""
        if not self.rows:
            return None
        keys = [key for key, ring in self.rows.items() for _ in ring]
        h = torch.stack([feature for ring in self.rows.values() for feature in ring])
        crop, disease = (torch.tensor(column, device=h.device) for column in zip(*keys))
        return h, crop, disease
