"""Read-only: tour length metric."""
from __future__ import annotations

import numpy as np


def tour_length(coords: np.ndarray, order: list[int]) -> float:
    pts = coords[order]
    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
