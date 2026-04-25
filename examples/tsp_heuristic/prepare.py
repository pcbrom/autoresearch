"""Read-only: fixed 50-city Euclidean TSP instance."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "autoresearch" / "tsp"
INSTANCE_PATH = CACHE / "tsp50.pkl"
N_CITIES = 50
SEED = 1729


def get_instance() -> np.ndarray:
    if INSTANCE_PATH.exists():
        with open(INSTANCE_PATH, "rb") as f:
            return pickle.load(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    coords = rng.uniform(0.0, 1.0, size=(N_CITIES, 2)).astype(np.float64)
    with open(INSTANCE_PATH, "wb") as f:
        pickle.dump(coords, f)
    return coords


if __name__ == "__main__":
    coords = get_instance()
    print(f"cities={coords.shape[0]} bbox=({coords.min():.3f}, {coords.max():.3f})")
