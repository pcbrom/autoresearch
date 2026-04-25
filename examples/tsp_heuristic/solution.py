"""solution.py — single mutable file.

Heuristic search for 50-city TSP. Baseline: random shuffle.
The agent should propose improvements (NN, 2-opt, SA, etc.).
Prints `tour_length: <float>` and `wall_s: <float>`.
"""
from __future__ import annotations

import time

import numpy as np

from prepare import get_instance
from evaluate import tour_length

SEED = 0


def search(coords: np.ndarray) -> list[int]:
    rng = np.random.default_rng(SEED)
    n = coords.shape[0]
    order = list(range(n))
    rng.shuffle(order)
    return order


def main() -> None:
    t0 = time.time()
    coords = get_instance()
    order = search(coords)
    length = tour_length(coords, order)
    wall = time.time() - t0
    print(f"tour_length: {length:.6f}")
    print(f"wall_s: {wall:.3f}")


if __name__ == "__main__":
    main()
