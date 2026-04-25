"""Read-only: California Housing split."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

CACHE = Path.home() / ".cache" / "autoresearch" / "xgb"
SPLIT_PATH = CACHE / "split.pkl"
SEED = 42


def get_split():
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH, "rb") as f:
            return pickle.load(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED,
    )
    split = (X_train.astype(np.float32), X_test.astype(np.float32),
             y_train.astype(np.float32), y_test.astype(np.float32))
    with open(SPLIT_PATH, "wb") as f:
        pickle.dump(split, f)
    return split


if __name__ == "__main__":
    X_train, X_test, _, _ = get_split()
    print(f"train={X_train.shape} test={X_test.shape}")
