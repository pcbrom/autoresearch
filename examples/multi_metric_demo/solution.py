"""solution.py — single mutable file.

Trains a regressor on California Housing and emits THREE raw metrics
(rmse, latency_ms, model_size_mb) plus the SINGLE `score` that evaluate.score
collapses them into. The agent edits hyperparameters to balance the trio.

Stdout MUST contain a line `score: <float>` (the only one the framework
classifies). The other lines are informational.
"""
from __future__ import annotations

import io
import pickle
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from prepare import get_split
from evaluate import score

# --- Hyperparameters (agent edits these) ---
N_ESTIMATORS = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1
SUBSAMPLE = 1.0
MIN_SAMPLES_LEAF = 1
SEED = 42

# --- Latency measurement ---
LATENCY_REPEATS = 50    # how many predict() runs to average


def _measure_latency(model, X_test) -> float:
    n = X_test.shape[0]
    # Warm-up
    model.predict(X_test[:1])
    t0 = time.time()
    for _ in range(LATENCY_REPEATS):
        model.predict(X_test)
    elapsed = time.time() - t0
    return (elapsed / LATENCY_REPEATS) / n * 1e6   # microseconds per sample
    # interpreted as latency_ms when divided by 1000 below


def _measure_size(model) -> float:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.tell() / (1024 * 1024)   # MB


def main() -> None:
    t0 = time.time()
    X_train, X_test, y_train, y_test = get_split()

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    latency_us_per_sample = _measure_latency(model, X_test)
    latency_ms = latency_us_per_sample / 1000.0  # whole-batch latency in ms
    size_mb = _measure_size(model)
    final_score = score(rmse, latency_ms, size_mb)

    wall = time.time() - t0
    # Informational raw metrics
    print(f"rmse:          {rmse:.6f}")
    print(f"latency_ms:    {latency_ms:.6f}")
    print(f"model_size_mb: {size_mb:.4f}")
    # The ONE the framework classifies on
    print(f"score:         {final_score:.6f}")
    print(f"wall_s:        {wall:.2f}")


if __name__ == "__main__":
    main()
