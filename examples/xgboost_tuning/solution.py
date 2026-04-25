"""solution.py — single mutable file.

XGBoost regressor on California Housing.
Prints `rmse: <float>` and `wall_s: <float>`.
"""
from __future__ import annotations

import time

import xgboost as xgb

from prepare import get_split
from evaluate import rmse

# --- Hyperparameters (agent edits these) ---
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
MIN_CHILD_WEIGHT = 1
SEED = 42


def main() -> None:
    t0 = time.time()
    X_train, X_test, y_train, y_test = get_split()
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        reg_alpha=REG_ALPHA,
        min_child_weight=MIN_CHILD_WEIGHT,
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = rmse(y_test, pred)
    wall = time.time() - t0
    print(f"rmse: {score:.6f}")
    print(f"wall_s: {wall:.2f}")


if __name__ == "__main__":
    main()
