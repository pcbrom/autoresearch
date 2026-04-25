"""Read-only: RMSE metric."""
from __future__ import annotations

import math
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
