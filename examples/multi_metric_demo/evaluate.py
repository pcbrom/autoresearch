"""Read-only: scalarization of three metrics into a single `score`.

The framework's invariant is one scalar per experiment. To honor it under
multi-metric optimization, we collapse (rmse, latency_ms, model_size_mb) into
a single `score` here. The active scalarizer is `convex_sum`. The other three
(`tchebycheff`, `lexicographic`, `constrained_soft`) are kept as alternatives
ready to swap in: pick one and reroute `score(...)` to it. Only ONE active
scalarizer per session — that is the invariant the framework relies on.

See docs/multi-metric.md for the theory and selection guide.
"""
from __future__ import annotations

import math
from typing import Sequence

# --- Reference budgets (used to normalize each metric) ---
RMSE_BUDGET = 0.50          # baseline RMSE on California Housing
LATENCY_BUDGET_MS = 50.0    # target inference latency
SIZE_BUDGET_MB = 5.0        # target model footprint

# --- Convex combination weights (sum to 1) ---
LAMBDAS = (0.6, 0.3, 0.1)   # rmse, latency, size


def _normalize(rmse: float, latency_ms: float, size_mb: float) -> tuple[float, float, float]:
    return (rmse / RMSE_BUDGET,
            latency_ms / LATENCY_BUDGET_MS,
            size_mb / SIZE_BUDGET_MB)


# ============================================================================
# Active scalarizer (this is what `score(...)` calls below)
# ============================================================================

def convex_sum(rmse: float, latency_ms: float, size_mb: float) -> float:
    """Weighted linear combination of normalized metrics.

    Pros: simplest; one free parameter (the weight vector).
    Cons: only recovers points on the convex hull of the Pareto frontier.
    """
    rmse_n, lat_n, size_n = _normalize(rmse, latency_ms, size_mb)
    return LAMBDAS[0] * rmse_n + LAMBDAS[1] * lat_n + LAMBDAS[2] * size_n


# ============================================================================
# Alternative scalarizers (commented usage). Pick ONE per session.
# ============================================================================

def tchebycheff(rmse: float, latency_ms: float, size_mb: float,
                z_star: Sequence[float] = (0.0, 0.0, 0.0)) -> float:
    """Weighted Chebyshev distance from an ideal point z*.

    Use when the Pareto frontier is non-convex. With all positive weights and
    a credible z*, it can recover points the convex sum misses.
    """
    rmse_n, lat_n, size_n = _normalize(rmse, latency_ms, size_mb)
    devs = (LAMBDAS[0] * abs(rmse_n - z_star[0]),
            LAMBDAS[1] * abs(lat_n  - z_star[1]),
            LAMBDAS[2] * abs(size_n - z_star[2]))
    return max(devs)


def lexicographic(rmse: float, latency_ms: float, size_mb: float) -> float:
    """Strict priority encoding via separated magnitudes.

    Primary: rmse. Secondary: latency. Tertiary: size. Use only when the
    operational budgets allow strict ordering (small secondary deltas must
    not be able to override a primary improvement).
    """
    return rmse * 1e6 + latency_ms * 1e3 + size_mb


def constrained_soft(rmse: float, latency_ms: float, size_mb: float,
                     latency_cap_ms: float = 100.0,
                     size_cap_mb: float = 50.0,
                     penalty: float = 10.0) -> float:
    """Optimize rmse subject to soft latency and size constraints.

    Penalty grows linearly past the cap. Returns +inf only if the runner
    crashed (NaN); soft penalty preserves gradient signal for the critic.
    """
    if not (math.isfinite(rmse) and math.isfinite(latency_ms) and math.isfinite(size_mb)):
        return float("inf")
    over_lat  = max(0.0, latency_ms - latency_cap_ms) / latency_cap_ms
    over_size = max(0.0, size_mb    - size_cap_mb)    / size_cap_mb
    return rmse + penalty * (over_lat + over_size)


# ============================================================================
# Public entry point
# ============================================================================

def score(rmse: float, latency_ms: float, size_mb: float) -> float:
    """Single-scalar score the runner prints. Active scalarizer: convex_sum.

    To switch: replace the body with `return tchebycheff(...)`,
    `return lexicographic(...)`, or `return constrained_soft(...)`.
    """
    return convex_sum(rmse, latency_ms, size_mb)
