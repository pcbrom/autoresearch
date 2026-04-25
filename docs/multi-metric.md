# Multi-metric optimization in `autoresearch`

The framework's hard contract is **one scalar per experiment**. That contract
is preserved when you have multiple metrics by computing a *scalarization* in
`evaluate.py` (read-only) and printing the resulting scalar. Nothing in the
framework changes.

This document covers the four standard scalarizations, their assumptions, and
when to use each. The companion example
[`examples/multi_metric_demo/`](../examples/multi_metric_demo/) implements
convex sum as the default and keeps the others as commented variants ready to
swap in.

---

## Setting

You have $k$ metrics $f_1, f_2, \dots, f_k$ produced by the runner (e.g.
`rmse`, `latency_ms`, `model_size_mb`). Each has a direction (lower or higher
is better). You need a single number to compare experiments.

Before scalarizing, **normalize**. Use min-max to a fixed reference budget, or
z-score against a calibrated baseline. Without normalization the weights are
uninterpretable and dominate scales drown out subtler ones.

```python
rmse_n  = rmse / 0.50          # baseline ~0.50, lower is better
lat_n   = latency_ms / 100.0   # budget ~100 ms,  lower is better
size_n  = model_size_mb / 50.0 # budget ~50 MB,   lower is better
```

After this every $\tilde f_i \in [0, \approx 2]$ with 1.0 being "on budget".

---

## 1. Convex sum (weighted linear combination)

```math
g(x) = \sum_{i=1}^k \lambda_i \, \tilde f_i(x), \quad \lambda_i \geq 0, \sum \lambda_i = 1
```

```python
def score(rmse_n, lat_n, size_n) -> float:
    return 0.6 * rmse_n + 0.3 * lat_n + 0.1 * size_n
```

**Use when**: metrics are commensurable post-normalization, you want **one
point** on the Pareto frontier, and you accept a fixed prior $\lambda$.

**Limitation**: only recovers points on the **convex hull** of the Pareto
frontier. For non-convex frontiers (typical when latency vs. accuracy has a
knee), no choice of $\lambda$ reaches the interior of the concave region.

---

## 2. Weighted Tchebycheff (Chebyshev)

```math
g(x; \lambda, z^*) = \max_{i} \, \lambda_i \cdot \big| \tilde f_i(x) - z_i^* \big|
```

where $z^*$ is an *ideal point* — the best each metric could achieve in
isolation (often estimated by single-objective baselines).

```python
def score(rmse_n, lat_n, size_n,
          z_star=(0.0, 0.0, 0.0),
          lam=(0.6, 0.3, 0.1)) -> float:
    devs = [lam[i] * abs(v - z_star[i])
            for i, v in enumerate((rmse_n, lat_n, size_n))]
    return max(devs)
```

**Use when**: the Pareto frontier is **non-convex** and you need to access
points the convex sum misses. Tchebycheff with all positive $\lambda$
recovers the entire Pareto frontier (under mild conditions).

**Cost**: needs a credible $z^*$. If $z^*$ is wildly wrong, the search
drifts. Estimate $z^*$ once, fix it for the session, document it.

---

## 3. Lexicographic ordering

Pick a strict priority of metrics. Compare experiments first by metric 1; only
if tied (within $\varepsilon_1$), break by metric 2; etc.

```python
def score(rmse, lat_ms, size_mb) -> float:
    # Encode lex order: dominant metric * 1e6 + secondary * 1e3 + tertiary
    # Works because the framework only sees a single number; small enough
    # secondary terms never override a larger primary delta.
    return rmse * 1e6 + lat_ms * 1e3 + size_mb
```

**Use when**: priorities are strict and you have orders-of-magnitude budget
between scales (e.g. accuracy is the hard goal, latency is a tiebreaker).

**Pitfall**: if the magnitudes are not separated enough, the encoding leaks —
a tiny accuracy loss can be hidden by a huge latency gain. Keep the
multipliers conservative.

---

## 4. Constrained scalarization (epsilon-constraint)

One metric is the objective; the others are hard constraints with thresholds.
If a constraint is violated, the experiment is rejected.

```python
def score(rmse, lat_ms, size_mb,
          lat_budget_ms=100.0, size_budget_mb=50.0) -> float:
    if lat_ms > lat_budget_ms or size_mb > size_budget_mb:
        return float("inf")    # framework treats inf as worst-possible
    return rmse
```

The framework's keep/discard logic naturally handles `inf` as "worse than any
finite value" (assuming `lower_is_better: true`). For symmetry under
`lower_is_better: false`, return `-inf` instead.

**Use when**: you have explicit operational ceilings (latency SLA, memory cap)
and only want to optimize one metric inside the feasible region.

**Cost**: the search loses information about the *amount* of constraint
violation. An idea that overshoots latency by 1ms scores the same as one that
overshoots by 1000ms — the critic cannot tell which is closer to viable. To
keep that signal, use a soft constraint:

```python
def score(rmse, lat_ms, size_mb,
          lat_budget_ms=100.0, size_budget_mb=50.0,
          penalty=10.0) -> float:
    over_lat  = max(0.0, lat_ms - lat_budget_ms) / lat_budget_ms
    over_size = max(0.0, size_mb - size_budget_mb) / size_budget_mb
    return rmse + penalty * (over_lat + over_size)
```

---

## Choosing

| Frontier shape | Strict priorities | Hard SLA | Pick |
|----------------|-------------------|----------|------|
| convex            | no  | no  | **convex sum** |
| non-convex        | no  | no  | **Tchebycheff** |
| any               | yes | no  | **lexicographic** |
| any               | no  | yes | **constrained** (hard or soft) |

If unsure: start with **convex sum + normalization**. It is the simplest, has
the fewest free parameters, and reveals quickly whether the frontier is convex
(by inspecting the running-best trajectory in `AUDIT_LOG.md`).

---

## Where to encode

Always inside `evaluate.py` (read-only). Reasons:

1. The framework's "metric is fixed for the session" invariant requires the
   scoring function to be pinned at session start. `evaluate.py` is read-only
   precisely to enforce this.
2. The scalarization parameters ($\lambda$, $z^*$, budgets) are research
   decisions, not optimization decisions. Versioning them in
   `evaluate.py` makes them auditable: changing a weight is a new
   `evaluate.py` and a new git branch / new session.
3. The Gemma critic sees only the final `score:` value. Hiding the
   decomposition keeps it focused on the goal you actually want to optimize.

---

## When to escalate beyond scalarization

If you genuinely need the **whole Pareto frontier** (not a point on it), the
single-metric framework is the wrong shape. Run multiple sessions in parallel
with different scalarization weights, then merge their `results.tsv` and
filter for non-dominated points externally. The framework does not orchestrate
this, but the per-session artifacts (`results.tsv`, `AUDIT_LOG.json`) are
exactly the right inputs for a downstream Pareto-frontier extraction script.
