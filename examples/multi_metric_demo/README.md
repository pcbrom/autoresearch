# Example: multi-metric optimization via scalarization

Demonstrates the framework's pattern for multiple metrics: collapse them into
a single `score` inside `evaluate.py` (read-only), preserving the
"single-metric" invariant. Nothing in `autoresearch` itself changes.

The runner trains a `GradientBoostingRegressor` on California Housing and
emits three raw metrics + the collapsed score:

```
rmse:          0.516413
latency_ms:    0.241752
model_size_mb: 0.0418
score:         0.766678
wall_s:        4.21
```

Only `score` is classified by the framework. The agent edits hyperparameters
to balance the three concerns.

## The four scalarizers

`evaluate.py` ships **four** scalarizers; only one is active per session
(this is the framework's invariant — see [docs/multi-metric.md](../../docs/multi-metric.md)).

| Scalarizer | Use when | Trade-off |
|------------|----------|-----------|
| `convex_sum` (default) | metrics commensurable, want one Pareto point | only convex hull of frontier |
| `tchebycheff` | non-convex frontier | needs ideal point z* |
| `lexicographic` | strict priority order | budgets must allow magnitude separation |
| `constrained_soft` | hard SLAs (latency cap, memory cap) | linear penalty past cap |

Switching scalarizer = edit ONE line in `evaluate.py`:

```python
def score(rmse, latency_ms, size_mb) -> float:
    return convex_sum(rmse, latency_ms, size_mb)        # default
    # return tchebycheff(rmse, latency_ms, size_mb)
    # return lexicographic(rmse, latency_ms, size_mb)
    # return constrained_soft(rmse, latency_ms, size_mb)
```

Because `evaluate.py` is read-only, this is a **session-level decision**: pick
the scalarizer, start a new branch, run the loop. The Gemma critic only sees
the final `score` and never confuses itself with implicit trade-offs.

## Requirements

```bash
pip install scikit-learn numpy pyyaml openai
ollama pull gemma3n:e2b
ollama serve   # in another terminal
```

## Run

```bash
TARGET=/tmp/multimetric_run
EXAMPLE=$(python3 -c "import autoresearch, pathlib; print(pathlib.Path(autoresearch.__file__).parent.parent / 'examples/multi_metric_demo')")

autoresearch init --problem $EXAMPLE/problem.yaml --target $TARGET --tag run1
cp $EXAMPLE/{prepare,evaluate,solution}.py $TARGET/
cd $TARGET && git add -A && git commit -q -m "scaffold multimetric"

export AUTORESEARCH_PROJECT=$TARGET
export AUTORESEARCH_PROBLEM=$TARGET/problem.yaml

autoresearch wizard next   # repeat until 'all_done'
echo "$(date -Iseconds)" > $TARGET/.autoresearch/loop_confirmed
autoresearch wizard step confirm_loop

autoresearch loop
```

## Why the scalarization lives in `evaluate.py`

1. **Comparability invariant**: every experiment in a session uses the same
   scoring function. `evaluate.py` being read-only enforces this; the agent
   cannot quietly retune the weights mid-session.
2. **Auditability**: changing the weights creates a new branch / new
   `evaluate.py`, fully visible in git history.
3. **Critic focus**: Gemma sees the final scalar and chooses hyperparameters
   that move it. It never gets confused trying to weigh `rmse` against
   `latency_ms` itself.

## Beyond a single Pareto point

If you need the **full Pareto frontier** (not one operating point), run several
sessions in parallel with different scalarizer weights and merge their
`results.tsv` externally. The framework gives you exactly the right inputs
for a downstream Pareto extraction; orchestrating multiple parallel sessions
is intentionally out of scope.

See [`docs/multi-metric.md`](../../docs/multi-metric.md) for the theory and
selection guide.

## Reference run (real execution, see [sample_run/](sample_run/))

| iter | RMSE | latency_ms | size_MB | score | status |
|------|------|------------|---------|-------|--------|
| 0 | 0.5422 | 0.000751 | 0.13 | 0.6532 | KEEP (baseline) |
| 1 | 0.5872 | 0.000515 | 0.075 | 0.7061 | discard (RMSE↑ swamped latency↓ + size↓) |
| 2 | 0.5093 | 0.000996 | 0.235 | 0.6159 | **KEEP** (depth↑ won despite size↑) |

The Gemma critic reasoned about the trade-off explicitly: it first proposed
shrinking the model (iter 1, lost on RMSE), then escalated capacity (iter 2,
won overall). [`sample_run/AUDIT_LOG.md`](sample_run/AUDIT_LOG.md) shows the
full reasoning chain attached to each iteration.

[`sample_run/`](sample_run/) artifacts:

- [`AUDIT_LOG.md`](sample_run/AUDIT_LOG.md) and [`AUDIT_LOG.json`](sample_run/AUDIT_LOG.json)
- [`results.tsv`](sample_run/results.tsv), [`STATE.md`](sample_run/STATE.md), [`wizard_state.json`](sample_run/wizard_state.json)
- [`critic_logs/`](sample_run/critic_logs/) — per-call critic prompts + responses
