# Example: XGBoost hyperparameter tuning

CPU-only smoke test of the framework. California Housing dataset + XGBoost
regressor, ~1s per iteration. The agent should discover good combinations of
`n_estimators`, `max_depth`, `learning_rate`, and regularization.

## Requirements

```bash
pip install xgboost scikit-learn pyyaml openai
ollama pull gemma3n:e2b
ollama serve   # in another terminal
```

## Run

```bash
TARGET=/tmp/xgb_run
EXAMPLE=$(python3 -c "import autoresearch, pathlib; print(pathlib.Path(autoresearch.__file__).parent.parent / 'examples/xgboost_tuning')")

autoresearch init --problem $EXAMPLE/problem.yaml --target $TARGET --tag run1
cp $EXAMPLE/{prepare,evaluate,solution}.py $TARGET/
cd $TARGET && git add -A && git commit -q -m "scaffold xgb"

export AUTORESEARCH_PROJECT=$TARGET
export AUTORESEARCH_PROBLEM=$TARGET/problem.yaml

autoresearch wizard next   # repeat until 'all_done'
echo "$(date -Iseconds)" > $TARGET/.autoresearch/loop_confirmed
autoresearch wizard step confirm_loop

autoresearch loop
```

## Harvest

```bash
autoresearch analyze
autoresearch audit
```

## Reference run (real execution, see [sample_run/](sample_run/))

| iter | RMSE | status | proposal |
|------|------|--------|----------|
| 0 | 0.4639 | KEEP | baseline (n_est=200, depth=6, LR=0.1) |
| 1 | 0.4503 | KEEP | n_estimators 200 → 500 |
| 2 | 0.4561 | discard | LR 0.1 → 0.05 (without more estimators) |
| 3 | 0.4485 | KEEP | n_estimators 500 → 1000 |
| 4 | **0.4449** | KEEP | LR 0.1 → 0.08 (with n_est=1000) |
| 5 | 0.4481 | discard | LR 0.08 → 0.05 |

4.1% RMSE reduction from baseline in 4 keeps. Best config:
`n_estimators=1000, max_depth=6, learning_rate=0.08`.

[`sample_run/`](sample_run/) contains the full execution artifacts:

- [`AUDIT_LOG.md`](sample_run/AUDIT_LOG.md) — navigable timeline with the Gemma critic reasoning attached to each iteration
- [`AUDIT_LOG.json`](sample_run/AUDIT_LOG.json) — same data, machine-readable
- [`results.tsv`](sample_run/results.tsv) — raw experiment log
- [`STATE.md`](sample_run/STATE.md) — final snapshot
- [`wizard_state.json`](sample_run/wizard_state.json) — preconditions log
- [`critic_logs/`](sample_run/critic_logs/) — one `.jsonl` per critic call with full prompt + raw Ollama response + parsed JSON + thinking blocks
