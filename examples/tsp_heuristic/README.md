# Example: 50-city TSP heuristic search

Validates `autoresearch` outside ML. Random shuffle baseline (~25), optimum near
~5.7. The agent should discover nearest-neighbor, 2-opt, simulated annealing,
multi-start, or Or-opt refinements.

## Requirements

```bash
pip install numpy pyyaml openai
ollama pull gemma3n:e2b
ollama serve   # in another terminal
```

## Run

```bash
TARGET=/tmp/tsp_run
EXAMPLE=$(python3 -c "import autoresearch, pathlib; print(pathlib.Path(autoresearch.__file__).parent.parent / 'examples/tsp_heuristic')")

autoresearch init --problem $EXAMPLE/problem.yaml --target $TARGET --tag run1
cp $EXAMPLE/{prepare,evaluate,solution}.py $TARGET/
cd $TARGET && git add -A && git commit -q -m "scaffold tsp"

export AUTORESEARCH_PROJECT=$TARGET
export AUTORESEARCH_PROBLEM=$TARGET/problem.yaml

autoresearch wizard next   # repeat until 'all_done'
echo "$(date -Iseconds)" > $TARGET/.autoresearch/loop_confirmed
autoresearch wizard step confirm_loop

autoresearch loop          # NEVER STOP — Ctrl+C to interrupt
```

## Harvest

```bash
autoresearch analyze
autoresearch audit         # writes AUDIT_LOG.md and AUDIT_LOG.json
```

## Reference run (real execution, see [sample_run/](sample_run/))

| iter | tour_length | status | proposal |
|------|-------------|--------|----------|
| 0 | 25.25 | KEEP | random shuffle (baseline) |
| 1 | 5.97 | KEEP | 2-opt on random |
| 2 | 5.89 | KEEP | nearest-neighbor + 2-opt |
| 3 | 5.89 | discard | simulated annealing on top |

77% reduction from baseline in 3 keeps.

[`sample_run/`](sample_run/) contains the full execution artifacts:

- [`AUDIT_LOG.md`](sample_run/AUDIT_LOG.md) — navigable timeline with the Gemma critic reasoning attached to each iteration
- [`AUDIT_LOG.json`](sample_run/AUDIT_LOG.json) — same data, machine-readable
- [`results.tsv`](sample_run/results.tsv) — raw experiment log
- [`STATE.md`](sample_run/STATE.md) — final snapshot
- [`wizard_state.json`](sample_run/wizard_state.json) — preconditions log
- [`critic_logs/`](sample_run/critic_logs/) — one `.jsonl` per critic call with full prompt + raw Ollama response + parsed JSON + thinking blocks (~13–20 KB each)
