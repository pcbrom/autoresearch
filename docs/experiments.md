# Experiments

This note documents demonstrations run with the package itself. They are
illustrative worked examples, not statistical benchmarks: each uses a single
run per arm and the critic is sampled at temperature 0.7, so the proposal
trajectory differs between runs. The purpose is to show that the loop behaves
as specified end to end, not to rank configurations.

## Loop-driven coder versus external agent

The loop can apply each `next_idea.json` in two ways: an external agent (a
human-launched coding agent, the historical default) or, with `coder.enabled:
true`, an agent the loop invokes headless every iteration. This demonstration
compares the two on the same problem and budget.

### Setup

| Parameter | Value |
|-----------|-------|
| Example | `examples/tsp_heuristic` (50-city Euclidean TSP, fixed instance `SEED=1729`) |
| Metric | `tour_length` (lower is better), deterministic baseline 25.251112 |
| Critic | Ollama `gemma4:e2b`, `thinking: true`, temperature 0.7 |
| Budget | baseline + 3 agent iterations per arm |
| External arm | `coder.enabled: false`; each proposal applied by hand |
| Autonomous arm | `coder.enabled: true`, `agent: claude`, `permission: acceptEdits` |

Both arms share the cached instance, so metrics are comparable on the
instance. The instance is fixed; the critic's idea stream is not.

### Results

| | External (hand-applied) | Autonomous (loop-driven `claude`) |
|---|---|---|
| Best `tour_length` | 5.694621 | 5.625659 |
| Improvement vs baseline | 77.45% | 77.72% |
| keep / discard | 2 / 2 | 3 / 1 |
| Human edits | 3 | 0 |
| Trajectory | 2-opt (keep) → pure-swap SA (discard) → 2-opt-seeded SA (discard, tie) | 2-opt (keep) → SA (keep) → SA + more iterations (discard) |

Both arms started from the random-shuffle baseline (25.251112) and converged to
the 5.6 to 5.7 region, the expected quality for 2-opt and simulated annealing on
a 50-city instance. The 1.21% gap between the two best tours is within the noise
of the critic's stochastic sampling and carries no significance at a single run
per arm.

### Observations

The autonomous arm reproduced the full optimization trajectory without human
intervention: the loop invoked the agent, the agent applied the critic's
proposals (a complete 2-opt with a distance matrix, then simulated annealing),
the runner evaluated and classified each result, and `git reset --hard HEAD~1`
reverted the discarded proposals. The discard invariant held identically in both
arms: a regression to 21.369624 (external, pure-swap SA) and to 5.738031
(autonomous, an over-iterated SA variant) were both reverted to the running
best.

The measurable difference between the arms is operational rather than
qualitative. The external arm required applying each proposal by hand; the
autonomous arm delivered an equivalent trajectory unattended.

### Reproduction

```bash
pip install -e ".[tsp]"
autoresearch init --problem examples/tsp_heuristic/problem.yaml --target /tmp/tsp_auto
# enable the coder block in /tmp/tsp_auto/problem.yaml (agent: claude), then
autoresearch wizard next        # repeat until confirm_loop
echo "$(date -Iseconds)" > /tmp/tsp_auto/.autoresearch/loop_confirmed
autoresearch loop --problem /tmp/tsp_auto/problem.yaml
```

For the external arm, leave `coder.enabled: false` and apply each
`next_idea.json` to `solution.py` between `autoresearch run` and `autoresearch
critic` calls.

## Convergence on the shipped examples

`docs/convergence.png` plots the best-so-far trajectory for the three shipped
examples (TSP heuristic, XGBoost tuning, multi-metric scalarization), with kept
proposals in green and discarded ones in red. The data is extracted directly
from each `examples/*/sample_run/results.tsv`.
