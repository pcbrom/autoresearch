# Installation

End-to-end install in dependency order. Skip optional steps you do not need.

## 1. System prerequisites

| Requirement | Minimum | Verified with |
|-------------|---------|---------------|
| OS | Linux x86_64 (also runs on macOS for the CPU examples) | `uname -a` |
| Python | 3.10+ | `python3 --version` |
| git | 2.25+ | `git --version` |
| curl | any modern | `curl --version` |

Optional (for GPU runners and the wizard's VRAM check):

| Requirement | Used by | Verified with |
|-------------|---------|---------------|
| NVIDIA driver + CUDA | runner workloads using PyTorch | `nvidia-smi` |
| `gh` CLI | optional `gh_auth` step (skipped automatically if missing) | `gh --version` |

Debian/Ubuntu one-liner:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl
```

## 2. Ollama (local LLM critic backend)

The default critic runs the **Gemma 3n e2b** model via [Ollama](https://ollama.com).
You can swap to OpenAI / vLLM / llama.cpp without reinstalling — see the
"Swapping the critic" section in the [README](README.md). To use the default:

```bash
# 2.1 install ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2.2 start the server (leave running in another terminal or as a service)
ollama serve            # foreground; or use systemctl on Linux

# 2.3 pull the default critic model (~7 GB download, ~7 GB VRAM at runtime)
ollama pull gemma4:e2b

# 2.4 sanity check
ollama list             # should show gemma4:e2b
curl -s http://localhost:11434/api/version
```

Alternative larger models (work without changing the rest of the install):

```bash
ollama pull gemma3:12b   # ~7 GB VRAM, sharper reasoning
ollama pull gemma3:27b   # ~15 GB VRAM, MoE, best quality
ollama pull qwen2.5:7b   # alternative family
```

Update `gemma_critic.model` in your `problem.yaml` to use them.

## 3. Python environment

Recommended: an isolated virtualenv per project.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Or use `uv` if you have it:

```bash
uv venv && source .venv/bin/activate
```

## 4. Install `autoresearch`

### From a local checkout (development install)

```bash
cd /path/to/autoresearch
pip install -e .
```

### Required Python deps (auto-resolved by `pip install -e .`)

| Package | Why |
|---------|-----|
| `pyyaml` | parse `problem.yaml` |
| `openai` | OpenAI-compatible client (talks to Ollama / vLLM / OpenAI / proxies) |

### Optional Python deps (install on demand)

| Extra | Install | Used by |
|-------|---------|---------|
| `torch` | `pip install -e ".[torch]"` | runtime CUDA cache GC between iterations |
| `xgb`   | `pip install -e ".[xgb]"`   | the `examples/xgboost_tuning` example |
| `tsp`   | `pip install -e ".[tsp]"`   | the `examples/tsp_heuristic` example |
| `all`   | `pip install -e ".[all]"`   | everything above |

Without `torch`, the loop still runs — it simply skips the CUDA cache flush.

## 5. Verify installation

```bash
which autoresearch                      # /path/to/.venv/bin/autoresearch
autoresearch --help                     # lists 8 subcommands
```

Should print:

```
usage: autoresearch [-h] <command> ...

Autonomous research loop with a local LLM critic.

positional arguments:
  <command>
    init      scaffold a new project from a problem.yaml
    wizard    stepwise preflight validator
    run       run a single experiment iteration
    critic    propose the next idea via Ollama
    loop      run the NEVER-STOP autonomous loop
    analyze   summary of results.tsv
    audit     consolidated timeline (AUDIT_LOG.md + .json)
    state     regenerate STATE.md snapshot
```

## 6. Smoke test (TSP example, no GPU needed)

```bash
pip install -e ".[tsp]"

TARGET=/tmp/tsp_smoke
EXAMPLE=$(python3 -c "import autoresearch, pathlib; print(pathlib.Path(autoresearch.__file__).parent.parent / 'examples/tsp_heuristic')")

autoresearch init --problem $EXAMPLE/problem.yaml --target $TARGET --tag smoke
cp $EXAMPLE/{prepare,evaluate,solution}.py $TARGET/
cd $TARGET && git add -A && git commit -q -m "scaffold"

export AUTORESEARCH_PROJECT=$TARGET
export AUTORESEARCH_PROBLEM=$TARGET/problem.yaml

# Wizard validates 9 preconditions
for step in repo_git tools_present problem_yaml ollama_model baseline_smoke critic_dry_run cleanup_check; do
    autoresearch wizard step $step
done
echo "$(date -Iseconds)" > $TARGET/.autoresearch/loop_confirmed
autoresearch wizard step confirm_loop

# Single baseline run + one critic call
autoresearch run
autoresearch critic
cat $TARGET/next_idea.json
```

If the last `cat` prints a JSON with `thought_process`, `hypothesis`,
`code_pseudocode` and friends — installation is complete.

## 7. (Optional) GPU runners with PyTorch

Only needed if your `runner` (the script `autoresearch run` invokes) uses
CUDA. Install the matching PyTorch wheel for your CUDA:

```bash
# CUDA 12.x (consult https://pytorch.org for the right URL for your CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 8. Uninstall

```bash
pip uninstall autoresearch
rm -rf ~/.cache/autoresearch          # loop pidfile + heartbeat
rm -rf <project_dir>/.autoresearch    # wizard state, audit logs, critic logs
```

The Ollama install (`/usr/local/bin/ollama`, `~/.ollama/`) is not removed by
the steps above; remove it separately if you no longer need any local LLM.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `wizard step ollama_model` returns `fail: ollama server unreachable` | server not running | `ollama serve` in another terminal |
| `wizard step ollama_model` returns `fail: model <name> not pulled` | model missing | `ollama pull <name>` |
| `wizard step critic_dry_run` returns `non-JSON` | model does not honor JSON Schema | switch to `gemma3:12b` or larger; tiny 1B models often fail |
| `wizard step baseline_smoke` returns `metric_regex did not match` | runner does not print the metric on a line matching the regex | print exactly `{metric_name}: <float>` (or update `metric_regex`) |
| `loop refuses to start: confirm_loop not done` | gate file missing | `echo $(date -Iseconds) > <project>/.autoresearch/loop_confirmed` |
| `loop` shows `noop` repeatedly | the agent (you / Claude / Codex) has not edited `solution.py` since the last commit | apply the proposal in `next_idea.json` to `solution.py`, save |
