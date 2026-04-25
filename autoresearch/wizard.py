"""Stepwise preflight validator. Each step returns structured JSON; the agent
consumer (Claude, Codex, etc.) only advances when the current step is `ok`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .helpers import load_yaml, problem_path, project_root

STATE_DIRNAME = ".autoresearch"
STATE_FILENAME = "wizard_state.json"

STEPS = [
    "repo_git",
    "tools_present",
    "problem_yaml",
    "ollama_model",
    "vram_budget",
    "baseline_smoke",
    "critic_dry_run",
    "cleanup_check",
    "confirm_loop",
]


def _state_path() -> Path:
    return project_root() / STATE_DIRNAME / STATE_FILENAME


def _load_state() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {"steps": {}, "started_at": time.time()}


def _save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, default=str))


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError as e:
        return 127, "", str(e)


def _load_problem_safe() -> dict:
    p = problem_path()
    if not p.exists():
        return {"_error": f"problem.yaml not found at {p}"}
    try:
        return load_yaml(p)
    except Exception as e:
        return {"_error": f"problem.yaml parse error: {e}"}


# -- step implementations ----------------------------------------------------

def step_repo_git() -> dict:
    root = project_root()
    rc, out, _ = _run(["git", "-C", str(root), "rev-parse", "--git-dir"])
    if rc == 0:
        _, branch, _ = _run(["git", "-C", str(root), "branch", "--show-current"])
        return {"status": "ok", "git_dir": out.strip(), "current_branch": branch.strip()}
    return {"status": "fail", "reason": "not a git repo",
            "fix": f"cd {root} && git init && git add -A && git commit -m 'initial'"}


def step_tools_present() -> dict:
    required = ["git", "python3", "ollama"]
    optional = ["uv", "nvidia-smi"]
    found, missing = {}, []
    for t in required:
        path = _which(t)
        if path:
            found[t] = path
        else:
            missing.append(t)
    opt = {t: _which(t) for t in optional}
    if missing:
        return {"status": "fail", "missing": missing, "found": found, "optional": opt,
                "fix": f"install: {' '.join(missing)}"}
    return {"status": "ok", "found": found, "optional": opt}


def step_problem_yaml() -> dict:
    prob = _load_problem_safe()
    if "_error" in prob:
        return {"status": "fail", "reason": prob["_error"],
                "fix": "copy templates/problem.yaml.template to ./problem.yaml and fill"}
    required = ["objective", "metric_name", "metric_regex", "lower_is_better",
                "time_budget_s", "hard_timeout_s", "mutable_file", "runner",
                "branch_prefix", "results_tsv_columns"]
    missing = [k for k in required if k not in prob]
    if missing:
        return {"status": "fail", "missing": missing, "fix": "fill missing keys in problem.yaml"}
    root = project_root()
    if not (root / prob["mutable_file"]).exists():
        return {"status": "fail", "reason": f"mutable_file not found: {prob['mutable_file']}"}
    for f in prob.get("readonly_files", []):
        if not (root / f).exists():
            return {"status": "fail", "reason": f"readonly_file not found: {f}"}
    return {"status": "ok", "problem_path": str(problem_path()),
            "objective": prob["objective"], "metric": prob["metric_name"]}


def step_ollama_model() -> dict:
    prob = _load_problem_safe()
    if "_error" in prob:
        return {"status": "blocked", "reason": "problem_yaml step failed first"}
    if not prob.get("gemma_critic", {}).get("enabled", False):
        return {"status": "skip", "reason": "gemma_critic.enabled=false"}
    model = prob["gemma_critic"].get("model", "gemma3n:e2b")
    if not _which("ollama"):
        return {"status": "fail", "reason": "ollama binary missing"}
    rc, out, _ = _run(["ollama", "list"], timeout=10)
    if rc != 0 or model not in out:
        return {"status": "fail", "reason": f"model {model} not pulled",
                "fix": f"ollama pull {model}"}
    rc2, out2, _ = _run(
        ["curl", "-sS", "-X", "POST", "http://localhost:11434/api/generate",
         "-d", json.dumps({"model": model, "prompt": "ping", "stream": False,
                           "options": {"num_predict": 1}})],
        timeout=20,
    )
    if rc2 != 0 or "response" not in (out2 or ""):
        return {"status": "fail", "reason": "ollama server unreachable",
                "fix": "ollama serve  # in another terminal"}
    return {"status": "ok", "model": model}


def step_vram_budget() -> dict:
    prob = _load_problem_safe()
    if not _which("nvidia-smi"):
        return {"status": "skip", "reason": "nvidia-smi missing (CPU-only is fine)"}
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                       "--format=csv,noheader,nounits"], timeout=5)
    if rc != 0 or not out.strip():
        return {"status": "fail", "reason": "nvidia-smi failed"}
    line = out.strip().splitlines()[0]
    free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
    free_gb, total_gb = free_mb / 1024, total_mb / 1024
    critic_gb = prob.get("gemma_critic", {}).get("max_vram_gb", 8) if isinstance(prob, dict) else 8
    if free_gb < critic_gb + 4:
        return {"status": "fail", "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1),
                "critic_reserve_gb": critic_gb,
                "fix": f"free VRAM (kill processes) until at least {critic_gb + 4} GB free"}
    return {"status": "ok", "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1),
            "critic_reserve_gb": critic_gb}


def step_baseline_smoke() -> dict:
    prob = _load_problem_safe()
    if "_error" in prob:
        return {"status": "blocked", "reason": "problem_yaml step failed first"}
    runner = prob["runner"]
    timeout_s = int(prob["hard_timeout_s"])
    log_path = project_root() / "run.log"
    rc, _, _ = _run(["bash", "-c", f"cd {project_root()} && timeout {timeout_s}s {runner} > {log_path} 2>&1"],
                    timeout=timeout_s + 30)
    if not log_path.exists():
        return {"status": "fail", "reason": "no run.log produced"}
    log_text = log_path.read_text(errors="replace")
    m = re.search(prob["metric_regex"], log_text, flags=re.MULTILINE)
    if not m:
        return {"status": "fail", "reason": "metric_regex did not match",
                "log_tail": log_text[-2000:],
                "fix": "verify regex and that runner prints the metric line"}
    return {"status": "ok", "metric_value": m.group(1), "rc": rc}


def step_critic_dry_run() -> dict:
    prob = _load_problem_safe()
    if not prob.get("gemma_critic", {}).get("enabled", False):
        return {"status": "skip", "reason": "gemma_critic disabled"}
    rc, out, err = _run(
        ["python3", "-m", "autoresearch.cli", "critic", "--dry-run",
         "--problem", str(problem_path())],
        timeout=120,
    )
    if rc != 0:
        return {"status": "fail", "reason": "critic dry-run failed", "stderr": err.strip()}
    try:
        parsed = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "fail", "reason": "critic returned non-JSON", "stdout": out[-500:]}
    required = {"hypothesis", "expected_delta", "justification", "code_pseudocode", "risk_level"}
    if not required.issubset(parsed.keys()):
        return {"status": "fail", "reason": "critic JSON missing keys",
                "missing": list(required - set(parsed.keys()))}
    return {"status": "ok", "sample_idea": parsed.get("hypothesis", "")[:100]}


def step_cleanup_check() -> dict:
    rc, out, _ = _run(["pgrep", "-fa", "ollama runner"], timeout=5)
    zombies = []
    if rc == 0 and out.strip():
        zombies = out.strip().splitlines()
    log_path = project_root() / "run.log"
    rotation_dir = project_root() / ".autoresearch" / "logs"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "ok",
            "ollama_runners": zombies,
            "advice": "the loop rotates run.log into .autoresearch/logs/ each iteration",
            "log_path": str(log_path),
            "rotation_dir": str(rotation_dir)}


def step_confirm_loop() -> dict:
    flag = project_root() / ".autoresearch" / "loop_confirmed"
    if flag.exists():
        return {"status": "ok", "confirmed_at": flag.read_text().strip()}
    return {"status": "fail",
            "reason": "loop not yet confirmed by user",
            "fix": f"create gate file: echo $(date -Iseconds) > {flag}"}


STEP_FNS = {
    "repo_git": step_repo_git,
    "tools_present": step_tools_present,
    "problem_yaml": step_problem_yaml,
    "ollama_model": step_ollama_model,
    "vram_budget": step_vram_budget,
    "baseline_smoke": step_baseline_smoke,
    "critic_dry_run": step_critic_dry_run,
    "cleanup_check": step_cleanup_check,
    "confirm_loop": step_confirm_loop,
}


def cmd_status(_args: argparse.Namespace) -> None:
    state = _load_state()
    snapshot = {"project_root": str(project_root()), "state_path": str(_state_path()), "steps": []}
    for name in STEPS:
        rec = state.get("steps", {}).get(name, {"status": "pending"})
        snapshot["steps"].append({"name": name, **rec})
    _emit(snapshot)


def cmd_step(args: argparse.Namespace) -> None:
    name = args.name
    if name not in STEP_FNS:
        _emit({"error": f"unknown step '{name}'", "valid": STEPS})
        sys.exit(2)
    result = STEP_FNS[name]()
    result["step"] = name
    result["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state = _load_state()
    state.setdefault("steps", {})[name] = result
    _save_state(state)
    _emit(result)
    if result.get("status") == "fail":
        sys.exit(1)


def cmd_next(_args: argparse.Namespace) -> None:
    state = _load_state()
    for name in STEPS:
        rec = state.get("steps", {}).get(name)
        if rec is None or rec.get("status") in (None, "pending", "fail"):
            cmd_step(argparse.Namespace(name=name))
            return
    _emit({"status": "all_done", "next_action": "run `autoresearch loop`"})


def cmd_reset(_args: argparse.Namespace) -> None:
    p = _state_path()
    if p.exists():
        p.unlink()
    flag = project_root() / ".autoresearch" / "loop_confirmed"
    if flag.exists():
        flag.unlink()
    _emit({"status": "reset"})
