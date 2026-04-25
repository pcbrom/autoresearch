"""Single experiment iteration: commit → run → classify → keep/discard/crash."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .helpers import (
    classify, gc_all, get_dotted, load_yaml, problem_path, project_root,
    read_tsv, running_best,
)


def _git(args: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd)] + args,
                          capture_output=capture, text=True, timeout=30)


def _extract(text: str, pattern: str) -> str:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1) if m else ""


def _transform_resource(value: str, transform: str) -> str:
    if not value:
        return "0.0"
    try:
        v = float(value)
    except ValueError:
        return "0.0"
    if transform == "divide_by_1024":
        return f"{v / 1024:.1f}"
    if transform == "divide_by_1000":
        return f"{v / 1000:.1f}"
    return value


def _emit(payload: dict) -> None:
    print(json.dumps(payload))


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem).resolve()
    problem = load_yaml(pp)
    project = pp.parent

    metric = problem["metric_name"]
    metric_re = problem["metric_regex"]
    lower = bool(problem.get("lower_is_better", True))
    runner_cmd = problem["runner"]
    mutable_file = problem["mutable_file"]
    hard_timeout = int(problem["hard_timeout_s"])
    resource_re = get_dotted(problem, "resource_metric_regex", "")
    resource_tx = get_dotted(problem, "resource_metric_transform", "identity")

    results_tsv = project / "results.tsv"
    logs_dir = project / ".autoresearch" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = project / "run.log"

    # GC: stop Ollama if downtime mode, rotate old logs.
    if get_dotted(problem, "gemma_critic.when", "always") == "downtime":
        try:
            subprocess.run(
                ["ollama", "stop", get_dotted(problem, "gemma_critic.model", "gemma3n:e2b")],
                capture_output=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    rotated = sorted(logs_dir.glob("run-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in rotated[20:]:
        old.unlink(missing_ok=True)

    # Description from next_idea.json (if present), else baseline.
    next_idea = project / "next_idea.json"
    description = "baseline"
    if next_idea.exists():
        try:
            description = json.loads(next_idea.read_text())["hypothesis"][:120]
        except (KeyError, json.JSONDecodeError):
            pass

    # Stage and commit pending edits.
    _git(["add", mutable_file], project)
    diff = _git(["diff", "--cached", "--quiet"], project)
    has_changes = diff.returncode != 0
    if has_changes:
        _git(["commit", "-m", description], project, capture=True)
    elif results_tsv.exists() and len(results_tsv.read_text().splitlines()) > 1:
        # No edits since last commit → wait for the agent to apply next_idea.
        _emit({
            "commit": "", "description": "noop", "metric_name": metric,
            "metric_value": None, "resource_value": "0.0", "status": "noop",
            "rc": 0, "elapsed_s": 0, "running_best": None,
        })
        return

    commit_short = _git(["rev-parse", "--short", "HEAD"], project, capture=True).stdout.strip()

    # Run with hard timeout.
    print(f"[{time.strftime('%H:%M:%S')}] commit={commit_short} description={description}", file=sys.stderr)
    t0 = time.time()
    proc = subprocess.run(
        ["timeout", "--signal=KILL", f"{hard_timeout}s", "bash", "-c", runner_cmd],
        cwd=project, capture_output=True, text=True,
    )
    elapsed = int(time.time() - t0)
    run_log.write_text((proc.stdout or "") + (proc.stderr or ""))
    rc = proc.returncode

    log_text = run_log.read_text(errors="replace")
    metric_value = _extract(log_text, metric_re)
    resource_raw = _extract(log_text, resource_re) if resource_re else ""
    resource_value = _transform_resource(resource_raw, resource_tx)

    rows = read_tsv(results_tsv)
    best = running_best(rows, metric, lower)
    status = classify(metric_value, best, lower)
    if rc in (124, 137) or not metric_value:
        status = "crash"

    if not results_tsv.exists():
        results_tsv.write_text(f"commit\t{metric}\tresource\tstatus\tdescription\n")

    safe_desc = description.replace("\t", " ").replace("\n", " ")[:200]
    metric_log = metric_value or "0.000000"
    with open(results_tsv, "a", encoding="utf-8") as f:
        f.write(f"{commit_short}\t{metric_log}\t{resource_value}\t{status}\t{safe_desc}\n")

    if status == "keep":
        print(f"[{time.strftime('%H:%M:%S')}] KEEP {metric}={metric_value} (was {best})", file=sys.stderr)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {status} {metric}={metric_value or 'NA'} (best {best})", file=sys.stderr)
        last_subj = _git(["log", "-1", "--pretty=%s"], project, capture=True).stdout.strip()
        if last_subj == description:
            _git(["reset", "--hard", "HEAD~1"], project, capture=True)

    # Rotate the run log into per-commit archive.
    archive = logs_dir / f"run-{commit_short}.log"
    try:
        archive.write_text(log_text)
    except OSError:
        pass

    gc_all()

    _emit({
        "commit": commit_short,
        "description": safe_desc,
        "metric_name": metric,
        "metric_value": metric_value or None,
        "resource_value": resource_value,
        "status": status,
        "rc": rc,
        "elapsed_s": elapsed,
        "running_best": str(best) if best is not None else None,
    })
