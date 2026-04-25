"""Regenerate STATE.md — navigable snapshot of current loop state."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .helpers import load_yaml, problem_path, project_root, read_tsv


def _git(args: list[str], cwd: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(cwd)] + args,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _running_best(rows: list[dict], metric: str, lower: bool):
    best = None
    best_row = None
    for r in rows:
        if r.get("status", "").strip().lower() != "keep":
            continue
        try:
            v = float(r.get(metric, ""))
        except (ValueError, TypeError):
            continue
        if best is None or (lower and v < best) or (not lower and v > best):
            best, best_row = v, r
    return best, best_row


def _baseline(rows: list[dict], metric: str) -> str:
    if not rows:
        return "n/a"
    try:
        return f"{float(rows[0].get(metric, 'nan')):.6f}"
    except (ValueError, TypeError):
        return rows[0].get(metric, "n/a")


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem).resolve()
    problem = load_yaml(pp)
    project = pp.parent

    cfg = problem.get("state", {})
    if not cfg.get("enabled", False):
        sys.exit("state.enabled=false in problem.yaml")

    out_path = project / "STATE.md"
    throttle = int(cfg.get("throttle_seconds", 300))
    if not args.force and out_path.exists():
        if (time.time() - out_path.stat().st_mtime) < throttle:
            print(f"[estado] skipped (mtime < {throttle}s)")
            return

    metric = problem["metric_name"]
    lower = problem.get("lower_is_better", True)
    rows = read_tsv(project / "results.tsv")
    best, best_row = _running_best(rows, metric, lower)
    baseline = _baseline(rows, metric)

    next_idea_path = project / "next_idea.json"
    next_idea_text = next_idea_path.read_text(errors="replace")[:1500] if next_idea_path.exists() else ""

    branch = _git(["branch", "--show-current"], project)
    commit = _git(["rev-parse", "--short", "HEAD"], project)
    last_commits = _git(["log", "--oneline", "-n", "10"], project)

    crashes = [r for r in rows if r.get("status", "").strip().lower() == "crash"][-5:]
    last10 = rows[-10:]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# State — {project.name}
_Generated at: {now}_

## Overview
- objective: {problem.get('objective','?')}
- metric: `{metric}` ({'lower is better' if lower else 'higher is better'})
- branch: `{branch}` @ `{commit}`
- total experiments: {len(rows)}
- baseline: {baseline}
- running-best: {f"{best:.6f}" if best is not None else 'n/a'}
{f"- best commit: `{best_row.get('commit','?')}` ({best_row.get('description','')[:80]})" if best_row else ""}

## Loop status
- heartbeat: `cat ~/.cache/autoresearch/loop.heartbeat` should show a recent timestamp

## Next idea (proposed by Gemma critic)
```json
{next_idea_text or '(none yet)'}
```

## Last 10 experiments
| commit | {metric} | resource | status | description |
|---|---|---|---|---|
"""
    for r in last10:
        md += f"| {r.get('commit','?')} | {r.get(metric,'?')} | {r.get('resource','?')} | {r.get('status','?')} | {r.get('description','')[:60]} |\n"

    md += "\n## Recent crashes\n"
    if crashes:
        for r in crashes:
            md += f"- `{r.get('commit','?')}` — {r.get('description','')[:80]}\n"
    else:
        md += "_none_\n"

    md += f"\n## Recent git commits\n```\n{last_commits}\n```\n"

    out_path.write_text(md)
    print(f"[estado] wrote {out_path}")
