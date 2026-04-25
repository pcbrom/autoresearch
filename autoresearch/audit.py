"""Consolidated timeline: wizard + iterations + critic reasoning + crashes.

Outputs:
  AUDIT_LOG.md   — markdown timeline (human-readable)
  AUDIT_LOG.json — same data, machine-readable
"""
from __future__ import annotations

import argparse
import csv
import json
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


def _wizard_events(project: Path) -> list[dict]:
    state_path = project / ".autoresearch" / "wizard_state.json"
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text())
    events = []
    for name, rec in state.get("steps", {}).items():
        events.append({
            "kind": "wizard", "ts": rec.get("ts"), "step": name,
            "status": rec.get("status"),
            "detail": {k: v for k, v in rec.items() if k not in {"ts", "status", "step"}},
        })
    return events


def _commit_meta(project: Path) -> dict[str, dict]:
    log = _git(["log", "--all", "--pretty=format:%h\t%aI\t%s"], project)
    out = {}
    for line in log.splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 3:
            out[parts[0]] = {"author_iso": parts[1], "subject": parts[2]}
    return out


def _critic_logs(project: Path) -> list[dict]:
    logs_dir = project / ".autoresearch" / "critic_logs"
    if not logs_dir.exists():
        return []
    items = []
    for fp in sorted(logs_dir.glob("*.jsonl")):
        try:
            data = json.loads(fp.read_text())
            data["_file"] = str(fp.relative_to(project))
            items.append(data)
        except json.JSONDecodeError:
            continue
    return items


def _crash_tail(project: Path, commit: str, n: int = 30) -> str:
    log = project / ".autoresearch" / "logs" / f"run-{commit}.log"
    if not log.exists():
        return ""
    text = log.read_text(errors="replace").splitlines()
    return "\n".join(text[-n:])


def _experiment_events(project: Path, problem: dict) -> list[dict]:
    rows = read_tsv(project / "results.tsv")
    metric = problem["metric_name"]
    commits = _commit_meta(project)
    events = []
    best = None
    lower = problem.get("lower_is_better", True)
    for i, row in enumerate(rows):
        commit = row.get("commit", "").strip()
        try:
            mv = float(row.get(metric, ""))
        except (ValueError, TypeError):
            mv = None
        status = row.get("status", "").strip().lower()
        is_new_best = False
        if status == "keep" and mv is not None:
            if best is None or (lower and mv < best) or (not lower and mv > best):
                best = mv
                is_new_best = True
        events.append({
            "kind": "experiment", "iter": i,
            "ts": commits.get(commit, {}).get("author_iso"),
            "commit": commit, "metric_name": metric, "metric_value": mv,
            "resource": row.get("resource"),
            "status": status, "description": row.get("description", "").strip(),
            "is_new_best": is_new_best, "running_best_after": best,
            "crash_tail": _crash_tail(project, commit) if status == "crash" else None,
        })
    return events


def _attach_critic(experiments: list[dict], critic_logs: list[dict]) -> None:
    """Match each experiment to the critic call that proposed it. We index
    critic logs by both timestamp and the captured commit (which is the commit
    BEFORE the experiment was committed, so we look one step back)."""
    by_commit_prev: dict[str, dict] = {}
    for c in critic_logs:
        commit = c.get("commit")
        if commit:
            by_commit_prev.setdefault(commit, c)

    for i, exp in enumerate(experiments):
        # The critic ran AFTER iteration i-1, with commit = experiments[i-1].commit.
        prev_commit = experiments[i - 1]["commit"] if i > 0 else None
        if prev_commit and prev_commit in by_commit_prev:
            exp["critic"] = by_commit_prev[prev_commit]
            continue
        # Fallback: timestamp-based match.
        exp_ts = exp.get("ts") or ""
        candidates = [c for c in critic_logs if (c.get("ts") or "") <= exp_ts]
        exp["critic"] = candidates[-1] if candidates else None


def build_timeline(project: Path, problem: dict) -> dict:
    wizard = _wizard_events(project)
    critic = _critic_logs(project)
    experiments = _experiment_events(project, problem)
    _attach_critic(experiments, critic)
    timeline = sorted(
        wizard + experiments,
        key=lambda e: (e.get("ts") or "", e.get("kind", ""), e.get("iter", -1) if "iter" in e else 0),
    )
    return {"wizard": wizard, "experiments": experiments, "critic_logs": critic,
            "timeline": timeline}


def summarize(experiments: list[dict], problem: dict) -> dict:
    counts = {"keep": 0, "discard": 0, "crash": 0}
    for e in experiments:
        s = e.get("status")
        if s in counts:
            counts[s] += 1
    metric = problem["metric_name"]
    keeps = [e for e in experiments if e.get("status") == "keep" and e.get("metric_value") is not None]
    baseline = experiments[0].get("metric_value") if experiments else None
    best = None
    best_exp = None
    lower = problem.get("lower_is_better", True)
    for e in keeps:
        mv = e["metric_value"]
        if best is None or (lower and mv < best) or (not lower and mv > best):
            best, best_exp = mv, e
    delta = None
    if baseline is not None and best is not None:
        delta = (baseline - best) if lower else (best - baseline)
    decided = counts["keep"] + counts["discard"]
    return {
        "metric": metric, "total": len(experiments), "counts": counts,
        "keep_rate": counts["keep"] / decided if decided else None,
        "baseline": baseline, "best": best,
        "best_commit": best_exp.get("commit") if best_exp else None,
        "best_description": best_exp.get("description") if best_exp else None,
        "total_improvement": delta,
    }


def render_md(timeline: dict, summary: dict, problem: dict, project: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric = summary["metric"]
    parts = [
        f"# Audit Log — {project.name}",
        f"_Generated at: {now}_\n",
        "## Summary\n",
        f"- **Objective**: {problem.get('objective','?')}",
        f"- **Metric**: `{metric}` ({'lower is better' if problem.get('lower_is_better', True) else 'higher is better'})",
        f"- **Experiments**: {summary['total']}",
        f"- **Keep / Discard / Crash**: {summary['counts']['keep']} / {summary['counts']['discard']} / {summary['counts']['crash']}",
        f"- **Keep rate**: {summary['keep_rate']:.1%}" if summary["keep_rate"] is not None else "- **Keep rate**: n/a",
        f"- **Baseline**: {summary['baseline']}",
        f"- **Best**: {summary['best']} (commit `{summary['best_commit']}`, `{summary['best_description']}`)" if summary['best'] is not None else "- **Best**: n/a",
        f"- **Total improvement**: {summary['total_improvement']:.6f}" if summary["total_improvement"] is not None else "- **Total improvement**: n/a",
        "",
        "## Wizard (preconditions)\n",
        "| step | status | detail |",
        "|---|---|---|",
    ]
    for w in timeline["wizard"]:
        detail = json.dumps(w.get("detail", {}), ensure_ascii=False)
        if len(detail) > 120:
            detail = detail[:117] + "..."
        parts.append(f"| {w['step']} | {w['status']} | {detail} |")
    parts.append("")

    parts.append("## Experiment timeline\n")
    for exp in timeline["experiments"]:
        marker = " (NEW BEST)" if exp.get("is_new_best") else ""
        parts.append(f"### Iter {exp['iter']} — `{exp.get('commit','?')}` — **{exp['status'].upper()}**{marker}")
        parts.append(f"- ts: `{exp.get('ts', '?')}`")
        parts.append(f"- {metric}: `{exp.get('metric_value')}`  resource: `{exp.get('resource')}`")
        parts.append(f"- description: {exp.get('description','')}")
        if exp.get("running_best_after") is not None:
            parts.append(f"- running-best after: `{exp['running_best_after']}`")
        critic = exp.get("critic")
        if critic:
            idea = critic.get("parsed_idea") or {}
            raw = critic.get("raw_response") or {}
            parts.append("\n**Gemma critic reasoning:**\n")
            if idea.get("thought_process"):
                parts.append(f"> _thought_process_: {idea['thought_process']}")
            if idea.get("alternatives_considered"):
                parts.append(f"> _alternatives_considered_:")
                for a in idea["alternatives_considered"]:
                    parts.append(f">   - {a}")
            parts.append(f"> _hypothesis_: **{idea.get('hypothesis','?')}**")
            parts.append(f"> _expected_delta_: {idea.get('expected_delta','?')}  _risk_: {idea.get('risk_level','?')}")
            if idea.get("justification"):
                parts.append(f"> _justification_: {idea['justification']}")
            if idea.get("code_pseudocode"):
                parts.append(f"> _code_pseudocode_: `{idea['code_pseudocode']}`")
            if raw.get("thinking_block"):
                parts.append(f"\n> _Ollama thought block_:\n> ```\n> {raw['thinking_block'][:600]}\n> ```")
            if raw.get("reasoning_field"):
                parts.append(f"\n> _reasoning field_:\n> ```\n> {raw['reasoning_field'][:600]}\n> ```")
            audit_file = critic.get("_file")
            if audit_file:
                parts.append(f"\n_audit_: [{audit_file}]({audit_file})")
        else:
            parts.append("\n_(no critic log linked to this iteration)_")
        if exp.get("crash_tail"):
            parts.append("\n**Stack trace (tail):**\n")
            parts.append("```")
            parts.append(exp["crash_tail"])
            parts.append("```")
        parts.append("")

    parts.append("## Top hits by delta\n")
    keeps = [e for e in timeline["experiments"] if e.get("status") == "keep" and e.get("metric_value") is not None]
    hits, prev = [], None
    for e in keeps:
        if prev is not None:
            delta = (prev - e["metric_value"]) if problem.get("lower_is_better", True) else (e["metric_value"] - prev)
            hits.append((delta, e))
        prev = e["metric_value"]
    hits.sort(key=lambda x: -x[0])
    if hits:
        parts.append("| Δ | commit | description |")
        parts.append("|---|---|---|")
        for delta, e in hits:
            parts.append(f"| +{delta:.6f} | `{e['commit']}` | {e['description'][:80]} |")
    else:
        parts.append("_(not enough keeps to compute deltas)_")
    parts.append("")
    return "\n".join(parts)


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem).resolve()
    problem = load_yaml(pp)
    project = pp.parent

    timeline = build_timeline(project, problem)
    summary = summarize(timeline["experiments"], problem)

    md_path = project / args.out_md
    json_path = project / args.out_json
    md_path.write_text(render_md(timeline, summary, problem, project))
    json_path.write_text(json.dumps({"summary": summary, **timeline}, indent=2,
                                    ensure_ascii=False, default=str))
    print(f"[audit] wrote {md_path}")
    print(f"[audit] wrote {json_path}")
    print(f"[audit] summary: total={summary['total']} keeps={summary['counts']['keep']} "
          f"discards={summary['counts']['discard']} crashes={summary['counts']['crash']} "
          f"best={summary['best']}")
