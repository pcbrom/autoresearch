"""LLM critic — proposes the next experiment as JSON Schema via Ollama."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .helpers import gc_all, load_yaml, problem_path

NEXT_IDEA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought_process": {
            "type": "string",
            "description": "Step-by-step reasoning: read the history, identify what worked / what failed, "
                           "infer the underlying pattern, derive the next experiment from that pattern.",
        },
        "alternatives_considered": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Two or three other ideas you considered and why you ranked them lower.",
        },
        "hypothesis": {"type": "string", "description": "One-line idea to try"},
        "expected_delta": {"type": "number", "description": "Expected change in metric (signed)"},
        "justification": {"type": "string", "description": "Why this specific change should help"},
        "code_pseudocode": {"type": "string", "description": "Minimal sketch of edits"},
        "risk_level": {"type": "string", "enum": ["low", "med", "high"]},
    },
    "required": [
        "thought_process",
        "alternatives_considered",
        "hypothesis",
        "expected_delta",
        "justification",
        "code_pseudocode",
        "risk_level",
    ],
}

THINKING_BLOCK_RE = re.compile(
    r"<\|channel\|>thought\s*(.*?)(?:<\|channel\|>|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _tail_tsv(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))[-n:]


def _read_mutable(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(errors="replace")
    if len(txt) > max_chars:
        return txt[: max_chars // 2] + "\n# ... [truncated] ...\n" + txt[-max_chars // 2 :]
    return txt


def _build_prompt(problem: dict, history: list[dict], mutable_src: str,
                  thinking_on: bool) -> tuple[str, str]:
    lower = problem["lower_is_better"]
    direction = "minimize" if lower else "maximize"
    metric = problem["metric_name"]

    history_block = "(none yet — propose baseline-perturbing first idea)"
    if history:
        lines = []
        for r in history:
            mv = r.get(metric, r.get("metric", "?"))
            lines.append(
                f"  - {r.get('commit','?')} | {mv} | {r.get('status','?')} | {r.get('description','')[:80]}"
            )
        history_block = "\n".join(lines)

    cot_clause = (
        "\nThink step by step BEFORE producing the JSON. Use `thought_process` for the full chain "
        "of reasoning, and `alternatives_considered` for at least two other ideas you ranked lower."
        if thinking_on
        else "\nFill `thought_process` with at least three sentences explaining how the history led "
             "to this hypothesis."
    )

    system = (
        "You are an autonomous research critic for a single-file optimization loop. "
        f"The goal is to {direction} the metric `{metric}`. You propose ONE next experiment. "
        "Respond ONLY with JSON conforming to the provided schema. No prose, no preamble. "
        "Prefer simple, low-complexity changes. Reject ideas that worsened the metric in past iterations."
        + cot_clause
    )

    user = f"""## Objective
{problem['objective']}

## Metric direction
{direction} `{metric}` (lower_is_better={lower})

## Constraints
- mutable_file: {problem['mutable_file']}
- runner: {problem['runner']}
- time_budget_s: {problem['time_budget_s']}
- hard_timeout_s: {problem['hard_timeout_s']}
- readonly_files: {problem.get('readonly_files', [])} (DO NOT propose edits to these)

## Recent history (last {len(history)})
{history_block}

## Current mutable file (snippet)
```
{mutable_src}
```

## Task
Reason explicitly about what to try next, then emit the JSON.
"""
    return system, user


def _call_ollama(model: str, system: str, user: str, base_url: str,
                 thinking_on: bool) -> tuple[dict, dict]:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai SDK missing — pip install openai")
    client = OpenAI(base_url=base_url, api_key="ollama")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "next_idea", "schema": NEXT_IDEA_SCHEMA, "strict": True},
        },
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    if thinking_on:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    raw = msg.content or ""

    raw_payload: dict[str, Any] = {
        "model_used": getattr(resp, "model", model),
        "raw_content": raw,
        "thinking_block": None,
        "reasoning_field": None,
        "usage": {
            "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
            "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
        },
    }
    if hasattr(msg, "reasoning") and getattr(msg, "reasoning", None):
        raw_payload["reasoning_field"] = msg.reasoning
    m = THINKING_BLOCK_RE.search(raw)
    if m:
        raw_payload["thinking_block"] = m.group(1).strip()

    parsed = json.loads(raw)
    return parsed, raw_payload


def _stop_ollama(model: str) -> None:
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _audit(project: Path, parsed: dict, raw: dict, system: str, user: str,
           commit_short: str | None) -> Path:
    audit_dir = project / ".autoresearch" / "critic_logs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}_{commit_short or 'na'}.jsonl"
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": commit_short,
        "system_prompt": system,
        "user_prompt": user,
        "raw_response": raw,
        "parsed_idea": parsed,
    }
    path = audit_dir / fname
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def _current_commit(project: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem)
    pp = pp.resolve()
    problem = load_yaml(pp)
    project = pp.parent

    cfg = problem.get("gemma_critic", {})
    if not cfg.get("enabled", False):
        sys.exit("gemma_critic.enabled=false in problem.yaml")
    model = cfg.get("model", "gemma4:e2b")
    when = cfg.get("when", "always")
    base_url = cfg.get("ollama_url", "http://localhost:11434/v1")
    n_history = int(cfg.get("context_last_n", 10))
    thinking_on = bool(cfg.get("thinking", True))

    history = _tail_tsv(project / "results.tsv", n_history)
    mutable_src = _read_mutable(project / problem["mutable_file"])
    system, user = _build_prompt(problem, history, mutable_src, thinking_on)

    commit_short = _current_commit(project)

    try:
        idea, raw = _call_ollama(model, system, user, base_url, thinking_on)
    except Exception as e:
        _audit(project, {}, {"error": str(e)}, system, user, commit_short)
        print(json.dumps({"error": str(e), "fallback": "skip iteration"}))
        if when == "downtime":
            _stop_ollama(model)
        gc_all()
        sys.exit(1)

    audit_path = _audit(project, idea, raw, system, user, commit_short)
    if when == "downtime":
        _stop_ollama(model)
    gc_all()

    if not args.dry_run:
        (project / "next_idea.json").write_text(json.dumps(idea, indent=2, ensure_ascii=False))
    print(json.dumps({**idea, "_audit_log": str(audit_path)}))
