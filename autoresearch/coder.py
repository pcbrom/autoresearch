"""Drive an external coding agent to apply next_idea.json to the mutable file.

This closes step 3 of the loop (apply the critic's proposal) without a human in
the seat. The agent is invoked headless in the project directory; the loop still
owns running the experiment and all git operations, so the agent only edits one
file and never commits.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .helpers import load_yaml, problem_path

# Headless invocation per agent and permission posture. The single {prompt}
# token is replaced with the instruction; nothing is passed through a shell.
PRESETS: dict[str, dict[str, list[str]]] = {
    "claude": {
        "acceptEdits": ["claude", "-p", "--permission-mode", "acceptEdits", "{prompt}"],
        "bypass": ["claude", "-p", "--dangerously-skip-permissions", "{prompt}"],
    },
    "codex": {
        "acceptEdits": ["codex", "exec", "--sandbox", "workspace-write", "{prompt}"],
        "bypass": ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "{prompt}"],
    },
    "opencode": {
        # opencode resolves permissions from its own agent config, so both
        # postures share one invocation.
        "acceptEdits": ["opencode", "run", "{prompt}"],
        "bypass": ["opencode", "run", "{prompt}"],
    },
}

DEFAULT_PROMPT = (
    "You are one step of an autonomous research loop. Read the file `next_idea.json` "
    "in the current directory and apply ONLY the change it proposes (fields `hypothesis` "
    "and `code_pseudocode`) to `{mutable_file}`. Edit `{mutable_file}` and nothing else; "
    "do not touch {readonly_files}. Do NOT run git or commit, and do NOT run the experiment; "
    "the loop handles execution and version control. Make the smallest edit that "
    "implements the idea."
)


def _build_prompt(problem: dict, cfg: dict) -> str:
    tmpl = cfg.get("prompt_template") or DEFAULT_PROMPT
    readonly = ", ".join(problem.get("readonly_files", [])) or "any other file"
    return tmpl.replace("{mutable_file}", problem["mutable_file"]).replace("{readonly_files}", readonly)


def _build_command(cfg: dict, prompt: str) -> list[str]:
    raw = cfg.get("command")
    if raw:
        parts = shlex.split(raw) if isinstance(raw, str) else list(raw)
    else:
        agent = str(cfg.get("agent", "claude")).lower()
        perm = str(cfg.get("permission", "acceptEdits"))
        if agent not in PRESETS:
            sys.exit(f"coder.agent '{agent}' has no preset; set coder.command explicitly")
        if perm not in PRESETS[agent]:
            sys.exit(f"coder.permission '{perm}' invalid; use acceptEdits or bypass")
        parts = PRESETS[agent][perm]
    return [prompt if p == "{prompt}" else p.replace("{prompt}", prompt) for p in parts]


def _mutable_changed(project: Path, mutable: str) -> bool:
    """True if the working tree's mutable_file differs from HEAD (mirrors the
    runner's noop check, without staging anything)."""
    r = subprocess.run(
        ["git", "-C", str(project), "diff", "--quiet", "HEAD", "--", mutable],
        capture_output=True, timeout=30,
    )
    return r.returncode != 0


def apply_next_idea(problem: dict, project: Path, *, dry_run: bool = False) -> dict:
    cfg = problem.get("coder", {})
    if not cfg.get("enabled", False):
        return {"status": "disabled"}
    idea = project / "next_idea.json"
    if not idea.exists():
        return {"status": "fail", "reason": "next_idea.json not found (run the critic first)"}
    cmd = _build_command(cfg, _build_prompt(problem, cfg))
    if dry_run:
        return {"status": "ok", "dry_run": True, "agent": cfg.get("agent"), "command": cmd}
    timeout_s = int(cfg.get("timeout_s", 300))
    try:
        proc = subprocess.run(cmd, cwd=str(project), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "fail", "reason": f"coder timed out after {timeout_s}s"}
    except FileNotFoundError:
        return {"status": "fail", "reason": f"coder binary not found: {cmd[0]}"}
    if proc.returncode != 0:
        return {"status": "fail", "reason": f"coder exited {proc.returncode}",
                "stderr": (proc.stderr or "")[-500:]}
    if not _mutable_changed(project, problem["mutable_file"]):
        return {"status": "noedit", "reason": "coder ran but left mutable_file unchanged",
                "stdout": (proc.stdout or "")[-300:]}
    return {"status": "ok", "agent": cfg.get("agent"), "stdout": (proc.stdout or "")[-300:]}


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem).resolve()
    problem = load_yaml(pp)
    res = apply_next_idea(problem, pp.parent, dry_run=getattr(args, "dry_run", False))
    print(json.dumps(res))
    if res.get("status") in ("fail", "noedit"):
        sys.exit(1)
