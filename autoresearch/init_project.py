"""Scaffold a new autoresearch project from a problem.yaml."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .helpers import get_dotted, load_yaml, render_template

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _git(args: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd)] + args,
                          capture_output=capture, text=True, timeout=30)


def run(args: argparse.Namespace) -> None:
    problem_src = Path(args.problem).resolve()
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    if not problem_src.exists():
        sys.exit(f"problem.yaml not found at {problem_src}")

    problem = load_yaml(problem_src)
    branch_prefix = problem["branch_prefix"]
    metric = problem["metric_name"]
    tag = args.tag or "run1"
    branch = f"{branch_prefix}/{tag}"

    if problem_src.resolve() != (target / "problem.yaml").resolve():
        shutil.copy(problem_src, target / "problem.yaml")

    if not (target / ".git").exists():
        _git(["init", "-q"], target)
        _git(["add", "-A"], target, capture=True)
        _git(["commit", "-q", "-m", "initial scaffold"], target, capture=True)

    branches = _git(["branch", "--list", branch], target, capture=True).stdout.strip()
    if branches:
        sys.exit(f"branch {branch} already exists. Pick another --tag.")
    _git(["checkout", "-b", branch, "-q"], target, capture=True)

    results_tsv = target / "results.tsv"
    if not results_tsv.exists():
        results_tsv.write_text(f"commit\t{metric}\tresource\tstatus\tdescription\n")

    program_tmpl = TEMPLATES_DIR / "program.md.template"
    if program_tmpl.exists():
        rendered = render_template(program_tmpl.read_text(), problem)
        (target / "program.md").write_text(rendered)

    gi = target / ".gitignore"
    additions = [
        "", "# autoresearch", "results.tsv", "run.log", "next_idea.json",
        "ESTADO_GLOBAL.md", "STATE.md", "AUDIT_LOG.md", "AUDIT_LOG.json",
        ".autoresearch/",
    ]
    existing = gi.read_text() if gi.exists() else ""
    if "# autoresearch" not in existing:
        gi.write_text(existing + "\n".join(additions) + "\n")

    print(f"[init] target={target} branch={branch} problem={problem_src}")
    print(f"[init] next: AUTORESEARCH_PROJECT={target} autoresearch wizard next")
