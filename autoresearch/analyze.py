"""CLI summary of results.tsv: counts, running-best, top hits by delta."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .helpers import problem_path, project_root, read_tsv, load_yaml


def _metric_col(rows: list[dict]) -> str:
    keys = list(rows[0].keys())
    for k in keys:
        if k not in {"commit", "resource", "status", "description"}:
            return k
    sys.exit("could not infer metric column")


def run(args: argparse.Namespace) -> None:
    proj = Path(args.project).resolve() if getattr(args, "project", None) else project_root()
    rows = read_tsv(proj / "results.tsv")
    if not rows:
        sys.exit("results.tsv empty or missing")
    metric = _metric_col(rows)

    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem)
    if pp.exists():
        problem = load_yaml(pp)
        lower = bool(problem.get("lower_is_better", True))
    else:
        lower = args.lower_is_better.lower() == "true"

    counts = {"keep": 0, "discard": 0, "crash": 0}
    for r in rows:
        s = r.get("status", "").strip().lower()
        if s in counts:
            counts[s] += 1
    total = sum(counts.values())
    print(f"\n=== {total} experiments ===")
    for k, v in counts.items():
        print(f"  {k:8s}: {v}")
    decided = counts["keep"] + counts["discard"]
    if decided:
        print(f"  keep rate: {counts['keep']}/{decided} = {counts['keep']/decided:.1%}")

    keeps = [r for r in rows if r.get("status", "").strip().lower() == "keep"]
    if not keeps:
        print("\nno KEEP experiments yet.")
        return

    def to_float(s):
        try: return float(s)
        except (ValueError, TypeError): return None

    print(f"\n=== KEPT experiments ({metric}) ===")
    best_so_far = None
    for r in keeps:
        v = to_float(r.get(metric, ""))
        if v is None:
            continue
        if best_so_far is None or (lower and v < best_so_far) or (not lower and v > best_so_far):
            best_so_far = v
            mark = "*"
        else:
            mark = " "
        print(f"  {mark} {r.get('commit','?'):>8s}  {v:.6f}  {r.get('description','')[:80]}")
    print(f"\nbest {metric} = {best_so_far}")

    print(f"\n=== Top hits by delta ===")
    hits = []
    prev = None
    for r in keeps:
        v = to_float(r.get(metric, ""))
        if v is None:
            continue
        if prev is not None:
            delta = (prev - v) if lower else (v - prev)
            hits.append((delta, r))
        prev = v
    hits.sort(key=lambda x: -x[0])
    for delta, r in hits:
        print(f"  +{delta:.6f}  {r.get('commit','?'):>8s}  {r.get('description','')[:80]}")
