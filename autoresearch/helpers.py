"""Shared helpers for the autoresearch CLI."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML missing — pip install pyyaml")
    return yaml.safe_load(path.read_text()) or {}


def project_root() -> Path:
    return Path(os.environ.get("AUTORESEARCH_PROJECT", os.getcwd()))


def problem_path() -> Path:
    explicit = os.environ.get("AUTORESEARCH_PROBLEM")
    if explicit:
        return Path(explicit)
    return project_root() / "problem.yaml"


def get_dotted(data: dict, key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def running_best(rows: list[dict], metric: str, lower_is_better: bool) -> float | None:
    best = None
    for r in rows:
        if r.get("status", "").strip().lower() != "keep":
            continue
        try:
            v = float(r.get(metric, ""))
        except (ValueError, TypeError):
            continue
        if best is None or (lower_is_better and v < best) or (not lower_is_better and v > best):
            best = v
    return best


def classify(value_str: str, best: float | None, lower_is_better: bool) -> str:
    if not value_str or value_str.lower() == "nan":
        return "crash"
    try:
        v = float(value_str)
    except ValueError:
        return "crash"
    if best is None:
        return "keep"
    if lower_is_better:
        return "keep" if v < best else "discard"
    return "keep" if v > best else "discard"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def render_template(template_text: str, ctx: dict) -> str:
    out = template_text
    flat = _flatten(ctx)
    for k, v in flat.items():
        out = out.replace("{{ " + k + " }}", str(v))
    out = re.sub(
        r"\{%-?\s*for\s+(\w+)\s+in\s+([\w.]+)\s*-?%\}(.*?)\{%-?\s*endfor\s*-?%\}",
        lambda m: _render_loop(m, ctx),
        out,
        flags=re.DOTALL,
    )
    out = re.sub(
        r"\{%-?\s*if\s+(\w+)\s*-?%\}(.*?)\{%-?\s*else\s*-?%\}(.*?)\{%-?\s*endif\s*-?%\}",
        lambda m: m.group(2) if ctx.get(m.group(1)) else m.group(3),
        out,
        flags=re.DOTALL,
    )
    return out


def _render_loop(m, ctx: dict) -> str:
    var, source, body = m.group(1), m.group(2), m.group(3)
    cur: Any = ctx
    for part in source.split("."):
        cur = cur.get(part, []) if isinstance(cur, dict) else []
    if not isinstance(cur, list):
        return ""
    pieces = []
    for i, item in enumerate(cur):
        rendered = body.replace("{{ " + var + " }}", str(item))
        rendered = rendered.replace('{{ "\\t" if not loop.last }}', "" if i == len(cur) - 1 else "\t")
        rendered = re.sub(
            r"\{%\s*if\s+not\s+loop\.last\s*%\}(.*?)\{%\s*endif\s*%\}",
            lambda mm: "" if i == len(cur) - 1 else mm.group(1),
            rendered,
            flags=re.DOTALL,
        )
        pieces.append(rendered)
    return "".join(pieces)


def gc_all() -> None:
    """Best-effort GC across Python and CUDA."""
    import gc as _gc
    _gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
