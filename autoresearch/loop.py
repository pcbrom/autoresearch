"""NEVER-STOP autonomous loop: run experiment → propose next idea → repeat."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .helpers import gc_all, get_dotted, load_yaml, problem_path, project_root, running_best, read_tsv

CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch"))
PID_FILE = CACHE_DIR / "loop.pid"
HEARTBEAT_FILE = CACHE_DIR / "loop.heartbeat"
SHUTDOWN_FLAG = CACHE_DIR / "loop.shutdown"


def _cleanup(model: str | None) -> None:
    PID_FILE.unlink(missing_ok=True)
    if model:
        try:
            subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    gc_all()


def run(args: argparse.Namespace) -> None:
    pp = problem_path() if not getattr(args, "problem", None) else Path(args.problem).resolve()
    problem = load_yaml(pp)
    project = pp.parent

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (project / ".autoresearch" / "logs").mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            old = int(PID_FILE.read_text().strip())
            os.kill(old, 0)
            sys.exit(f"another loop already running (pid={old}). Stop it first.")
        except (OSError, ValueError):
            pass
    PID_FILE.write_text(str(os.getpid()))
    SHUTDOWN_FLAG.unlink(missing_ok=True)

    confirm_flag = project / ".autoresearch" / "loop_confirmed"
    if not confirm_flag.exists():
        sys.exit(
            f"wizard step `confirm_loop` not done. Run: autoresearch wizard step confirm_loop"
        )

    gemma_enabled = bool(get_dotted(problem, "gemma_critic.enabled", False))
    gemma_model = get_dotted(problem, "gemma_critic.model", "gemma3n:e2b")
    state_enabled = bool(get_dotted(problem, "state.enabled", False))
    state_every = int(get_dotted(problem, "state.rebuild_every_n_iter", 5))
    metric = problem["metric_name"]
    lower = bool(problem.get("lower_is_better", True))

    def shutdown(signum, frame):
        print(f"\n[{time.strftime('%H:%M:%S')}] received signal, shutting down...", file=sys.stderr)
        SHUTDOWN_FLAG.touch()
        _cleanup(gemma_model if gemma_enabled else None)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    i = 0
    last_best = None
    noop_streak = 0
    try:
        while True:
            if SHUTDOWN_FLAG.exists():
                shutdown(0, None)

            print(f"===== iter {i} =====", file=sys.stderr)
            from . import runner as _runner
            class _A: pass
            ns = _A(); ns.problem = str(pp)
            try:
                _runner.run(ns)  # prints JSON to stdout via _emit
            except SystemExit:
                pass
            # Re-read TSV to peek the latest status (since runner.run printed to stdout
            # in the same process and we cannot capture it from inside the same run).
            rows = read_tsv(project / "results.tsv")
            last_status = rows[-1].get("status", "?") if rows else "?"
            if last_status == "noop":
                noop_streak += 1
                if gemma_enabled and noop_streak <= 1:
                    from . import critic as _critic
                    cns = _A(); cns.problem = str(pp); cns.dry_run = False
                    try:
                        _critic.run(cns)
                    except SystemExit:
                        pass
                sleep_s = 5 if noop_streak < 5 else 30
                print(f"[loop] noop (streak={noop_streak}), sleeping {sleep_s}s for agent to edit",
                      file=sys.stderr)
                for _ in range(sleep_s):
                    if SHUTDOWN_FLAG.exists():
                        shutdown(0, None)
                    time.sleep(1)
                continue
            noop_streak = 0

            if gemma_enabled:
                from . import critic as _critic
                cns = _A(); cns.problem = str(pp); cns.dry_run = False
                try:
                    _critic.run(cns)
                except SystemExit as e:
                    print(f"[warn] critic failed this iter: {e}", file=sys.stderr)

            current_best = running_best(read_tsv(project / "results.tsv"), metric, lower)
            should_rebuild = False
            if state_enabled:
                if i % state_every == 0:
                    should_rebuild = True
                if current_best != last_best:
                    should_rebuild = True
            if should_rebuild:
                from . import state as _state
                ens = _A(); ens.problem = str(pp); ens.force = False
                try:
                    _state.run(ens)
                except SystemExit:
                    pass
            last_best = current_best

            HEARTBEAT_FILE.write_text(str(int(time.time())))
            i += 1
    finally:
        _cleanup(gemma_model if gemma_enabled else None)
