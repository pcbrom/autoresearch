"""Single CLI entrypoint: `autoresearch <subcommand>`."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="autoresearch",
        description="Autonomous research loop with a local LLM critic.",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # init
    sp = sub.add_parser("init", help="scaffold a new project from a problem.yaml")
    sp.add_argument("--problem", required=True)
    sp.add_argument("--target", required=True)
    sp.add_argument("--tag", default=None)

    # wizard
    sp = sub.add_parser("wizard", help="stepwise preflight validator")
    wsub = sp.add_subparsers(dest="wizard_cmd", required=True)
    wsub.add_parser("status")
    s2 = wsub.add_parser("step"); s2.add_argument("name")
    wsub.add_parser("next")
    wsub.add_parser("reset")

    # run
    sp = sub.add_parser("run", help="run a single experiment iteration")
    sp.add_argument("--problem", default=None)

    # critic
    sp = sub.add_parser("critic", help="propose the next idea via Ollama")
    sp.add_argument("--problem", default=None)
    sp.add_argument("--dry-run", action="store_true")

    # loop
    sp = sub.add_parser("loop", help="run the NEVER-STOP autonomous loop")
    sp.add_argument("--problem", default=None)

    # analyze
    sp = sub.add_parser("analyze", help="summary of results.tsv")
    sp.add_argument("--project", default=None)
    sp.add_argument("--problem", default=None)
    sp.add_argument("--lower-is-better", default="true")

    # audit
    sp = sub.add_parser("audit", help="consolidated timeline (AUDIT_LOG.md + .json)")
    sp.add_argument("--problem", default=None)
    sp.add_argument("--out-md", default="AUDIT_LOG.md")
    sp.add_argument("--out-json", default="AUDIT_LOG.json")

    # estado / state
    sp = sub.add_parser("state", help="regenerate STATE.md snapshot")
    sp.add_argument("--problem", default=None)
    sp.add_argument("--force", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "init":
        from . import init_project; init_project.run(args)
    elif args.cmd == "wizard":
        from . import wizard
        if args.wizard_cmd == "status": wizard.cmd_status(args)
        elif args.wizard_cmd == "step": wizard.cmd_step(args)
        elif args.wizard_cmd == "next": wizard.cmd_next(args)
        elif args.wizard_cmd == "reset": wizard.cmd_reset(args)
    elif args.cmd == "run":
        from . import runner; runner.run(args)
    elif args.cmd == "critic":
        from . import critic; critic.run(args)
    elif args.cmd == "loop":
        from . import loop; loop.run(args)
    elif args.cmd == "analyze":
        from . import analyze; analyze.run(args)
    elif args.cmd == "audit":
        from . import audit; audit.run(args)
    elif args.cmd == "state":
        from . import state; state.run(args)


if __name__ == "__main__":
    main()
