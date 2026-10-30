from __future__ import annotations

import argparse
import json

from .config import load_experiment_config
from .core import backend_names
from .loop import build_initial_program, evaluate_program_locally, run_chia_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CHIA trusted verification loop with pluggable backends"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/experiment.yaml")
    run.add_argument("--engine", choices=("openevolve", "alphaevolve", "llm_agent"))
    run.add_argument("--backend")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("program")
    evaluate.add_argument("--config", default="configs/experiment.yaml")
    evaluate.add_argument("--engine", choices=("openevolve", "alphaevolve", "llm_agent"))
    evaluate.add_argument("--backend")

    render = subparsers.add_parser("render-program")
    render.add_argument("--config", default="configs/experiment.yaml")
    render.add_argument("--engine", choices=("openevolve", "alphaevolve", "llm_agent"))
    render.add_argument("--backend")

    subparsers.add_parser("list-backends")

    args = parser.parse_args()
    if args.command == "list-backends":
        print("\n".join(backend_names()))
        return

    config = load_experiment_config(
        args.config,
        engine_override=args.engine,
        backend_override=args.backend,
    )
    if args.command == "run":
        result = run_chia_loop(config)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        result = evaluate_program_locally(args.program, config)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        path, _ = build_initial_program(config)
        print(str(path))


if __name__ == "__main__":
    main()
