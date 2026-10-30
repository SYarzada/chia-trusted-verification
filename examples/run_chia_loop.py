from __future__ import annotations

import argparse
import json

from chia_trusted_verification.config import load_experiment_config
from chia_trusted_verification.loop import run_chia_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--engine", choices=("openevolve", "alphaevolve", "llm_agent"))
    parser.add_argument("--backend")
    args = parser.parse_args()
    config = load_experiment_config(
        args.config,
        engine_override=args.engine,
        backend_override=args.backend,
    )
    print(json.dumps(run_chia_loop(config).to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
