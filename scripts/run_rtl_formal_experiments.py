#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from chia_trusted_verification.backends.rtl_formal.intents import load_intent_catalog


REQS = [
    "assertions_compile",
    "assertions_non_vacuous",
    "prove_selected_intents",
    "reach_selected_covers",
    "trusted_assumption_policy",
]


MODULE_CONFIGS = {
    "ReservationStation": "configs/rtl_formal_reservation_station_local.yaml",
    "ExecuteController": "configs/rtl_formal_execute_controller_local.yaml",
    "LoadController": "configs/rtl_formal_load_controller_local.yaml",
    "ScratchpadBank": "configs/rtl_formal_scratchpad_bank_local.yaml",
    "StoreController": "configs/rtl_formal_store_controller_local.yaml",
}


def post_verify(
    *,
    root: Path,
    config_path: Path,
    intent_id: str,
    seed: int,
) -> dict:
    out_dir = (
        root
        / "runs"
        / "rtl_formal_gemmini"
        / intent_id
        / f"seed{seed}"
        / "llm_agent"
    )
    candidate = out_dir / "best_candidate.sv"
    result_path = out_dir / "final_trusted_result.json"

    if not candidate.exists():
        raise RuntimeError(
            f"Agent run did not produce expected candidate: {candidate}"
        )

    command = [
        "chia-trusted-verify",
        "evaluate",
        str(candidate),
        "--config",
        str(config_path),
        "--engine",
        "llm_agent",
    ]

    print("FINAL TRUSTED VERIFY", intent_id, "seed", seed)
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")

    if completed.returncode != 0:
        print(completed.stdout)
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print(completed.stdout)
        raise

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = result.get("metrics", {})
    print(
        "AUTHORITATIVE RESULT",
        f"intent={intent_id}",
        f"seed={seed}",
        f"status={result.get('status')}",
        f"score={metrics.get('combined_score', metrics.get('score'))}",
        f"compile={metrics.get('compile_ok')}",
        f"bmc={metrics.get('bounded_check_ok')}",
        f"induction={metrics.get('induction_ok')}",
        f"cover={metrics.get('cover_ok')}",
        f"proved={metrics.get('proved')}",
    )
    print("WROTE", result_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Backend-II intent/seed matrix sequentially."
    )
    parser.add_argument(
        "--config",
        default="configs/rtl_formal_gemmini_local.yaml",
    )
    parser.add_argument(
        "--catalog",
        default="rtl_formal/gemmini/intents.yaml",
    )
    parser.add_argument(
        "--intents",
        default="all",
        help="all or comma-separated intent IDs",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip intent/seed runs that already have final_trusted_result.json.",
    )

    # Re-check an already-generated best_candidate.sv without another LLM call.
    parser.add_argument(
        "--post-verify-only",
        action="store_true",
        help="Skip the LLM and trusted-verify the existing best_candidate.sv.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    catalog_path = (root / args.catalog).resolve()

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _, specs = load_intent_catalog(catalog_path)

    intent_ids = (
        list(specs)
        if args.intents == "all"
        else [x.strip() for x in args.intents.split(",") if x.strip()]
    )

    unknown = [item for item in intent_ids if item not in specs]
    if unknown:
        raise SystemExit("Unknown intent(s): " + ", ".join(unknown))

    seeds = [
        int(x.strip())
        for x in args.seeds.split(",")
        if x.strip()
    ]

    for intent_id in intent_ids:
        for seed in seeds:
            existing_final = (
                root
                / "runs"
                / "rtl_formal_gemmini"
                / intent_id
                / f"seed{seed}"
                / "llm_agent"
                / "final_trusted_result.json"
            )

            if args.skip_existing and existing_final.is_file():
                print(
                    f"SKIP {intent_id} seed {seed}: "
                    f"{existing_final}"
                )
                continue

            spec = specs[intent_id]
            module_config = MODULE_CONFIGS.get(spec.target_module)

            if module_config is None:
                raise RuntimeError(
                    f"No RTL source-closure config for target module {spec.target_module}"
                )

            selected_config_path = (root / module_config).resolve()
            if not selected_config_path.exists():
                raise SystemExit(
                    f"Missing source files for {spec.target_module}. Run "
                    "scripts/configure_gemmini_generated_rtl.py on your generated RTL first."
                )
            cfg = yaml.safe_load(
                selected_config_path.read_text(encoding="utf-8")
            )

            print(
                f"BASE CONFIG {intent_id}: "
                f"{spec.target_module} -> {selected_config_path.name}"
            )
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["iterations"] = args.max_attempts
            cfg["experiment"]["run_id"] = (
                f"rtl-formal-{intent_id}-seed{seed}"
            )
            cfg["experiment"]["output_dir"] = (
                f"../runs/rtl_formal_gemmini/{intent_id}/seed{seed}"
            )

            cfg["verification"]["intent"] = {
                "intent_id": intent_id,
                "artifact_type": "systemverilog_formal_collateral",
                "requirements": REQS,
                "parameters": {},
            }

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                prefix="rtl_formal_",
                dir=config_path.parent,
                delete=False,
                encoding="utf-8",
            ) as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
                tmp = Path(handle.name)

            command = [
                sys.executable,
                "examples/run_chia_loop.py",
                "--config",
                str(tmp),
                "--engine",
                "llm_agent",
            ]

            try:
                if args.dry_run:
                    print("RUN", intent_id, "seed", seed)
                    print(" ".join(command))
                    continue

                if not args.post_verify_only:
                    print("RUN", intent_id, "seed", seed)
                    print(" ".join(command))
                    subprocess.run(command, cwd=root, check=True)
                else:
                    print(
                        "SKIP LLM; POST-VERIFY EXISTING",
                        intent_id,
                        "seed",
                        seed,
                    )

                post_verify(
                    root=root,
                    config_path=tmp,
                    intent_id=intent_id,
                    seed=seed,
                )

            finally:
                tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
