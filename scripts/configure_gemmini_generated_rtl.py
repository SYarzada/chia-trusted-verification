#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re

import yaml

from chia_trusted_verification.backends.rtl_formal.intents import load_intent_catalog

MODULE_FILE_SETS = {
    "ReservationStation": ["plusarg_reader.v", "ReservationStation.sv"],
    "ExecuteController": [
        "plusarg_reader.v", "MeshWithDelays.sv", "Queue_98.sv", "TagQueue.sv",
        "Mesh.sv", "AlwaysOutTransposer.sv", "PE.sv", "Tile.sv",
        "ram_combMem_6.sv", "PE_256.sv", "MacUnit.sv", "Queue_97.sv",
        "ram_combMem_18.sv", "MultiHeadedQueue_1.sv",
        "TransposePreloadUnroller.sv", "MultiHeadedQueue.sv",
        "ExecuteController.sv",
    ],
    "LoadController": [
        "plusarg_reader.v", "DMACommandTracker.sv", "Queue_95.sv",
        "ram_combMem_44.sv", "LoadController.sv",
    ],
    "ScratchpadBank": [
        "plusarg_reader.v", "Queue_84.sv", "ram_combMem_30.sv",
        "mem.sv", "chipyard.TestHarness.GemminiRocketConfig.top.mems.v",
        "ScratchpadBank.sv",
    ],
    "StoreController": [
        "plusarg_reader.v", "DMACommandTracker_1.sv", "Queue_91.sv",
        "ram_combMem_43.sv", "StoreController.sv",
    ],
}

MODULE_CONFIG_NAMES = {
    "ReservationStation": "rtl_formal_reservation_station_local.yaml",
    "ExecuteController": "rtl_formal_execute_controller_local.yaml",
    "LoadController": "rtl_formal_load_controller_local.yaml",
    "ScratchpadBank": "rtl_formal_scratchpad_bank_local.yaml",
    "StoreController": "rtl_formal_store_controller_local.yaml",
}


def write_module_configs(generated: Path, base_config: dict, output_dir: Path) -> list[str]:
    """Write the original per-module RTL file sets for this local build."""
    paths_by_name: dict[str, list[Path]] = {}
    for path in sorted([*generated.rglob("*.sv"), *generated.rglob("*.v")]):
        paths_by_name.setdefault(path.name, []).append(path)

    warnings: list[str] = []
    for module, filenames in MODULE_FILE_SETS.items():
        missing = [name for name in filenames if name not in paths_by_name]
        if missing:
            warnings.append(f"{module}: missing RTL files: {', '.join(missing)}")
            continue
        config = deepcopy(base_config)
        settings = config["backend"]["settings"]
        settings["rtl_files"] = [str(paths_by_name[name][0]) for name in filenames]
        settings.pop("rtl_globs", None)
        settings["formal_timeout_s"] = 60
        config["experiment"]["output_dir"] = "../runs/rtl_formal_gemmini/local"
        module_path = output_dir / MODULE_CONFIG_NAMES[module]
        module_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"Wrote {module_path}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point Backend II at an already elaborated Gemmini generated-src tree."
    )
    parser.add_argument("generated_src", help="Gemmini/Chipyard generated-src directory")
    parser.add_argument("--base", default="configs/rtl_formal_gemmini.yaml")
    parser.add_argument("--catalog", default="rtl_formal/gemmini/intents.yaml")
    parser.add_argument("--output", default="configs/rtl_formal_gemmini_local.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    generated = Path(args.generated_src).expanduser().resolve()
    if not generated.is_dir():
        raise SystemExit(f"generated-src directory not found: {generated}")
    rtl_files = sorted(
        [*generated.rglob("*.sv"), *generated.rglob("*.v")],
        key=lambda p: str(p),
    )
    if not rtl_files:
        raise SystemExit(f"no .sv/.v files found below {generated}")

    module_to_files: dict[str, list[str]] = {}
    module_re = re.compile(r"(?m)^\s*module\s+([A-Za-z_$][A-Za-z0-9_$]*)\b")
    for path in rtl_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in module_re.finditer(text):
            module_to_files.setdefault(match.group(1), []).append(str(path))

    _, specs = load_intent_catalog(root / args.catalog)
    targets = sorted({spec.target_module for spec in specs.values()})
    missing = [name for name in targets if name not in module_to_files]

    raw = yaml.safe_load((root / args.base).read_text(encoding="utf-8"))
    config = deepcopy(raw)
    settings = config["backend"]["settings"]
    settings["rtl_files"] = [str(path) for path in rtl_files]
    settings.pop("rtl_globs", None)
    settings.pop("target_file", None)
    settings.pop("top_module", None)
    config["experiment"]["output_dir"] = "../runs/rtl_formal_gemmini/local"

    output = root / args.output
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {output}")
    for warning in write_module_configs(generated, raw, output.parent):
        print(f"WARNING: {warning}")
    print(f"Indexed {len(rtl_files)} RTL files and {len(module_to_files)} module names.")
    print(f'For the Docker/CHIA formal worker: export GEMMINI_GENERATED_SRC="{generated}"')
    if missing:
        print("WARNING: expected target module names not found:")
        for name in missing:
            print(f"  - {name}")
        print("Generated Chisel module names can vary. Inspect the generated RTL and adjust target_module/target_file for those intents before running them.")
    else:
        print("All target module names referenced by the 24-intent catalog were found.")


if __name__ == "__main__":
    main()
