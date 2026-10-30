#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess


def version(exe: str, flag: str = "--version") -> str | None:
    path = shutil.which(exe)
    if not path:
        return None
    try:
        out = subprocess.run(
            [path, flag], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=10, check=False
        ).stdout.strip().splitlines()
        return out[0] if out else path
    except Exception:
        return path


result = {
    "yosys": version("yosys", "-V"),
    "yosys-smtbmc": version("yosys-smtbmc"),
    "z3": version("z3"),
    "sby": version("sby"),
    "boolector": version("boolector"),
}
print(json.dumps(result, indent=2, sort_keys=True))
missing_required = [name for name in ("yosys", "yosys-smtbmc", "z3") if result[name] is None]
if missing_required:
    raise SystemExit("Missing required direct-formal tool(s): " + ", ".join(missing_required))
