from __future__ import annotations

import glob
import hashlib
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping

from .intents import FormalIntentSpec


class FormalSetupError(RuntimeError):
    pass


def describe_target_rtl(
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
    *,
    max_chars: int = 12000,
) -> str:
    """Return a short RTL excerpt for the agent prompt.

    Formal verification checks the configured RTL files.
    """
    try:
        config_dir = Path(str(settings.get("_config_dir", "."))).resolve()
        rtl_files = _resolve_rtl_files(config_dir, settings)
        if not rtl_files:
            return ""
        target_module = str(settings.get("top_module") or spec.target_module)
        target_file_setting = settings.get("target_file")
        target_file = (
            _resolve(config_dir, target_file_setting)
            if target_file_setting
            else _find_module_file(rtl_files, target_module)
        )
        text = target_file.read_text(encoding="utf-8", errors="replace")
        module_text, start_line = _extract_module_text(text, target_module)
    except (FormalSetupError, OSError, ValueError):
        return ""

    lines = module_text.splitlines()
    selected: set[int] = set(range(min(len(lines), 80)))
    needles: set[str] = set()
    for hint in spec.signal_hints:
        raw = hint.lower()
        needles.add(raw)
        needles.add(raw.replace(".", "_"))
        if raw.startswith("io_"):
            needles.add(raw[3:])
        for token in re.split(r"[^a-zA-Z0-9_$]+", raw):
            if len(token) >= 5:
                needles.add(token)

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(needle and needle in lowered for needle in needles):
            for neighbor in range(max(0, idx - 2), min(len(lines), idx + 3)):
                selected.add(neighbor)

    rendered: list[str] = []
    previous = -1
    for idx in sorted(selected):
        if idx != previous + 1:
            rendered.append("    ...")
        rendered.append(f"{start_line + idx:6d}: {lines[idx]}")
        previous = idx
    result = "\n".join(rendered)
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "\n    ... [RTL context truncated]"
    return f"Target RTL file: {target_file}\n" + result


def prepare_formal_workspace(
    *,
    candidate_source: str,
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    config_dir = Path(str(settings.get("_config_dir", "."))).resolve()
    workspace_root = _resolve(config_dir, settings.get("workspace_root", "../runs/rtl_formal/workspaces"))
    digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
    workspace = workspace_root / spec.intent_id / digest[:16]
    workspace.mkdir(parents=True, exist_ok=True)

    rtl_files = _resolve_rtl_files(config_dir, settings)
    if not rtl_files:
        raise FormalSetupError(
            "no RTL files configured; set backend.settings.rtl_files and/or rtl_globs"
        )
    missing = [str(path) for path in rtl_files if not path.is_file()]
    if missing:
        raise FormalSetupError("configured RTL file(s) do not exist: " + ", ".join(missing))

    target_module = str(settings.get("top_module") or spec.target_module)
    target_file_setting = settings.get("target_file")
    target_file = (
        _resolve(config_dir, target_file_setting) if target_file_setting else _find_module_file(rtl_files, target_module)
    )
    if target_file not in rtl_files:
        rtl_files.append(target_file)
    original = target_file.read_text(encoding="utf-8", errors="replace")
    trusted_collateral, trusted_files = _load_trusted_collateral(config_dir, spec, settings)
    combined_collateral = candidate_source
    if trusted_collateral.strip():
        combined_collateral = (
            "// CHIA-BEGIN-TRUSTED-HARNESS\n"
            + trusted_collateral.rstrip()
            + "\n// CHIA-END-TRUSTED-HARNESS\n"
            + candidate_source
        )
    instrumented = _instrument_module(original, target_module, combined_collateral)
    instrumented_path = workspace / f"instrumented_{target_file.name}"
    instrumented_path.write_text(instrumented, encoding="utf-8")

    final_files = [instrumented_path if path == target_file else path for path in rtl_files]
    include_dirs = [_resolve(config_dir, item) for item in settings.get("include_dirs", [])]
    include_dirs.extend(path.parent for path in rtl_files)
    include_dirs = _dedupe_paths(include_dirs)

    return {
        "workspace": str(workspace),
        "rtl_files": [str(path) for path in final_files],
        "target_file": str(target_file),
        "instrumented_file": str(instrumented_path),
        "top_module": target_module,
        "include_dirs": [str(path) for path in include_dirs],
        "defines": _formal_defines(settings),
        "candidate_sha256": digest,
        "trusted_collateral_files": [str(path) for path in trusted_files],
    }


def run_formal(
    prepared: Mapping[str, Any],
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    engine = str(settings.get("formal_engine", "smtbmc")).strip().lower()
    if engine not in {"smtbmc", "sby"}:
        raise FormalSetupError("formal_engine must be 'smtbmc' or 'sby'")
    if engine == "sby":
        return _run_sby(prepared, spec, settings)
    return _run_smtbmc(prepared, spec, settings)


def tool_versions(settings: Mapping[str, Any]) -> dict[str, str | None]:
    tools = {
        "yosys": str(settings.get("yosys_bin", "yosys")),
        "yosys_smtbmc": str(settings.get("yosys_smtbmc_bin", "yosys-smtbmc")),
        "sby": str(settings.get("sby_bin", "sby")),
        "solver": str(settings.get("solver", "z3")),
    }
    versions: dict[str, str | None] = {}
    for name, executable in tools.items():
        resolved = shutil.which(executable)
        if not resolved:
            versions[name] = None
            continue
        try:
            flag = "-V" if name == "yosys" else "--version"
            completed = subprocess.run(
                [resolved, flag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            )
            first = (completed.stdout or "").strip().splitlines()
            versions[name] = first[0] if first else resolved
        except Exception:
            versions[name] = resolved
    return versions


def _run_smtbmc(
    prepared: Mapping[str, Any],
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    workspace = Path(str(prepared["workspace"]))
    yosys = _which(str(settings.get("yosys_bin", "yosys")), "Yosys")
    smtbmc_name = str(settings.get("yosys_smtbmc_bin", "yosys-smtbmc"))
    smtbmc = shutil.which(smtbmc_name)
    if not smtbmc:
        sibling = Path(yosys).with_name("yosys-smtbmc")
        if sibling.is_file():
            smtbmc = str(sibling)
    if not smtbmc:
        raise FormalSetupError("yosys-smtbmc was not found in PATH")
    solver = str(settings.get("solver", "z3"))
    if not shutil.which(solver):
        raise FormalSetupError(f"SMT solver {solver!r} was not found in PATH")

    depth = int(settings.get("proof_depth", spec.proof_depth))
    timeout_s = int(settings.get("formal_timeout_s", 300))
    model = workspace / "design.smt2"
    yosys_script = workspace / "formal.ys"
    yosys_script.write_text(_yosys_script(prepared, model), encoding="utf-8")

    started = time.perf_counter()
    compile_run = _run([yosys, "-ql", "yosys.log", str(yosys_script)], cwd=workspace, timeout_s=timeout_s)
    result: dict[str, Any] = {
        "formal_engine": "smtbmc",
        "compile": compile_run,
        "bmc": None,
        "induction": None,
        "cover": None,
        "proof_depth": depth,
        "tool_versions": tool_versions(settings),
    }
    if compile_run["returncode"] != 0:
        result["status"] = "compile_error"
        result["runtime_s"] = time.perf_counter() - started
        return result

    # Reject results if Yosys removes required formal properties.
    smt_text = model.read_text(encoding="utf-8", errors="replace")

    if "assert" in spec.required_property_kinds and "yosys-smt2-assert" not in smt_text:
        result["status"] = "compile_error"
        result["diagnostics"] = [{
            "stage": "formal",
            "kind": "MissingFormalProperties",
            "message": "Candidate requires assertions but Yosys exported no assertions to SMT2.",
        }]
        result["runtime_s"] = time.perf_counter() - started
        result["artifacts"] = _artifact_paths(workspace)
        return result

    if "cover" in spec.required_property_kinds and "yosys-smt2-cover" not in smt_text:
        result["status"] = "compile_error"
        result["diagnostics"] = [{
            "stage": "formal",
            "kind": "MissingFormalProperties",
            "message": "Candidate requires covers but Yosys exported no covers to SMT2.",
        }]
        result["runtime_s"] = time.perf_counter() - started
        result["artifacts"] = _artifact_paths(workspace)
        return result

    common = [smtbmc, "-s", solver, "-t", str(depth), "--noprogress"]
    bmc = _run(
        common + ["--dump-vcd", "counterexample.vcd", str(model)],
        cwd=workspace,
        timeout_s=timeout_s,
    )
    result["bmc"] = bmc

    induction = None
    cover = None

    # Run induction only if bounded checking passes.
    if bmc["returncode"] == 0:
        induction = _run(
            common[:3] + ["-i", "-t", str(depth), "--noprogress", str(model)],
            cwd=workspace,
            timeout_s=timeout_s,
        )
        result["induction"] = induction

    # Check cover only if bounded checking and induction pass.
    if (
        bmc["returncode"] == 0
        and induction is not None
        and induction["returncode"] == 0
        and "cover" in spec.required_property_kinds
    ):
        cover = _run(
            common[:3]
            + [
                "-c",
                "-t",
                str(depth),
                "--noprogress",
                "--dump-vcd",
                "cover.vcd",
                str(model),
            ],
            cwd=workspace,
            timeout_s=timeout_s,
        )
        result["cover"] = cover

    if bmc.get("timed_out") or bmc["returncode"] == 124:
        status = "timeout"
    elif bmc["returncode"] != 0:
        status = "counterexample"
    elif (
        induction is not None
        and (induction.get("timed_out") or induction["returncode"] == 124)
    ):
        status = "timeout"
    elif induction is not None and induction["returncode"] != 0:
        status = "induction_failed"
    elif cover is not None and (
        cover.get("timed_out") or cover["returncode"] == 124
    ):
        status = "timeout"
    elif cover is not None and cover["returncode"] != 0:
        status = "cover_unreached"
    else:
        status = "proved"

    result["status"] = status
    result["runtime_s"] = time.perf_counter() - started
    result["artifacts"] = _artifact_paths(workspace)
    return result


def _run_sby(
    prepared: Mapping[str, Any],
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    workspace = Path(str(prepared["workspace"]))
    sby = _which(str(settings.get("sby_bin", "sby")), "sby")
    solver = str(settings.get("solver", "z3"))
    depth = int(settings.get("proof_depth", spec.proof_depth))
    timeout_s = int(settings.get("formal_timeout_s", 300))
    started = time.perf_counter()

    prove_file = workspace / "prove.sby"
    prove_file.write_text(_sby_file(prepared, solver, depth, mode="prove"), encoding="utf-8")
    prove = _run([sby, "-f", str(prove_file)], cwd=workspace, timeout_s=timeout_s)
    result: dict[str, Any] = {
        "formal_engine": "sby",
        "compile": prove,
        "bmc": prove,
        "induction": prove,
        "cover": None,
        "proof_depth": depth,
        "tool_versions": tool_versions(settings),
    }
    if prove["returncode"] != 0:
        result["status"] = "counterexample_or_tool_error"
    else:
        result["status"] = "proved"
    if "cover" in spec.required_property_kinds:
        cover_file = workspace / "cover.sby"
        cover_file.write_text(_sby_file(prepared, solver, depth, mode="cover"), encoding="utf-8")
        cover = _run([sby, "-f", str(cover_file)], cwd=workspace, timeout_s=timeout_s)
        result["cover"] = cover
        if result["status"] == "proved" and cover["returncode"] != 0:
            result["status"] = "cover_unreached"
    result["runtime_s"] = time.perf_counter() - started
    result["artifacts"] = _artifact_paths(workspace)
    return result


def _yosys_script(prepared: Mapping[str, Any], model: Path) -> str:
    include_flags = " ".join(f"-I{_ys_quote(path)}" for path in prepared.get("include_dirs", []))
    define_flags = " ".join(f"-D{item}" for item in prepared.get("defines", []))
    files = " ".join(_ys_quote(path) for path in prepared["rtl_files"])
    top = str(prepared["top_module"])
    return (
        f"read_verilog -formal -sv -noautowire {include_flags} {define_flags} {files}\n"
        f"hierarchy -check -top {top}\n"
        f"prep -top {top}\n"
        "chformal -assert -remove */c:assert__*\n"
        "chformal -cover -remove */c:cover__*\n"
        "async2sync\n"
        "dffunmap\n"
        f"write_smt2 -wires {_ys_quote(model)}\n"
    )


def _sby_file(prepared: Mapping[str, Any], solver: str, depth: int, *, mode: str) -> str:
    include_flags = " ".join(f"-I{_ys_quote(path)}" for path in prepared.get("include_dirs", []))
    define_flags = " ".join(f"-D{item}" for item in prepared.get("defines", []))
    files = " ".join(_ys_quote(path) for path in prepared["rtl_files"])
    top = str(prepared["top_module"])
    return (
        "[options]\n"
        f"mode {mode}\n"
        f"depth {depth}\n\n"
        "[engines]\n"
        f"smtbmc {solver}\n\n"
        "[script]\n"
        f"read_verilog -formal -sv -noautowire {include_flags} {define_flags} {files}\n"
        f"prep -top {top}\n"
        "chformal -assert -remove */c:assert__*\n"
        "chformal -cover -remove */c:cover__*\n"
        "async2sync\n"
        "dffunmap\n"
    )


def _run(command: list[str], *, cwd: Path, timeout_s: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        output = completed.stdout or ""
        return {
            "command": " ".join(shlex.quote(item) for item in command),
            "returncode": int(completed.returncode),
            "timed_out": False,
            "runtime_s": time.perf_counter() - started,
            "output_tail": _tail(output),
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "command": " ".join(shlex.quote(item) for item in command),
            "returncode": 124,
            "timed_out": True,
            "runtime_s": time.perf_counter() - started,
            "output_tail": _tail(output + "\nFORMAL TIMEOUT"),
        }


def _formal_defines(settings: Mapping[str, Any]) -> list[str]:
    values = ["FORMAL", "SYNTHESIS", *(str(item) for item in settings.get("defines", []))]
    return list(dict.fromkeys(values))


def _load_trusted_collateral(
    config_dir: Path,
    spec: FormalIntentSpec,
    settings: Mapping[str, Any],
) -> tuple[str, list[Path]]:
    entries: list[Any] = list(settings.get("trusted_collateral_files", []) or [])
    by_intent = settings.get("trusted_collateral_by_intent", {}) or {}
    if isinstance(by_intent, Mapping):
        selected = by_intent.get(spec.intent_id, [])
        if isinstance(selected, (str, Path)):
            entries.append(selected)
        else:
            entries.extend(list(selected or []))
    files = [_resolve(config_dir, item) for item in entries]
    chunks: list[str] = []
    for path in files:
        if not path.is_file():
            raise FormalSetupError(f"trusted collateral file does not exist: {path}")
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks), files


def _extract_module_text(text: str, module_name: str) -> tuple[str, int]:
    start_match = re.search(rf"(?m)^\s*module\s+{re.escape(module_name)}\b", text)
    if start_match is None:
        raise FormalSetupError(f"module {module_name!r} was not found in target RTL")
    end_match = re.search(r"(?m)^\s*endmodule\b", text[start_match.end():])
    if end_match is None:
        raise FormalSetupError(f"endmodule for module {module_name!r} was not found")
    end = start_match.end() + end_match.end()
    start_line = text.count("\n", 0, start_match.start()) + 1
    return text[start_match.start():end], start_line


def _instrument_module(text: str, module_name: str, collateral: str) -> str:
    marker = "// CHIA_FORMAL_COLLATERAL"
    if marker in text:
        if text.count(marker) != 1:
            raise FormalSetupError("target RTL contains multiple CHIA_FORMAL_COLLATERAL markers")
        return text.replace(marker, collateral.rstrip() + "\n", 1)

    start_match = re.search(rf"(?m)^\s*module\s+{re.escape(module_name)}\b", text)
    if start_match is None:
        raise FormalSetupError(f"module {module_name!r} was not found in target RTL")
    end_match = re.search(r"(?m)^\s*endmodule\b", text[start_match.end():])
    if end_match is None:
        raise FormalSetupError(f"endmodule for module {module_name!r} was not found")
    insert_at = start_match.end() + end_match.start()
    banner = "\n`ifdef FORMAL\n" + collateral.rstrip() + "\n`endif\n"
    return text[:insert_at] + banner + text[insert_at:]


def _find_module_file(files: Iterable[Path], module_name: str) -> Path:
    pattern = re.compile(rf"(?m)^\s*module\s+{re.escape(module_name)}\b")
    matches: list[Path] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            matches.append(path)
    if not matches:
        raise FormalSetupError(
            f"module {module_name!r} was not found in configured RTL files; "
            "set backend.settings.target_file/top_module after elaborating Gemmini"
        )
    if len(matches) > 1:
        raise FormalSetupError(
            f"module {module_name!r} appears in multiple RTL files: " + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def _resolve_rtl_files(base: Path, settings: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for item in settings.get("rtl_files", []):
        files.append(_resolve(base, item))
    for pattern in settings.get("rtl_globs", []):
        raw = Path(str(pattern)).expanduser()
        full = raw if raw.is_absolute() else base / raw
        for match in sorted(glob.glob(str(full), recursive=True)):
            path = Path(match)
            if path.is_file():
                files.append(path.resolve())
    return _dedupe_paths(files)


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key not in seen:
            result.append(resolved)
            seen.add(key)
    return result


def _which(executable: str, label: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        raise FormalSetupError(f"{label} executable {executable!r} was not found in PATH")
    return resolved


def _ys_quote(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    if '"' in text or "\n" in text:
        raise FormalSetupError(f"unsupported path for Yosys script: {text!r}")
    return f'"{text}"'


def _tail(text: str, *, lines: int = 80, chars: int = 12000) -> str:
    selected = "\n".join(text.splitlines()[-lines:])
    return selected[-chars:]


def _artifact_paths(workspace: Path) -> dict[str, str]:
    names = [
        "yosys.log",
        "counterexample.vcd",
        "cover.vcd",
        "design.smt2",
        "prove/status",
        "cover/status",
    ]
    return {
        name.replace("/", "_"): str(workspace / name)
        for name in names
        if (workspace / name).exists()
    }
