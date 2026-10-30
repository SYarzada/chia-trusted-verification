from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


COMMAND_HINTS = {
    "git": "Install/update Git in chia_env: conda install -n chia_env -c conda-forge git -y",
    "docker": "Install Docker Engine (admin usually required) and ensure `docker info` works without sudo.",
    "ssh": "Install OpenSSH client/server and configure public-key SSH to CHIA_HEAD_IP.",
    "chia": "Clone https://github.com/ucb-bar/chia.git and run `pip install -e /path/to/chia` in chia_env.",
    "gcloud": "Install user-locally with `curl -fsSL https://sdk.cloud.google.com -o /tmp/install_google_cloud_sdk.sh && bash /tmp/install_google_cloud_sdk.sh --disable-prompts --install-dir=\"$HOME\"`, then source `$HOME/google-cloud-sdk/path.bash.inc`.",
}

MODULE_HINTS = {
    "chia": "Install CHIA in chia_env with `pip install -e /path/to/chia`.",
    "ray": "Ray is installed by CHIA. Reinstall CHIA if this is missing.",
    "yaml": "Install this repository with `pip install -e \".[search,test,analysis]\"`.",
    "z3": "Install this repository with `pip install -e \".[search,test,analysis]\"`.",
    "evolve_flows": "Install the search extra: `pip install -e \".[search]\"`.",
    "skydiscover": "Install the search extra: `pip install -e \".[search]\"`.",
    "openevolve": "Install the search extra: `pip install -e \".[search]\"`.",
    "alpha_evolve": "Install the search extra: `pip install -e \".[search]\"`.",
    "google.auth": "Install the search extra: `pip install -e \".[search]\"`.",
    "pandas": "Install the analysis extra: `pip install -e \".[analysis]\"`.",
    "matplotlib": "Install the analysis extra: `pip install -e \".[analysis]\"`.",
}


def ok(label: str, detail: str = "") -> None:
    print(f"[OK] {label}" + (f": {detail}" if detail else ""))


def fail(label: str, detail: str = "", hint: str = "") -> None:
    print(f"[MISSING] {label}" + (f": {detail}" if detail else ""))
    if hint:
        print(f"          Fix: {hint}")


def command(name: str, required: bool = True) -> bool:
    path = shutil.which(name)
    if path:
        ok(name, path)
        return True
    if required:
        fail(name, hint=COMMAND_HINTS.get(name, ""))
    return not required


def module(name: str) -> bool:
    if importlib.util.find_spec(name) is not None:
        ok(f"python module {name}")
        return True
    fail(f"python module {name}", hint=MODULE_HINTS.get(name, ""))
    return False


def env(name: str, required: bool = True) -> bool:
    value = os.environ.get(name)
    if value:
        shown = "set" if "KEY" in name or "TOKEN" in name or "SECRET" in name else value
        ok(name, shown)
        return True
    if required:
        fail(name, hint=f"Export {name} before running this check; see docs/TEAAL_EVOLVE_QUICKSTART.md.")
    return not required


def run_check(cmd: list[str], label: str, hint: str = "") -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=20)
        ok(label)
        return True
    except Exception:
        fail(label, hint=hint)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["openevolve", "alphaevolve", "llm_agent"], required=True)
    parser.add_argument("--analysis", action="store_true")
    args = parser.parse_args()

    checks: list[bool] = []
    if sys.version_info[:3] == (3, 10, 19):
        ok("Python", sys.version.split()[0])
        checks.append(True)
    elif sys.version_info[:2] == (3, 10):
        ok("Python 3.10.x", sys.version.split()[0])
        print("[WARN] CHIA's worker images use Python 3.10.19. Matching it is recommended.")
        checks.append(True)
    else:
        fail("Python 3.10.19", sys.version.split()[0], "Create/activate it with `conda create -n chia_env -c conda-forge python=3.10.19 git -y`.")
        checks.append(False)

    for name in ("git", "docker", "ssh", "chia"):
        checks.append(command(name))

    if shutil.which("docker"):
        checks.append(run_check(["docker", "info"], "Docker daemon access", "Ensure Docker is running and your user can access the daemon without sudo."))

    checks.extend(module(name) for name in ("chia", "ray", "yaml", "z3"))
    checks.append(env("CHIA_PROJECT_ROOT"))
    checks.append(env("CHIA_HEAD_IP"))

    head_ip = os.environ.get("CHIA_HEAD_IP")
    if head_ip and shutil.which("ssh"):
        user = os.environ.get("USER", "")
        target = f"{user}@{head_ip}" if user else head_ip
        checks.append(
            run_check(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "true"],
                "Passwordless SSH to CHIA_HEAD_IP",
                "Create/authorize an SSH key and verify `ssh -o BatchMode=yes $USER@$CHIA_HEAD_IP true` succeeds.",
            )
        )

    if args.engine in {"openevolve", "alphaevolve"}:
        checks.append(module("evolve_flows"))
        checks.append(module("skydiscover"))

    if args.engine == "openevolve":
        checks.append(module("openevolve"))
        checks.append(module("google.auth"))
        checks.append(command("gcloud"))
        checks.append(env("GOOGLE_CLOUD_PROJECT"))
        if shutil.which("gcloud"):
            checks.append(
                run_check(
                    ["gcloud", "auth", "application-default", "print-access-token"],
                    "Google ADC",
                    "Run `gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/cloud-platform` on a headless server.",
                )
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION") or "global"
        ok("OpenEvolve Vertex location", location)
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            print("[WARN] GEMINI_API_KEY/GOOGLE_API_KEY is not used by this repository's OpenEvolve Vertex ADC path.")

    if args.engine == "alphaevolve":
        checks.append(module("alpha_evolve"))
        checks.append(module("google.auth"))
        checks.append(command("gcloud"))
        checks.append(env("ALPHAEVOLVE_PROJECT_ID"))
        checks.append(env("ALPHAEVOLVE_ENGINE_ID"))
        if shutil.which("gcloud"):
            checks.append(run_check(["gcloud", "auth", "application-default", "print-access-token"], "Google ADC", "Run `gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/cloud-platform` on a headless server."))

    if args.engine == "llm_agent":
        checks.append(command("gcloud"))
        checks.append(env("GOOGLE_CLOUD_PROJECT"))
        if shutil.which("gcloud"):
            checks.append(run_check(["gcloud", "auth", "application-default", "print-access-token"], "Google ADC", "Run `gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/cloud-platform` on a headless server."))

    if args.analysis:
        checks.extend(module(name) for name in ("pandas", "matplotlib"))

    project_root = os.environ.get("CHIA_PROJECT_ROOT")
    if project_root and Path(project_root).resolve() == Path.cwd().resolve():
        ok("CHIA_PROJECT_ROOT points to current repository")
        checks.append(True)
    elif project_root:
        fail("CHIA_PROJECT_ROOT points to current repository", project_root, "Run `export CHIA_PROJECT_ROOT=\"$PWD\"` from the repository root.")
        checks.append(False)

    if all(checks):
        print("Environment check passed.")
        return
    print("Environment check failed. Resolve the missing items above before starting the cluster.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
