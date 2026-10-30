from __future__ import annotations

import argparse

from chia_trusted_verification.chia.runtime import ChiaFunction, get, require_chia


@ChiaFunction(num_cpus=0.1, max_retries=0)
def probe_worker(resource_name: str) -> dict[str, str]:
    import os
    import socket

    result = {
        "resource": resource_name,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
    }
    if resource_name in {"teaal_evolve", "trusted_verification_tool"}:
        project_root = os.environ.get("CHIA_PROJECT_ROOT", "")
        teaal_root = os.path.join(project_root, "teaal_evolve") if project_root else ""
        runs_root = os.path.join(project_root, "runs") if project_root else ""
        result.update(
            {
                "project_root": project_root,
                "teaal_root": teaal_root,
                "teaal_root_exists": bool(teaal_root) and os.path.isdir(teaal_root),
                "runs_root": runs_root,
                "runs_root_writable": bool(runs_root) and os.path.isdir(runs_root) and os.access(runs_root, os.W_OK),
            }
        )
        if not result["teaal_root_exists"]:
            raise RuntimeError(f"TeAAL-Evolve root is not visible on {resource_name}: {teaal_root}")
        if not result["runs_root_writable"]:
            raise RuntimeError(f"Trusted results directory is not writable on {resource_name}: {runs_root}")

    if resource_name in {"opencode_creds", "vertex_creds"}:
        adc = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
        )
        result.update(
            {
                "adc_path": adc,
                "adc_exists": os.path.exists(adc),
                "adc_readable": os.access(adc, os.R_OK),
            }
        )
        if not result["adc_readable"]:
            raise RuntimeError(f"Google ADC is not readable on {resource_name}: {adc}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "resources",
        nargs="*",
        default=["teaal_evolve", "trusted_verification_tool", "vertex_creds"],
    )
    args = parser.parse_args()
    require_chia()
    for resource in args.resources:
        print(f"Probing {resource}...", flush=True)
        ref = probe_worker.options(resources={resource: 0.01}).chia_remote(resource)
        print(f"Submitted {resource}, waiting for result...", flush=True)
        result = get(ref, timeout=15)
        print(result, flush=True)


if __name__ == "__main__":
    main()
