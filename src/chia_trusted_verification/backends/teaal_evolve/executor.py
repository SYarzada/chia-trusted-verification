from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


def run_worker_stage(
    stage: str,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one trusted stage in a fresh process and return its JSON envelope."""

    timeout = int(config.get("node_timeout_s", 300))
    request = {"stage": stage, "payload": dict(payload), "config": dict(config)}
    with tempfile.TemporaryDirectory(prefix=f"chia-teaal-{stage}-") as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "chia_trusted_verification.backends.teaal_evolve.worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _error(stage, "TimeoutExpired", f"stage exceeded {timeout}s")
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            return _error(
                stage,
                "WorkerProcessError",
                detail or f"worker exited with code {completed.returncode}",
            )
        if not response_path.exists():
            return _error(stage, "MissingWorkerResponse", "worker wrote no response")
        try:
            return dict(json.loads(response_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return _error(stage, type(exc).__name__, f"invalid worker response: {exc}")


def _error(stage: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "error",
        "diagnostics": [
            {"code": f"{stage}.{error_type}", "message": message[:8000]}
        ],
    }
