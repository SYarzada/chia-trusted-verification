from __future__ import annotations

import time
from typing import Any, Mapping

from chia_trusted_verification.core import TrustedVerificationLayer
from chia_trusted_verification.types import PIPELINE_SCHEMA_VERSION, VerificationIntent

from .runtime import ChiaFunction


def _layer(
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> TrustedVerificationLayer:
    return TrustedVerificationLayer(
        backend_name=backend_name,
        backend_settings=backend_settings,
        intent=VerificationIntent.from_dict(intent),
    )


def _with_timing(stage: str, result: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    value = dict(result)
    timings = dict(value.get("stage_timings_s", {}))
    timings[stage] = float(elapsed_s)
    value["stage_timings_s"] = timings
    return value


def _error(
    stage: str,
    backend_name: str,
    intent: Mapping[str, Any],
    exc: Exception,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(prior or {})
    result.update(
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "stage": stage,
            "backend": backend_name,
            "intent_id": str(intent.get("intent_id", "default")),
            "status": "error",
            "diagnostics": [
                {
                    "code": f"{stage}.{type(exc).__name__}",
                    "message": str(exc),
                }
            ],
        }
    )
    return result


def prepare_artifact_local(
    candidate_source: str,
    context: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        return _with_timing(
            "prepare",
            _layer(backend_name, backend_settings, intent).prepare(candidate_source, context),
            time.perf_counter() - start,
        )
    except Exception as exc:
        return _with_timing(
            "prepare",
            _error("prepare", backend_name, intent, exc),
            time.perf_counter() - start,
        )


def verify_artifact_local(
    prepared: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        return _with_timing(
            "verify",
            _layer(backend_name, backend_settings, intent).verify(prepared),
            time.perf_counter() - start,
        )
    except Exception as exc:
        return _with_timing(
            "verify",
            _error("verify", backend_name, intent, exc, prepared),
            time.perf_counter() - start,
        )


def evaluate_artifact_local(
    verified: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        return _with_timing(
            "evaluate",
            _layer(backend_name, backend_settings, intent).evaluate(verified),
            time.perf_counter() - start,
        )
    except Exception as exc:
        return _with_timing(
            "evaluate",
            _error("evaluate", backend_name, intent, exc, verified),
            time.perf_counter() - start,
        )


def feedback_for_search_local(
    evaluated: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = _layer(backend_name, backend_settings, intent).feedback(evaluated)
    except Exception as exc:
        result = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "stage": "feedback",
            "backend": backend_name,
            "intent_id": str(intent.get("intent_id", "default")),
            "status": "error",
            "metrics": {
                "combined_score": 0.0,
                "score": 0.0,
                "runs_successfully": 0.0,
            },
            "feedback": {
                "summary": "Trusted verification feedback generation failed.",
                "suggested_action": "Inspect the backend diagnostic and repair the candidate or backend configuration.",
                "messages": [str(exc)],
            },
            "evidence": {},
            "trace": {
                "terminal_stage": str(evaluated.get("stage", "evaluate")),
                "terminal_status": str(evaluated.get("status", "error")),
            },
            "diagnostics": [
                {
                    "code": f"feedback.{type(exc).__name__}",
                    "message": str(exc),
                }
            ],
            "stage_timings_s": dict(evaluated.get("stage_timings_s", {})),
        }
    return _with_timing("feedback", result, time.perf_counter() - start)


@ChiaFunction(num_cpus=1, max_retries=1)
def prepare_artifact(
    candidate_source: str,
    context: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    return prepare_artifact_local(
        candidate_source,
        context,
        backend_name,
        backend_settings,
        intent,
    )


@ChiaFunction(num_cpus=1, max_retries=1)
def verify_artifact(
    prepared: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    return verify_artifact_local(prepared, backend_name, backend_settings, intent)


@ChiaFunction(num_cpus=1, max_retries=1)
def evaluate_artifact(
    verified: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    return evaluate_artifact_local(verified, backend_name, backend_settings, intent)


@ChiaFunction(num_cpus=0.1, max_retries=1)
def feedback_for_search(
    evaluated: Mapping[str, Any],
    backend_name: str,
    backend_settings: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    return feedback_for_search_local(evaluated, backend_name, backend_settings, intent)
