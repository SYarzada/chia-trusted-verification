from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

from .certificate import validate_pipeline_certificate

SCHEMA_VERSION = 1


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_api(config: Mapping[str, Any]) -> dict[str, Any]:
    teaal_root = Path(str(config["teaal_root"])).resolve()
    if not teaal_root.is_dir():
        raise FileNotFoundError(f"TeAAL-Evolve root not found: {teaal_root}")
    if str(teaal_root) not in sys.path:
        sys.path.insert(0, str(teaal_root))

    from src.trans.certify.candidate import Candidate
    from src.trans.certify.checker_backend import check_candidate_with_backend
    from src.trans.certify.mutate import apply_knob_specs_atomic
    from src.trans.certify.mutation_profiles import get_mutation_options
    from src.trans.certify.objective_backend import compute_objective_with_backend

    return {
        "Candidate": Candidate,
        "check_candidate_with_backend": check_candidate_with_backend,
        "apply_knob_specs_atomic": apply_knob_specs_atomic,
        "get_mutation_options": get_mutation_options,
        "compute_objective_with_backend": compute_objective_with_backend,
    }


def _seed_candidate(api: Mapping[str, Any], config: Mapping[str, Any]):
    return api["Candidate"].from_files(
        [str(config["kernel_file"]), str(config["architecture_file"])],
        label=str(config["kernel_label"]),
    )


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    return {
        "label": str(candidate.label),
        "spec_dict": candidate.spec_dict,
        "knobs": candidate.knobs,
    }


def _candidate_sha(candidate_payload: Mapping[str, Any]) -> str:
    return _stable_sha256(
        {
            "spec_dict": candidate_payload["spec_dict"],
            "knobs": candidate_payload.get("knobs", {}),
        }
    )


def _knob_id(knob_spec: str) -> str:
    return knob_spec.split("=", 1)[0].strip()


def _build_catalog(api: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = _seed_candidate(api, config)
    options = list(api["get_mutation_options"](str(config["kernel_label"]), purpose="search"))
    if int(config.get("catalog_seed", 0)):
        random.Random(int(config["catalog_seed"])).shuffle(options)

    seed_sha = _stable_sha256(seed.spec_dict)
    effective: list[dict[str, Any]] = []
    for option in options:
        try:
            mutated = api["apply_knob_specs_atomic"](
                seed,
                [option.knob_spec],
                label=f"catalog-{len(effective)}",
                knob_space=str(config.get("knob_space", "rich")),
            )
        except Exception:
            continue
        if _stable_sha256(mutated.spec_dict) == seed_sha:
            continue
        effective.append(
            {
                "mutation_id": f"m{len(effective):03d}",
                "knob_spec": option.knob_spec,
                "knob_id": _knob_id(option.knob_spec),
                "family": option.family,
                "stage": option.stage,
                "severity": option.severity,
                "expected_legality": option.expected_legality,
                "objective_direction": option.objective_direction,
                "evidence_source": option.evidence_source,
            }
        )
    if not effective:
        raise ValueError(f"curated catalog for {config['kernel_label']!r} is empty")
    return effective


def catalog_stage(_: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    api = _load_api(config)
    catalog = _build_catalog(api, config)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "catalog",
        "status": "pass",
        "catalog": catalog,
        "catalog_sha256": _stable_sha256(catalog),
    }


def prepare_stage(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    api = _load_api(config)
    catalog = _build_catalog(api, config)
    proposal = dict(payload.get("proposal", {}))
    adapter_error = proposal.get("_adapter_error")
    if isinstance(adapter_error, Mapping):
        raise ValueError(
            f"search strategy rejected candidate: {adapter_error.get('message', 'unknown error')}"
        )
    raw_ids = proposal.get("mutation_ids", [])
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise ValueError("CANDIDATE.mutation_ids must be a list")
    mutation_ids = [str(item) for item in raw_ids]
    allow_seed = bool(payload.get("allow_seed_candidate", False))
    minimum = int(config.get("min_mutations", 1))
    maximum = int(config.get("max_mutations", 2))
    if not mutation_ids and not allow_seed:
        raise ValueError(f"proposal must select at least {minimum} mutation id(s)")
    if mutation_ids and not minimum <= len(mutation_ids) <= maximum:
        raise ValueError(
            f"proposal selected {len(mutation_ids)} mutations; expected {minimum}..{maximum}"
        )
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("mutation_ids must be unique")

    by_id = {row["mutation_id"]: row for row in catalog}
    unknown = [mutation_id for mutation_id in mutation_ids if mutation_id not in by_id]
    if unknown:
        raise ValueError("unknown mutation id(s): " + ", ".join(unknown))
    selected = [by_id[mutation_id] for mutation_id in mutation_ids]
    knob_ids = [row["knob_id"] for row in selected]
    if len(knob_ids) != len(set(knob_ids)):
        raise ValueError("a proposal may not select two mutations for the same knob")

    seed = _seed_candidate(api, config)
    label = str(proposal.get("label", "candidate"))
    candidate = (
        api["apply_knob_specs_atomic"](
            seed,
            [row["knob_spec"] for row in selected],
            label=label,
            knob_space=str(config.get("knob_space", "rich")),
        )
        if selected
        else seed.clone(label=label)
    )
    encoded_candidate = _candidate_dict(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare",
        "status": "pass",
        "proposal": {"label": label, "mutation_ids": mutation_ids},
        "selected_mutations": selected,
        "catalog_sha256": _stable_sha256(catalog),
        "candidate": encoded_candidate,
        "candidate_sha256": _candidate_sha(encoded_candidate),
        "diagnostics": [],
    }


def _status(value: str | None) -> str:
    normalized = str(value or "pass").lower()
    if normalized in {"pass", "passed", "ok", "legal", "success"}:
        return "pass"
    if normalized in {"skip", "skipped", "disabled"}:
        return "skip"
    if normalized == "error":
        return "error"
    return "fail"


def _report_checks(report: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for stage in report.legality_obligations:
        checks.append(
            {
                "check_id": f"legality.{stage.stage_name}",
                "category": "legality_stage",
                "status": "pass" if stage.ok else "fail",
                "message": stage.message or ("passed" if stage.ok else "failed"),
                "evidence": {"error_type": stage.error_type},
            }
        )
    for group_name, obligations in (
        ("local", report.local_legality_obligations),
        ("stage", report.stage_obligations),
        ("interface", report.interface_obligations),
    ):
        for obligation in obligations:
            row = asdict(obligation)
            check_status = _status(row.pop("status", None))
            obligation_id = str(row.pop("obligation_id"))
            message = str(row.pop("detail", ""))
            checks.append(
                {
                    "check_id": f"{group_name}.{obligation_id}",
                    "category": group_name,
                    "status": check_status,
                    "message": message,
                    "evidence": row,
                }
            )
    return checks


def verify_stage(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "pass":
        return {
            **dict(payload),
            "stage": "verify",
            "status": "error",
            "diagnostics": list(payload.get("diagnostics", [])),
        }
    api = _load_api(config)
    encoded = dict(payload["candidate"])
    candidate = api["Candidate"].from_dict(
        encoded["spec_dict"],
        label=str(encoded["label"]),
        knobs=dict(encoded.get("knobs", {})),
    )
    checker = api["check_candidate_with_backend"](
        candidate,
        backend_name=str(config.get("checker_backend", "hierarchical")),
        include_bottleneck_objective=False,
        knob_space=str(config.get("knob_space", "rich")),
    )
    report = checker.candidate_report
    checks = _report_checks(report)
    required_obligations = len(checks)
    reused_obligations = int(checker.total_hits)
    freshly_checked_obligations = (
        int(checker.total_misses)
        if checker.total_hits or checker.total_misses
        else required_obligations
    )
    certificate_body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha256": str(payload["candidate_sha256"]),
        "catalog_sha256": str(payload["catalog_sha256"]),
        "checker_backend": str(config.get("checker_backend", "hierarchical")),
        "legal": bool(report.is_legal),
        "required_obligation_count": required_obligations,
        "reused_obligation_count": reused_obligations,
        "freshly_checked_obligation_count": freshly_checked_obligations,
        "checks": checks,
        "execution": {
            "mode": checker.execution_mode,
            "cache_hits": checker.total_hits,
            "cache_misses": checker.total_misses,
            "total_obligations_processed": checker.total_obligations_processed,
            "cache_hit_rate": checker.hit_rate,
            "stage_hit_ids": list(checker.stage_hit_ids),
            "stage_miss_ids": list(checker.stage_miss_ids),
            "interface_hit_ids": list(checker.interface_hit_ids),
            "interface_miss_ids": list(checker.interface_miss_ids),
            "legality_hit_ids": list(checker.legality_hit_ids),
            "legality_miss_ids": list(checker.legality_miss_ids),
            "syntactic_legality_reuse_rate": checker.syntactic_legality_reuse_rate,
            "rewrite_legality_reuse_rate": checker.rewrite_legality_reuse_rate,
            "extra_reused_legality_ids": list(checker.extra_reused_legality_ids),
            "extra_reused_legality_count": len(checker.extra_reused_legality_ids),
            "incremental_fallback_reason": checker.incremental_fallback_reason,
            "incremental_delta_source": checker.incremental_delta_source,
        },
    }
    certificate = {
        **certificate_body,
        "certificate_sha256": _stable_sha256(certificate_body),
    }
    certificate_size_bytes = len(
        json.dumps(certificate, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return {
        **dict(payload),
        "stage": "verify",
        "status": "pass" if report.is_legal else "fail",
        "legal": bool(report.is_legal),
        "checks": checks,
        "certificate": certificate,
        "certificate_size_bytes": certificate_size_bytes,
        "diagnostics": [],
    }


def _validate_certificate(
    payload: Mapping[str, Any],
    *,
    expected_checker_backend: str | None = None,
) -> dict[str, Any]:
    # Kept as a small compatibility wrapper for existing callers/tests. The
    # independent validator itself lives in certificate.py and does not import
    # TeAAL or the worker/checker implementation.
    return validate_pipeline_certificate(
        payload,
        expected_checker_backend=expected_checker_backend,
    )


def evaluate_stage(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "verify" or "certificate" not in payload:
        return {
            **dict(payload),
            "stage": "evaluate",
            "status": "error",
            "diagnostics": list(payload.get("diagnostics", [])),
        }
    validation_started = time.perf_counter()
    certificate_validation = _validate_certificate(
        payload,
        expected_checker_backend=str(config.get("checker_backend", "hierarchical")),
    )
    certificate_validation_time_s = time.perf_counter() - validation_started
    if not bool(payload.get("legal", False)):
        return {
            **dict(payload),
            "stage": "evaluate",
            "status": "fail",
            "certificate_validation": certificate_validation,
            "certificate_validation_time_s": certificate_validation_time_s,
            "objective": {
                "status": "skipped",
                "value": None,
                "message": "objective skipped because trusted verification failed",
            },
        }

    api = _load_api(config)
    encoded = dict(payload["candidate"])
    candidate = api["Candidate"].from_dict(
        encoded["spec_dict"],
        label=str(encoded["label"]),
        knobs=dict(encoded.get("knobs", {})),
    )
    objective = api["compute_objective_with_backend"](
        candidate,
        backend_name=str(config.get("objective_backend", "monolithic_bottleneck")),
        shapes=dict(config.get("objective_shapes", {})) or None,
    )
    objective_payload = {
        "status": str(objective.status),
        "value": None if objective.value is None else float(objective.value),
        "name": objective.objective_name,
        "error_type": objective.error_type,
        "message": objective.message,
    }
    return {
        **dict(payload),
        "stage": "evaluate",
        "status": "pass" if objective.status == "ok" and objective.value is not None else "error",
        "certificate_validation": certificate_validation,
        "certificate_validation_time_s": certificate_validation_time_s,
        "objective": objective_payload,
    }


STAGES = {
    "catalog": catalog_stage,
    "prepare": prepare_stage,
    "verify": verify_stage,
    "evaluate": evaluate_stage,
}


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(request.get("stage", ""))
    if stage not in STAGES:
        raise ValueError(f"unsupported worker stage: {stage!r}")
    return STAGES[stage](
        dict(request.get("payload", {})),
        dict(request.get("config", {})),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        response = execute(request)
    except Exception as exc:
        stage = str(request.get("stage", "unknown"))
        response = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "status": "error",
            "diagnostics": [
                {
                    "code": f"{stage}.{type(exc).__name__}",
                    "message": str(exc),
                }
            ],
        }
    Path(args.response).write_text(
        json.dumps(response, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
