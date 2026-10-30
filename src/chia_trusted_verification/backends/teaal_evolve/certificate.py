from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

CERTIFICATE_VALIDATOR_VERSION = 1
SUPPORTED_CERTIFICATE_SCHEMA_VERSIONS = {1}
SUPPORTED_CHECKER_BACKENDS = {
    "full",
    "hierarchical",
    "hierarchical_delta_syntactic",
    "hierarchical_delta_egraph",
}
VALID_CHECK_STATUSES = {"pass", "fail", "skip", "error"}


class CertificateValidationError(ValueError):
    """Raised when a TeAAL-Evolve certificate is internally inconsistent."""


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_sha(candidate_payload: Mapping[str, Any]) -> str:
    return _stable_sha256(
        {
            "spec_dict": candidate_payload["spec_dict"],
            "knobs": candidate_payload.get("knobs", {}),
        }
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateValidationError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _as_string_list(value: Any, field: str) -> list[str]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"certificate execution field {field!r} must be a list",
    )
    result = [str(item) for item in value]
    _require(len(result) == len(set(result)), f"certificate execution field {field!r} contains duplicates")
    return result


def _validate_execution(certificate: Mapping[str, Any]) -> None:
    execution = certificate.get("execution", {})
    _require(isinstance(execution, Mapping), "certificate execution must be an object")

    hit_fields = ("stage_hit_ids", "interface_hit_ids", "legality_hit_ids")
    miss_fields = ("stage_miss_ids", "interface_miss_ids", "legality_miss_ids")
    hits = {field: _as_string_list(execution.get(field, []), field) for field in hit_fields}
    misses = {field: _as_string_list(execution.get(field, []), field) for field in miss_fields}

    for hit_field, miss_field in zip(hit_fields, miss_fields):
        overlap = set(hits[hit_field]) & set(misses[miss_field])
        _require(not overlap, f"certificate marks obligations as both reused and checked in {hit_field}/{miss_field}")

    hit_count = sum(len(values) for values in hits.values())
    miss_count = sum(len(values) for values in misses.values())

    if "cache_hits" in execution:
        _require(int(execution["cache_hits"]) == hit_count, "certificate cache-hit count is inconsistent")
    if "cache_misses" in execution:
        _require(int(execution["cache_misses"]) == miss_count, "certificate cache-miss count is inconsistent")
    if "total_obligations_processed" in execution:
        _require(
            int(execution["total_obligations_processed"]) == hit_count + miss_count,
            "certificate processed-obligation count is inconsistent",
        )
    if "cache_hit_rate" in execution and hit_count + miss_count:
        expected_rate = hit_count / (hit_count + miss_count)
        _require(
            math.isclose(float(execution["cache_hit_rate"]), expected_rate, rel_tol=1e-9, abs_tol=1e-12),
            "certificate cache-hit rate is inconsistent",
        )

    if "reused_obligation_count" in certificate:
        _require(
            int(certificate["reused_obligation_count"]) == hit_count,
            "certificate reused-obligation count is inconsistent",
        )
    if "freshly_checked_obligation_count" in certificate and hit_count + miss_count:
        _require(
            int(certificate["freshly_checked_obligation_count"]) == miss_count,
            "certificate freshly-checked obligation count is inconsistent",
        )

    extra_ids = _as_string_list(execution.get("extra_reused_legality_ids", []), "extra_reused_legality_ids")
    if "extra_reused_legality_count" in execution:
        _require(
            int(execution["extra_reused_legality_count"]) == len(extra_ids),
            "certificate rewrite-reuse count is inconsistent",
        )
    _require(
        set(extra_ids).issubset(set(hits["legality_hit_ids"])),
        "rewrite-only reused obligations must also appear in legality_hit_ids",
    )


def validate_certificate_record(
    certificate_value: Mapping[str, Any],
    *,
    expected_candidate_sha256: str | None = None,
    expected_catalog_sha256: str | None = None,
    expected_checker_backend: str | None = None,
    expected_checks: Sequence[Mapping[str, Any]] | None = None,
    expected_legal: bool | None = None,
) -> dict[str, Any]:
    """Validate a certificate without invoking TeAAL or any trusted checker.

    This validator checks certificate integrity, identity binding, obligation
    composition, and reuse-accounting consistency. It deliberately does not
    rerun primitive TeAAL semantic or legality checks.
    """

    _require(isinstance(certificate_value, Mapping), "verification certificate must be an object")
    certificate = dict(certificate_value)
    recorded_digest = certificate.pop("certificate_sha256", None)
    _require(_is_sha256(recorded_digest), "verification certificate is missing a valid digest")
    _require(recorded_digest == _stable_sha256(certificate), "verification certificate digest mismatch")

    schema_version = certificate.get("schema_version")
    _require(
        schema_version in SUPPORTED_CERTIFICATE_SCHEMA_VERSIONS,
        f"unsupported verification certificate schema: {schema_version!r}",
    )

    candidate_sha = certificate.get("candidate_sha256")
    catalog_sha = certificate.get("catalog_sha256")
    _require(_is_sha256(candidate_sha), "certificate candidate_sha256 is invalid")
    _require(_is_sha256(catalog_sha), "certificate catalog_sha256 is invalid")
    if expected_candidate_sha256 is not None:
        _require(candidate_sha == expected_candidate_sha256, "certificate does not identify this candidate")
    if expected_catalog_sha256 is not None:
        _require(catalog_sha == expected_catalog_sha256, "certificate does not identify this mutation catalog")

    checker_backend = certificate.get("checker_backend")
    _require(checker_backend in SUPPORTED_CHECKER_BACKENDS, "certificate checker backend is unsupported")
    if expected_checker_backend is not None:
        _require(checker_backend == expected_checker_backend, "certificate checker backend does not match configuration")

    checks = certificate.get("checks")
    _require(isinstance(checks, list) and checks, "certificate must contain at least one verification check")
    normalized_checks: list[Mapping[str, Any]] = []
    for index, check in enumerate(checks):
        _require(isinstance(check, Mapping), f"certificate check {index} must be an object")
        _require(bool(str(check.get("check_id", ""))), f"certificate check {index} is missing check_id")
        _require(bool(str(check.get("category", ""))), f"certificate check {index} is missing category")
        _require(check.get("status") in VALID_CHECK_STATUSES, f"certificate check {index} has invalid status")
        normalized_checks.append(check)

    if "required_obligation_count" in certificate:
        _require(
            int(certificate["required_obligation_count"]) == len(checks),
            "certificate required-obligation count is inconsistent",
        )

    if expected_checks is not None:
        _require(list(expected_checks) == checks, "certificate checks do not match the verified payload")

    # TeAAL-Evolve's global legality result is the conjunction of its emitted
    # verification obligations. No objective result participates in legality.
    composed_legal = all(check.get("status") == "pass" for check in normalized_checks)
    _require(bool(certificate.get("legal")) == composed_legal, "certificate legal result does not compose from its checks")
    if expected_legal is not None:
        _require(bool(certificate.get("legal")) == bool(expected_legal), "certificate legal result does not match payload")

    _validate_execution(certificate)

    return {
        "valid": True,
        "validator_version": CERTIFICATE_VALIDATOR_VERSION,
        "certificate_sha256": recorded_digest,
        "candidate_sha256": candidate_sha,
        "catalog_sha256": catalog_sha,
        "checker_backend": checker_backend,
        "legal": bool(certificate.get("legal")),
        "check_count": len(checks),
    }


def validate_pipeline_certificate(
    payload: Mapping[str, Any],
    *,
    expected_checker_backend: str | None = None,
) -> dict[str, Any]:
    """Validate a certificate against the concrete CHIA pipeline payload."""

    candidate = payload.get("candidate")
    _require(isinstance(candidate, Mapping), "verified payload is missing candidate")
    candidate_sha = payload.get("candidate_sha256")
    _require(_is_sha256(candidate_sha), "verified payload candidate_sha256 is invalid")
    _require(_candidate_sha(candidate) == candidate_sha, "candidate payload digest mismatch")

    catalog_sha = payload.get("catalog_sha256")
    _require(_is_sha256(catalog_sha), "verified payload catalog_sha256 is invalid")

    return validate_certificate_record(
        payload.get("certificate", {}),
        expected_candidate_sha256=str(candidate_sha),
        expected_catalog_sha256=str(catalog_sha),
        expected_checker_backend=expected_checker_backend,
        expected_checks=payload.get("checks") if isinstance(payload.get("checks"), list) else None,
        expected_legal=bool(payload.get("legal", False)),
    )
