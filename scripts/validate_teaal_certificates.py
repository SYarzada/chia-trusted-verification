#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chia_trusted_verification.backends.teaal_evolve.certificate import (
    CertificateValidationError,
    validate_certificate_record,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("evidence file must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently validate saved TeAAL-Evolve certificates without invoking TeAAL."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="artifacts",
        help="artifact directory (default: artifacts)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first invalid certificate",
    )
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.rglob("trusted_evidence_json"))
    valid = 0
    invalid = 0
    skipped = 0

    for path in paths:
        try:
            evidence = _load(path)
            certificate = evidence.get("certificate")
            if not isinstance(certificate, dict):
                skipped += 1
                continue
            report = validate_certificate_record(
                certificate,
                expected_checks=evidence.get("checks") if isinstance(evidence.get("checks"), list) else None,
                expected_legal=bool(certificate.get("legal", False)),
            )
            valid += 1
            if args.fail_fast:
                # Keep output quiet for valid records in fail-fast mode.
                continue
        except (CertificateValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid += 1
            print(f"INVALID {path}: {exc}")
            if args.fail_fast:
                return 1

    print(
        json.dumps(
            {
                "root": str(root),
                "evidence_files": len(paths),
                "valid_certificates": valid,
                "invalid_certificates": invalid,
                "skipped_without_certificate": skipped,
            },
            sort_keys=True,
        )
    )
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
