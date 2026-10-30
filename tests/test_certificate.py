from __future__ import annotations

import copy
import unittest

from chia_trusted_verification.backends.teaal_evolve.certificate import (
    CertificateValidationError,
    validate_certificate_record,
)
from chia_trusted_verification.backends.teaal_evolve.worker import (
    _candidate_sha,
    _stable_sha256,
    _validate_certificate,
)


class CertificateTests(unittest.TestCase):
    def _payload(self):
        candidate = {"label": "x", "spec_dict": {"einsum": {}}, "knobs": {}}
        candidate_sha = _candidate_sha(candidate)
        catalog_sha = _stable_sha256([{"mutation_id": "m000"}])
        checks = [
            {
                "check_id": "legality.options",
                "category": "legality_stage",
                "status": "pass",
                "message": "passed",
                "evidence": {"error_type": None},
            },
            {
                "check_id": "local.res:x",
                "category": "local",
                "status": "pass",
                "message": "passed",
                "evidence": {},
            },
        ]
        body = {
            "schema_version": 1,
            "candidate_sha256": candidate_sha,
            "catalog_sha256": catalog_sha,
            "checker_backend": "hierarchical",
            "legal": True,
            "required_obligation_count": len(checks),
            "reused_obligation_count": 0,
            "freshly_checked_obligation_count": 1,
            "checks": checks,
            "execution": {
                "mode": "complete_hierarchical",
                "cache_hits": 0,
                "cache_misses": 1,
                "total_obligations_processed": 1,
                "cache_hit_rate": 0.0,
                "stage_hit_ids": [],
                "stage_miss_ids": [],
                "interface_hit_ids": [],
                "interface_miss_ids": [],
                "legality_hit_ids": [],
                "legality_miss_ids": ["res:x"],
                "extra_reused_legality_ids": [],
                "extra_reused_legality_count": 0,
            },
        }
        return {
            "stage": "verify",
            "status": "pass",
            "candidate": candidate,
            "candidate_sha256": candidate_sha,
            "catalog_sha256": catalog_sha,
            "legal": True,
            "checks": checks,
            "certificate": {**body, "certificate_sha256": _stable_sha256(body)},
        }

    @staticmethod
    def _rehash(payload) -> None:
        certificate = payload["certificate"]
        body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
        certificate["certificate_sha256"] = _stable_sha256(body)

    def test_valid_certificate(self) -> None:
        report = _validate_certificate(self._payload(), expected_checker_backend="hierarchical")
        self.assertTrue(report["valid"])
        self.assertEqual(report["check_count"], 2)

    def test_tampered_candidate_is_rejected(self) -> None:
        payload = self._payload()
        payload["candidate"]["spec_dict"]["einsum"] = {"changed": True}
        with self.assertRaises(CertificateValidationError):
            _validate_certificate(payload)

    def test_label_is_not_part_of_semantic_candidate_identity(self) -> None:
        payload = self._payload()
        payload["candidate"]["label"] = "renamed"
        _validate_certificate(payload)

    def test_rehashed_missing_obligation_is_rejected(self) -> None:
        payload = self._payload()
        payload["certificate"]["checks"].pop()
        self._rehash(payload)
        with self.assertRaisesRegex(CertificateValidationError, "required-obligation"):
            _validate_certificate(payload)

    def test_rehashed_flipped_global_result_is_rejected(self) -> None:
        payload = self._payload()
        payload["certificate"]["legal"] = False
        self._rehash(payload)
        with self.assertRaisesRegex(CertificateValidationError, "does not compose"):
            _validate_certificate(payload)

    def test_rehashed_wrong_catalog_is_rejected(self) -> None:
        payload = self._payload()
        payload["certificate"]["catalog_sha256"] = "0" * 64
        self._rehash(payload)
        with self.assertRaisesRegex(CertificateValidationError, "mutation catalog"):
            _validate_certificate(payload)

    def test_rehashed_bad_reuse_accounting_is_rejected(self) -> None:
        payload = self._payload()
        payload["certificate"]["execution"]["cache_hits"] = 1
        self._rehash(payload)
        with self.assertRaisesRegex(CertificateValidationError, "cache-hit count"):
            _validate_certificate(payload)

    def test_saved_evidence_record_can_be_validated_without_checker(self) -> None:
        payload = self._payload()
        certificate = copy.deepcopy(payload["certificate"])
        report = validate_certificate_record(
            certificate,
            expected_candidate_sha256=payload["candidate_sha256"],
            expected_catalog_sha256=payload["catalog_sha256"],
            expected_checker_backend="hierarchical",
            expected_checks=payload["checks"],
            expected_legal=True,
        )
        self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
