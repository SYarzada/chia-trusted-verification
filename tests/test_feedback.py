from __future__ import annotations

import unittest

from chia_trusted_verification.backends.teaal_evolve.backend import TeaalEvolveBackend
from chia_trusted_verification.types import VerificationIntent


INTENT = VerificationIntent(
    intent_id="test",
    artifact_type="structured_accelerator_candidate",
    requirements=(
        "mapping_format_binding_legality",
        "local_semantics",
        "producer_consumer_interfaces",
        "structured_resource_legality",
    ),
)


class FeedbackTests(unittest.TestCase):
    def test_legal_objective_result_gets_positive_score(self) -> None:
        result = TeaalEvolveBackend().feedback(
            {
                "stage": "evaluate",
                "status": "pass",
                "legal": True,
                "objective": {"status": "ok", "value": 9.0},
                "checks": [],
                "proposal": {"mutation_ids": ["m001"]},
            },
            INTENT,
            {},
        )
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["metrics"]["combined_score"], 1.0)

    def test_illegal_result_preserves_failed_obligations(self) -> None:
        result = TeaalEvolveBackend().feedback(
            {
                "stage": "evaluate",
                "status": "fail",
                "legal": False,
                "objective": {"status": "skipped", "value": None},
                "checks": [
                    {
                        "check_id": "interface.i0",
                        "category": "interface",
                        "status": "fail",
                        "message": "producer/consumer mismatch",
                    }
                ],
                "proposal": {"mutation_ids": ["m004"]},
            },
            INTENT,
            {},
        )
        self.assertEqual(result["metrics"]["combined_score"], -1.0)
        self.assertEqual(result["feedback"]["failed_check_ids"], ["interface.i0"])

    def test_more_failed_obligations_receive_lower_score(self) -> None:
        result = TeaalEvolveBackend().feedback(
            {
                "stage": "evaluate",
                "status": "fail",
                "legal": False,
                "objective": {"status": "skipped", "value": None},
                "checks": [
                    {
                        "check_id": "legality.binding",
                        "category": "legality_stage",
                        "status": "fail",
                        "message": "binding mismatch",
                    },
                    {
                        "check_id": "legality.architecture",
                        "category": "legality_stage",
                        "status": "fail",
                        "message": "architecture mismatch",
                    },
                    {
                        "check_id": "interface.i0",
                        "category": "interface",
                        "status": "fail",
                        "message": "interface mismatch",
                    },
                ],
                "proposal": {"mutation_ids": ["m015", "m017"]},
            },
            INTENT,
            {},
        )
        self.assertEqual(result["metrics"]["combined_score"], -3.0)
        self.assertIn("Do not repeat this exact mutation set", result["feedback"]["suggested_action"])

    def test_pipeline_error_receives_large_negative_penalty(self) -> None:
        result = TeaalEvolveBackend().feedback(
            {
                "stage": "evaluate",
                "status": "error",
                "legal": False,
                "objective": {"status": "error", "value": None},
                "checks": [],
                "diagnostics": [{"message": "candidate parse failed"}],
                "proposal": {},
            },
            INTENT,
            {},
        )
        self.assertEqual(result["metrics"]["combined_score"], -100.0)
        self.assertEqual(result["status"], "error")

    def test_legal_candidate_with_missing_objective_is_penalized(self) -> None:
        result = TeaalEvolveBackend().feedback(
            {
                "stage": "evaluate",
                "status": "fail",
                "legal": True,
                "objective": {"status": "error", "value": None},
                "checks": [],
                "proposal": {"mutation_ids": ["m006"]},
            },
            INTENT,
            {},
        )
        self.assertEqual(result["metrics"]["combined_score"], -50.0)


if __name__ == "__main__":
    unittest.main()
