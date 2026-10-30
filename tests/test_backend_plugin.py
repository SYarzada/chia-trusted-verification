from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chia_trusted_verification import TrustedVerificationBackend, register_backend
from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.types import SearchProblem, VerificationIntent


class ToyBackend(TrustedVerificationBackend):
    name = "toy"

    def build_search_problem(self, settings):
        return SearchProblem("TOY = 1\n", "toy.py")

    def prepare(self, candidate_source, context, intent, settings):
        return {"status": "pass", "artifact": candidate_source, "context": dict(context)}

    def verify(self, prepared, intent, settings):
        return {**dict(prepared), "status": "pass", "verified": True}

    def evaluate(self, verified, intent, settings):
        return {**dict(verified), "status": "pass", "value": 4.0}

    def feedback(self, evaluated, intent, settings):
        return {
            "status": "pass",
            "metrics": {"combined_score": float(evaluated["value"])},
            "feedback": {"summary": "toy pass"},
            "evidence": {"verified": bool(evaluated["verified"])},
            "trace": {"terminal_stage": "evaluate"},
        }


class BackendPluginTests(unittest.TestCase):
    def test_new_backend_runs_without_changing_chia_nodes(self) -> None:
        register_backend("toy", ToyBackend, replace=True)
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = TrustedVerificationEvaluator(
                backend_name="toy",
                backend_settings={},
                backend_resources={},
                intent=VerificationIntent("toy-intent", "toy"),
                engine="openevolve",
                initial_source="TOY = 1\n",
                result_log=Path(tmp) / "results.jsonl",
                use_chia=False,
            )
            result = evaluator.evaluate_source("TOY = 2\n")
        self.assertEqual(result.backend, "toy")
        self.assertEqual(result.intent_id, "toy-intent")
        self.assertEqual(result.metrics["combined_score"], 4.0)
        self.assertTrue(result.evidence["verified"])


if __name__ == "__main__":
    unittest.main()
