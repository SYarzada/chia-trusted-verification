from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.types import VerificationIntent


class _FakeNode:
    def __init__(self, name, function, calls):
        self.name = name
        self.function = function
        self.calls = calls

    def options(self, **kwargs):
        del kwargs
        return self

    def chia_remote(self, *args, **kwargs):
        self.calls.append((self.name, kwargs.get("_chia_tag")))
        clean = {key: value for key, value in kwargs.items() if not key.startswith("_chia_")}
        return self.function(*args, **clean)


class ChiaGraphTests(unittest.TestCase):
    def test_evaluator_chains_all_four_generic_nodes(self) -> None:
        calls = []
        source = "candidate source"
        prepare = _FakeNode(
            "prepare",
            lambda source, context, backend, settings, intent: {
                "stage": "prepare",
                "status": "pass",
                "source": source,
            },
            calls,
        )
        verify = _FakeNode(
            "verify",
            lambda prepared, backend, settings, intent: {
                **prepared,
                "stage": "verify",
                "status": "pass",
            },
            calls,
        )
        evaluate = _FakeNode(
            "evaluate",
            lambda verified, backend, settings, intent: {
                **verified,
                "stage": "evaluate",
                "status": "pass",
            },
            calls,
        )
        feedback = _FakeNode(
            "feedback",
            lambda evaluated, backend, settings, intent: {
                "backend": backend,
                "intent_id": intent["intent_id"],
                "status": "pass",
                "metrics": {"combined_score": 1.0},
                "feedback": {"summary": "ok"},
                "evidence": {},
                "trace": {"terminal_stage": evaluated["stage"]},
            },
            calls,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "chia_trusted_verification.chia.evaluator.require_chia"
        ), patch(
            "chia_trusted_verification.chia.evaluator.CHIA_AVAILABLE", True
        ), patch(
            "chia_trusted_verification.chia.evaluator.get", side_effect=lambda value: value
        ), patch(
            "chia_trusted_verification.chia.evaluator.prepare_artifact", prepare
        ), patch(
            "chia_trusted_verification.chia.evaluator.verify_artifact", verify
        ), patch(
            "chia_trusted_verification.chia.evaluator.evaluate_artifact", evaluate
        ), patch(
            "chia_trusted_verification.chia.evaluator.feedback_for_search", feedback
        ):
            evaluator = TrustedVerificationEvaluator(
                backend_name="toy",
                backend_settings={},
                backend_resources={},
                intent=VerificationIntent("toy-intent", "toy-artifact"),
                engine="alphaevolve",
                initial_source=source,
                result_log=Path(tmp) / "results.jsonl",
                use_chia=True,
            )
            result = evaluator.evaluate_source(source)

        self.assertEqual([name for name, _ in calls], ["prepare", "verify", "evaluate", "feedback"])
        self.assertEqual(result.trace["terminal_stage"], "evaluate")
        self.assertTrue(all(tag for _, tag in calls))


if __name__ == "__main__":
    unittest.main()
