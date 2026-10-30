from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chia_trusted_verification.search.skydiscover import (
    _make_alphaevolve_evaluator,
)


class AlphaEvolveFeedbackBridgeTests(unittest.TestCase):
    def test_metrics_and_structured_insights_are_both_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator_path = Path(tmp) / "evaluator.py"
            evaluator_path.write_text(
                "def evaluate(path):\n"
                "    return {\n"
                "        'combined_score': 1.25,\n"
                "        '__trusted_status': 'pass',\n"
                "        '__trusted_feedback_json': '{\"summary\": \"ok\"}',\n"
                "    }\n",
                encoding="utf-8",
            )
            evaluate = _make_alphaevolve_evaluator(str(evaluator_path))
            result = evaluate(
                {
                    "content": {
                        "files": [{"path": "candidate.py", "content": "x = 1\n"}]
                    }
                }
            )

        self.assertEqual(
            result["scores"]["scores"],
            [{"metric": "combined_score", "score": 1.25}],
        )
        self.assertEqual(
            result["insights"],
            {
                "insights": [
                    {
                        "label": "trusted_verification",
                        "text": '{"backend": null, "intent_id": null, "status": "pass"}',
                    },
                    {
                        "label": "trusted_feedback",
                        "text": "ok",
                    },
                    {
                        "label": "trusted_metrics",
                        "text": '{"combined_score": 1.25}',
                    },
                ]
            },
        )

    def test_failed_trusted_feedback_is_compact_and_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator_path = Path(tmp) / "evaluator.py"
            evaluator_path.write_text(
                "def evaluate(path):\n"
                "    return {\n"
                "        'combined_score': -2.0,\n"
                "        '__trusted_status': 'fail',\n"
                "        '__trusted_feedback_json': "
                "'{\\\"summary\\\": \\\"Candidate failed trusted verification.\\\", "
                "\\\"suggested_action\\\": \\\"Do not repeat this exact mutation set unchanged.\\\", "
                "\\\"mutation_ids\\\": [\\\"m015\\\", \\\"m017\\\"], "
                "\\\"failed_check_ids\\\": [\\\"legality.binding\\\", \\\"legality.architecture\\\"], "
                "\\\"messages\\\": [\\\"binding mismatch\\\", \\\"architecture mismatch\\\"]}',\n"
                "    }\n",
                encoding="utf-8",
            )
            evaluate = _make_alphaevolve_evaluator(str(evaluator_path))
            result = evaluate(
                {
                    "content": {
                        "files": [{"path": "candidate.py", "content": "x = 1\n"}]
                    }
                }
            )

        self.assertEqual(
            result["scores"]["scores"],
            [{"metric": "combined_score", "score": -2.0}],
        )
        feedback = next(
            item
            for item in result["insights"]["insights"]
            if item["label"] == "trusted_feedback"
        )["text"]
        self.assertIn("Candidate mutations: m015, m017", feedback)
        self.assertIn("Failed trusted checks: legality.binding, legality.architecture", feedback)
        self.assertIn("Next action: Do not repeat this exact mutation set unchanged.", feedback)
        self.assertNotIn("{\"summary\"", feedback)

    def test_only_combined_score_is_submitted_as_alphaevolve_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator_path = Path(tmp) / "evaluator.py"
            evaluator_path.write_text(
                "def evaluate(path):\n"
                "    return {\n"
                "        'combined_score': 1.75,\n"
                "        'objective_value': 0.000004,\n"
                "        'failed_checks': 0.0,\n"
                "        'legal': 1.0,\n"
                "    }\n",
                encoding="utf-8",
            )
            evaluate = _make_alphaevolve_evaluator(str(evaluator_path))
            result = evaluate(
                {
                    "content": {
                        "files": [{"path": "candidate.py", "content": "x = 1\n"}]
                    }
                }
            )

        self.assertEqual(
            result["scores"]["scores"],
            [{"metric": "combined_score", "score": 1.75}],
        )
        metrics_insight = next(
            item
            for item in result["insights"]["insights"]
            if item["label"] == "trusted_metrics"
        )
        self.assertIn('"objective_value": 4e-06', metrics_insight["text"])
        self.assertIn('"failed_checks": 0.0', metrics_insight["text"])

    def test_error_insight_uses_alphaevolve_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator_path = Path(tmp) / "evaluator.py"
            evaluator_path.write_text(
                "def evaluate(path):\n"
                "    raise RuntimeError('boom')\n",
                encoding="utf-8",
            )
            evaluate = _make_alphaevolve_evaluator(str(evaluator_path))
            result = evaluate(
                {
                    "content": {
                        "files": [{"path": "candidate.py", "content": "x = 1\n"}]
                    }
                }
            )

        self.assertEqual(result["scores"], {"scores": []})
        self.assertEqual(
            result["insights"],
            {
                "insights": [
                    {
                        "label": "evaluation_error",
                        "text": "RuntimeError: boom",
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
