from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from chia_trusted_verification.chia.tool import TrustedVerificationTool
from chia_trusted_verification.types import VerificationIntent


class _FakeNode:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.resources = None

    def options(self, **kwargs):
        self.resources = kwargs.get("resources")
        return self

    def chia_remote_blocking(self, *args, **kwargs):
        self.calls.append((self.name, self.resources, kwargs.get("_chia_tag")))
        if self.name == "prepare":
            return {"status": "pass", "candidate": args[0]}
        if self.name == "verify":
            return {**args[0], "status": "pass", "verified": True}
        if self.name == "evaluate":
            return {**args[0], "status": "pass", "score": 3.0}
        return {
            "backend": "toy",
            "intent_id": "toy-intent",
            "status": "pass",
            "metrics": {"combined_score": 3.0},
            "feedback": {"summary": "ok"},
            "evidence": {"verified": True},
            "trace": {},
            "diagnostics": [{"code": "toy.ok", "message": "checked"}],
        }


class ChiaToolTests(unittest.TestCase):
    def test_tool_runs_complete_trusted_pipeline_with_blocking_nodes(self) -> None:
        calls = []
        tool = object.__new__(TrustedVerificationTool)
        tool.backend_name = "toy"
        tool.backend_settings = {}
        tool.backend_resources = {
            "prepare": {"toy_verify": 1.0},
            "verify": {"toy_verify": 1.0},
            "evaluate": {"toy_verify": 1.0},
            "feedback": {"toy_verify": 1.0},
        }
        tool.intent = VerificationIntent("toy-intent", "toy")
        tool.initial_source = "seed"
        tool.result_log = None
        tool.run_metadata = {}
        tool.run_started_epoch = None
        tool.max_evaluations = 2
        tool.evaluations = 0
        nodes = {name: _FakeNode(name, calls) for name in ("prepare", "verify", "evaluate", "feedback")}
        with patch("chia_trusted_verification.chia.tool.prepare_artifact", nodes["prepare"]), patch(
            "chia_trusted_verification.chia.tool.verify_artifact", nodes["verify"]
        ), patch("chia_trusted_verification.chia.tool.evaluate_artifact", nodes["evaluate"]), patch(
            "chia_trusted_verification.chia.tool.feedback_for_search", nodes["feedback"]
        ):
            result = tool.evaluate_candidate("candidate")
        self.assertEqual([name for name, _, _ in calls], ["prepare", "verify", "evaluate", "feedback"])
        self.assertTrue(all(resources == {"toy_verify": 1.0} for _, resources, _ in calls))
        self.assertTrue(all(tag for _, _, tag in calls))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["evaluation_index"], 1)
        self.assertEqual(result["remaining_evaluations"], 1)
        self.assertEqual(result["diagnostics"][0]["code"], "toy.ok")


    def test_tool_archives_proposal_source_for_trusted_reconciliation(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tool = object.__new__(TrustedVerificationTool)
            tool.backend_name = "toy"
            tool.backend_settings = {}
            tool.backend_resources = {}
            tool.intent = VerificationIntent("toy-intent", "toy")
            tool.initial_source = "seed"
            tool.result_log = Path(tmp) / "trusted_results.jsonl"
            tool.run_metadata = {"run_instance_id": "run_test"}
            tool.run_started_epoch = None
            tool.max_evaluations = 2
            tool.evaluations = 0
            nodes = {name: _FakeNode(name, calls) for name in ("prepare", "verify", "evaluate", "feedback")}
            with patch("chia_trusted_verification.chia.tool.prepare_artifact", nodes["prepare"]), patch(
                "chia_trusted_verification.chia.tool.verify_artifact", nodes["verify"]
            ), patch("chia_trusted_verification.chia.tool.evaluate_artifact", nodes["evaluate"]), patch(
                "chia_trusted_verification.chia.tool.feedback_for_search", nodes["feedback"]
            ):
                tool.evaluate_candidate("candidate source")

            row = json.loads(tool.result_log.read_text(encoding="utf-8").strip())
            archived = tool.result_log.parent / row["candidate_source_file"]
            self.assertTrue(archived.exists())
            self.assertEqual(archived.read_text(encoding="utf-8"), "candidate source")
            self.assertTrue(row["candidate_source_sha256"])


    def test_tool_state_does_not_embed_thread_lock(self) -> None:
        tool = object.__new__(TrustedVerificationTool)
        tool.backend_name = "toy"
        tool.backend_settings = {}
        tool.backend_resources = {}
        tool.intent = VerificationIntent("toy-intent", "toy")
        tool.initial_source = "seed"
        tool.max_evaluations = 2
        tool.evaluations = 0
        tool.result_log = None
        tool.run_metadata = {}
        tool.run_started_epoch = None

        payload = pickle.dumps(tool)
        restored = pickle.loads(payload)
        self.assertEqual(restored.backend_name, "toy")
        self.assertFalse(hasattr(restored, "_write_lock"))


if __name__ == "__main__":
    unittest.main()
