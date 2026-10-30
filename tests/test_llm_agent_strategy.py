from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import types

from chia_trusted_verification.config import ExperimentConfig
from chia_trusted_verification.search.llm_agent import LLMAgentSearchStrategy
from chia_trusted_verification.types import (
    PipelineResult,
    SearchGuidance,
    SearchProblem,
    VerificationIntent,
)


class _FakeEvaluator:
    result_log = Path("results.jsonl")
    run_metadata = {}

    def evaluate_source(self, source: str, *, role: str = "proposal") -> PipelineResult:
        score = 2.0 if "VALUE = 2" in source else 1.0
        return PipelineResult(
            backend="toy",
            intent_id="toy-intent",
            status="pass",
            metrics={"combined_score": score},
            feedback={"summary": f"score={score}"},
            evidence={},
            trace={},
        )


class _FakeTool:
    def __init__(self) -> None:
        self.name = "trusted_verification"
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakePrompt:
    def __init__(self, owner) -> None:
        self.owner = owner

    def chia_remote(self, agent, prompt, tools):
        self.owner.calls.append((agent, prompt, tools))
        return type("Response", (), {"result": "```python\nVALUE = 2\n```"})()


class _FakeChiaAgent:
    def __init__(self) -> None:
        self.calls = []
        self.prompt = _FakePrompt(self)


class _FakeStrategy(LLMAgentSearchStrategy):
    def __init__(self) -> None:
        self.tool = _FakeTool()
        self.agent = _FakeChiaAgent()
        self.agent_settings = None

    def _build_tool(self, **kwargs):
        self.tool_kwargs = kwargs
        return self.tool

    def _build_agent(self, problem, config, settings):
        self.agent_settings = settings
        self.system_message = problem.guidance.system_message
        return self.agent


class LLMAgentStrategyTests(unittest.TestCase):
    def test_agent_receives_chiatool_and_backend_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search_config = root / "agent.yaml"
            search_config.write_text(
                "agent:\n  provider: opencode_vertex\n  model: google-vertex/test\n  max_evaluations: 2\n"
                "tool:\n  resources:\n    trusted_verification_tool: 1\n",
                encoding="utf-8",
            )
            initial = root / "candidate.py"
            initial.write_text("VALUE = 0\n", encoding="utf-8")
            problem = SearchProblem(
                initial_source="VALUE = 0\n",
                file_name="candidate.py",
                guidance=SearchGuidance(
                    system_message="toy backend instructions",
                    problem_description="improve VALUE",
                ),
            )
            config = ExperimentConfig(
                engine="llm_agent",
                backend_name="toy",
                iterations=2,
                model="fallback-model",
                experiment_name="toy-exp",
                run_id="toy-run",
                seed=0,
                output_dir=str(root / "out"),
                result_log=str(root / "results.jsonl"),
                search_config=str(search_config),
                backend_settings={},
                intent=VerificationIntent("toy-intent", "toy"),
            )
            strategy = _FakeStrategy()
            with patch("chia_trusted_verification.search.llm_agent.require_chia_tool"), patch(
                "chia_trusted_verification.search.llm_agent.get", side_effect=lambda value: value
            ):
                result = strategy.run(
                    problem=problem,
                    initial_program=initial,
                    evaluator=_FakeEvaluator(),
                    config=config,
                )

        self.assertEqual(result.engine, "llm_agent")
        self.assertEqual(result.best_score, 2.0)
        self.assertIn("VALUE = 2", result.best_solution)
        self.assertEqual(strategy.system_message, "toy backend instructions")
        self.assertTrue(strategy.tool.stopped)
        self.assertEqual(len(strategy.agent.calls), 1)
        _, prompt, tools = strategy.agent.calls[0]
        self.assertEqual(tools, [strategy.tool])
        self.assertIn("trusted_verification_evaluate_candidate", prompt)
        self.assertIn("Evaluate at least one non-seed proposal", prompt)
        self.assertIn("Do not return a final candidate that has not received", prompt)
        self.assertIn("improve VALUE", prompt)
        self.assertEqual(strategy.tool_kwargs["task_options"]["resources"], {"trusted_verification_tool": 1.0})


    def test_shared_result_log_is_made_appendable_for_remote_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_log = root / "runs" / "trusted_results.jsonl"
            result_log.parent.mkdir(parents=True)
            result_log.write_text("{}\n", encoding="utf-8")
            lock = result_log.with_suffix(result_log.suffix + ".lock")
            seq = result_log.with_suffix(result_log.suffix + ".seq")
            lock.write_text("", encoding="utf-8")
            seq.write_text("1\n", encoding="utf-8")
            result_log.chmod(0o600)
            lock.chmod(0o600)
            seq.chmod(0o600)
            result_log.parent.chmod(0o700)

            LLMAgentSearchStrategy()._prepare_shared_result_log(result_log)

            self.assertEqual(result_log.parent.stat().st_mode & 0o777, 0o777)
            self.assertEqual(
                (result_log.parent / "candidate_programs").stat().st_mode & 0o777,
                0o777,
            )
            self.assertEqual(result_log.stat().st_mode & 0o666, 0o666)
            self.assertEqual(lock.stat().st_mode & 0o666, 0o666)
            self.assertEqual(seq.stat().st_mode & 0o666, 0o666)

    def test_vertex_gemini_builder_receives_forced_tool_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search_config = root / "agent.yaml"
            search_config.write_text(
                "agent:\n"
                "  provider: vertex_gemini\n"
                "  model: google-vertex/test-model\n"
                "  max_evaluations: 20\n"
                "  max_tokens: 65536\n"
                "  max_tool_iterations: 25\n"
                "  require_first_tool_call: true\n"
                "  thinking_level: medium\n"
                "  first_tool_thinking_level: low\n"
                "  first_tool_retry_thinking_level: minimal\n"
                "  max_forced_tool_retries: 1\n",
                encoding="utf-8",
            )
            config = ExperimentConfig(
                engine="llm_agent",
                backend_name="toy",
                iterations=20,
                model="",
                experiment_name="toy-exp",
                run_id="toy-run",
                seed=0,
                output_dir=str(root / "out"),
                result_log=str(root / "results.jsonl"),
                search_config=str(search_config),
                backend_settings={},
                intent=VerificationIntent("toy-intent", "toy"),
            )
            problem = SearchProblem(
                initial_source="VALUE = 0\n",
                file_name="candidate.py",
                guidance=SearchGuidance(system_message="system", problem_description=""),
            )
            settings = LLMAgentSearchStrategy()._settings(search_config)["agent"]
            captured = {}

            def fake_builder(**kwargs):
                captured.update(kwargs)
                return object()

            fake_module = types.ModuleType("chia_trusted_verification.search.vertex_gemini")
            fake_module.build_vertex_gemini_agent = fake_builder
            with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project"}), patch.dict(
                sys.modules,
                {"chia_trusted_verification.search.vertex_gemini": fake_module},
            ):
                LLMAgentSearchStrategy()._build_agent(problem, config, settings)

            self.assertEqual(captured["model"], "test-model")
            self.assertEqual(captured["max_tokens"], 65536)
            self.assertEqual(captured["max_tool_iterations"], 25)
            self.assertTrue(captured["require_first_tool_call"])
            self.assertEqual(captured["thinking_level"], "medium")
            self.assertEqual(captured["first_tool_thinking_level"], "low")
            self.assertEqual(captured["first_tool_retry_thinking_level"], "minimal")
            self.assertEqual(captured["max_forced_tool_retries"], 1)


if __name__ == "__main__":
    unittest.main()
