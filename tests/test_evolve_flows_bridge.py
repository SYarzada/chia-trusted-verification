from __future__ import annotations
import json

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from chia_trusted_verification.search.skydiscover import (
    AlphaEvolveSearchStrategy,
    _TrustedEvolveFlowsEvaluator,
    _trusted_evaluator_spawn_spec,
    _write_spawn_safe_evaluator_shim,
    _infer_population_size,
)
from chia_trusted_verification.types import SearchGuidance, SearchProblem


class _FakeTrustedEvaluator:
    def __init__(self, result_log: Path) -> None:
        self.result_log = result_log
        self.sources: list[str] = []
        self.roles: list[str] = []
        self.initial_source = "SEED = 1\n"

    def evaluate_source(self, source: str, *, role: str = "proposal"):
        self.sources.append(source)
        self.roles.append(role)
        return SimpleNamespace(value="trusted")

    def format_engine_result(self, result):
        return {"combined_score": 1.25, "__trusted_status": "pass"}


class EvolveFlowsBridgeTests(unittest.TestCase):
    def test_async_adapter_preserves_engine_specific_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _FakeTrustedEvaluator(Path(tmp) / "results.jsonl")
            adapter = _TrustedEvolveFlowsEvaluator(trusted, output_dir=tmp)
            result = asyncio.run(adapter.evaluate_program("VALUE = 2\n"))

        self.assertEqual(trusted.sources, ["VALUE = 2\n"])
        self.assertEqual(result.metrics["combined_score"], 1.25)
        self.assertEqual(result.metrics["__trusted_status"], "pass")

    def test_initial_search_seed_is_not_counted_as_generated_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _FakeTrustedEvaluator(Path(tmp) / "results.jsonl")
            adapter = _TrustedEvolveFlowsEvaluator(trusted, output_dir=tmp)
            asyncio.run(adapter.evaluate_program("SEED = 1\n"))
            asyncio.run(adapter.evaluate_program("VALUE = 2\n"))
            # Only the first initial program evaluation is marked as search_seed.
            # Later evaluations of the same candidate count as proposals.
            asyncio.run(adapter.evaluate_program("SEED = 1\n"))

        self.assertEqual(trusted.roles, ["search_seed", "proposal", "proposal"])

    def test_spawn_safe_shim_does_not_depend_on_process_local_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = SimpleNamespace(
                backend_name="teaal_evolve",
                backend_settings={"kernel": "attention"},
                backend_resources={"prepare": {"teaal_evolve": 0.01}},
                intent=SimpleNamespace(
                    to_dict=lambda: {
                        "intent_id": "teaal_structured_design",
                        "artifact_type": "structured_accelerator_candidate",
                        "requirements": [],
                        "parameters": {},
                    }
                ),
                engine="openevolve",
                initial_source="CANDIDATE = {'mutation_ids': []}\n",
                result_log=Path(tmp) / "results.jsonl",
                run_metadata={"run_id": "test"},
            )
            spec = _trusted_evaluator_spawn_spec(evaluator)
            path = Path(_write_spawn_safe_evaluator_shim(spec))
            try:
                source = path.read_text(encoding="utf-8")
            finally:
                path.unlink(missing_ok=True)

        self.assertIn("TrustedVerificationEvaluator(", source)
        self.assertIn("use_chia=True", source)
        self.assertNotIn("_EVALUATOR_REGISTRY", source)
        self.assertEqual(spec["backend_name"], "teaal_evolve")
        self.assertEqual(spec["engine"], "openevolve")


    def test_population_size_uses_saved_metadata_over_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "checkpoint_20"
            programs = checkpoint / "programs"
            programs.mkdir(parents=True)

            for index in range(6):
                (programs / f"{index}.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "metadata.json").write_text(
                json.dumps({"islands": [["0", "1"], ["2"], [], [], []]}) + "\n",
                encoding="utf-8",
            )

            raw = SimpleNamespace(population=[], iteration_count=20)
            size, source = _infer_population_size(raw, root)

        self.assertEqual(size, 3)
        self.assertEqual(source, "latest_checkpoint_metadata")

    def test_population_size_uses_saved_files_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = root / "checkpoints" / "checkpoint_20" / "programs"
            programs.mkdir(parents=True)
            for index in range(3):
                (programs / f"{index}.json").write_text("{}\n", encoding="utf-8")
            raw = SimpleNamespace(population=[], iteration_count=20)
            size, source = _infer_population_size(raw, root)
        self.assertEqual(size, 3)
        self.assertEqual(source, "latest_checkpoint_files")

    def test_alphaevolve_routes_through_evolve_flows_with_search_type(self) -> None:
        captured = {}

        class EvolverInput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def run_skydiscover(evolver_input, build_fn, run_fn, result_mapper_fn, **kwargs):
            captured["input"] = evolver_input
            captured["evaluator"] = kwargs["evaluator"]
            return SimpleNamespace(
                terminal_status="completed",
                error_message=None,
                best_metrics={"combined_score": 2.5},
                best_program="VALUE = 3\n",
                iteration_count=7,
                population=[],
                metrics_log_path=None,
            )

        evolve_flows = ModuleType("evolve_flows")
        evolver = ModuleType("evolve_flows.evolver")
        bridge = ModuleType("evolve_flows.evolver.bridge")
        types = ModuleType("evolve_flows.evolver.types")
        bridge.run_skydiscover = run_skydiscover
        types.EvolverInput = EvolverInput

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search_config = root / "alpha.yaml"
            search_config.write_text(
                "max_iterations: 20\nsearch:\n  type: alphaevolve\nalphaevolve:\n  title: test\n",
                encoding="utf-8",
            )
            initial = root / "candidate.py"
            initial.write_text("VALUE = 1\n", encoding="utf-8")
            problem = SearchProblem(
                initial_source="VALUE = 1\n",
                file_name="candidate.py",
                guidance=SearchGuidance(
                    system_message="backend guidance",
                    problem_description="improve VALUE",
                ),
            )
            config = SimpleNamespace(
                search_config=str(search_config),
                model="gemini-3.5-flash",
                iterations=11,
                seed=3,
                output_dir=str(root / "out"),
            )
            trusted = _FakeTrustedEvaluator(root / "results.jsonl")

            modules = {
                "evolve_flows": evolve_flows,
                "evolve_flows.evolver": evolver,
                "evolve_flows.evolver.bridge": bridge,
                "evolve_flows.evolver.types": types,
            }
            with patch.dict(sys.modules, modules), patch(
                "chia_trusted_verification.search.skydiscover._alphaevolve_structured_feedback_bridge",
                return_value=nullcontext(),
            ):
                result = AlphaEvolveSearchStrategy().run(
                    problem=problem,
                    initial_program=initial,
                    evaluator=trusted,
                    config=config,
                )

        parsed = yaml.safe_load(captured["input"].config_content)
        self.assertEqual(parsed["search"]["type"], "alphaevolve")
        self.assertEqual(parsed["max_iterations"], 12)
        self.assertEqual(
            parsed["alphaevolve"]["problem_description"],
            "improve VALUE",
        )
        self.assertIsInstance(captured["evaluator"], _TrustedEvolveFlowsEvaluator)
        self.assertEqual(result.engine, "alphaevolve")
        self.assertEqual(result.best_solution, "VALUE = 3\n")
        self.assertEqual(result.best_score, 2.5)
        self.assertEqual(result.metrics["search_iteration_count"], 7)


if __name__ == "__main__":
    unittest.main()
