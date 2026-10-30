from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import asyncio
import os
import sys
import unittest
from unittest.mock import patch

from chia_trusted_verification.config import load_experiment_config
from chia_trusted_verification.search.factory import build_search_strategy
from chia_trusted_verification.search.llm_agent import LLMAgentSearchStrategy
from chia_trusted_verification.search.structured import StructuredRandomSearchStrategy, StructuredWalkSearchStrategy
from chia_trusted_verification.search.skydiscover import (
    AlphaEvolveSearchStrategy,
    EvolutionSearchStrategy,
    OpenEvolveSearchStrategy,
    _apply_vertex_adc_to_openevolve_config,
    _configure_openevolve_vertex_adc,
    _configure_openevolve_process_pool,
    _inject_problem_guidance,
    _openevolve_vertex_bridge,
    _vertex_model_name,
    _vertex_openai_base_url,
)
from chia_trusted_verification.types import SearchGuidance, SearchProblem


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndAdapterTests(unittest.TestCase):
    def test_config_paths_and_override(self) -> None:
        config = load_experiment_config(
            ROOT / "configs" / "experiment.yaml",
            engine_override="alphaevolve",
        )
        self.assertEqual(config.engine, "alphaevolve")
        self.assertEqual(config.backend_name, "teaal_evolve")
        self.assertEqual(config.intent.artifact_type, "structured_accelerator_candidate")
        self.assertTrue(Path(config.search_config).is_file())

    def test_strategy_switch(self) -> None:
        open_strategy = build_search_strategy("openevolve")
        alpha_strategy = build_search_strategy("alphaevolve")
        self.assertIsInstance(open_strategy, OpenEvolveSearchStrategy)
        self.assertIsInstance(alpha_strategy, AlphaEvolveSearchStrategy)
        self.assertIsInstance(open_strategy, EvolutionSearchStrategy)
        self.assertIsInstance(alpha_strategy, EvolutionSearchStrategy)
        self.assertIsInstance(build_search_strategy("llm_agent"), LLMAgentSearchStrategy)
        self.assertIsInstance(build_search_strategy("structured_random"), StructuredRandomSearchStrategy)
        self.assertIsInstance(build_search_strategy("structured_walk"), StructuredWalkSearchStrategy)
        self.assertIsInstance(build_search_strategy("openevolve_structured"), OpenEvolveSearchStrategy)
        self.assertIsInstance(build_search_strategy("openevolve_guided"), OpenEvolveSearchStrategy)
        self.assertIsInstance(build_search_strategy("alphaevolve_structured"), AlphaEvolveSearchStrategy)
        self.assertIsInstance(build_search_strategy("alphaevolve_guided"), AlphaEvolveSearchStrategy)
        self.assertIsInstance(build_search_strategy("llm_agent_structured"), LLMAgentSearchStrategy)
        self.assertIsInstance(build_search_strategy("llm_agent_guided"), LLMAgentSearchStrategy)
        self.assertEqual(build_search_strategy("openevolve_structured").name, "openevolve_structured")
        self.assertEqual(build_search_strategy("llm_agent_guided").name, "llm_agent_guided")

    def test_backend_guidance_is_injected_into_evolution_engine(self) -> None:
        problem = SearchProblem(
            initial_source="VALUE = 1\n",
            file_name="formal_candidate.sv",
            guidance=SearchGuidance(
                system_message="backend-owned search instructions",
                evaluator_system_message="backend-owned evaluator instructions",
                problem_description="backend-owned problem description",
            ),
        )
        raw = {"prompt": {"template": "default"}, "alphaevolve": {"title": "test"}}
        configured = _inject_problem_guidance(
            raw, problem, "alphaevolve", "gemini-3.5-flash"
        )
        self.assertEqual(
            configured["prompt"]["system_message"],
            "backend-owned search instructions",
        )
        self.assertEqual(
            configured["prompt"]["evaluator_system_message"],
            "backend-owned evaluator instructions",
        )
        self.assertEqual(configured["prompt"]["template"], "default")
        self.assertEqual(
            configured["alphaevolve"]["problem_description"],
            "backend-owned problem description",
        )
        self.assertEqual(configured["file_suffix"], ".sv")
        self.assertEqual(
            configured["alphaevolve"]["generation_settings"]["models"][0]["name"],
            "gemini-3.5-flash",
        )

    def test_openevolve_model_override_uses_vertex_model_name(self) -> None:
        problem = SearchProblem(
            initial_source="VALUE = 1\n",
            file_name="candidate.py",
            guidance=SearchGuidance(system_message="instructions"),
        )
        raw = {
            "llm": {
                "models": [
                    {"name": "google/gemini-2.5-flash", "weight": 1.0, "temperature": 0.35}
                ]
            }
        }
        configured = _inject_problem_guidance(
            raw, problem, "openevolve", "gemini/gemini-2.5-flash"
        )
        self.assertEqual(
            configured["llm"]["models"][0]["name"],
            "google/gemini-2.5-flash",
        )
        self.assertEqual(configured["llm"]["models"][0]["weight"], 1.0)

    def test_vertex_openai_endpoint_and_model_normalization(self) -> None:
        self.assertEqual(
            _vertex_openai_base_url("project-1", "global"),
            "https://aiplatform.googleapis.com/v1/projects/project-1/locations/global/endpoints/openapi",
        )
        self.assertEqual(
            _vertex_openai_base_url("project-1", "us-central1"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project-1/locations/us-central1/endpoints/openapi",
        )
        self.assertEqual(
            _vertex_model_name("gemini/gemini-2.5-flash"),
            "google/gemini-2.5-flash",
        )
        self.assertEqual(
            _vertex_model_name("google-vertex/gemini-2.5-flash"),
            "google/gemini-2.5-flash",
        )

    def test_openevolve_vertex_adc_configuration(self) -> None:
        primary = SimpleNamespace(
            name="gemini/gemini-2.5-flash",
            api_base=None,
            api_key=None,
        )
        evaluator = SimpleNamespace(
            name="google-vertex/gemini-2.5-flash",
            api_base=None,
            api_key=None,
        )
        config = SimpleNamespace(
            llm=SimpleNamespace(
                api_base=None,
                models=[primary],
                evaluator_models=[evaluator],
            )
        )
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "project-1",
                "GOOGLE_CLOUD_LOCATION": "global",
            },
            clear=False,
        ):
            endpoint = _configure_openevolve_vertex_adc(config)

        expected = (
            "https://aiplatform.googleapis.com/v1/projects/project-1/"
            "locations/global/endpoints/openapi"
        )
        self.assertEqual(endpoint, expected)
        self.assertEqual(config.llm.api_base, expected)
        self.assertEqual(primary.name, "google/gemini-2.5-flash")
        self.assertEqual(evaluator.name, "google/gemini-2.5-flash")
        self.assertEqual(primary.api_base, expected)
        self.assertEqual(evaluator.api_base, expected)
        self.assertIsNone(primary.api_key)
        self.assertIsNone(evaluator.api_key)

        mapped_primary = SimpleNamespace(
            name=primary.name,
            api_base=expected,
            api_key=None,
        )
        mapped_evaluator = SimpleNamespace(
            name=evaluator.name,
            api_base=expected,
            api_key=None,
        )
        mapped = SimpleNamespace(
            llm=SimpleNamespace(
                api_base=expected,
                models=[mapped_primary],
                evaluator_models=[mapped_evaluator],
            )
        )
        with patch(
            "chia_trusted_verification.search.skydiscover.vertex_adc_access_token",
            return_value="fresh-adc-token",
        ):
            _apply_vertex_adc_to_openevolve_config(mapped, expected)
        self.assertEqual(mapped.llm.api_key, "fresh-adc-token")
        self.assertEqual(mapped_primary.api_key, "fresh-adc-token")
        self.assertEqual(mapped_evaluator.api_key, "fresh-adc-token")


    def test_openevolve_bridge_filters_newer_evolve_flows_optional_kwargs(self) -> None:
        calls = {}

        async def legacy_runner(
            program_path,
            evaluator_path,
            config_obj,
            iterations,
            output_dir,
            monitor_callback=None,
            feedback_reader=None,
        ):
            calls["program_path"] = program_path
            calls["iterations"] = iterations
            return "ok"

        skydiscover = ModuleType("skydiscover")
        extras = ModuleType("skydiscover.extras")
        external = ModuleType("skydiscover.extras.external")
        backend = ModuleType("skydiscover.extras.external.openevolve_backend")
        backend._map_config = lambda config, iterations, output_dir: config
        external._REGISTRY = {"openevolve": legacy_runner}
        external.openevolve_backend = backend
        extras.external = external
        skydiscover.extras = extras

        modules = {
            "skydiscover": skydiscover,
            "skydiscover.extras": extras,
            "skydiscover.extras.external": external,
            "skydiscover.extras.external.openevolve_backend": backend,
        }
        with patch.dict(sys.modules, modules):
            with _openevolve_vertex_bridge("https://vertex.example/v1"):
                wrapped = external._REGISTRY["openevolve"]
                result = asyncio.run(
                    wrapped(
                        program_path="seed.py",
                        evaluator_path="eval.py",
                        config_obj=SimpleNamespace(),
                        iterations=4,
                        output_dir="out",
                        monitor_callback=None,
                        feedback_reader=None,
                        stop_check=None,
                        status_callback=None,
                    )
                )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, {"program_path": "seed.py", "iterations": 4})
        self.assertIs(external._REGISTRY["openevolve"], legacy_runner)

    def test_openevolve_mapped_config_forces_spawn_safe_pool(self) -> None:
        config = SimpleNamespace(max_tasks_per_child=None)
        _configure_openevolve_process_pool(config)
        self.assertEqual(config.max_tasks_per_child, 1)

    def test_engine_config_files_are_backend_agnostic(self) -> None:
        for name in ("openevolve.yaml", "alphaevolve.yaml", "llm_agent.yaml"):
            text = (ROOT / "configs" / "search" / name).read_text(encoding="utf-8").lower()
            self.assertNotIn("teaal", text)
            self.assertNotIn("mutation_catalog", text)


    def test_local_cluster_caps_ray_worker_and_library_threads(self) -> None:
        text = Path("cluster/local_gemini.yaml").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("--num-cpus=4"), 2)
        for name in (
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()



def test_alphaevolve_guidance_includes_full_program_and_backend_context():
    from chia_trusted_verification.search.skydiscover import _inject_problem_guidance
    from chia_trusted_verification.types import SearchGuidance, SearchProblem

    problem = SearchProblem(
        initial_source='# EVOLVE-BLOCK-START\nVALUE = 1\n# EVOLVE-BLOCK-END\n',
        file_name="initial_program.py",
        guidance=SearchGuidance(
            system_message="Backend-owned immutable-context instructions.",
            problem_description="Backend-owned problem description.",
        ),
    )
    cfg = {
        "search": {"type": "alphaevolve"},
        "alphaevolve": {
            "generation_settings": {
                "models": [{"name": "gemini-3.5-flash", "weight": 1.0}]
            }
        },
    }

    out = _inject_problem_guidance(cfg, problem, "alphaevolve")
    alpha = out["alphaevolve"]
    generation = alpha["generation_settings"]

    assert alpha["problem_description"] == "Backend-owned problem description."
    assert generation["include_full_program_in_prompt"] is True
    assert generation["context"] == "Backend-owned immutable-context instructions."
    assert generation["models"] == [{"name": "gemini-3.5-flash", "weight": 1.0}]


def test_alphaevolve_model_override_preserves_full_program_setting():
    from chia_trusted_verification.search.skydiscover import _inject_problem_guidance
    from chia_trusted_verification.types import SearchProblem

    problem = SearchProblem(initial_source="pass", file_name="initial_program.py")
    cfg = {
        "search": {"type": "alphaevolve"},
        "alphaevolve": {"generation_settings": {"include_full_program_in_prompt": False}},
    }

    out = _inject_problem_guidance(
        cfg, problem, "alphaevolve", model_override="google/gemini-3.5-flash"
    )
    generation = out["alphaevolve"]["generation_settings"]
    assert generation["include_full_program_in_prompt"] is True
    assert generation["models"] == [{"name": "gemini-3.5-flash", "weight": 1.0}]
