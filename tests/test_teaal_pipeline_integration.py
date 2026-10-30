from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import importlib.util

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.config import load_experiment_config
from chia_trusted_verification.loop import build_initial_program


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("z3") is not None, "z3-solver is not installed")
class TeaalPipelineIntegrationTests(unittest.TestCase):
    def test_seed_and_known_illegal_curated_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = load_experiment_config(ROOT / "configs" / "experiment.yaml")
            config = replace(
                base,
                output_dir=tmp,
                result_log=str(Path(tmp) / "results.jsonl"),
            )
            _, initial = build_initial_program(config)
            evaluator = TrustedVerificationEvaluator(
                backend_name=config.backend_name,
                backend_settings=config.backend_settings,
                backend_resources=config.backend_resources,
                intent=config.intent,
                engine=config.engine,
                initial_source=initial,
                result_log=config.result_log,
                use_chia=False,
            )

            seed = evaluator.evaluate_source(initial)
            illegal_source = initial.replace(
                '"mutation_ids": [],',
                '"mutation_ids": ["m015"],',
            )
            illegal = evaluator.evaluate_source(illegal_source)

            self.assertEqual(seed.backend, "teaal_evolve")
            self.assertEqual(seed.status, "pass")
            self.assertEqual(seed.metrics["certification_passed"], 1.0)
            self.assertEqual(illegal.status, "fail")
            self.assertEqual(illegal.metrics["objective_available"], 0.0)
            self.assertTrue(illegal.feedback["failed_check_ids"])


if __name__ == "__main__":
    unittest.main()
