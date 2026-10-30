from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from chia_trusted_verification.search.structured import (
    StructuredRandomSearchStrategy,
    StructuredWalkSearchStrategy,
)
from chia_trusted_verification.types import PipelineResult, SearchProblem


PROGRAM = '''MUTATION_CATALOG = [
    {"mutation_id": "m000"},
    {"mutation_id": "m001"},
    {"mutation_id": "m002"},
]
MIN_MUTATIONS = 1
MAX_MUTATIONS = 2
# EVOLVE-BLOCK-START
CANDIDATE = {"label": "seed", "mutation_ids": []}
# EVOLVE-BLOCK-END
'''


class _Evaluator:
    def __init__(self):
        self.sources = []

    def evaluate_source(self, source, role="proposal"):
        self.sources.append((source, role))
        return PipelineResult(
            backend="teaal_evolve",
            intent_id="test",
            status="pass",
            metrics={
                "combined_score": float(len(self.sources)),
                "legal": 1.0,
                "objective_available": 1.0,
                "objective_value": 1.0,
            },
            feedback={},
        )


class StructuredBaselineTests(unittest.TestCase):
    def _run(self, strategy):
        evaluator = _Evaluator()
        config = SimpleNamespace(seed=7, iterations=5, output_dir="out")
        problem = SearchProblem(initial_source=PROGRAM)
        with tempfile.TemporaryDirectory() as td:
            result = strategy.run(
                problem=problem,
                initial_program=Path(td) / "initial.py",
                evaluator=evaluator,
                config=config,
            )
        self.assertEqual(len(evaluator.sources), 5)
        self.assertEqual(result.best_score, 5.0)
        for source, role in evaluator.sources:
            self.assertEqual(role, "proposal")
            self.assertIn("# EVOLVE-BLOCK-START", source)
            self.assertIn('"mutation_ids"', source)

    def test_structured_random(self):
        self._run(StructuredRandomSearchStrategy())

    def test_structured_walk(self):
        self._run(StructuredWalkSearchStrategy())


if __name__ == "__main__":
    unittest.main()
