from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.config import ExperimentConfig
from chia_trusted_verification.types import SearchProblem, SearchResult

from .base import SearchStrategy


def _literal_assignment(source: str, name: str) -> Any:
    tree = ast.parse(source, mode="exec")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"initial program has no literal assignment for {name}")


def _candidate_source(initial_source: str, mutation_ids: list[str], label: str) -> str:
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start = initial_source.index(start_marker)
    end = initial_source.index(end_marker, start)
    prefix = initial_source[: start + len(start_marker)]
    suffix = initial_source[end:]
    block = (
        "\nCANDIDATE = {\n"
        f"    \"label\": {label!r},\n"
        f"    \"mutation_ids\": {mutation_ids!r},\n"
        "}\n"
    )
    return prefix + block + suffix


def _space(problem: SearchProblem) -> tuple[list[str], int, int]:
    catalog = _literal_assignment(problem.initial_source, "MUTATION_CATALOG")
    if not isinstance(catalog, list):
        raise TypeError("MUTATION_CATALOG must be a literal list")
    ids = [str(row["mutation_id"]) for row in catalog]
    minimum = int(_literal_assignment(problem.initial_source, "MIN_MUTATIONS"))
    maximum = int(_literal_assignment(problem.initial_source, "MAX_MUTATIONS"))
    return ids, minimum, maximum


class StructuredRandomSearchStrategy(SearchStrategy):
    name = "structured_random"

    def run(
        self,
        *,
        problem: SearchProblem,
        initial_program: Path,
        evaluator: TrustedVerificationEvaluator,
        config: ExperimentConfig,
    ) -> SearchResult:
        del initial_program
        ids, minimum, maximum = _space(problem)
        rng = random.Random(config.seed)
        seen: set[tuple[str, ...]] = set()
        best_score = float("-inf")
        best_source = problem.initial_source
        legal_count = 0

        for index in range(1, config.iterations + 1):
            choice: tuple[str, ...] | None = None
            for _ in range(200):
                count = rng.randint(minimum, min(maximum, len(ids)))
                candidate = tuple(sorted(rng.sample(ids, count)))
                if candidate not in seen:
                    choice = candidate
                    seen.add(candidate)
                    break
            if choice is None:
                count = rng.randint(minimum, min(maximum, len(ids)))
                choice = tuple(sorted(rng.sample(ids, count)))
            source = _candidate_source(problem.initial_source, list(choice), f"random-{index}")
            result = evaluator.evaluate_source(source, role="proposal")
            score = float(result.metrics.get("combined_score", float("-inf")))
            if float(result.metrics.get("legal", 0.0)) > 0.0:
                legal_count += 1
            if score > best_score:
                best_score = score
                best_source = source

        return SearchResult(
            engine=self.name,
            best_score=best_score if best_score != float("-inf") else 0.0,
            best_solution=best_source,
            metrics={
                "search_iteration_count": config.iterations,
                "search_unique_proposals": len(seen),
                "search_legal_proposals": legal_count,
                "baseline_type": "non_llm_structured_random",
            },
            output_dir=config.output_dir,
        )


class StructuredWalkSearchStrategy(SearchStrategy):
    name = "structured_walk"

    def run(
        self,
        *,
        problem: SearchProblem,
        initial_program: Path,
        evaluator: TrustedVerificationEvaluator,
        config: ExperimentConfig,
    ) -> SearchResult:
        del initial_program
        ids, minimum, maximum = _space(problem)
        rng = random.Random(config.seed)
        current: tuple[str, ...] = ()
        best_score = float("-inf")
        best_source = problem.initial_source
        legal_count = 0
        seen: set[tuple[str, ...]] = set()

        for index in range(1, config.iterations + 1):
            pool = set(current)
            if not pool or (len(pool) < maximum and rng.random() < 0.55):
                available = [item for item in ids if item not in pool]
                if available:
                    pool.add(rng.choice(available))
            elif len(pool) > minimum and rng.random() < 0.35:
                pool.remove(rng.choice(sorted(pool)))
            else:
                if pool:
                    pool.remove(rng.choice(sorted(pool)))
                available = [item for item in ids if item not in pool]
                if available:
                    pool.add(rng.choice(available))
            while len(pool) < minimum:
                available = [item for item in ids if item not in pool]
                if not available:
                    break
                pool.add(rng.choice(available))
            while len(pool) > maximum:
                pool.remove(rng.choice(sorted(pool)))
            proposal = tuple(sorted(pool))
            seen.add(proposal)
            source = _candidate_source(problem.initial_source, list(proposal), f"walk-{index}")
            result = evaluator.evaluate_source(source, role="proposal")
            score = float(result.metrics.get("combined_score", float("-inf")))
            if float(result.metrics.get("legal", 0.0)) > 0.0:
                current = proposal
                legal_count += 1
            if score > best_score:
                best_score = score
                best_source = source

        return SearchResult(
            engine=self.name,
            best_score=best_score if best_score != float("-inf") else 0.0,
            best_solution=best_source,
            metrics={
                "search_iteration_count": config.iterations,
                "search_unique_proposals": len(seen),
                "search_legal_proposals": legal_count,
                "baseline_type": "non_llm_structured_walk",
            },
            output_dir=config.output_dir,
        )
