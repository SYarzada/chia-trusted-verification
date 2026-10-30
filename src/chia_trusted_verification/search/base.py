from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.config import ExperimentConfig
from chia_trusted_verification.types import SearchProblem, SearchResult


class SearchStrategy(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        *,
        problem: SearchProblem,
        initial_program: Path,
        evaluator: TrustedVerificationEvaluator,
        config: ExperimentConfig,
    ) -> SearchResult:
        raise NotImplementedError
