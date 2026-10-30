from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from chia_trusted_verification.types import SearchProblem, VerificationIntent


class TrustedVerificationBackend(ABC):
    name: str

    @abstractmethod
    def build_search_problem(self, settings: Mapping[str, Any]) -> SearchProblem:
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        candidate_source: str,
        context: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        prepared: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        verified: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def feedback(
        self,
        evaluated: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError
