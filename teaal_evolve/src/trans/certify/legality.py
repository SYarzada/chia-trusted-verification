

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ir import (
    ElaboratedArchitecture,
    ElaboratedBinding,
    ElaboratedEinsum,
    ElaboratedFormat,
    ElaboratedMapping,
)

from .candidate import Candidate, CandidateArtifacts
from .obligation import LegalityStageResult


@dataclass
class LegalityResult:
    candidate: Candidate
    artifacts: CandidateArtifacts
    stage_results: list[LegalityStageResult]
    is_legal: bool

    @property
    def failed_stage(self) -> Optional[str]:
        for stage in self.stage_results:
            if not stage.ok:
                return stage.stage_name
        return None


def run_monolithic_legality(candidate: Candidate) -> LegalityResult:
    

    artifacts = candidate.empty_artifacts()
    results: list[LegalityStageResult] = []

    try:
        artifacts.elaborated_einsum = ElaboratedEinsum(
            artifacts.parsers.einsum,
            artifacts.parsers.options,
        )
        artifacts.elaborated_einsum.check_concrete()
        results.append(LegalityStageResult(stage_name="einsum", ok=True))
    except Exception as exc:
        results.append(
            LegalityStageResult(
                stage_name="einsum",
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        return LegalityResult(candidate, artifacts, results, False)

    try:
        artifacts.elaborated_mapping = ElaboratedMapping(
            artifacts.elaborated_einsum,
            artifacts.parsers.mapping,
        )
        artifacts.elaborated_mapping.check_concrete()
        results.append(LegalityStageResult(stage_name="mapping", ok=True))
    except Exception as exc:
        results.append(
            LegalityStageResult(
                stage_name="mapping",
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        return LegalityResult(candidate, artifacts, results, False)

    try:
        artifacts.elaborated_format = ElaboratedFormat(
            artifacts.elaborated_einsum,
            artifacts.elaborated_mapping,
            artifacts.parsers.format,
        )
        artifacts.elaborated_format.check_concrete()
        results.append(LegalityStageResult(stage_name="format", ok=True))
    except Exception as exc:
        results.append(
            LegalityStageResult(
                stage_name="format",
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        return LegalityResult(candidate, artifacts, results, False)

    try:
        artifacts.elaborated_architecture = ElaboratedArchitecture(
            artifacts.parsers.architecture,
        )
        results.append(LegalityStageResult(stage_name="architecture", ok=True))
    except Exception as exc:
        results.append(
            LegalityStageResult(
                stage_name="architecture",
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        return LegalityResult(candidate, artifacts, results, False)

    try:
        artifacts.elaborated_binding = ElaboratedBinding(
            artifacts.elaborated_einsum,
            artifacts.elaborated_mapping,
            artifacts.elaborated_format,
            artifacts.parsers.binding,
            artifacts.elaborated_architecture,
        )
        results.append(LegalityStageResult(stage_name="binding", ok=True))
    except Exception as exc:
        results.append(
            LegalityStageResult(
                stage_name="binding",
                ok=False,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        return LegalityResult(candidate, artifacts, results, False)

    return LegalityResult(candidate, artifacts, results, True)