

from __future__ import annotations

from typing import Dict, Mapping, Optional

from .candidate import Candidate, CandidateArtifacts
from .legality import run_monolithic_legality
from .objective import compute_bottleneck_objective
from .objective_shapes import (
    ObjectiveShapeResolutionError,
    required_objective_symbols,
    resolve_objective_shape_report,
    validate_objective_shape_report,
)
from .obligation import ObjectiveResult

OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK = "monolithic_bottleneck"
SUPPORTED_OBJECTIVE_BACKENDS = {OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK}


def get_default_objective_shapes(candidate: Candidate) -> Dict[str, int]:
    

    return resolve_objective_shape_report(candidate).shapes


def resolve_objective_shapes(
    candidate: Candidate,
    explicit_shapes: Optional[Mapping[str, int]],
) -> Dict[str, int]:
    

    return resolve_objective_shape_report(candidate, explicit_shapes).shapes


def compute_objective_with_backend(
    candidate: Candidate,
    *,
    backend_name: str = OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    artifacts: Optional[CandidateArtifacts] = None,
    shapes: Optional[Mapping[str, int]] = None,
) -> ObjectiveResult:
    

    if backend_name not in SUPPORTED_OBJECTIVE_BACKENDS:
        raise ValueError(f"Unsupported objective backend: {backend_name}")

    if backend_name == OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK:
        used_artifacts = artifacts
        if used_artifacts is None:
            legality = run_monolithic_legality(candidate)
            if not legality.is_legal:
                return ObjectiveResult(
                    enabled=True,
                    objective_name="bottleneck_latency",
                    status="skipped",
                    message=(
                        "Objective not evaluated because the candidate is illegal under "
                        "the monolithic elaboration/model path."
                    ),
                )
            used_artifacts = legality.artifacts

        assert used_artifacts is not None
        required = required_objective_symbols(used_artifacts)
        report = resolve_objective_shape_report(
            candidate,
            shapes,
            required_symbols=required,
        )
        try:
            validate_objective_shape_report(report)
        except ObjectiveShapeResolutionError as exc:
            return ObjectiveResult(
                enabled=True,
                objective_name="bottleneck_latency",
                status="error",
                error_type=type(exc).__name__,
                message=str(exc),
            )

        return compute_bottleneck_objective(used_artifacts, report.shapes)

    raise AssertionError(f"Unhandled objective backend: {backend_name}")
