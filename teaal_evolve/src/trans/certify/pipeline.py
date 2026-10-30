

from __future__ import annotations

from typing import Dict, Optional

from .candidate import Candidate
from .extract import (
    extract_interface_obligations,
    extract_legality_obligations,
    extract_selected_local_legality_obligations,
    extract_stage_obligations,
    extract_symbolic_condition_obligations,
)
from .knobs import KNOB_SPACE_RESTRICTED
from .legality import run_monolithic_legality
from .objective_backend import (
    OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    compute_objective_with_backend,
)
from .obligation import CandidateReport, ObjectiveResult


def certify_candidate(
    candidate: Candidate,
    *,
    include_bottleneck_objective: bool = False,
    objective_shapes: Optional[Dict[str, int]] = None,
    objective_backend_name: str = OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> CandidateReport:
    legality = run_monolithic_legality(candidate)
    legality_obligations = extract_legality_obligations(legality)

    if not legality.is_legal:
        objective = ObjectiveResult(
            enabled=include_bottleneck_objective,
            objective_name="bottleneck_latency" if include_bottleneck_objective else None,
            status="skipped" if include_bottleneck_objective else "disabled",
            message=("Objective not evaluated because the candidate is illegal." if include_bottleneck_objective else None),
        )
        return CandidateReport(
            candidate_label=candidate.label,
            is_legal=False,
            legality_obligations=legality_obligations,
            local_legality_obligations=[],
            stage_obligations=[],
            interface_obligations=[],
            objective=objective,
        )

    local_legality_obligations = extract_selected_local_legality_obligations(candidate, knob_space=knob_space)
    local_legality_obligations.extend(extract_symbolic_condition_obligations(legality.artifacts))
    stage_obligations = extract_stage_obligations(legality.artifacts)
    interface_obligations = extract_interface_obligations(legality.artifacts)

    if include_bottleneck_objective:
        objective = compute_objective_with_backend(candidate, backend_name=objective_backend_name, artifacts=legality.artifacts, shapes=objective_shapes)
    else:
        objective = ObjectiveResult(enabled=False)

    return CandidateReport(
        candidate_label=candidate.label,
        is_legal=True,
        legality_obligations=legality_obligations,
        local_legality_obligations=local_legality_obligations,
        stage_obligations=stage_obligations,
        interface_obligations=interface_obligations,
        objective=objective,
    )
