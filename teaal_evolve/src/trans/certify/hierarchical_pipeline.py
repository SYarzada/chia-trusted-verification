

from __future__ import annotations

from typing import Dict, Optional

from .candidate import Candidate
from .extract import extract_selected_local_legality_obligations
from .hierarchical_interface import check_interface_obligations
from .hierarchical_legality import check_local_legality
from .hierarchical_stage import check_stage_obligations
from .hierarchical_view import HierarchicalViewError, build_candidate_hierarchical_view
from .knobs import KNOB_SPACE_RESTRICTED
from .objective_backend import (
    OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    compute_objective_with_backend,
)
from .obligation import CandidateReport, LegalityStageResult, ObjectiveResult


def certify_candidate_hierarchical(
    candidate: Candidate,
    *,
    include_bottleneck_objective: bool = False,
    objective_shapes: Optional[Dict[str, int]] = None,
    objective_backend_name: str = OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> CandidateReport:
    try:
        view = build_candidate_hierarchical_view(candidate)
    except HierarchicalViewError as exc:
        return CandidateReport(
            candidate_label=candidate.label,
            is_legal=False,
            legality_obligations=[LegalityStageResult(stage_name=exc.stage_name, ok=False, error_type=type(exc).__name__, message=exc.message)],
            local_legality_obligations=[],
            stage_obligations=[],
            interface_obligations=[],
            objective=_skipped_or_disabled_objective(include_bottleneck_objective),
        )

    legality = check_local_legality(candidate, view)
                                                                                    
                                                                                       
                                            
    selected_local_legality = extract_selected_local_legality_obligations(candidate, knob_space=knob_space)
    stages = check_stage_obligations(view)
    interfaces = check_interface_obligations(view)

    stage_results = list(legality.stage_results) + list(stages.stage_results) + list(interfaces.stage_results)
    local_legality_ok = all(ob.status == "pass" for ob in selected_local_legality)
    is_legal = legality.is_legal and local_legality_ok and stages.is_legal and interfaces.is_legal

    if include_bottleneck_objective and is_legal:
        objective = compute_objective_with_backend(candidate, backend_name=objective_backend_name, shapes=objective_shapes)
    else:
        objective = _skipped_or_disabled_objective(include_bottleneck_objective)

    return CandidateReport(
        candidate_label=candidate.label,
        is_legal=is_legal,
        legality_obligations=stage_results,
        local_legality_obligations=list(selected_local_legality),
        stage_obligations=list(stages.obligations),
        interface_obligations=list(interfaces.obligations),
        objective=objective,
    )


def _skipped_or_disabled_objective(include_bottleneck_objective: bool) -> ObjectiveResult:
    if include_bottleneck_objective:
        return ObjectiveResult(
            enabled=True,
            objective_name="bottleneck_latency",
            status="skipped",
            message="Objective not evaluated because the candidate is illegal.",
        )
    return ObjectiveResult(enabled=False)
