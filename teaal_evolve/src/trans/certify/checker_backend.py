

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .candidate import Candidate
from .incremental_pipeline import (
    DELTA_SOURCE_RECONSTRUCTED,
    INCREMENTAL_FALLBACK_MODE,
    IncrementalCertificationResult,
    TrustedKnobDelta,
    certify_candidate_hierarchical_incremental,
)
from .knobs import KNOB_SPACE_RESTRICTED
from .objective_backend import OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK
from .obligation import CandidateReport
from .pipeline import certify_candidate
from .hierarchical_pipeline import certify_candidate_hierarchical


CHECKER_BACKEND_FULL = "full"
CHECKER_BACKEND_HIERARCHICAL = "hierarchical"
CHECKER_BACKEND_HIERARCHICAL_DELTA_SYNTACTIC = "hierarchical_delta_syntactic"
CHECKER_BACKEND_HIERARCHICAL_DELTA_EGRAPH = "hierarchical_delta_egraph"

SUPPORTED_CHECKER_BACKENDS = {
    CHECKER_BACKEND_FULL,
    CHECKER_BACKEND_HIERARCHICAL,
    CHECKER_BACKEND_HIERARCHICAL_DELTA_SYNTACTIC,
    CHECKER_BACKEND_HIERARCHICAL_DELTA_EGRAPH,
}


@dataclass(frozen=True)
class CheckerBackendResult:
    backend_name: str
    candidate_report: CandidateReport
    failed_stage: Optional[str]
    total_hits: int
    total_misses: int
    total_obligations_processed: int
    hit_rate: float
    stage_hit_ids: List[str]
    stage_miss_ids: List[str]
    interface_hit_ids: List[str]
    interface_miss_ids: List[str]
    legality_hit_ids: List[str]
    legality_miss_ids: List[str]
    syntactic_legality_reuse_rate: Optional[float]
    rewrite_legality_reuse_rate: Optional[float]
    extra_reused_legality_ids: List[str]
    execution_mode: str = "unspecified"
    incremental_fallback_reason: Optional[str] = None
    incremental_delta_source: Optional[str] = None


def check_candidate_with_backend(
    mutated_candidate: Candidate,
    *,
    backend_name: str,
    base_candidate: Optional[Candidate] = None,
    base_report: Optional[CandidateReport] = None,
    include_bottleneck_objective: bool = False,
    objective_shapes: Optional[Dict[str, int]] = None,
    objective_backend_name: str = OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    knob_space: str = KNOB_SPACE_RESTRICTED,
    trusted_knob_delta: Optional[TrustedKnobDelta] = None,
) -> CheckerBackendResult:
    if backend_name not in SUPPORTED_CHECKER_BACKENDS:
        raise ValueError(f"Unsupported checker backend: {backend_name}")

    if backend_name == CHECKER_BACKEND_FULL:
        report = certify_candidate(
            mutated_candidate,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
        )
        return CheckerBackendResult(
            backend_name=backend_name,
            candidate_report=report,
            failed_stage=_first_failed_stage(report),
            total_hits=0,
            total_misses=0,
            total_obligations_processed=0,
            hit_rate=0.0,
            stage_hit_ids=[],
            stage_miss_ids=[],
            interface_hit_ids=[],
            interface_miss_ids=[],
            legality_hit_ids=[],
            legality_miss_ids=[],
            syntactic_legality_reuse_rate=None,
            rewrite_legality_reuse_rate=None,
            extra_reused_legality_ids=[],
            execution_mode="complete_monolithic",
            incremental_delta_source=None,
        )

    if backend_name == CHECKER_BACKEND_HIERARCHICAL:
        report = certify_candidate_hierarchical(
            mutated_candidate,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
        )
        return _complete_hierarchical_result(
            backend_name,
            report,
            execution_mode="complete_hierarchical",
        )

    rewrite_aware = backend_name == CHECKER_BACKEND_HIERARCHICAL_DELTA_EGRAPH
    if base_candidate is None or base_report is None:
        report = certify_candidate_hierarchical(
            mutated_candidate,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
        )
        return _complete_hierarchical_result(
            backend_name,
            report,
            execution_mode=INCREMENTAL_FALLBACK_MODE,
            fallback_reason="missing_base_candidate_or_report",
            rewrite_aware=rewrite_aware,
        )

    incremental = certify_candidate_hierarchical_incremental(
        mutated_candidate,
        base_candidate=base_candidate,
        base_report=base_report,
        rewrite_aware=rewrite_aware,
        include_bottleneck_objective=include_bottleneck_objective,
        objective_shapes=objective_shapes,
        objective_backend_name=objective_backend_name,
        knob_space=knob_space,
        trusted_knob_delta=trusted_knob_delta,
    )
    return _incremental_backend_result(backend_name, incremental)


def _complete_hierarchical_result(
    backend_name: str,
    report: CandidateReport,
    *,
    execution_mode: str,
    fallback_reason: Optional[str] = None,
    rewrite_aware: bool = False,
) -> CheckerBackendResult:
    stage_miss_ids = [
        obligation.obligation_id for obligation in report.stage_obligations
    ]
    interface_miss_ids = [
        obligation.obligation_id for obligation in report.interface_obligations
    ]
    legality_miss_ids = [
        obligation.obligation_id
        for obligation in report.local_legality_obligations
    ]
    total_misses = (
        len(stage_miss_ids) + len(interface_miss_ids) + len(legality_miss_ids)
    )
    return CheckerBackendResult(
        backend_name=backend_name,
        candidate_report=report,
        failed_stage=_first_failed_stage(report),
        total_hits=0,
        total_misses=total_misses,
        total_obligations_processed=total_misses,
        hit_rate=0.0,
        stage_hit_ids=[],
        stage_miss_ids=stage_miss_ids,
        interface_hit_ids=[],
        interface_miss_ids=interface_miss_ids,
        legality_hit_ids=[],
        legality_miss_ids=legality_miss_ids,
        syntactic_legality_reuse_rate=(0.0 if rewrite_aware else None),
        rewrite_legality_reuse_rate=(0.0 if rewrite_aware else None),
        extra_reused_legality_ids=[],
        execution_mode=execution_mode,
        incremental_fallback_reason=fallback_reason,
        incremental_delta_source=(
            DELTA_SOURCE_RECONSTRUCTED
            if execution_mode == INCREMENTAL_FALLBACK_MODE
            else None
        ),
    )


def _incremental_backend_result(
    backend_name: str,
    incremental: IncrementalCertificationResult,
) -> CheckerBackendResult:
    return CheckerBackendResult(
        backend_name=backend_name,
        candidate_report=incremental.report,
        failed_stage=_first_failed_stage(incremental.report),
        total_hits=incremental.num_hits(),
        total_misses=incremental.num_misses(),
        total_obligations_processed=incremental.num_total(),
        hit_rate=incremental.hit_rate(),
        stage_hit_ids=list(incremental.stage_hit_ids),
        stage_miss_ids=list(incremental.stage_miss_ids),
        interface_hit_ids=list(incremental.interface_hit_ids),
        interface_miss_ids=list(incremental.interface_miss_ids),
        legality_hit_ids=list(incremental.legality_hit_ids),
        legality_miss_ids=list(incremental.legality_miss_ids),
        syntactic_legality_reuse_rate=(
            incremental.syntactic_legality_reuse_rate
        ),
        rewrite_legality_reuse_rate=incremental.rewrite_legality_reuse_rate,
        extra_reused_legality_ids=list(incremental.extra_reused_legality_ids),
        execution_mode=incremental.execution_mode,
        incremental_fallback_reason=incremental.fallback_reason,
        incremental_delta_source=incremental.delta_source,
    )


def _first_failed_stage(report: CandidateReport) -> Optional[str]:
    failed = [stage for stage in report.legality_obligations if not stage.ok]
    return None if not failed else failed[0].stage_name
