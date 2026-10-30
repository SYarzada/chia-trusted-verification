

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .candidate import Candidate
from .deps import (
    DeltaImpactReport,
    analyze_changed_knob_impact,
    analyze_delta_impact,
)
from .diff import ChangedKnob, diff_candidate_knobs
from .extract import (
    extract_selected_local_legality_obligations,
    extract_selected_local_legality_requests,
)
from .hierarchical_interface import check_interface_obligations
from .hierarchical_legality import check_local_legality
from .hierarchical_pipeline import certify_candidate_hierarchical
from .hierarchical_stage import check_stage_obligations
from .hierarchical_view import HierarchicalViewError, build_candidate_hierarchical_view
from .knobs import KNOB_SPACE_RESTRICTED, extract_knob_map
from .mutate import apply_knob_specs_atomic
from .objective_backend import (
    OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    compute_objective_with_backend,
)
from .obligation import (
    CandidateReport,
    LegalityStageResult,
    LocalLegalityObligation,
    ObjectiveResult,
)
from .fragments import build_local_legality_request_fragment, render_expr
from .rewrite import normalize_expr


INCREMENTAL_EXECUTION_MODE = "dependency_driven_incremental"
INCREMENTAL_FALLBACK_MODE = "complete_hierarchical_fallback"
DELTA_SOURCE_RECONSTRUCTED = "reconstructed_candidate_diff"
DELTA_SOURCE_TRUSTED_PROVENANCE = "trusted_structured_provenance"


@dataclass(frozen=True)
class TrustedKnobDelta:
    

    changed_knobs: Tuple[ChangedKnob, ...]
    knob_space: str
    source: str
    base_candidate_fingerprint: str
    mutated_candidate_fingerprint: str
    impact: Optional[DeltaImpactReport] = None


@dataclass(frozen=True)
class IncrementalCertificationResult:
    report: CandidateReport
    stage_hit_ids: List[str]
    stage_miss_ids: List[str]
    interface_hit_ids: List[str]
    interface_miss_ids: List[str]
    legality_hit_ids: List[str]
    legality_miss_ids: List[str]
    syntactic_legality_reuse_rate: Optional[float]
    rewrite_legality_reuse_rate: Optional[float]
    extra_reused_legality_ids: List[str]
    execution_mode: str
    fallback_reason: Optional[str] = None
    delta_source: str = DELTA_SOURCE_RECONSTRUCTED

    def num_hits(self) -> int:
        return (
            len(self.stage_hit_ids)
            + len(self.interface_hit_ids)
            + len(self.legality_hit_ids)
        )

    def num_misses(self) -> int:
        return (
            len(self.stage_miss_ids)
            + len(self.interface_miss_ids)
            + len(self.legality_miss_ids)
        )

    def num_total(self) -> int:
        return self.num_hits() + self.num_misses()

    def hit_rate(self) -> float:
        total = self.num_total()
        return 0.0 if total == 0 else self.num_hits() / total


@dataclass(frozen=True)
class _DeltaCoverage:
    covered: bool
    reason: Optional[str]
    impact: Optional[DeltaImpactReport]


def certify_candidate_hierarchical_incremental(
    mutated_candidate: Candidate,
    *,
    base_candidate: Candidate,
    base_report: CandidateReport,
    rewrite_aware: bool,
    include_bottleneck_objective: bool = False,
    objective_shapes: Optional[Dict[str, int]] = None,
    objective_backend_name: str = OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    knob_space: str = KNOB_SPACE_RESTRICTED,
    trusted_knob_delta: Optional[TrustedKnobDelta] = None,
) -> IncrementalCertificationResult:
    

    delta_source = (
        DELTA_SOURCE_TRUSTED_PROVENANCE
        if trusted_knob_delta is not None
        else DELTA_SOURCE_RECONSTRUCTED
    )

    if not base_report.is_legal:
        return _fallback(
            mutated_candidate,
            reason="base_report_not_legal",
            rewrite_aware=rewrite_aware,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
            delta_source=delta_source,
        )

    coverage = _validate_knob_delta_coverage(
        base_candidate,
        base_report,
        mutated_candidate,
        knob_space=knob_space,
        trusted_knob_delta=trusted_knob_delta,
    )
    if not coverage.covered or coverage.impact is None:
        return _fallback(
            mutated_candidate,
            reason=coverage.reason or "unmodeled_candidate_delta",
            rewrite_aware=rewrite_aware,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
            delta_source=delta_source,
        )

    try:
        view = build_candidate_hierarchical_view(mutated_candidate)
    except HierarchicalViewError as exc:
                                                                                
                                                                       
        report = CandidateReport(
            candidate_label=mutated_candidate.label,
            is_legal=False,
            legality_obligations=[
                LegalityStageResult(
                    stage_name=exc.stage_name,
                    ok=False,
                    error_type=type(exc).__name__,
                    message=exc.message,
                )
            ],
            local_legality_obligations=[],
            stage_obligations=[],
            interface_obligations=[],
            objective=_skipped_or_disabled_objective(
                include_bottleneck_objective
            ),
        )
        return IncrementalCertificationResult(
            report=report,
            stage_hit_ids=[],
            stage_miss_ids=[],
            interface_hit_ids=[],
            interface_miss_ids=[],
            legality_hit_ids=[],
            legality_miss_ids=[],
            syntactic_legality_reuse_rate=(0.0 if rewrite_aware else None),
            rewrite_legality_reuse_rate=(0.0 if rewrite_aware else None),
            extra_reused_legality_ids=[],
            execution_mode=INCREMENTAL_EXECUTION_MODE,
            delta_source=delta_source,
        )

    local_legality_requests = extract_selected_local_legality_requests(
        mutated_candidate,
        knob_space=knob_space,
    )

    universe_error = _obligation_universe_error(
        base_report,
        view_stage_ids=[f"stage:{stage.stage_name}" for stage in view.stage_views],
        view_interface_ids=[interface.obligation_id for interface in view.interface_views],
        local_ids=[obligation.obligation_id for obligation in local_legality_requests],
    )
    if universe_error is not None:
        return _fallback(
            mutated_candidate,
            reason=universe_error,
            rewrite_aware=rewrite_aware,
            include_bottleneck_objective=include_bottleneck_objective,
            objective_shapes=objective_shapes,
            objective_backend_name=objective_backend_name,
            knob_space=knob_space,
            delta_source=delta_source,
        )

    impact = coverage.impact
    stage_miss_set = set(impact.affected_stage_obligation_ids)
    interface_miss_set = set(impact.affected_interface_obligation_ids)
    stage_hit_ids, stage_miss_ids = _partition_in_base_order(
        [obligation.obligation_id for obligation in base_report.stage_obligations],
        stage_miss_set,
    )
    interface_hit_ids, interface_miss_ids = _partition_in_base_order(
        [
            obligation.obligation_id
            for obligation in base_report.interface_obligations
        ],
        interface_miss_set,
    )

    (
        legality_hit_ids,
        legality_miss_ids,
        syntactic_rate,
        rewrite_rate,
        extra_reused,
    ) = _classify_local_legality_reuse(
        base_report=base_report,
        current_obligations=local_legality_requests,
        syntactic_affected_ids=set(impact.affected_legality_obligation_ids),
        rewrite_aware=rewrite_aware,
    )

    base_stage_map = {
        obligation.obligation_id: obligation
        for obligation in base_report.stage_obligations
    }
    base_interface_map = {
        obligation.obligation_id: obligation
        for obligation in base_report.interface_obligations
    }
    base_local_map = {
        obligation.obligation_id: obligation
        for obligation in base_report.local_legality_obligations
    }
    syntactic_legality_hit_ids = [
        obligation.obligation_id
        for obligation in local_legality_requests
        if obligation.obligation_id
        not in set(impact.affected_legality_obligation_ids)
    ]
    selected_local_legality = extract_selected_local_legality_obligations(
        mutated_candidate,
        knob_space=knob_space,
        cached_obligations=base_local_map,
        reuse_obligation_ids=set(syntactic_legality_hit_ids),
        rewrite_reuse_obligation_ids=set(extra_reused),
    )

                                                                            
                                                                              
                                                                          
    legality = check_local_legality(
        mutated_candidate,
        view,
        include_local_obligations=False,
    )
    stages = check_stage_obligations(
        view,
        cached_obligations=base_stage_map,
        reuse_obligation_ids=set(stage_hit_ids),
    )
    interfaces = check_interface_obligations(
        view,
        cached_obligations=base_interface_map,
        reuse_obligation_ids=set(interface_hit_ids),
    )

    stage_results = (
        list(legality.stage_results)
        + list(stages.stage_results)
        + list(interfaces.stage_results)
    )
    local_legality_ok = all(
        obligation.status == "pass" for obligation in selected_local_legality
    )
    is_legal = (
        legality.is_legal
        and local_legality_ok
        and stages.is_legal
        and interfaces.is_legal
    )

    if include_bottleneck_objective and is_legal:
        objective = compute_objective_with_backend(
            mutated_candidate,
            backend_name=objective_backend_name,
            shapes=objective_shapes,
        )
    else:
        objective = _skipped_or_disabled_objective(include_bottleneck_objective)

    report = CandidateReport(
        candidate_label=mutated_candidate.label,
        is_legal=is_legal,
        legality_obligations=stage_results,
        local_legality_obligations=list(selected_local_legality),
        stage_obligations=list(stages.obligations),
        interface_obligations=list(interfaces.obligations),
        objective=objective,
    )

    return IncrementalCertificationResult(
        report=report,
        stage_hit_ids=stage_hit_ids,
        stage_miss_ids=stage_miss_ids,
        interface_hit_ids=interface_hit_ids,
        interface_miss_ids=interface_miss_ids,
        legality_hit_ids=legality_hit_ids,
        legality_miss_ids=legality_miss_ids,
        syntactic_legality_reuse_rate=syntactic_rate,
        rewrite_legality_reuse_rate=rewrite_rate,
        extra_reused_legality_ids=extra_reused,
        execution_mode=INCREMENTAL_EXECUTION_MODE,
        delta_source=delta_source,
    )


def verify_trusted_knob_delta(
    base_candidate: Candidate,
    mutated_candidate: Candidate,
    knob_specs: Sequence[str],
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
    source: str = "immutable_candidate_corpus",
    base_candidate_fingerprint: str,
    mutated_candidate_fingerprint: str,
    base_report: Optional[CandidateReport] = None,
) -> TrustedKnobDelta:
    

    raw_specs: List[str] = []
    knob_ids: List[str] = []
    for spec in knob_specs:
        text = str(spec).strip()
        knob_id = text.split("=", 1)[0].strip()
        if not knob_id:
            raise ValueError(f"Malformed trusted knob spec: {text!r}")
        raw_specs.append(text)
        knob_ids.append(knob_id)

    if len(set(knob_ids)) != len(knob_ids):
        raise ValueError("Trusted knob specs contain duplicate knob ids")

    # Reconstruct candidates using the same mutation rules as generation.
    # Implicit mutations like mapping.loop_order::KS are valid without =value.
    reconstructed = (
        apply_knob_specs_atomic(
            base_candidate,
            raw_specs,
            label=mutated_candidate.label,
            knob_space=knob_space,
        )
        if raw_specs
        else base_candidate.clone(label=mutated_candidate.label)
    )
    if reconstructed.spec_dict != mutated_candidate.spec_dict:
        raise ValueError(
            "Trusted knob specs do not reconstruct the materialized candidate"
        )

    base_knobs = extract_knob_map(base_candidate, knob_space=knob_space)
    reconstructed_knobs = extract_knob_map(reconstructed, knob_space=knob_space)
    changed_knobs: List[ChangedKnob] = []
    for knob_id in knob_ids:
        base_knob = base_knobs.get(knob_id)
        after_knob = reconstructed_knobs.get(knob_id)
        if base_knob is None or after_knob is None:
            raise ValueError(f"Trusted delta references unknown knob {knob_id!r}")
        changed_knobs.append(
            ChangedKnob(
                knob_id=knob_id,
                before_value=base_knob.current_value,
                after_value=after_knob.current_value,
                detail=base_knob.detail,
            )
        )

    impact = (
        None
        if base_report is None
        else analyze_changed_knob_impact(
            base_candidate,
            base_report,
            changed_knobs,
            knob_space=knob_space,
        )
    )
    return TrustedKnobDelta(
        changed_knobs=tuple(changed_knobs),
        knob_space=knob_space,
        source=str(source),
        base_candidate_fingerprint=str(base_candidate_fingerprint),
        mutated_candidate_fingerprint=str(mutated_candidate_fingerprint),
        impact=impact,
    )


def _validate_knob_delta_coverage(
    base_candidate: Candidate,
    base_report: CandidateReport,
    mutated_candidate: Candidate,
    *,
    knob_space: str,
    trusted_knob_delta: Optional[TrustedKnobDelta] = None,
) -> _DeltaCoverage:
    

    if trusted_knob_delta is not None:
        if trusted_knob_delta.knob_space != knob_space:
            return _DeltaCoverage(
                covered=False,
                reason="trusted_delta_knob_space_mismatch",
                impact=None,
            )
        try:
            if trusted_knob_delta.impact is not None:
                impact = trusted_knob_delta.impact
                if list(impact.changed_knobs) != list(
                    trusted_knob_delta.changed_knobs
                ):
                    return _DeltaCoverage(
                        covered=False,
                        reason="trusted_delta_precomputed_impact_mismatch",
                        impact=None,
                    )
            else:
                impact = analyze_changed_knob_impact(
                    base_candidate,
                    base_report,
                    trusted_knob_delta.changed_knobs,
                    knob_space=knob_space,
                )
        except Exception as exc:
            return _DeltaCoverage(
                covered=False,
                reason=f"trusted_delta_impact_analysis_failed:{type(exc).__name__}",
                impact=None,
            )
        return _DeltaCoverage(covered=True, reason=None, impact=impact)

    try:
        changes = diff_candidate_knobs(
            base_candidate,
            mutated_candidate,
            knob_space=knob_space,
        )
        knob_specs = [
            f"{change.knob_id}={change.after_value}" for change in changes
        ]
        reconstructed = (
            apply_knob_specs_atomic(
                base_candidate,
                knob_specs,
                label=mutated_candidate.label,
                knob_space=knob_space,
            )
            if knob_specs
            else base_candidate.clone(label=mutated_candidate.label)
        )
    except Exception as exc:
        return _DeltaCoverage(
            covered=False,
            reason=f"knob_delta_reconstruction_failed:{type(exc).__name__}",
            impact=None,
        )

    if reconstructed.spec_dict != mutated_candidate.spec_dict:
        return _DeltaCoverage(
            covered=False,
            reason="raw_spec_delta_outside_declared_knob_space",
            impact=None,
        )

    try:
        impact = analyze_delta_impact(
            base_candidate,
            base_report,
            mutated_candidate,
            knob_space=knob_space,
        )
    except Exception as exc:
        return _DeltaCoverage(
            covered=False,
            reason=f"delta_impact_analysis_failed:{type(exc).__name__}",
            impact=None,
        )

    return _DeltaCoverage(covered=True, reason=None, impact=impact)


def _obligation_universe_error(
    base_report: CandidateReport,
    *,
    view_stage_ids: Sequence[str],
    view_interface_ids: Sequence[str],
    local_ids: Sequence[str],
) -> Optional[str]:
    base_stage_ids = [
        obligation.obligation_id for obligation in base_report.stage_obligations
    ]
    base_interface_ids = [
        obligation.obligation_id
        for obligation in base_report.interface_obligations
    ]
    base_local_ids = [
        obligation.obligation_id
        for obligation in base_report.local_legality_obligations
    ]

    if list(view_stage_ids) != base_stage_ids:
        return "stage_obligation_universe_changed"
    if list(view_interface_ids) != base_interface_ids:
        return "interface_obligation_universe_changed"
    if list(local_ids) != base_local_ids:
        return "local_legality_obligation_universe_changed"
    return None


def _partition_in_base_order(
    obligation_ids: Sequence[str],
    miss_ids: Set[str],
) -> Tuple[List[str], List[str]]:
    hits = [obligation_id for obligation_id in obligation_ids if obligation_id not in miss_ids]
    misses = [obligation_id for obligation_id in obligation_ids if obligation_id in miss_ids]
    return hits, misses


def _classify_local_legality_reuse(
    *,
    base_report: CandidateReport,
    current_obligations: Sequence[LocalLegalityObligation],
    syntactic_affected_ids: Set[str],
    rewrite_aware: bool,
) -> Tuple[List[str], List[str], Optional[float], Optional[float], List[str]]:
    current_ids = [obligation.obligation_id for obligation in current_obligations]
    syntactic_hits = [
        obligation_id
        for obligation_id in current_ids
        if obligation_id not in syntactic_affected_ids
    ]
    syntactic_misses = [
        obligation_id
        for obligation_id in current_ids
        if obligation_id in syntactic_affected_ids
    ]

    if not rewrite_aware:
        return syntactic_hits, syntactic_misses, None, None, []

                                                                               
                                                                             
                                                                             
                                           
    base_tensor_rewrite_keys = {
        _local_rewrite_key(obligation)
        for obligation in base_report.local_legality_obligations
        if obligation.kind == "tensor_component_group"
        and obligation.status == "pass"
    }
    syntactic_hit_set = set(syntactic_hits)
    rewrite_hits: List[str] = []
    for obligation in current_obligations:
        if obligation.obligation_id in syntactic_hit_set:
            rewrite_hits.append(obligation.obligation_id)
            continue
        if (
            obligation.kind == "tensor_component_group"
            and _local_rewrite_key(obligation) in base_tensor_rewrite_keys
        ):
            rewrite_hits.append(obligation.obligation_id)

    rewrite_hit_set = set(rewrite_hits)
    rewrite_misses = [
        obligation_id
        for obligation_id in current_ids
        if obligation_id not in rewrite_hit_set
    ]
    total = len(current_ids)
    syntactic_rate = 0.0 if total == 0 else len(syntactic_hits) / total
    rewrite_rate = 0.0 if total == 0 else len(rewrite_hits) / total
    extra_reused = sorted(set(rewrite_hits) - set(syntactic_hits))
    return (
        rewrite_hits,
        rewrite_misses,
        syntactic_rate,
        rewrite_rate,
        extra_reused,
    )


def _local_rewrite_key(obligation: LocalLegalityObligation) -> str:
    fragment = build_local_legality_request_fragment(obligation)
    return render_expr(normalize_expr(fragment))


def _fallback(
    candidate: Candidate,
    *,
    reason: str,
    rewrite_aware: bool,
    include_bottleneck_objective: bool,
    objective_shapes: Optional[Dict[str, int]],
    objective_backend_name: str,
    knob_space: str,
    delta_source: str = DELTA_SOURCE_RECONSTRUCTED,
) -> IncrementalCertificationResult:
    report = certify_candidate_hierarchical(
        candidate,
        include_bottleneck_objective=include_bottleneck_objective,
        objective_shapes=objective_shapes,
        objective_backend_name=objective_backend_name,
        knob_space=knob_space,
    )
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
    return IncrementalCertificationResult(
        report=report,
        stage_hit_ids=[],
        stage_miss_ids=stage_miss_ids,
        interface_hit_ids=[],
        interface_miss_ids=interface_miss_ids,
        legality_hit_ids=[],
        legality_miss_ids=legality_miss_ids,
        syntactic_legality_reuse_rate=(0.0 if rewrite_aware else None),
        rewrite_legality_reuse_rate=(0.0 if rewrite_aware else None),
        extra_reused_legality_ids=[],
        execution_mode=INCREMENTAL_FALLBACK_MODE,
        fallback_reason=reason,
        delta_source=delta_source,
    )


def _skipped_or_disabled_objective(
    include_bottleneck_objective: bool,
) -> ObjectiveResult:
    if include_bottleneck_objective:
        return ObjectiveResult(
            enabled=True,
            objective_name="bottleneck_latency",
            status="skipped",
            message="Objective not evaluated because the candidate is illegal.",
        )
    return ObjectiveResult(enabled=False)
