

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from .candidate import Candidate
from .diff import ChangedKnob, diff_candidate_knobs
from .knobs import KNOB_SPACE_RESTRICTED, extract_knob_map
from .obligation import CandidateReport


@dataclass(frozen=True)
class ObligationDependency:
    obligation_id: str
    obligation_kind: str
    depends_on_knobs: List[str]
    detail: str


@dataclass(frozen=True)
class DeltaImpactReport:
    changed_knobs: List[ChangedKnob]
    stage_dependencies: List[ObligationDependency]
    interface_dependencies: List[ObligationDependency]
    legality_dependencies: List[ObligationDependency]
    affected_stage_obligation_ids: List[str]
    unaffected_stage_obligation_ids: List[str]
    affected_interface_obligation_ids: List[str]
    unaffected_interface_obligation_ids: List[str]
    affected_legality_obligation_ids: List[str]
    unaffected_legality_obligation_ids: List[str]


def extract_stage_obligation_dependencies(
    candidate: Candidate,
    report: CandidateReport,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[ObligationDependency]:
    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    dependencies: List[ObligationDependency] = []

    for obligation in report.stage_obligations:
        knob_ids: Set[str] = set()
        for tensor_name in list(obligation.input_tensors) + [obligation.output_tensor]:
            component_id = f"binding.tensor.component::{tensor_name}"
            if component_id in knob_map:
                knob_ids.add(component_id)
            format_id = f"format.rank_order::{tensor_name}::{obligation.einsum_name}"
            if format_id in knob_map:
                knob_ids.add(format_id)

        loop_id = f"mapping.loop_order::{obligation.einsum_name}"
        space_id = f"mapping.spacetime::{obligation.einsum_name}"
        if loop_id in knob_map:
            knob_ids.add(loop_id)
        if space_id in knob_map:
            knob_ids.add(space_id)

                                                                              
                                                                               
                                                                          
                                                            
        for knob_id, knob in knob_map.items():
            if knob.kind != "engine_instruction":
                continue
            covered_stages = {
                stage
                for stage in knob.metadata.get("stage_name", "").split("+")
                if stage
            }
            if obligation.einsum_name in covered_stages:
                knob_ids.add(knob_id)

        dependencies.append(
            ObligationDependency(
                obligation_id=obligation.obligation_id,
                obligation_kind="stage",
                depends_on_knobs=sorted(knob_ids),
                detail=(
                    "Stage obligation depends on input/output tensor placement, "
                    "stage-local mapping knobs, and stage-local format rank-order knobs."
                ),
            )
        )

    return dependencies


def extract_interface_obligation_dependencies(
    candidate: Candidate,
    report: CandidateReport,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[ObligationDependency]:
    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    dependencies: List[ObligationDependency] = []

    for obligation in report.interface_obligations:
        knob_ids: Set[str] = set()
        component_id = f"binding.tensor.component::{obligation.tensor_name}"
        if component_id in knob_map:
            knob_ids.add(component_id)

        producer_format_id = f"format.rank_order::{obligation.tensor_name}::{obligation.producer_einsum}"
        consumer_format_id = f"format.rank_order::{obligation.tensor_name}::{obligation.consumer_einsum}"
        if producer_format_id in knob_map:
            knob_ids.add(producer_format_id)
        if consumer_format_id in knob_map:
            knob_ids.add(consumer_format_id)

        dependencies.append(
            ObligationDependency(
                obligation_id=obligation.obligation_id,
                obligation_kind="interface",
                depends_on_knobs=sorted(knob_ids),
                detail=(
                    "Interface obligation depends on the interfacing tensor placement "
                    "and the producer/consumer format rank-order knobs when present."
                ),
            )
        )

    return dependencies


def extract_local_legality_obligation_dependencies(
    candidate: Candidate,
    report: CandidateReport,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[ObligationDependency]:
    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    dependencies: List[ObligationDependency] = []

    for obligation in report.local_legality_obligations:
        knob_ids: Set[str] = {obligation.knob_id}
        detail = "Selected local legality obligation depends directly on its associated knob."

                                                                             
                                                                           
                                                                     
        if obligation.kind == "mapping_spacetime" and obligation.stage_name:
            loop_id = f"mapping.loop_order::{obligation.stage_name}"
            if loop_id in knob_map:
                knob_ids.add(loop_id)
                detail = (
                    "Spacetime legality depends on both the spacetime knob and "
                    "the stage-local loop-order it must cover."
                )

        dependencies.append(
            ObligationDependency(
                obligation_id=obligation.obligation_id,
                obligation_kind="legality",
                depends_on_knobs=sorted(knob_ids),
                detail=detail,
            )
        )

    return dependencies


def analyze_delta_impact(
    base_candidate: Candidate,
    base_report: CandidateReport,
    mutated_candidate: Candidate,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> DeltaImpactReport:
    changed_knobs = diff_candidate_knobs(base_candidate, mutated_candidate, knob_space=knob_space)
    return analyze_changed_knob_impact(
        base_candidate,
        base_report,
        changed_knobs,
        knob_space=knob_space,
    )


def analyze_changed_knob_impact(
    base_candidate: Candidate,
    base_report: CandidateReport,
    changed_knobs: Sequence[ChangedKnob],
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> DeltaImpactReport:
    

    changed_knobs = list(changed_knobs)
    changed_knob_ids: Set[str] = {change.knob_id for change in changed_knobs}

    stage_dependencies = extract_stage_obligation_dependencies(base_candidate, base_report, knob_space=knob_space)
    interface_dependencies = extract_interface_obligation_dependencies(base_candidate, base_report, knob_space=knob_space)
    legality_dependencies = extract_local_legality_obligation_dependencies(base_candidate, base_report, knob_space=knob_space)

    affected_stage_obligation_ids, unaffected_stage_obligation_ids = _split_affected(stage_dependencies, changed_knob_ids)
    affected_interface_obligation_ids, unaffected_interface_obligation_ids = _split_affected(interface_dependencies, changed_knob_ids)
    affected_legality_obligation_ids, unaffected_legality_obligation_ids = _split_affected(legality_dependencies, changed_knob_ids)

    return DeltaImpactReport(
        changed_knobs=changed_knobs,
        stage_dependencies=stage_dependencies,
        interface_dependencies=interface_dependencies,
        legality_dependencies=legality_dependencies,
        affected_stage_obligation_ids=affected_stage_obligation_ids,
        unaffected_stage_obligation_ids=unaffected_stage_obligation_ids,
        affected_interface_obligation_ids=affected_interface_obligation_ids,
        unaffected_interface_obligation_ids=unaffected_interface_obligation_ids,
        affected_legality_obligation_ids=affected_legality_obligation_ids,
        unaffected_legality_obligation_ids=unaffected_legality_obligation_ids,
    )


def _split_affected(
    dependencies: List[ObligationDependency],
    changed_knob_ids: Set[str],
) -> tuple[List[str], List[str]]:
    affected: List[str] = []
    unaffected: List[str] = []
    for dependency in dependencies:
        if any(knob_id in changed_knob_ids for knob_id in dependency.depends_on_knobs):
            affected.append(dependency.obligation_id)
        else:
            unaffected.append(dependency.obligation_id)
    return affected, unaffected
