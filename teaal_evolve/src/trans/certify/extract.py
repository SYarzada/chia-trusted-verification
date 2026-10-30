

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Optional

from src.ir import BoundTensor
from .schema_utils import component_name

from .candidate import Candidate, CandidateArtifacts
from .condition_obligations import extract_symbolic_condition_obligations
from .knobs import (
    CandidateKnob,
    KNOB_SPACE_RESTRICTED,
    extract_candidate_knobs,
    parse_engine_instruction_value,
    parse_rank_list_value,
    parse_spacetime_value,
)
from .legality import LegalityResult
from .obligation import (
    InterfaceObligation,
    LegalityStageResult,
    LocalLegalityObligation,
    StageObligation,
)


@dataclass(frozen=True)
class _TensorUse:
    tensor_name: str
    einsum_name: str
    instruction_name: str
    instruction_position: int
    bound_tensor: BoundTensor


def extract_legality_obligations(result: LegalityResult) -> List[LegalityStageResult]:
    return list(result.stage_results)


def extract_selected_local_legality_obligations(
    candidate: Candidate,
    knob_space: str = KNOB_SPACE_RESTRICTED,
    *,
    cached_obligations: Optional[Mapping[str, LocalLegalityObligation]] = None,
    reuse_obligation_ids: Optional[AbstractSet[str]] = None,
    rewrite_reuse_obligation_ids: Optional[AbstractSet[str]] = None,
) -> List[LocalLegalityObligation]:
    
    tensor_bindings = candidate.spec_dict.get("binding", {}).get("tensor", [])
    declaration = candidate.spec_dict.get("einsum", {}).get("declaration", {})
    loop_order = candidate.spec_dict.get("mapping", {}).get("loop-order", {})
    obligations: List[LocalLegalityObligation] = []
    cached = dict(cached_obligations or {})
    reuse_ids = set(reuse_obligation_ids or set())
    rewrite_reuse_ids = set(rewrite_reuse_obligation_ids or set())

    for knob in extract_candidate_knobs(candidate, knob_space=knob_space):
        obligation_id = f"res:{knob.knob_id}"
        cached_obligation = cached.get(obligation_id)
        if (
            obligation_id in reuse_ids
            and cached_obligation is not None
            and cached_obligation.status == "pass"
        ):
            obligations.append(cached_obligation)
            continue
        if (
            obligation_id in rewrite_reuse_ids
            and knob.kind == "tensor_component_group"
        ):
            obligations.append(_build_rewrite_reused_tensor_component_legality(knob))
            continue

        if knob.kind == "tensor_component_group":
            obligations.append(_build_tensor_component_legality(knob, tensor_bindings))
        elif knob.kind == "engine_instruction":
            stage_name = knob.metadata.get("stage_name", "")
            status = "pass"
            detail = (
                "Engine-instruction knob value is syntactically well-formed; "
                "architecture compatibility is checked by the hierarchical architecture stage."
            )
            try:
                component, instruction = parse_engine_instruction_value(knob.current_value)
            except ValueError as exc:
                status = "fail"
                detail = str(exc)
            else:
                if knob.current_value not in knob.allowed_values:
                    status = "fail"
                    detail = (
                        f"Engine-instruction value {knob.current_value!r} is not in the curated "
                        f"allowed set {knob.allowed_values}"
                    )
                elif not component or not instruction:
                    status = "fail"
                    detail = "Engine-instruction value must contain a component and instruction"

            obligations.append(
                LocalLegalityObligation(
                    obligation_id=f"res:{knob.knob_id}",
                    knob_id=knob.knob_id,
                    kind=knob.kind,
                    stage_name=stage_name,
                    current_value=knob.current_value,
                    allowed_values=list(knob.allowed_values),
                    binding_tensor_indices=list(knob.binding_tensor_indices),
                    status=status,
                    detail=detail,
                )
            )
        elif knob.kind == "mapping_loop_order":
            stage_name = knob.metadata.get("stage_name", "")
            ranks = parse_rank_list_value(knob.current_value)
            status = "pass"
            detail = "Loop-order knob value is syntactically well-formed."
            if not ranks:
                status = "fail"
                detail = "Loop-order must contain at least one rank"
            elif len(set(ranks)) != len(ranks):
                status = "fail"
                detail = "Loop-order contains duplicate ranks"
            obligations.append(
                LocalLegalityObligation(
                    obligation_id=f"res:{knob.knob_id}",
                    knob_id=knob.knob_id,
                    kind=knob.kind,
                    stage_name=stage_name,
                    current_value=knob.current_value,
                    allowed_values=list(knob.allowed_values),
                    status=status,
                    detail=detail,
                )
            )
        elif knob.kind == "mapping_spacetime":
            stage_name = knob.metadata.get("stage_name", "")
            space_ranks, time_ranks = parse_spacetime_value(knob.current_value)
            expected_loop_order = [str(rank) for rank in loop_order.get(stage_name, [])]
            status = "pass"
            detail = "Spacetime knob value covers the current loop-order."
            all_ranks = list(space_ranks) + list(time_ranks)
            if len(set(all_ranks)) != len(all_ranks):
                status = "fail"
                detail = "Spacetime ranks contain duplicates"
            elif set(all_ranks) != set(expected_loop_order) or len(all_ranks) != len(expected_loop_order):
                status = "fail"
                detail = "Spacetime ranks do not cover the current loop-order"
            obligations.append(
                LocalLegalityObligation(
                    obligation_id=f"res:{knob.knob_id}",
                    knob_id=knob.knob_id,
                    kind=knob.kind,
                    stage_name=stage_name,
                    current_value=knob.current_value,
                    allowed_values=list(knob.allowed_values),
                    status=status,
                    detail=detail,
                )
            )
        elif knob.kind == "format_rank_order":
            tensor_name = knob.metadata.get("tensor_name", knob.tensor_name)
            consumer_name = knob.metadata.get("consumer_name", "")
            ranks = parse_rank_list_value(knob.current_value)
            declared = [str(rank) for rank in declaration.get(tensor_name, [])]
            options = candidate.spec_dict.get("options", {})
            problem = candidate.spec_dict.get("problem", {})
            kernel_name = str(options.get("kernel-name") or problem.get("kernel-name") or "")
            status = "pass"
            detail = "Format rank-order knob value is locally well-formed."

            if not ranks:
                status = "fail"
                detail = "Format rank-order must contain at least one rank"
            elif len(set(ranks)) != len(ranks):
                status = "fail"
                detail = "Format rank-order contains duplicate ranks"
            elif kernel_name in {"attn", "matmul", "mamba"}:
                                                                                   
                                                                                 
                                                                                
                                                                                   
                                                                              
                detail = (
                    "Structured-kernel format rank-order is valid. Derived partition and "
                    "padding ranks are checked by the hierarchical and interface stages."
                )
            elif set(ranks) != set(declared) or len(ranks) != len(declared):
                status = "fail"
                detail = "Format rank-order does not match the tensor declaration ranks"
            obligations.append(
                LocalLegalityObligation(
                    obligation_id=f"res:{knob.knob_id}",
                    knob_id=knob.knob_id,
                    kind=knob.kind,
                    tensor_name=tensor_name,
                    consumer_name=consumer_name,
                    current_value=knob.current_value,
                    allowed_values=list(knob.allowed_values),
                    status=status,
                    detail=detail,
                )
            )
        else:
            obligations.append(
                LocalLegalityObligation(
                    obligation_id=f"res:{knob.knob_id}",
                    knob_id=knob.knob_id,
                    kind=knob.kind,
                    tensor_name=knob.tensor_name,
                    current_value=knob.current_value,
                    allowed_values=list(knob.allowed_values),
                    binding_tensor_indices=list(knob.binding_tensor_indices),
                    status="fail",
                    detail=f"Unsupported local legality knob kind: {knob.kind}",
                )
            )

    return obligations


def extract_selected_local_legality_requests(
    candidate: Candidate,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[LocalLegalityObligation]:
    

    return [
        _build_local_legality_request(knob)
        for knob in extract_candidate_knobs(candidate, knob_space=knob_space)
    ]


def _build_local_legality_request(knob: CandidateKnob) -> LocalLegalityObligation:
    return LocalLegalityObligation(
        obligation_id=f"res:{knob.knob_id}",
        knob_id=knob.knob_id,
        kind=knob.kind,
        tensor_name=knob.metadata.get("tensor_name", knob.tensor_name),
        stage_name=knob.metadata.get("stage_name", ""),
        consumer_name=knob.metadata.get("consumer_name", ""),
        current_value=knob.current_value,
        allowed_values=list(knob.allowed_values),
        binding_tensor_indices=list(knob.binding_tensor_indices),
        status="unchecked",
        detail="Local legality premise has not been discharged.",
    )


def _build_rewrite_reused_tensor_component_legality(
    knob: CandidateKnob,
) -> LocalLegalityObligation:
    return LocalLegalityObligation(
        obligation_id=f"res:{knob.knob_id}",
        knob_id=knob.knob_id,
        kind=knob.kind,
        tensor_name=knob.tensor_name,
        current_value=knob.current_value,
        allowed_values=list(knob.allowed_values),
        binding_tensor_indices=list(knob.binding_tensor_indices),
        status="pass",
        detail=(
            "Grouped tensor-component binding entries are locally consistent and "
            "use an allowed component value."
        ),
    )


def _build_tensor_component_legality(knob, tensor_bindings) -> LocalLegalityObligation:
    status = "pass"
    detail = (
        "Grouped tensor-component binding entries are locally consistent and "
        "use an allowed component value."
    )
    current_value = "<missing>"
    values: List[str] = []

    for index in knob.binding_tensor_indices:
        if index >= len(tensor_bindings):
            status = "fail"
            detail = f"binding.tensor index {index} is missing for grouped tensor {knob.tensor_name}"
            break

        component = component_name(tensor_bindings[index].get("component"))
        if component not in knob.allowed_values:
            status = "fail"
            detail = (
                f"Component {component!r} is not allowed for grouped tensor {knob.tensor_name}; "
                f"allowed values are {knob.allowed_values}"
            )
            break

        values.append(component)

    if status == "pass":
        unique_values = sorted(set(values))
        if len(unique_values) != 1:
            status = "fail"
            detail = f"Grouped tensor-component entries for {knob.tensor_name} are inconsistent: {unique_values}"
        else:
            current_value = unique_values[0]

    return LocalLegalityObligation(
        obligation_id=f"res:{knob.knob_id}",
        knob_id=knob.knob_id,
        kind=knob.kind,
        tensor_name=knob.tensor_name,
        current_value=current_value,
        allowed_values=list(knob.allowed_values),
        binding_tensor_indices=list(knob.binding_tensor_indices),
        status=status,
        detail=detail,
    )


def extract_stage_obligations(artifacts: CandidateArtifacts) -> List[StageObligation]:
    assert artifacts.elaborated_binding is not None

    obligations: List[StageObligation] = []

    for instruction_position, instruction in enumerate(artifacts.elaborated_binding.get_instructions()):
        instruction_name = f"{instruction.get_engine()}.{instruction.get_instruction()}"

        for bound_einsum in instruction.get_einsums():
            reuse_spec = bound_einsum.get_reuse_spec()
            tensor_accesses = reuse_spec.get_tensor_accesses()

            obligations.append(
                StageObligation(
                    obligation_id=f"stage:{bound_einsum.get_name()}",
                    einsum_name=bound_einsum.get_name(),
                    instruction_name=instruction_name,
                    instruction_position=instruction_position,
                    engine=instruction.get_engine(),
                    output_tensor=reuse_spec.get_output_tensor().get_name(),
                    input_tensors=[tensor_access.get_tensor().get_name() for tensor_access in tensor_accesses[1:]],
                    bound_ranks=[rank.get_name() for rank in bound_einsum.get_bound_ranks()],
                    unbound_ranks=[rank.get_name() for rank in bound_einsum.get_unbound_ranks()],
                )
            )

    return obligations


def extract_interface_obligations(artifacts: CandidateArtifacts) -> List[InterfaceObligation]:
    assert artifacts.elaborated_binding is not None
    assert artifacts.elaborated_einsum is not None

    cascade_inputs = {tensor.get_name() for tensor in artifacts.elaborated_einsum.get_cascade_inputs()}

    outputs_by_tensor: Dict[str, List[_TensorUse]] = {}
    obligations: List[InterfaceObligation] = []

    for instruction_position, instruction in enumerate(artifacts.elaborated_binding.get_instructions()):
        instruction_name = f"{instruction.get_engine()}.{instruction.get_instruction()}"

        for bound_einsum in instruction.get_einsums():
            reuse_spec = bound_einsum.get_reuse_spec()

            for tensor_access in reuse_spec.get_tensor_accesses()[1:]:
                input_tensor = tensor_access.get_tensor()
                tensor_name = input_tensor.get_name()

                if tensor_name in cascade_inputs:
                    continue

                producer = _latest_prior_use(outputs_by_tensor.get(tensor_name, []), instruction_position)

                if producer is None:
                    obligations.append(
                        InterfaceObligation(
                            obligation_id=f"iface:{tensor_name}:{bound_einsum.get_name()}",
                            tensor_name=tensor_name,
                            producer_einsum="<unknown>",
                            consumer_einsum=bound_einsum.get_name(),
                            producer_instruction="<unknown>",
                            consumer_instruction=instruction_name,
                            producer_position=-1,
                            consumer_position=instruction_position,
                            explicit_reshape_required=False,
                            status="unresolved",
                            detail="No earlier producer found for non-input tensor",
                        )
                    )
                    continue

                status = "pass"
                detail = "Producer and consumer tensor bindings are layout-compatible"
                explicit_reshape_required = False

                try:
                    producer.bound_tensor.match_with_reshape(input_tensor)
                    explicit_reshape_required = producer.bound_tensor.need_explicit_reshape(input_tensor)
                    if explicit_reshape_required:
                        detail = "Layout-compatible but an explicit reshape is required"
                except Exception as exc:
                    status = "fail"
                    detail = str(exc)

                obligations.append(
                    InterfaceObligation(
                        obligation_id=f"iface:{tensor_name}:{producer.einsum_name}->{bound_einsum.get_name()}",
                        tensor_name=tensor_name,
                        producer_einsum=producer.einsum_name,
                        consumer_einsum=bound_einsum.get_name(),
                        producer_instruction=producer.instruction_name,
                        consumer_instruction=instruction_name,
                        producer_position=producer.instruction_position,
                        consumer_position=instruction_position,
                        explicit_reshape_required=explicit_reshape_required,
                        status=status,
                        detail=detail,
                    )
                )

            output_tensor = reuse_spec.get_output_tensor()
            outputs_by_tensor.setdefault(output_tensor.get_name(), []).append(
                _TensorUse(
                    tensor_name=output_tensor.get_name(),
                    einsum_name=bound_einsum.get_name(),
                    instruction_name=instruction_name,
                    instruction_position=instruction_position,
                    bound_tensor=output_tensor,
                )
            )

    return obligations


def _latest_prior_use(uses: List[_TensorUse], instruction_position: int) -> Optional[_TensorUse]:
    candidates = [use for use in uses if use.instruction_position <= instruction_position]
    if not candidates:
        return None
    return max(candidates, key=lambda use: use.instruction_position)
