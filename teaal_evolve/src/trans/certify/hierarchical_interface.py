

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, List, Mapping, Optional

from .hierarchical_view import CandidateHierarchicalView, InterfaceView
from .obligation import InterfaceObligation, LegalityStageResult


@dataclass(frozen=True)
class HierarchicalInterfaceResult:
    obligations: List[InterfaceObligation]
    stage_results: List[LegalityStageResult]

    @property
    def is_legal(self) -> bool:
        return all(ob.status == "pass" for ob in self.obligations) and all(
            result.ok for result in self.stage_results
        )


def check_interface_obligations(
    view: CandidateHierarchicalView,
    *,
    cached_obligations: Optional[Mapping[str, InterfaceObligation]] = None,
    reuse_obligation_ids: Optional[AbstractSet[str]] = None,
) -> HierarchicalInterfaceResult:
    

    obligations: List[InterfaceObligation] = []
    results: List[LegalityStageResult] = []
    cached = dict(cached_obligations or {})
    reuse_ids = set(reuse_obligation_ids or set())

    for interface in view.interface_views:
        cached_obligation = cached.get(interface.obligation_id)
        if (
            interface.obligation_id in reuse_ids
            and cached_obligation is not None
            and cached_obligation.status == "pass"
        ):
            obligation = cached_obligation
            result = LegalityStageResult(
                stage_name=f"interface:{interface.tensor_name}",
                ok=True,
            )
        else:
            obligation, result = _check_one_interface(
                interface,
                kernel_name=view.kernel_name,
            )
        obligations.append(obligation)
        results.append(result)

    return HierarchicalInterfaceResult(obligations=obligations, stage_results=results)


def _check_one_interface(interface: InterfaceView, *, kernel_name: str):
    status = "pass"
    detail = "Producer and consumer agree on tensor identity, component, layout, and format"
    explicit_reshape_required = False

    attention_family = kernel_name in {"attn", "matmul", "mamba"}

    if interface.producer_stage_position >= interface.consumer_stage_position:
        status = "fail"
        detail = (
            f"Producer stage {interface.producer_stage_name} must appear before consumer stage "
            f"{interface.consumer_stage_name}"
        )
    elif interface.producer_binding.tensor_name != interface.consumer_binding.tensor_name:
        status = "fail"
        detail = "Producer and consumer bindings refer to different tensor names"
    elif interface.producer_binding.current_component != interface.consumer_binding.current_component:
        if attention_family:
            explicit_reshape_required = True
            detail = (
                f"Structured-kernel tensor {interface.tensor_name} crosses component boundary "
                f"{interface.producer_binding.current_component!r}->{interface.consumer_binding.current_component!r}; "
                "recorded as an explicit materialization interface."
            )
        else:
            status = "fail"
            detail = (
                f"Tensor {interface.tensor_name} uses component {interface.producer_binding.current_component!r} "
                f"at the producer and {interface.consumer_binding.current_component!r} at the consumer"
            )
    elif interface.producer_format_order != interface.consumer_format_order:
        if (
            interface.producer_normalized_format_order
            == interface.consumer_normalized_format_order
        ):
                                                                             
                                                                       
                                                                              
                                                                                
            explicit_reshape_required = True
            detail = (
                f"Tensor {interface.tensor_name} uses partition-equivalent rank orders "
                f"{list(interface.producer_format_order)} and "
                f"{list(interface.consumer_format_order)}; "
                "recorded as an explicit partition reshape interface."
            )
        else:
            status = "fail"
            detail = (
                f"Tensor {interface.tensor_name} rank-order mismatch: producer "
                f"{list(interface.producer_format_order)} "
                f"(normalized {list(interface.producer_normalized_format_order)}) vs "
                f"consumer {list(interface.consumer_format_order)} "
                f"(normalized {list(interface.consumer_normalized_format_order)})"
            )
    elif interface.producer_bit_width != interface.consumer_bit_width:
        if attention_family:
            explicit_reshape_required = True
            detail = (
                f"Structured-kernel tensor {interface.tensor_name} changes bit-width from "
                f"{interface.producer_bit_width} to {interface.consumer_bit_width}; "
                "recorded as an explicit conversion interface."
            )
        else:
            status = "fail"
            detail = (
                f"Tensor {interface.tensor_name} bit-width mismatch: producer "
                f"{interface.producer_bit_width} vs consumer {interface.consumer_bit_width}"
            )
    elif interface.producer_data_type != interface.consumer_data_type:
        if attention_family:
            explicit_reshape_required = True
            detail = (
                f"Structured-kernel tensor {interface.tensor_name} changes data type from "
                f"{interface.producer_data_type!r} to {interface.consumer_data_type!r}; "
                "recorded as an explicit conversion interface."
            )
        else:
            status = "fail"
            detail = (
                f"Tensor {interface.tensor_name} data-type mismatch: producer "
                f"{interface.producer_data_type!r} vs consumer {interface.consumer_data_type!r}"
            )

    obligation = InterfaceObligation(
        obligation_id=interface.obligation_id,
        tensor_name=interface.tensor_name,
        producer_einsum=interface.producer_stage_name,
        consumer_einsum=interface.consumer_stage_name,
        producer_instruction=interface.producer_stage_name,
        consumer_instruction=interface.consumer_stage_name,
        producer_position=interface.producer_stage_position,
        consumer_position=interface.consumer_stage_position,
        explicit_reshape_required=explicit_reshape_required,
        status=status,
        detail=detail,
    )

    result = LegalityStageResult(
        stage_name=f"interface:{interface.tensor_name}",
        ok=(status == "pass"),
        error_type=None if status == "pass" else "HierarchicalInterfaceError",
        message=None if status == "pass" else detail,
    )

    return obligation, result
