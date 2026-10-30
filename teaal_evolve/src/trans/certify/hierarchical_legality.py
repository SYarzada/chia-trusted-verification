

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .candidate import Candidate
from .extract import extract_selected_local_legality_obligations
from .hierarchical_view import (
    ArchitectureTemplateView,
    CandidateHierarchicalView,
    StageView,
)
from .obligation import LegalityStageResult, LocalLegalityObligation


_ALLOWED_ONCHIP_COMPONENTS = {"SBUF", "PSUM"}
_ATTENTION_LOCAL_COMPONENTS = {
    "SBUF",
    "PSUM",
    "DMAEnginePipelineRegisters",
    "ScalarEnginePipelineRegisters",
    "VectorEnginePipelineRegisters",
}

_KERNEL_RULES: Dict[str, Dict[str, object]] = {
    "unary": {
        "external_tensors": {"AH", "ZH"},
        "intermediate_tensors": {"AS", "ZS"},
        "expected_outputs": ("ZH",),
        "expected_engine_ranks": {
            "AS": ("N", "M"),
            "ZS": ("N", "M"),
            "ZH": ("N", "M"),
        },
    },
    "unary_reduce_one_tile": {
        "external_tensors": {"AH", "ZH", "RH"},
        "intermediate_tensors": {"AS", "ZS", "RS"},
        "expected_outputs": ("ZH", "RH"),
        "expected_engine_ranks": {
            "AS": ("N", "K"),
            "ZS": ("N", "K"),
            "RS": ("N", "K"),
            "ZH": ("N", "K"),
            "RH": ("N", "M"),
        },
    },
    "vector_scalar_overlap": {
        "external_tensors": {"AH", "ZH"},
        "intermediate_tensors": {"AS", "AT", "BS", "ZS"},
        "expected_outputs": ("ZH",),
        "expected_engine_ranks": {
            "AS": ("N", "M"),
            "AT": ("N", "M"),
            "BS": ("M", "N"),
            "ZS_init": ("M", "N"),
            "ZS": ("M", "N"),
            "ZH": ("M", "N"),
        },
    },
}


@dataclass(frozen=True)
class HierarchicalLegalityResult:
    obligations: List[LocalLegalityObligation]
    stage_results: List[LegalityStageResult]

    @property
    def is_legal(self) -> bool:
        return all(ob.status == "pass" for ob in self.obligations) and all(
            result.ok for result in self.stage_results
        )


def check_local_legality(
    candidate: Candidate,
    view: CandidateHierarchicalView,
    *,
    include_local_obligations: bool = True,
) -> HierarchicalLegalityResult:
    obligations = (
        extract_selected_local_legality_obligations(candidate)
        if include_local_obligations
        else []
    )
    results: List[LegalityStageResult] = []

    if view.kernel_name in {"attn", "matmul", "mamba"}:
        results.append(_check_attention_cascade_outputs(view))
        results.append(_check_format_rank_coverage(view))
        results.append(_check_format_binding_order(view))
        results.append(_check_stage_bindings_against_architecture(view))
        results.append(_check_attention_tensor_component_domains(view))
        results.append(_check_attention_mapping_shape_consistency(view))
        results.append(_check_fusion_spacetime_consistency(view))
        return HierarchicalLegalityResult(obligations=obligations, stage_results=results)

    rules = _KERNEL_RULES.get(view.kernel_name)
    if rules is None:
        results.append(
            LegalityStageResult(
                stage_name="options",
                ok=False,
                error_type="HierarchicalLocalLegalityError",
                message=f"Unsupported kernel-name for local legality: {view.kernel_name!r}",
            )
        )
        return HierarchicalLegalityResult(obligations=obligations, stage_results=results)

    results.append(_check_cascade_outputs(view, tuple(rules["expected_outputs"])))
    results.append(_check_format_rank_coverage(view))
    results.append(_check_format_binding_order(view))
    results.append(_check_stage_bindings_against_architecture(view))
    results.append(
        _check_tensor_component_domains(
            view,
            external_tensors=set(rules["external_tensors"]),
            intermediate_tensors=set(rules["intermediate_tensors"]),
        )
    )
    results.append(
        _check_stage_binding_shapes(
            view,
            expected_engine_ranks={
                stage_name: tuple(ranks)
                for stage_name, ranks in dict(rules["expected_engine_ranks"]).items()
            },
        )
    )
    results.append(_check_fusion_spacetime_consistency(view))

    return HierarchicalLegalityResult(obligations=obligations, stage_results=results)


def _check_cascade_outputs(
    view: CandidateHierarchicalView,
    expected_outputs: Tuple[str, ...],
) -> LegalityStageResult:
    if tuple(view.cascade_outputs) != expected_outputs:
        return LegalityStageResult(
            stage_name="options",
            ok=False,
            error_type="HierarchicalLocalLegalityError",
            message=(
                f"Restricted hierarchical checker expects cascade-outputs {expected_outputs}, "
                f"but found {list(view.cascade_outputs)}"
            ),
        )
    return LegalityStageResult(stage_name="options", ok=True)


def _check_attention_cascade_outputs(view: CandidateHierarchicalView) -> LegalityStageResult:
    if not view.cascade_outputs:
        return LegalityStageResult(
            stage_name="options",
            ok=False,
            error_type="HierarchicalLocalLegalityError",
            message="Attention-family checker expects at least one cascade output",
        )

    produced_outputs = {stage.output_tensor for stage in view.stage_views}
    declared_tensors = set(view.declaration)
    missing = [name for name in view.cascade_outputs if name not in produced_outputs]
    undeclared = [name for name in view.cascade_outputs if name not in declared_tensors]
    if missing or undeclared:
        return LegalityStageResult(
            stage_name="options",
            ok=False,
            error_type="HierarchicalLocalLegalityError",
            message=(
                "Attention-family cascade outputs must be declared and produced; "
                f"missing={missing}, undeclared={undeclared}"
            ),
        )

    return LegalityStageResult(stage_name="options", ok=True)


def _stage_tensor_uses(stage: StageView):
    yield (
        stage.output_tensor,
        stage.output_binding,
        stage.output_format,
    )
    yield from zip(
        stage.input_tensors,
        stage.input_bindings,
        stage.input_formats,
    )


def _check_format_rank_coverage(
    view: CandidateHierarchicalView,
) -> LegalityStageResult:
    

    for stage in view.stage_views:
        for tensor_name, binding, tensor_format in _stage_tensor_uses(stage):
            rank_order = tensor_format.rank_orders.get(stage.stage_name)
            if rank_order is None:
                return LegalityStageResult(
                    stage_name="format",
                    ok=False,
                    error_type="HierarchicalFormatCoverageError",
                    message=(
                        f"Missing format rank-order for tensor {tensor_name} "
                        f"when computing stage {stage.stage_name}"
                    ),
                )

            expected = tuple(
                rank
                for group_ranks in binding.rank_groups.values()
                for rank in group_ranks
            )
            if (
                len(rank_order) != len(expected)
                or len(set(rank_order)) != len(rank_order)
                or set(rank_order) != set(expected)
            ):
                missing = [rank for rank in expected if rank not in rank_order]
                extra = [rank for rank in rank_order if rank not in expected]
                return LegalityStageResult(
                    stage_name="format",
                    ok=False,
                    error_type="HierarchicalFormatCoverageError",
                    message=(
                        f"Tensor {tensor_name} format for stage {stage.stage_name} "
                        f"must contain exactly the bound ranks {list(expected)}; "
                        f"found {list(rank_order)}, missing={missing}, extra={extra}"
                    ),
                )

    return LegalityStageResult(stage_name="format", ok=True)


def _check_format_binding_order(
    view: CandidateHierarchicalView,
) -> LegalityStageResult:
    

    for stage in view.stage_views:
        for tensor_name, binding, tensor_format in _stage_tensor_uses(stage):
            rank_order = tensor_format.rank_orders.get(stage.stage_name, ())
            unbound = tuple(binding.rank_groups.get("unbound", ()))
            if tuple(rank_order[: len(unbound)]) != unbound:
                return LegalityStageResult(
                    stage_name="binding",
                    ok=False,
                    error_type="HierarchicalFormatBindingOrderError",
                    message=(
                        f"Tensor {tensor_name} format for stage {stage.stage_name} "
                        f"must begin with unbound ranks {list(unbound)}; "
                        f"found {list(rank_order)}"
                    ),
                )

            for group_name, group_ranks in binding.rank_groups.items():
                if group_name == "unbound":
                    continue
                group_set = set(group_ranks)
                observed = tuple(rank for rank in rank_order if rank in group_set)
                if observed != tuple(group_ranks):
                    return LegalityStageResult(
                        stage_name="binding",
                        ok=False,
                        error_type="HierarchicalFormatBindingOrderError",
                        message=(
                            f"Tensor {tensor_name} format for stage {stage.stage_name} "
                            f"orders ranks bound to memory rank {group_name!r} as "
                            f"{list(observed)}, expected {list(group_ranks)}"
                        ),
                    )

    return LegalityStageResult(stage_name="binding", ok=True)


def _check_stage_bindings_against_architecture(
    view: CandidateHierarchicalView,
) -> LegalityStageResult:
    for stage in view.stage_views:
        key = (stage.engine_binding.component, stage.engine_binding.instruction)
        matching_templates = view.architecture_templates.get(key, ())
        if not matching_templates:
            instruction_lower = stage.engine_binding.instruction.lower()
            if view.kernel_name in {"matmul", "mamba", "attn"} and stage.engine_binding.component == "None" and instruction_lower.startswith("renamehbm"):
                continue
            return LegalityStageResult(
                stage_name="architecture",
                ok=False,
                error_type="HierarchicalLocalLegalityError",
                message=(
                    f"No architecture template found for engine {stage.engine_binding.component} "
                    f"instruction {stage.engine_binding.instruction}"
                ),
            )

        matched, mismatch_details = _match_stage_to_architecture_templates(
            stage,
            matching_templates,
        )
        if not matched:
            detail = "; ".join(mismatch_details[:4])
            if len(mismatch_details) > 4:
                detail += f"; ... ({len(mismatch_details) - 4} more template-stage mismatches)"
            return LegalityStageResult(
                stage_name="architecture",
                ok=False,
                error_type="HierarchicalArchitectureTemplateMismatch",
                message=(
                    f"No architecture template stage matched {stage.stage_name} for "
                    f"{stage.engine_binding.component}.{stage.engine_binding.instruction}. "
                    f"{detail}"
                ),
            )

    return LegalityStageResult(stage_name="architecture", ok=True)



def _match_stage_to_architecture_templates(
    stage: StageView,
    templates: Tuple[ArchitectureTemplateView, ...],
) -> Tuple[bool, List[str]]:
    mismatch_details: List[str] = []
    for template in templates:
        ok, reason = _architecture_template_stage_matches(template, stage)
        if ok:
            return True, []
        mismatch_details.append(
            f"{template.template_name}/{template.template_stage_name}: {reason}"
        )
    return False, mismatch_details


def _architecture_template_stage_matches(
    template: ArchitectureTemplateView,
    stage: StageView,
) -> Tuple[bool, str]:
    if not _template_can_implement_stage(template.op_kind, stage.op_kind):
        return False, f"operation {template.op_kind!r} cannot implement {stage.op_kind!r}"                                                                 
                                    
    enforce_components = (
        stage.engine_binding.component == "DMAEngine"
        and stage.engine_binding.instruction in {"Load", "Store"}
    )
    if enforce_components and template.input_components:
        if len(template.input_components) != len(stage.input_bindings):
            return False, (
                f"tensor-input arity {len(stage.input_bindings)} != template "
                f"{len(template.input_components)}"
            )
        for index, (allowed, binding) in enumerate(
            zip(template.input_components, stage.input_bindings)
        ):
            if allowed and binding.current_component not in allowed:
                return False, (
                    f"input {index} tensor {binding.tensor_name} uses "
                    f"{binding.current_component!r}, expected one of {list(allowed)}"
                )

    if (
        enforce_components
        and template.output_components
        and stage.output_binding.current_component not in template.output_components
    ):
        return False, (
            f"output tensor {stage.output_tensor} uses "
            f"{stage.output_binding.current_component!r}, expected one of "
            f"{list(template.output_components)}"
        )

    return True, "matched"


def _template_can_implement_stage(template_op_kind: str, stage_op_kind: str) -> bool:
    

    if template_op_kind == stage_op_kind:
        return True

                                                                            
                                                                               
    if template_op_kind == "unary_reduce" and stage_op_kind in {"unary", "reduce", "unary_reduce", "binary_reduce"}:
        return True
    if template_op_kind == "binary_reduce" and stage_op_kind in {"binary", "reduce", "binary_reduce"}:
        return True
    if template_op_kind == "map_reduce" and stage_op_kind in {"unary", "binary", "reduce", "unary_reduce", "binary_reduce"}:
        return True

    if template_op_kind == "binary" and stage_op_kind in {"binary", "binary_reduce"}:
        return True
    if template_op_kind == "reduce" and stage_op_kind in {"reduce", "unary_reduce", "binary_reduce"}:
        return True
    if template_op_kind == "unary" and stage_op_kind == "unary":
        return True
    if template_op_kind == "init" and stage_op_kind == "init":
        return True
    if template_op_kind == "rename" and stage_op_kind == "rename":
        return True
    if template_op_kind == "unknown":
        return True
    return False


def _check_tensor_component_domains(
    view: CandidateHierarchicalView,
    *,
    external_tensors: Set[str],
    intermediate_tensors: Set[str],
) -> LegalityStageResult:
    for stage in view.stage_views:
        for binding in stage.input_bindings + (stage.output_binding,):
            if binding.tensor_name in intermediate_tensors:
                if binding.current_component not in _ALLOWED_ONCHIP_COMPONENTS:
                    return LegalityStageResult(
                        stage_name="binding",
                        ok=False,
                        error_type="HierarchicalLocalLegalityError",
                        message=(
                            f"Intermediate tensor {binding.tensor_name} must use one of "
                            f"{sorted(_ALLOWED_ONCHIP_COMPONENTS)}, but found {binding.current_component!r}"
                        ),
                    )
            elif binding.tensor_name in external_tensors:
                if binding.current_component != "HBM":
                    return LegalityStageResult(
                        stage_name="binding",
                        ok=False,
                        error_type="HierarchicalLocalLegalityError",
                        message=(
                            f"External tensor {binding.tensor_name} must stay on HBM, "
                            f"but found {binding.current_component!r}"
                        ),
                    )

    return LegalityStageResult(stage_name="binding", ok=True)


def _check_attention_tensor_component_domains(view: CandidateHierarchicalView) -> LegalityStageResult:
    

    cascade_outputs = set(view.cascade_outputs)
    input_tensors = set()
    produced_tensors = set()
    for stage in view.stage_views:
        produced_tensors.add(stage.output_tensor)
        input_tensors.update(stage.input_tensors)

    source_tensors = input_tensors - produced_tensors

    for stage in view.stage_views:
        for binding in stage.input_bindings + (stage.output_binding,):
            tensor = binding.tensor_name
            component = binding.current_component
            if component is None:
                return LegalityStageResult(
                    stage_name="binding",
                    ok=False,
                    error_type="HierarchicalLocalLegalityError",
                    message=f"Tensor {tensor} has no selected component",
                )

                                                                             
                                                                               
                                                                               
                                             
            external_like = tensor in source_tensors or tensor in cascade_outputs or (tensor.endswith("H") and len(tensor) > 1)
            if external_like:
                if component != "HBM":
                    return LegalityStageResult(
                        stage_name="binding",
                        ok=False,
                        error_type="HierarchicalLocalLegalityError",
                        message=(
                            f"Structured-kernel external/output tensor {tensor} must use HBM, "
                            f"but found {component!r}"
                        ),
                    )
            elif tensor in produced_tensors or tensor in input_tensors:
                if component not in _ATTENTION_LOCAL_COMPONENTS:
                    return LegalityStageResult(
                        stage_name="binding",
                        ok=False,
                        error_type="HierarchicalLocalLegalityError",
                        message=(
                            f"Structured-kernel intermediate tensor {tensor} must use a local or pipeline component, "
                            f"but found {component!r}"
                        ),
                    )

    return LegalityStageResult(stage_name="binding", ok=True)


def _check_stage_binding_shapes(
    view: CandidateHierarchicalView,
    *,
    expected_engine_ranks: Dict[str, Tuple[str, ...]],
) -> LegalityStageResult:
    for stage in view.stage_views:
        expected_ranks = expected_engine_ranks.get(stage.stage_name)
        if expected_ranks is not None and tuple(stage.engine_binding.ranks) != tuple(expected_ranks):
            return LegalityStageResult(
                stage_name="binding",
                ok=False,
                error_type="HierarchicalLocalLegalityError",
                message=(
                    f"Stage {stage.stage_name} uses engine ranks {list(stage.engine_binding.ranks)} "
                    f"but restricted checker expects {list(expected_ranks)}"
                ),
            )

        spacetime_result = _check_one_stage_spacetime(stage.stage_name, stage.loop_order, stage.space_ranks, stage.time_ranks)
        if spacetime_result is not None:
            return spacetime_result

        subset_result = _check_engine_ranks_in_loop_order(stage.stage_name, stage.engine_binding.ranks, stage.loop_order)
        if subset_result is not None:
            return subset_result

    return LegalityStageResult(stage_name="mapping", ok=True)


def _check_attention_mapping_shape_consistency(view: CandidateHierarchicalView) -> LegalityStageResult:
    for stage in view.stage_views:
        spacetime_result = _check_one_stage_spacetime(stage.stage_name, stage.loop_order, stage.space_ranks, stage.time_ranks)
        if spacetime_result is not None:
            return spacetime_result

        subset_result = _check_engine_ranks_in_loop_order(stage.stage_name, stage.engine_binding.ranks, stage.loop_order)
        if subset_result is not None:
            return subset_result

        if not stage.output_indices:
            return LegalityStageResult(
                stage_name="mapping",
                ok=False,
                error_type="HierarchicalLocalLegalityError",
                message=f"Attention stage {stage.stage_name} must have output indices",
            )

    return LegalityStageResult(stage_name="mapping", ok=True)


def _check_one_stage_spacetime(
    stage_name: str,
    loop_order: Tuple[str, ...],
    space_ranks: Tuple[str, ...],
    time_ranks: Tuple[str, ...],
) -> Optional[LegalityStageResult]:
    spacetime_ranks = tuple(space_ranks) + tuple(time_ranks)
    if set(spacetime_ranks) != set(loop_order) or len(spacetime_ranks) != len(loop_order):
        return LegalityStageResult(
            stage_name="mapping",
            ok=False,
            error_type="HierarchicalLocalLegalityError",
            message=(
                f"Stage {stage_name} spacetime ranks do not cover its loop-order "
                f"in the restricted checker"
            ),
        )

    expected_space_order = tuple(rank for rank in loop_order if rank in set(space_ranks))
    if expected_space_order != tuple(space_ranks):
        return LegalityStageResult(
            stage_name="mapping",
            ok=False,
            error_type="HierarchicalMappingOrderMismatch",
            message=(
                f"Stage {stage_name} space-rank order {list(space_ranks)} does not "
                f"preserve loop-order {list(loop_order)}"
            ),
        )

    expected_time_order = tuple(rank for rank in loop_order if rank in set(time_ranks))
    if expected_time_order != tuple(time_ranks):
        return LegalityStageResult(
            stage_name="mapping",
            ok=False,
            error_type="HierarchicalMappingOrderMismatch",
            message=(
                f"Stage {stage_name} time-rank order {list(time_ranks)} does not "
                f"preserve loop-order {list(loop_order)}"
            ),
        )
    return None



def _check_fusion_spacetime_consistency(
    view: CandidateHierarchicalView,
) -> LegalityStageResult:
    

    rank_is_spatial: Dict[Tuple[str, str], bool] = {}
    for stage in view.stage_views:
        space_set = set(stage.space_ranks)
        for rank in stage.loop_order:
            rank_is_spatial[(stage.stage_name, rank)] = rank in space_set

    for group in view.fusion_rank_groups:
        statuses: List[bool] = []
        missing: List[str] = []
        for stage_name, rank_name in group:
            key = (stage_name, rank_name)
            if key not in rank_is_spatial:
                missing.append(f"{stage_name}.{rank_name}")
            else:
                statuses.append(rank_is_spatial[key])

        if missing:
            return LegalityStageResult(
                stage_name="mapping",
                ok=False,
                error_type="HierarchicalFusionRankError",
                message=f"Fusion references unknown mapped ranks: {missing}",
            )
        if statuses and any(statuses) and not all(statuses):
            names = [f"{stage}.{rank}" for stage, rank in group]
            return LegalityStageResult(
                stage_name="mapping",
                ok=False,
                error_type="HierarchicalFusionSpacetimeMismatch",
                message=(
                    "Fused ranks must be either all spatial or all temporal, "
                    f"but found a mixture in {names}"
                ),
            )

    return LegalityStageResult(stage_name="mapping", ok=True)


def _check_engine_ranks_in_loop_order(
    stage_name: str,
    engine_ranks: Tuple[str, ...],
    loop_order: Tuple[str, ...],
) -> Optional[LegalityStageResult]:
    if not set(engine_ranks).issubset(set(loop_order)):
        return LegalityStageResult(
            stage_name="binding",
            ok=False,
            error_type="HierarchicalLocalLegalityError",
            message=(
                f"Stage {stage_name} engine ranks {list(engine_ranks)} are not a subset "
                f"of mapping.loop-order {list(loop_order)}"
            ),
        )
    return None
