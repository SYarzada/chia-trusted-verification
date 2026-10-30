

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Optional, Tuple

from .hierarchical_view import CandidateHierarchicalView, StageView, parse_affine_token
from .obligation import LegalityStageResult, StageObligation


_REFERENCE_CASCADES: Dict[str, List[Dict[str, object]]] = {
    "unary": [
        {
            "stage_name": "AS",
            "position": 0,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AH",),
            "engine_component": "DMAEngine",
            "instruction": "Load",
        },
        {
            "stage_name": "ZS",
            "position": 1,
            "op_kind": "unary",
            "semantic_kind": "unary_identity",
            "unary_operator": "square",
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AS",),
            "engine_component": "ScalarEngine",
            "instruction": "Unary",
        },
        {
            "stage_name": "ZH",
            "position": 2,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("ZS",),
            "engine_component": "DMAEngine",
            "instruction": "Store",
        },
    ],
    "unary_reduce_one_tile": [
        {
            "stage_name": "AS",
            "position": 0,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AH",),
            "engine_component": "DMAEngine",
            "instruction": "Load",
        },
        {
            "stage_name": "ZS",
            "position": 1,
            "op_kind": "unary",
            "semantic_kind": "unary_identity",
            "unary_operator": "square",
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AS",),
            "engine_component": "ScalarEngine",
            "instruction": "UnaryReduce",
        },
        {
            "stage_name": "RS",
            "position": 2,
            "op_kind": "reduce",
            "semantic_kind": "reduce_drop_one",
            "unary_operator": None,
            "reduction_operator": "add",
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("ZS",),
            "engine_component": "ScalarEngine",
            "instruction": "UnaryReduce",
        },
        {
            "stage_name": "ZH",
            "position": 3,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("ZS",),
            "engine_component": "DMAEngine",
            "instruction": "Store",
        },
        {
            "stage_name": "RH",
            "position": 4,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("RS",),
            "engine_component": "DMAEngine",
            "instruction": "Store",
        },
    ],
    "vector_scalar_overlap": [
        {
            "stage_name": "AS",
            "position": 0,
            "op_kind": "rename",
            "semantic_kind": "rename_identity",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AH",),
            "engine_component": "DMAEngine",
            "instruction": "Load",
        },
        {
            "stage_name": "AT",
            "position": 1,
            "op_kind": "rename",
            "semantic_kind": "expand_one_fresh_index",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AS",),
            "engine_component": "TensorEngine",
            "instruction": "Transpose32",
        },
        {
            "stage_name": "BS",
            "position": 2,
            "op_kind": "unary",
            "semantic_kind": "unary_identity",
            "unary_operator": "exp",
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("AT",),
            "engine_component": "ScalarEngine",
            "instruction": "Unary",
        },
        {
            "stage_name": "ZS_init",
            "position": 3,
            "op_kind": "init",
            "semantic_kind": "init_zero",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": "0",
            "input_tensors": (),
            "engine_component": "VectorEngine",
            "instruction": "MemSet",
        },
        {
            "stage_name": "ZS",
            "position": 4,
            "op_kind": "binary",
            "semantic_kind": "recurrent_binary_add",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": "add",
            "init_value": None,
            "input_tensors": ("ZS", "BS"),
            "engine_component": "VectorEngine",
            "instruction": "TensorTensorSP",
        },
        {
            "stage_name": "ZH",
            "position": 5,
            "op_kind": "rename",
            "semantic_kind": "read_terminal_slice",
            "unary_operator": None,
            "reduction_operator": None,
            "binary_operator": None,
            "init_value": None,
            "input_tensors": ("ZS",),
            "engine_component": "DMAEngine",
            "instruction": "Store",
        },
    ],
}


@dataclass(frozen=True)
class HierarchicalStageResult:
    obligations: List[StageObligation]
    stage_results: List[LegalityStageResult]

    @property
    def is_legal(self) -> bool:
        return all(ob.status == "pass" for ob in self.obligations) and all(
            result.ok for result in self.stage_results
        )


def check_stage_obligations(
    view: CandidateHierarchicalView,
    *,
    cached_obligations: Optional[Mapping[str, StageObligation]] = None,
    reuse_obligation_ids: Optional[AbstractSet[str]] = None,
) -> HierarchicalStageResult:
    

    obligations: List[StageObligation] = []
    results: List[LegalityStageResult] = []
    cached = dict(cached_obligations or {})
    reuse_ids = set(reuse_obligation_ids or set())

    def reuse_or_check_attention(stage: StageView):
        obligation_id = f"stage:{stage.stage_name}"
        cached_obligation = cached.get(obligation_id)
        if (
            obligation_id in reuse_ids
            and cached_obligation is not None
            and cached_obligation.status == "pass"
        ):
            return cached_obligation, LegalityStageResult(
                stage_name=obligation_id,
                ok=True,
            )
        return _check_attention_stage(view, stage)

    def reuse_or_check_reference(stage: StageView, reference: Dict[str, object]):
        obligation_id = f"stage:{stage.stage_name}"
        cached_obligation = cached.get(obligation_id)
        if (
            obligation_id in reuse_ids
            and cached_obligation is not None
            and cached_obligation.status == "pass"
        ):
            return cached_obligation, LegalityStageResult(
                stage_name=obligation_id,
                ok=True,
            )
        return _check_one_stage(view, stage, reference)

    if view.kernel_name in {"attn", "matmul", "mamba"}:
        for stage in view.stage_views:
            obligation, result = reuse_or_check_attention(stage)
            obligations.append(obligation)
            results.append(result)
        return HierarchicalStageResult(obligations=obligations, stage_results=results)

    expected_stages = _REFERENCE_CASCADES.get(view.kernel_name)
    if expected_stages is None:
        results.append(
            LegalityStageResult(
                stage_name="options",
                ok=False,
                error_type="HierarchicalStageError",
                message=f"Unsupported kernel-name for stage checking: {view.kernel_name!r}",
            )
        )
        return HierarchicalStageResult(obligations=obligations, stage_results=results)

    actual_stage_names = [stage.stage_name for stage in view.stage_views]
    expected_stage_names = [str(spec["stage_name"]) for spec in expected_stages]
    if actual_stage_names != expected_stage_names:
        results.append(
            LegalityStageResult(
                stage_name="einsum",
                ok=False,
                error_type="HierarchicalStageError",
                message=(
                    f"Expected stage sequence {expected_stage_names} for kernel {view.kernel_name}, "
                    f"but found {actual_stage_names}"
                ),
            )
        )
        return HierarchicalStageResult(obligations=obligations, stage_results=results)

    for stage, reference in zip(view.stage_views, expected_stages):
        obligation, result = reuse_or_check_reference(stage, reference)
        obligations.append(obligation)
        results.append(result)

    return HierarchicalStageResult(obligations=obligations, stage_results=results)


def _check_one_stage(
    view: CandidateHierarchicalView,
    stage: StageView,
    reference: Dict[str, object],
) -> Tuple[StageObligation, LegalityStageResult]:
    status = "pass"
    detail = "Stage matches the restricted reference cascade specification"

    def fail(message: str) -> None:
        nonlocal status, detail
        status = "fail"
        detail = message

    if stage.stage_position != int(reference["position"]):
        fail(
            f"Stage {stage.stage_name} appears at position {stage.stage_position}, expected {reference['position']}"
        )
    elif stage.op_kind != reference["op_kind"]:
        fail(
            f"Stage {stage.stage_name} has op_kind {stage.op_kind!r}, expected {reference['op_kind']!r}"
        )
    elif stage.unary_operator != reference["unary_operator"]:
        fail(
            f"Stage {stage.stage_name} has unary operator {stage.unary_operator!r}, expected {reference['unary_operator']!r}"
        )
    elif stage.reduction_operator != reference["reduction_operator"]:
        fail(
            f"Stage {stage.stage_name} has reduction operator {stage.reduction_operator!r}, expected {reference['reduction_operator']!r}"
        )
    elif stage.binary_operator != reference["binary_operator"]:
        fail(
            f"Stage {stage.stage_name} has binary operator {stage.binary_operator!r}, expected {reference['binary_operator']!r}"
        )
    elif stage.init_value != reference["init_value"]:
        fail(
            f"Stage {stage.stage_name} has init value {stage.init_value!r}, expected {reference['init_value']!r}"
        )
    elif tuple(stage.input_tensors) != tuple(reference["input_tensors"]):
        fail(
            f"Stage {stage.stage_name} uses input tensors {list(stage.input_tensors)}, expected {list(reference['input_tensors'])}"
        )
    elif stage.engine_binding.component != reference["engine_component"]:
        fail(
            f"Stage {stage.stage_name} binds to engine {stage.engine_binding.component!r}, expected {reference['engine_component']!r}"
        )
    elif stage.engine_binding.instruction != reference["instruction"]:
        fail(
            f"Stage {stage.stage_name} binds to instruction {stage.engine_binding.instruction!r}, expected {reference['instruction']!r}"
        )
    else:
        semantic_error = _check_stage_semantics(view, stage, str(reference["semantic_kind"]))
        if semantic_error is not None:
            fail(semantic_error)

    obligation = StageObligation(
        obligation_id=f"stage:{stage.stage_name}",
        einsum_name=stage.stage_name,
        instruction_name=f"{stage.engine_binding.component}.{stage.engine_binding.instruction}",
        instruction_position=stage.stage_position,
        engine=stage.engine_binding.component,
        output_tensor=stage.output_tensor,
        input_tensors=list(stage.input_tensors),
        bound_ranks=list(stage.engine_binding.ranks),
        unbound_ranks=[],
        status=status,
        detail=detail,
    )

    result = LegalityStageResult(
        stage_name=f"stage:{stage.stage_name}",
        ok=(status == "pass"),
        error_type=None if status == "pass" else "HierarchicalStageError",
        message=None if status == "pass" else detail,
    )

    return obligation, result


def _check_stage_semantics(
    view: CandidateHierarchicalView,
    stage: StageView,
    semantic_kind: str,
) -> Optional[str]:
    if stage.output_tensor not in view.declaration:
        return f"Stage {stage.stage_name} output tensor is not declared"

    if len(stage.output_indices) != len(view.declaration[stage.output_tensor]):
        return (
            f"Stage {stage.stage_name} output indices {list(stage.output_indices)} do not match "
            f"declaration rank count {len(view.declaration[stage.output_tensor])}"
        )

    if semantic_kind == "rename_identity":
        return _check_rename_identity(view, stage)
    if semantic_kind == "unary_identity":
        return _check_unary_identity(view, stage)
    if semantic_kind == "reduce_drop_one":
        return _check_reduce_drop_one(view, stage)
    if semantic_kind == "expand_one_fresh_index":
        return _check_expand_one_fresh_index(view, stage)
    if semantic_kind == "init_zero":
        return _check_init_zero(stage)
    if semantic_kind == "recurrent_binary_add":
        return _check_recurrent_binary_add(view, stage)
    if semantic_kind == "read_terminal_slice":
        return _check_read_terminal_slice(view, stage)

    return f"Unsupported semantic kind for restricted checker: {semantic_kind!r}"


def _check_rename_identity(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Restricted checker expects exactly one input tensor for stage {stage.stage_name}"
    input_tensor = stage.input_tensors[0]
    input_indices = stage.input_indices[0]
    if input_tensor not in view.declaration:
        return f"Stage {stage.stage_name} input tensor {input_tensor!r} is not declared"
    if len(input_indices) != len(view.declaration[input_tensor]):
        return (
            f"Stage {stage.stage_name} input indices {list(input_indices)} do not match "
            f"declaration rank count of {input_tensor}"
        )
    if tuple(stage.output_indices) != tuple(input_indices):
        return (
            f"Stage {stage.stage_name} expects identical input/output index order, "
            f"but found output {list(stage.output_indices)} vs input {list(input_indices)}"
        )
    return None


def _check_unary_identity(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    return _check_rename_identity(view, stage)


def _check_reduce_drop_one(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Restricted checker expects exactly one input tensor for stage {stage.stage_name}"
    input_tensor = stage.input_tensors[0]
    input_indices = stage.input_indices[0]
    if input_tensor not in view.declaration:
        return f"Stage {stage.stage_name} input tensor {input_tensor!r} is not declared"
    if len(input_indices) != len(view.declaration[input_tensor]):
        return (
            f"Stage {stage.stage_name} input indices {list(input_indices)} do not match "
            f"declaration rank count of {input_tensor}"
        )
    if len(input_indices) != len(stage.output_indices) + 1:
        return (
            f"Reduction stage {stage.stage_name} must drop exactly one index from the input, "
            f"but found input {list(input_indices)} and output {list(stage.output_indices)}"
        )
    expected_output = tuple(index for index in input_indices if index in stage.output_indices)
    if tuple(stage.output_indices) != expected_output:
        return (
            f"Reduction stage {stage.stage_name} output indices {list(stage.output_indices)} are not the "
            f"ordered non-reduced subset of input indices {list(input_indices)}"
        )
    return None


def _check_expand_one_fresh_index(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Restricted checker expects exactly one input tensor for stage {stage.stage_name}"
    input_tensor = stage.input_tensors[0]
    input_indices = stage.input_indices[0]
    if input_tensor not in view.declaration:
        return f"Stage {stage.stage_name} input tensor {input_tensor!r} is not declared"
    if len(stage.output_indices) != len(input_indices) + 1:
        return (
            f"Stage {stage.stage_name} must add exactly one fresh index, but found output "
            f"{list(stage.output_indices)} vs input {list(input_indices)}"
        )
    extras = [index for index in stage.output_indices if index not in input_indices]
    if len(extras) != 1:
        return (
            f"Stage {stage.stage_name} should introduce exactly one fresh output index, but found {extras}"
        )
    filtered_output = tuple(index for index in stage.output_indices if index in input_indices)
    if filtered_output != tuple(input_indices):
        return (
            f"Stage {stage.stage_name} must preserve the relative order of input indices when expanding, "
            f"but found output {list(stage.output_indices)} vs input {list(input_indices)}"
        )
    return None


def _check_init_zero(stage: StageView) -> Optional[str]:
    if stage.input_tensors or stage.input_indices:
        return f"Init stage {stage.stage_name} must not read any input tensors"
    if stage.init_value != "0":
        return f"Init stage {stage.stage_name} must initialize from literal 0"
    if not stage.output_indices:
        return f"Init stage {stage.stage_name} must write a concrete output slice"
    _, offset, symbolic = parse_affine_token(stage.output_indices[-1])
    if symbolic is not None or offset != 0:
        return (
            f"Init stage {stage.stage_name} must write the base recurrence slice, but found "
            f"terminal index {stage.output_indices[-1]!r}"
        )
    return None


def _check_recurrent_binary_add(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 2 or len(stage.input_indices) != 2:
        return f"Recurrent binary stage {stage.stage_name} must read exactly two input tensors"
    if stage.input_tensors[0] != stage.output_tensor:
        return (
            f"Recurrent binary stage {stage.stage_name} must read its prior state from {stage.output_tensor}, "
            f"but found {stage.input_tensors[0]}"
        )
    prev_state_indices = stage.input_indices[0]
    update_indices = stage.input_indices[1]
    if len(prev_state_indices) != len(stage.output_indices):
        return (
            f"Recurrent binary stage {stage.stage_name} prior-state indices must match output rank count"
        )
    if len(update_indices) != len(stage.output_indices):
        return (
            f"Recurrent binary stage {stage.stage_name} update tensor indices must match output rank count"
        )

    for output_index, state_index in zip(stage.output_indices[:-1], prev_state_indices[:-1]):
        if output_index != state_index:
            return (
                f"Recurrent binary stage {stage.stage_name} must preserve non-recurrence indices between "
                f"output and prior state"
            )
                                                                                                         
    if tuple(update_indices[:-1]) != tuple(stage.output_indices[:-1]):
        return (
            f"Recurrent binary stage {stage.stage_name} update input must agree on leading indices with the output"
        )

    out_base, out_offset, out_symbolic = parse_affine_token(stage.output_indices[-1])
    prev_base, prev_offset, prev_symbolic = parse_affine_token(prev_state_indices[-1])
    upd_base, upd_offset, upd_symbolic = parse_affine_token(update_indices[-1])

    if out_symbolic is not None or prev_symbolic is not None or upd_symbolic is not None:
        return (
            f"Recurrent binary stage {stage.stage_name} uses unsupported symbolic recurrence indices"
        )
    if out_base is None or prev_base is None or upd_base is None:
        return f"Recurrent binary stage {stage.stage_name} must use affine recurrence indices"
    if out_base != prev_base or prev_base != upd_base:
        return (
            f"Recurrent binary stage {stage.stage_name} must use a common recurrence loop variable across state and update"
        )
    if out_offset != 1 or prev_offset != 0 or upd_offset != 0:
        return (
            f"Recurrent binary stage {stage.stage_name} must write p+1 from p+0 and update[p], but found "
            f"output {stage.output_indices[-1]!r}, state {prev_state_indices[-1]!r}, update {update_indices[-1]!r}"
        )
    return None


def _check_read_terminal_slice(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Restricted checker expects exactly one input tensor for stage {stage.stage_name}"
    input_tensor = stage.input_tensors[0]
    input_indices = stage.input_indices[0]
    if input_tensor not in view.declaration:
        return f"Stage {stage.stage_name} input tensor {input_tensor!r} is not declared"
    if len(input_indices) != len(stage.output_indices) + 1:
        return (
            f"Readout stage {stage.stage_name} must drop exactly one terminal slice index, but found output "
            f"{list(stage.output_indices)} vs input {list(input_indices)}"
        )
    if tuple(input_indices[:-1]) != tuple(stage.output_indices):
        return (
            f"Readout stage {stage.stage_name} must preserve leading indices when reading a terminal slice"
        )
    base, offset, symbolic = parse_affine_token(input_indices[-1])
    if base is not None or offset is not None or symbolic is None:
        return (
            f"Readout stage {stage.stage_name} must read a symbolic terminal slice like P, but found {input_indices[-1]!r}"
        )
    return None


                                                                             
                                             
                                                                             

def _check_attention_stage(
    view: CandidateHierarchicalView,
    stage: StageView,
) -> Tuple[StageObligation, LegalityStageResult]:
    status = "pass"
    detail = "Attention-family stage has locally well-formed tensor, index, and engine structure"

    error = _check_generic_stage_semantics(view, stage)
    if error is not None:
        status = "fail"
        detail = error

    obligation = StageObligation(
        obligation_id=f"stage:{stage.stage_name}",
        einsum_name=stage.stage_name,
        instruction_name=f"{stage.engine_binding.component}.{stage.engine_binding.instruction}",
        instruction_position=stage.stage_position,
        engine=stage.engine_binding.component,
        output_tensor=stage.output_tensor,
        input_tensors=list(stage.input_tensors),
        bound_ranks=list(stage.engine_binding.ranks),
        unbound_ranks=[],
        status=status,
        detail=detail,
    )

    result = LegalityStageResult(
        stage_name=f"stage:{stage.stage_name}",
        ok=(status == "pass"),
        error_type=None if status == "pass" else "HierarchicalStageError",
        message=None if status == "pass" else detail,
    )
    return obligation, result


def _check_generic_stage_semantics(
    view: CandidateHierarchicalView,
    stage: StageView,
) -> Optional[str]:
    if stage.output_tensor not in view.declaration:
        return f"Stage {stage.stage_name} output tensor {stage.output_tensor!r} is not declared"
    if len(stage.output_indices) != len(view.declaration[stage.output_tensor]):
        return (
            f"Stage {stage.stage_name} output indices {list(stage.output_indices)} do not match "
            f"declaration rank count {len(view.declaration[stage.output_tensor])}"
        )
    if not stage.engine_binding.component or not stage.engine_binding.instruction:
        return f"Stage {stage.stage_name} is missing an engine component/instruction binding"

    if stage.op_kind == "init":
        return _check_generic_init(stage)
    if stage.op_kind == "rename":
        return _check_generic_rename(view, stage)
    if stage.op_kind == "unary":
        return _check_generic_unary(view, stage)
    if stage.op_kind == "reduce":
        return _check_generic_reduce(view, stage)
    if stage.op_kind == "binary":
        return _check_generic_binary(view, stage, allow_reduction_indices=False)
    if stage.op_kind == "binary_reduce":
        return _check_generic_binary(view, stage, allow_reduction_indices=True)

    return f"Unsupported stage op_kind for attention-family checker: {stage.op_kind!r}"


def _check_generic_init(stage: StageView) -> Optional[str]:
    if stage.input_tensors or stage.input_indices:
        return f"Init stage {stage.stage_name} must not read input tensors"
    if stage.init_value is None:
        return f"Init stage {stage.stage_name} must have a literal initial value"
    if not stage.output_indices:
        return f"Init stage {stage.stage_name} must write at least one output index"
    return None


def _check_generic_rename(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Rename stage {stage.stage_name} must read exactly one tensor"
    return _check_index_coverage(
        view,
        stage,
        allow_extra_output_indices=True,
        allow_extra_input_indices=True,
    )


def _check_generic_unary(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Unary stage {stage.stage_name} must read exactly one tensor"
    if stage.unary_operator is None:
        return f"Unary stage {stage.stage_name} is missing its unary operator"
    return _check_index_coverage(
        view,
        stage,
        allow_extra_output_indices=False,
        allow_extra_input_indices=False,
    )


def _check_generic_reduce(view: CandidateHierarchicalView, stage: StageView) -> Optional[str]:
    if len(stage.input_tensors) != 1 or len(stage.input_indices) != 1:
        return f"Reduce stage {stage.stage_name} must read exactly one tensor"
    if stage.reduction_operator is None:
        return f"Reduce stage {stage.stage_name} is missing its reduction operator"
    return _check_index_coverage(
        view,
        stage,
        allow_extra_output_indices=False,
        allow_extra_input_indices=True,
    )


def _check_generic_binary(
    view: CandidateHierarchicalView,
    stage: StageView,
    *,
    allow_reduction_indices: bool,
) -> Optional[str]:
    if len(stage.input_tensors) < 1 or len(stage.input_indices) < 1:
        return f"Binary stage {stage.stage_name} must read at least one tensor operand"
    if len(stage.input_tensors) > 2:
        return f"Binary stage {stage.stage_name} reads more than two tensor operands"
    if stage.binary_operator is None:
        return f"Binary stage {stage.stage_name} is missing its map-side binary operator"
    if allow_reduction_indices and stage.reduction_operator is None:
        return f"Binary-reduce stage {stage.stage_name} is missing its reduction operator"
    return _check_index_coverage(
        view,
        stage,
        allow_extra_output_indices=False,
        allow_extra_input_indices=allow_reduction_indices,
    )


def _check_index_coverage(
    view: CandidateHierarchicalView,
    stage: StageView,
    *,
    allow_extra_output_indices: bool,
    allow_extra_input_indices: bool,
) -> Optional[str]:
    input_bases: List[str] = []
    for input_tensor, indices in zip(stage.input_tensors, stage.input_indices):
        if input_tensor not in view.declaration:
            return f"Stage {stage.stage_name} input tensor {input_tensor!r} is not declared"
        if len(indices) != len(view.declaration[input_tensor]):
            return (
                f"Stage {stage.stage_name} input indices {list(indices)} do not match "
                f"declaration rank count of {input_tensor}"
            )
        input_bases.extend(_semantic_index_base(index) for index in indices)

    output_bases = [_semantic_index_base(index) for index in stage.output_indices]
    input_set = set(input_bases)
    output_set = set(output_bases)

    missing_from_inputs = sorted(base for base in output_set if base not in input_set and not _is_terminal_symbol(base))
    if missing_from_inputs and not allow_extra_output_indices:
        return (
            f"Stage {stage.stage_name} output indices {missing_from_inputs} are not supplied by its tensor inputs"
        )

    extra_inputs = sorted(base for base in input_set if base not in output_set and not _is_terminal_symbol(base))
    if extra_inputs and not allow_extra_input_indices:
        return (
            f"Stage {stage.stage_name} input indices {extra_inputs} are not preserved in the output"
        )

    return None


def _semantic_index_base(index: str) -> str:
    base, literal, symbolic = parse_affine_token(index)
    if base is not None:
        return base.lower()
    if symbolic is not None:
        return symbolic.lower()
    if literal is not None:
        return str(literal)
    return str(index).strip().lower()


def _is_terminal_symbol(base: str) -> bool:
                                                                                 
                                                                                
                                                                              
                                                                                 
                                                                                
                          
    if not base:
        return False
    if base.lstrip("-").isdigit():
        return True
    return any(ch.isdigit() for ch in base) and base[0].isalpha()
