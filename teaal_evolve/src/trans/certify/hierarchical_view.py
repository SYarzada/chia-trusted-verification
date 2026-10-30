

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .candidate import Candidate
from .schema_utils import component_name, component_names


@dataclass(frozen=True)
class TensorFormatView:
    tensor_name: str
    bit_width: Optional[int]
    data_type: Optional[str]
    rank_orders: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class TensorBindingView:
    tensor_name: str
    einsum_name: str
    index: int
    component: Optional[str]
    allowed_components: Tuple[str, ...]
    rank_groups: Dict[str, Tuple[str, ...]]
    constructor: Optional[str] = None

    @property
    def current_component(self) -> Optional[str]:
        return self.component


@dataclass(frozen=True)
class EngineBindingView:
    einsum_name: str
    component: str
    instruction: str
    ranks: Tuple[str, ...]


@dataclass(frozen=True)
class ArchitectureTemplateView:
    

    engine_component: str
    instruction_name: str
    template_name: str
    template_stage_name: str
    op_kind: str
    engine_ranks: Tuple[str, ...]
    loop_order: Tuple[str, ...]
    space_ranks: Tuple[str, ...]
    time_ranks: Tuple[str, ...]
    input_components: Tuple[Tuple[str, ...], ...]
    output_components: Tuple[str, ...]
    input_count: int
    tensor_components: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class StageView:
    stage_name: str
    stage_position: int
    raw_expression: str
    op_kind: str
    unary_operator: Optional[str]
    reduction_operator: Optional[str]
    binary_operator: Optional[str]
    init_value: Optional[str]
    output_tensor: str
    output_indices: Tuple[str, ...]
    input_tensors: Tuple[str, ...]
    input_indices: Tuple[Tuple[str, ...], ...]
    declaration_ranks: Tuple[str, ...]
    loop_order: Tuple[str, ...]
    space_ranks: Tuple[str, ...]
    time_ranks: Tuple[str, ...]
    engine_binding: EngineBindingView
    output_binding: TensorBindingView
    input_bindings: Tuple[TensorBindingView, ...]
    output_format: TensorFormatView
    input_formats: Tuple[TensorFormatView, ...]
    rank_roots: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class InterfaceView:
    obligation_id: str
    tensor_name: str
    producer_stage_name: str
    consumer_stage_name: str
    producer_stage_position: int
    consumer_stage_position: int
    producer_binding: TensorBindingView
    consumer_binding: TensorBindingView
    producer_format_order: Tuple[str, ...]
    consumer_format_order: Tuple[str, ...]
    producer_normalized_format_order: Tuple[str, ...]
    consumer_normalized_format_order: Tuple[str, ...]
    producer_bit_width: Optional[int]
    consumer_bit_width: Optional[int]
    producer_data_type: Optional[str]
    consumer_data_type: Optional[str]


@dataclass(frozen=True)
class CandidateHierarchicalView:
    candidate_label: str
    kernel_name: str
    declaration: Dict[str, Tuple[str, ...]]
    shapes: Dict[str, int]
    cascade_outputs: Tuple[str, ...]
    stage_views: Tuple[StageView, ...]
    interface_views: Tuple[InterfaceView, ...]
    architecture_templates: Dict[Tuple[str, str], Tuple[ArchitectureTemplateView, ...]]
    fusion_rank_groups: Tuple[Tuple[Tuple[str, str], ...], ...]


@dataclass
class HierarchicalViewError(Exception):
    stage_name: str
    message: str

    def __str__(self) -> str:
        return f"{self.stage_name}: {self.message}"


_TENSOR_REF_RE = re.compile(r"^(?P<tensor>[A-Za-z_][A-Za-z0-9_]*)\[(?P<idx>[^\]]+)\]$")
_OP_RE = re.compile(r"^(?P<tag>[MV])\s+(?P<kind>unary|binary)\[(?P<op>[^\]]+)\]$")
_OP_TOKEN_RE = re.compile(r"(?P<tag>[MV])\s+(?P<kind>unary|binary)\[(?P<op>[^\]]+)\]")
_AFFINE_TOKEN_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\+\s*(?P<offset>-?\d+))?$")
_INT_TOKEN_RE = re.compile(r"^-?\d+$")

_SUPPORTED_KERNEL_NAMES = {"unary", "unary_reduce_one_tile", "vector_scalar_overlap", "attn", "matmul", "mamba"}


def build_candidate_hierarchical_view(candidate: Candidate) -> CandidateHierarchicalView:
    spec = candidate.spec_dict

    einsum = _expect_mapping(spec, "einsum")
    mapping = _expect_mapping(spec, "mapping")
    format_spec = _expect_mapping(spec, "format")
    binding = _expect_mapping(spec, "binding")
    options = _expect_mapping(spec, "options")
    architecture = _expect_mapping(spec, "architecture")

    kernel_name = str(options.get("kernel-name", ""))
    if kernel_name not in _SUPPORTED_KERNEL_NAMES:
        raise HierarchicalViewError(
            "options",
            f"Restricted true hierarchical checker does not support kernel-name {kernel_name!r}",
        )

    declaration_raw = _expect_mapping(einsum, "declaration", stage_name="einsum")
    declaration = {
        tensor_name: tuple(_expect_string_list(ranks, f"einsum.declaration[{tensor_name}]"))
        for tensor_name, ranks in declaration_raw.items()
    }

    expressions = _expect_list(einsum, "expressions", stage_name="einsum")
    if not expressions:
        raise HierarchicalViewError("einsum", "At least one expression is required")

    shapes_raw = einsum.get("shape", {})
    if shapes_raw is None:
        shapes_raw = {}
    if not isinstance(shapes_raw, dict):
        raise HierarchicalViewError("einsum", "einsum.shape must be a mapping when present")
    shapes = {str(rank): int(value) for rank, value in shapes_raw.items()}

    loop_order = _expect_mapping(mapping, "loop-order", stage_name="mapping")
    spacetime = _expect_mapping(mapping, "spacetime", stage_name="mapping")

    tensor_formats = _build_tensor_formats(format_spec)
    stage_rank_roots = _build_stage_rank_roots(mapping)
    binding_engines = _build_engine_bindings(binding)
    binding_tensors = _build_tensor_bindings(binding)
    architecture_templates = _collect_architecture_templates(architecture)
    cascade_outputs = tuple(
        _expect_string_list(options.get("cascade-outputs", []), "options.cascade-outputs")
    )

    stage_views: List[StageView] = []

    for stage_position, raw_expression in enumerate(expressions):
        stage_name_override, expression = _normalize_expression_entry(
            raw_expression,
            stage_position,
        )

        parsed = _parse_expression(expression, stage_position)
        stage_name = stage_name_override or str(parsed["stage_name"])
        output_tensor = str(parsed["output_tensor"])

        if output_tensor not in declaration:
            raise HierarchicalViewError(
                f"stage:{stage_name}",
                f"Output tensor {output_tensor!r} is not declared in einsum.declaration",
            )

        stage_loop_order = tuple(
            _expect_string_list(
                loop_order.get(stage_name),
                f"mapping.loop-order[{stage_name}]",
                stage_name="mapping",
            )
        )
        stage_spacetime = _expect_mapping(spacetime, stage_name, stage_name="mapping")
        stage_space = tuple(
            _expect_string_list(
                stage_spacetime.get("space", []), f"mapping.spacetime[{stage_name}].space"
            )
        )
        stage_time = tuple(
            _expect_string_list(
                stage_spacetime.get("time", []), f"mapping.spacetime[{stage_name}].time"
            )
        )

        engine_binding = binding_engines.get(stage_name)
        if engine_binding is None:
            raise HierarchicalViewError(
                "binding",
                f"No binding.engine entry found for stage {stage_name}",
            )

        stage_tensor_bindings = binding_tensors.get(stage_name, {})
        output_binding = stage_tensor_bindings.get(output_tensor)
        if output_binding is None:
            raise HierarchicalViewError(
                "binding",
                f"No output binding.tensor entry found for stage {stage_name} output tensor {output_tensor}",
            )

        input_tensors = tuple(str(name) for name in parsed["input_tensors"])
        input_indices = tuple(tuple(group) for group in parsed["input_indices"])
        input_bindings: List[TensorBindingView] = []
        input_formats: List[TensorFormatView] = []

        for input_tensor in input_tensors:
            tensor_binding = stage_tensor_bindings.get(input_tensor)
            if tensor_binding is None:
                raise HierarchicalViewError(
                    "binding",
                    f"Stage {stage_name} is missing a binding.tensor entry for input tensor {input_tensor}",
                )
            input_bindings.append(tensor_binding)

            tensor_format = tensor_formats.get(input_tensor)
            if tensor_format is None:
                raise HierarchicalViewError(
                    "format",
                    f"Missing format entry for tensor {input_tensor}",
                )
            input_formats.append(tensor_format)

        output_format = tensor_formats.get(output_tensor)
        if output_format is None:
            raise HierarchicalViewError(
                "format",
                f"Missing format entry for tensor {output_tensor}",
            )

        stage_views.append(
            StageView(
                stage_name=stage_name,
                stage_position=stage_position,
                raw_expression=expression,
                op_kind=str(parsed["op_kind"]),
                unary_operator=parsed["unary_operator"],
                reduction_operator=parsed["reduction_operator"],
                binary_operator=parsed["binary_operator"],
                init_value=parsed["init_value"],
                output_tensor=output_tensor,
                output_indices=tuple(parsed["output_indices"]),
                input_tensors=input_tensors,
                input_indices=input_indices,
                declaration_ranks=declaration[output_tensor],
                loop_order=stage_loop_order,
                space_ranks=stage_space,
                time_ranks=stage_time,
                engine_binding=engine_binding,
                output_binding=output_binding,
                input_bindings=tuple(input_bindings),
                output_format=output_format,
                input_formats=tuple(input_formats),
                rank_roots=stage_rank_roots.get(stage_name, {}),
            )
        )

    fusion_rank_groups = _collect_fusion_rank_groups(
        mapping,
        stage_views=stage_views,
    )
    _validate_engine_binding_structure(
        binding,
        stage_views=stage_views,
        fusion_rank_groups=fusion_rank_groups,
    )
    _validate_pipelining_structure(
        binding,
        mapping,
        stage_views=stage_views,
    )
    interface_views = _build_interface_views(stage_views)

    return CandidateHierarchicalView(
        candidate_label=candidate.label,
        kernel_name=kernel_name,
        declaration=declaration,
        shapes=shapes,
        cascade_outputs=cascade_outputs,
        stage_views=tuple(stage_views),
        interface_views=tuple(interface_views),
        architecture_templates=architecture_templates,
        fusion_rank_groups=fusion_rank_groups,
    )



def _normalize_expression_entry(
    raw_expression: object,
    stage_position: int,
) -> tuple[Optional[str], str]:
    if isinstance(raw_expression, str):
        return None, raw_expression

    if isinstance(raw_expression, dict):
        if len(raw_expression) != 1:
            raise HierarchicalViewError(
                "einsum",
                f"Expression at position {stage_position} must be a string or a single-entry mapping",
            )

        stage_name, expression = next(iter(raw_expression.items()))
        if not isinstance(stage_name, str):
            raise HierarchicalViewError(
                "einsum",
                f"Expression label at position {stage_position} must be a string",
            )
        if not isinstance(expression, str):
            raise HierarchicalViewError(
                "einsum",
                f"Labeled expression at position {stage_position} must map to a string expression",
            )
        return stage_name, expression

    raise HierarchicalViewError(
        "einsum",
        f"Expression at position {stage_position} must be a string or a single-entry mapping",
    )


def _parse_expression(expression: str, stage_position: int) -> Dict[str, object]:
    label: Optional[str] = None
    body = expression.strip()
    if ":" in body:
        prefix, remainder = body.split(":", 1)
        if "=" in remainder:
            label = prefix.strip()
            body = remainder.strip()

    if "=" not in body:
        raise HierarchicalViewError(
            "einsum",
            f"Expression at position {stage_position} is missing '=': {expression}",
        )

    lhs_text, rhs_text = [piece.strip() for piece in body.split("=", 1)]
    lhs_tensor, lhs_idx = _parse_tensor_ref(lhs_text, stage_position)
    stage_name = label or lhs_tensor

    unary_operator: Optional[str] = None
    reduction_operator: Optional[str] = None
    binary_operator: Optional[str] = None
    init_value: Optional[str] = None
    input_tensors: List[str] = []
    input_indices: List[Tuple[str, ...]] = []

    if ";;" not in rhs_text:
        if _is_literal_operand(rhs_text):
            op_kind = "init"
            init_value = rhs_text.strip()
        else:
            rhs_tensor, rhs_idx = _parse_tensor_ref(rhs_text, stage_position)
            op_kind = "rename"
            input_tensors = [rhs_tensor]
            input_indices = [rhs_idx]
    else:
        core_text, op_text = [piece.strip() for piece in rhs_text.split(";;", 1)]
        operators = _parse_operator_suffix(op_text, stage_position, expression)

        map_unary = [op for op in operators if op[0] == "M" and op[1] == "unary"]
        map_binary = [op for op in operators if op[0] == "M" and op[1] == "binary"]
        reductions = [op for op in operators if op[0] == "V" and op[1] == "binary"]

        if map_unary and not map_binary and not reductions and len(operators) == 1:
            rhs_tensor, rhs_idx = _parse_tensor_ref(core_text, stage_position)
            op_kind = "unary"
            unary_operator = map_unary[0][2]
            input_tensors = [rhs_tensor]
            input_indices = [rhs_idx]
        elif reductions and not map_unary and not map_binary and len(operators) == 1:
            rhs_tensor, rhs_idx = _parse_tensor_ref(core_text, stage_position)
            op_kind = "reduce"
            reduction_operator = reductions[0][2]
            input_tensors = [rhs_tensor]
            input_indices = [rhs_idx]
        elif map_binary:
            binary_operator = map_binary[0][2]
            reduction_operator = reductions[0][2] if reductions else None
            op_kind = "binary_reduce" if reductions else "binary"

            operand_tensors: List[str] = []
            operand_indices: List[Tuple[str, ...]] = []
            for operand_text in _split_operand_text(core_text):
                parsed_operand = _parse_optional_tensor_operand(operand_text, stage_position)
                if parsed_operand is None:
                    continue
                operand_tensor, operand_idx = parsed_operand
                operand_tensors.append(operand_tensor)
                operand_indices.append(operand_idx)

            if not operand_tensors:
                raise HierarchicalViewError(
                    "einsum",
                    f"Expression at position {stage_position} has no tensor operands: {expression}",
                )
            if len(operand_tensors) > 2:
                raise HierarchicalViewError(
                    "einsum",
                    f"Expression at position {stage_position} has too many tensor operands: {expression}",
                )
            input_tensors = operand_tensors
            input_indices = operand_indices
        else:
            raise HierarchicalViewError(
                "einsum",
                f"Expression at position {stage_position} is outside the supported kernel-family syntax: {expression}",
            )

    return {
        "stage_name": stage_name,
        "output_tensor": lhs_tensor,
        "input_tensors": input_tensors,
        "output_indices": lhs_idx,
        "input_indices": input_indices,
        "op_kind": op_kind,
        "unary_operator": unary_operator,
        "reduction_operator": reduction_operator,
        "binary_operator": binary_operator,
        "init_value": init_value,
    }


def _parse_operator_suffix(
    op_text: str,
    stage_position: int,
    expression: str,
) -> List[Tuple[str, str, str]]:
    matches = list(_OP_TOKEN_RE.finditer(op_text))
    if not matches:
        raise HierarchicalViewError(
            "einsum",
            f"Expression at position {stage_position} uses an unsupported operator suffix: {expression}",
        )

                                                                                
    consumed = "".join(match.group(0) for match in matches)
    normalized_consumed = "".join(consumed.split())
    normalized_original = "".join(op_text.split())
    if normalized_consumed != normalized_original:
        raise HierarchicalViewError(
            "einsum",
            f"Expression at position {stage_position} has unsupported operator text: {op_text!r}",
        )

    return [
        (match.group("tag"), match.group("kind"), match.group("op"))
        for match in matches
    ]


def _split_operand_text(core_text: str) -> List[str]:
    return [piece.strip() for piece in core_text.split(".") if piece.strip()]


def _parse_optional_tensor_operand(
    operand_text: str,
    stage_position: int,
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    if _is_literal_operand(operand_text):
        return None
    return _parse_tensor_ref(operand_text, stage_position)


def _is_literal_operand(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def _parse_tensor_ref(text: str, stage_position: int) -> Tuple[str, Tuple[str, ...]]:
    match = _TENSOR_REF_RE.match(text.strip())
    if match is None:
        raise HierarchicalViewError(
            "einsum",
            f"Expression at position {stage_position} uses an unsupported tensor reference: {text}",
        )
    return match.group("tensor"), tuple(_split_indices(match.group("idx")))


                                                            

def _build_stage_rank_roots(
    mapping: Dict[str, object],
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    

    raw_partitioning = mapping.get("partitioning", {})
    if not isinstance(raw_partitioning, dict):
        return {}

    by_stage: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for raw_stage_name, raw_directives in raw_partitioning.items():
        stage_name = str(raw_stage_name)
        if not isinstance(raw_directives, dict):
            continue

                                                                          
                                                                            
                               
        aliases: Dict[str, Tuple[str, ...]] = {}

        for raw_sources, raw_ops in raw_directives.items():
            sources = _parse_partitioning_sources(raw_sources)
            if not sources or not isinstance(raw_ops, list) or not raw_ops:
                continue

            op_names = [str(op).strip().lower() for op in raw_ops]

            if len(sources) > 1:
                if any(name.startswith("flatten") for name in op_names):
                    aliases["".join(sources)] = tuple(sources)
                continue

            source = sources[0]
            uniform_ops = [
                name for name in op_names if name.startswith("uniform_shape")
            ]
            total = len(uniform_ops)
            for index in range(total):
                upper = source + str(total - index)
                lower = source + str(total - index - 1)
                if index < total - 1:
                    lower += "I"

                                                                        
                                                                   
                aliases[upper] = (source,)
                aliases[lower] = (source,)

        memo: Dict[str, Tuple[str, ...]] = {}

        def resolve(rank: str, stack: Tuple[str, ...] = ()) -> Tuple[str, ...]:
            cached = memo.get(rank)
            if cached is not None:
                return cached

            parts = aliases.get(rank)
            if parts is None:
                return (rank,)

                                                                            
                                                                         
                                                                      
            if rank in stack:
                return (rank,)

            resolved = tuple(
                root
                for part in parts
                for root in resolve(part, stack + (rank,))
            )
            memo[rank] = resolved
            return resolved

        by_stage[stage_name] = {
            alias: resolve(alias)
            for alias in sorted(aliases)
        }

    return by_stage

def _parse_partitioning_sources(raw_sources: object) -> Tuple[str, ...]:
    text = str(raw_sources).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
        return tuple(piece.strip() for piece in text.split(",") if piece.strip())
    return (text,) if text else tuple()


def _normalize_format_order(
    rank_order: Sequence[str],
    rank_roots: Dict[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    

    normalized: List[str] = []
    for rank in rank_order:
        for root in rank_roots.get(rank, (rank,)):
            if not normalized or normalized[-1] != root:
                normalized.append(root)
    return tuple(normalized)


def _build_tensor_formats(format_spec: Dict[str, object]) -> Dict[str, TensorFormatView]:
    tensor_formats: Dict[str, TensorFormatView] = {}

    for tensor_name, raw in format_spec.items():
        if not isinstance(raw, dict):
            raise HierarchicalViewError("format", f"format[{tensor_name}] must be a mapping")

        rank_order_raw = _expect_mapping(raw, "rank-order", stage_name="format")
        rank_orders = {
            stage_name: tuple(
                _expect_string_list(order, f"format[{tensor_name}].rank-order[{stage_name}]")
            )
            for stage_name, order in rank_order_raw.items()
        }

        bit_width = raw.get("bit-width")
        if bit_width is None and "bit-widths" in raw:
            widths = raw.get("bit-widths")
            if isinstance(widths, list) and widths:
                bit_width = int(widths[0])
        if bit_width is not None:
            bit_width = int(bit_width)

        data_type = raw.get("data-type")
        if data_type is None and "data-types" in raw:
            values = raw.get("data-types")
            if isinstance(values, list) and values:
                data_type = str(values[0])
        if data_type is not None:
            data_type = str(data_type)

        tensor_formats[str(tensor_name)] = TensorFormatView(
            tensor_name=str(tensor_name),
            bit_width=bit_width,
            data_type=data_type,
            rank_orders=rank_orders,
        )

    return tensor_formats


def _build_engine_bindings(binding: Dict[str, object]) -> Dict[str, EngineBindingView]:
    engines = _expect_list(binding, "engine", stage_name="binding")
    by_stage: Dict[str, EngineBindingView] = {}

    for index, entry in enumerate(engines):
        if not isinstance(entry, dict):
            raise HierarchicalViewError("binding", "Each binding.engine entry must be a mapping")

        component = component_name(entry.get("component"))
        if component is None:
            raise HierarchicalViewError(
                "binding",
                f"binding.engine entry {index} is missing a component",
            )
        instruction = str(entry.get("instruction"))
        if "einsum" in entry:
            stage_name = str(entry.get("einsum"))
            if stage_name in by_stage:
                raise HierarchicalViewError(
                    "binding",
                    f"Multiple binding.engine entries found for stage {stage_name}",
                )
            by_stage[stage_name] = EngineBindingView(
                einsum_name=stage_name,
                component=component,
                instruction=instruction,
                ranks=tuple(
                    _expect_string_list(
                        entry.get("ranks", []), f"binding.engine[{stage_name}].ranks"
                    )
                ),
            )
            continue

        if "einsums" in entry:
            stage_names = _expect_string_list(entry.get("einsums", []), f"binding.engine[{index}].einsums")
            raw_ranks = entry.get("ranks", {})
            if isinstance(raw_ranks, dict):
                ranks_by_stage = {
                    stage_name: tuple(
                        _expect_string_list(
                            raw_ranks.get(stage_name, []),
                            f"binding.engine[{index}].ranks[{stage_name}]",
                        )
                    )
                    for stage_name in stage_names
                }
            else:
                shared_ranks = tuple(_expect_string_list(raw_ranks, f"binding.engine[{index}].ranks"))
                ranks_by_stage = {stage_name: shared_ranks for stage_name in stage_names}

            for stage_name in stage_names:
                if stage_name in by_stage:
                    raise HierarchicalViewError(
                        "binding",
                        f"Multiple binding.engine entries found for stage {stage_name}",
                    )
                by_stage[stage_name] = EngineBindingView(
                    einsum_name=stage_name,
                    component=component,
                    instruction=instruction,
                    ranks=ranks_by_stage[stage_name],
                )
            continue

        raise HierarchicalViewError(
            "binding",
            f"binding.engine entry {index} must specify either einsum or einsums",
        )

    return by_stage


def _build_tensor_bindings(binding: Dict[str, object]) -> Dict[str, Dict[str, TensorBindingView]]:
    tensors = _expect_list(binding, "tensor", stage_name="binding")
    by_stage: Dict[str, Dict[str, TensorBindingView]] = {}

    for index, entry in enumerate(tensors):
        if not isinstance(entry, dict):
            raise HierarchicalViewError("binding", "Each binding.tensor entry must be a mapping")

        stage_name = str(entry.get("einsum"))
        tensor_name = str(entry.get("name"))
        component = entry.get("component")
        components = entry.get("components")

        if components is not None:
            allowed_components = component_names(components)
        elif component is not None:
            name = component_name(component)
            allowed_components = tuple() if name is None else (name,)
        else:
            allowed_components = tuple()

        rank_groups_raw = _expect_mapping(entry, "ranks", stage_name="binding")
        rank_groups = {
            group_name: tuple(_expect_string_list(values, f"binding.tensor[{index}].ranks[{group_name}]"))
            for group_name, values in rank_groups_raw.items()
        }

        by_stage.setdefault(stage_name, {})[tensor_name] = TensorBindingView(
            tensor_name=tensor_name,
            einsum_name=stage_name,
            index=index,
            component=component_name(component),
            allowed_components=allowed_components,
            rank_groups=rank_groups,
            constructor=None if entry.get("constructor") is None else str(entry.get("constructor")),
        )

    return by_stage


def _collect_fusion_rank_groups(
    mapping: Dict[str, object],
    *,
    stage_views: Sequence[StageView],
) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    

    raw_fusion = mapping.get("fusion", [])
    if raw_fusion is None:
        return tuple()
    if not isinstance(raw_fusion, list):
        raise HierarchicalViewError(
            "mapping",
            "mapping.fusion must be a list when present",
        )
    if not raw_fusion:
        return tuple()

    stage_by_name = {stage.stage_name: stage for stage in stage_views}
    expected_stage_names = set(stage_by_name)
    given_stage_names: List[str] = []
    prepared_entries: List[Tuple[int, Tuple[str, ...], object]] = []

                                                                             
                                                                      
    for fusion_index, fusion_entry in enumerate(raw_fusion):
        if not isinstance(fusion_entry, dict):
            raise HierarchicalViewError(
                "mapping",
                f"mapping.fusion[{fusion_index}] must be a mapping",
            )

        stage_names = tuple(
            _expect_string_list(
                fusion_entry.get("einsums", []),
                f"mapping.fusion[{fusion_index}].einsums",
                stage_name="mapping",
            )
        )
        given_stage_names.extend(stage_names)
        prepared_entries.append(
            (fusion_index, stage_names, fusion_entry.get("ranks", []))
        )

    duplicate_stage = _first_duplicate(given_stage_names)
    if duplicate_stage is not None:
        raise HierarchicalViewError(
            "mapping",
            f"Duplicate stage {duplicate_stage!r} in mapping.fusion",
        )

    given_stage_set = set(given_stage_names)
    unknown_stages = sorted(given_stage_set - expected_stage_names)
    if unknown_stages:
        raise HierarchicalViewError(
            "mapping",
            f"mapping.fusion references unknown stage {unknown_stages[0]!r}",
        )

    missing_stages = sorted(expected_stage_names - given_stage_set)
    if missing_stages:
        raise HierarchicalViewError(
            "mapping",
            f"mapping.fusion is missing stage {missing_stages[0]!r}",
        )

    seen_scoped_ranks: set[Tuple[str, str]] = set()
    groups: List[Tuple[Tuple[str, str], ...]] = []

    for fusion_index, stage_names, raw_rank_groups in prepared_entries:
        stage_group = set(stage_names)
        if not isinstance(raw_rank_groups, list):
            raise HierarchicalViewError(
                "mapping",
                f"mapping.fusion[{fusion_index}].ranks must be a list",
            )

        for rank_group_index, raw_group in enumerate(raw_rank_groups):
            if isinstance(raw_group, dict):
                if len(raw_group) != 1:
                    raise HierarchicalViewError(
                        "mapping",
                        "Each named fusion-rank group must contain exactly one entry",
                    )
                rank_refs_raw = next(iter(raw_group.values()))
            elif isinstance(raw_group, list):
                rank_refs_raw = raw_group
            else:
                raise HierarchicalViewError(
                    "mapping",
                    (
                        f"mapping.fusion[{fusion_index}].ranks[{rank_group_index}] "
                        "must be a list or a single-entry mapping"
                    ),
                )

            rank_refs = _expect_string_list(
                rank_refs_raw,
                (
                    f"mapping.fusion[{fusion_index}].ranks"
                    f"[{rank_group_index}]"
                ),
                stage_name="mapping",
            )
            parsed_group: List[Tuple[str, str]] = []

            for raw_ref in rank_refs:
                if raw_ref.count(".") != 1:
                    raise HierarchicalViewError(
                        "mapping",
                        f"Fusion rank reference {raw_ref!r} must use 'stage.rank' syntax",
                    )
                stage_name, rank_name = raw_ref.split(".", 1)

                stage = stage_by_name.get(stage_name)
                if stage is None:
                    raise HierarchicalViewError(
                        "mapping",
                        f"Fusion references unknown stage {stage_name!r}",
                    )
                if stage_name not in stage_group:
                    raise HierarchicalViewError(
                        "mapping",
                        (
                            f"Fusion rank {raw_ref} does not belong to its enclosing "
                            f"stage group {list(stage_names)}"
                        ),
                    )
                if rank_name not in set(stage.loop_order):
                    raise HierarchicalViewError(
                        "mapping",
                        f"Fusion references unknown mapped rank {raw_ref}",
                    )

                scoped_rank = (stage_name, rank_name)
                if scoped_rank in seen_scoped_ranks:
                    raise HierarchicalViewError(
                        "mapping",
                        f"Duplicate fused rank {raw_ref}",
                    )
                seen_scoped_ranks.add(scoped_rank)
                parsed_group.append(scoped_rank)

            if parsed_group:
                groups.append(tuple(parsed_group))

    return tuple(groups)



def _validate_engine_binding_structure(
    binding: Dict[str, object],
    *,
    stage_views: Sequence[StageView],
    fusion_rank_groups: Tuple[Tuple[Tuple[str, str], ...], ...],
) -> None:
    

    raw_engines = binding.get("engine", [])
    if not isinstance(raw_engines, list):
        raise HierarchicalViewError(
            "binding",
            "binding.engine must be a list",
        )

    stage_by_name = {stage.stage_name: stage for stage in stage_views}

                                                                               
                                                                             
                                                                              
                                                                       
    fusion_identity_by_rank: Dict[
        Tuple[str, str], frozenset[Tuple[str, str]]
    ] = {}
    for group in fusion_rank_groups:
        identity = frozenset(group)
        for scoped_rank in group:
            fusion_identity_by_rank[scoped_rank] = identity

    def fusion_identity(
        scoped_rank: Tuple[str, str],
    ) -> frozenset[Tuple[str, str]]:
        return fusion_identity_by_rank.get(
            scoped_rank,
            frozenset((scoped_rank,)),
        )

    for engine_index, raw_engine in enumerate(raw_engines):
        if not isinstance(raw_engine, dict):
            raise HierarchicalViewError(
                "binding",
                f"binding.engine[{engine_index}] must be a mapping",
            )

        if "einsum" in raw_engine:
            stage_names = (str(raw_engine.get("einsum")),)
        elif "einsums" in raw_engine:
            stage_names = tuple(
                _expect_string_list(
                    raw_engine.get("einsums", []),
                    f"binding.engine[{engine_index}].einsums",
                    stage_name="binding",
                )
            )
        else:
                                                                           
            continue

        raw_ranks = raw_engine.get("ranks", [])
        if isinstance(raw_ranks, dict):
            ranks_by_stage = {
                stage_name: tuple(
                    _expect_string_list(
                        raw_ranks.get(stage_name, []),
                        (
                            f"binding.engine[{engine_index}]"
                            f".ranks[{stage_name}]"
                        ),
                        stage_name="binding",
                    )
                )
                for stage_name in stage_names
            }
        else:
            shared_ranks = tuple(
                _expect_string_list(
                    raw_ranks,
                    f"binding.engine[{engine_index}].ranks",
                    stage_name="binding",
                )
            )
            ranks_by_stage = {
                stage_name: shared_ranks for stage_name in stage_names
            }

        unbound_by_stage: Dict[str, Tuple[str, ...]] = {}
        for stage_name in stage_names:
            stage = stage_by_name.get(stage_name)
            if stage is None:
                                                                           
                                                                          
                continue

            bound_ranks = ranks_by_stage[stage_name]
            if (
                bound_ranks
                and tuple(stage.loop_order[-len(bound_ranks):]) != bound_ranks
            ):
                raise HierarchicalViewError(
                    "binding",
                    (
                        f"Stage {stage_name} engine-bound ranks "
                        f"{list(bound_ranks)} must be the exact innermost "
                        "suffix of its loop order; "
                        f"found {list(stage.loop_order)}"
                    ),
                )

                                                                             
                                                        
            unbound_by_stage[stage_name] = (
                tuple(stage.loop_order[:-len(bound_ranks)])
                if bound_ranks
                else tuple()
            )

        present_stage_names = tuple(
            stage_name
            for stage_name in stage_names
            if stage_name in unbound_by_stage
        )
        if len(present_stage_names) > 1:
            counts = [
                len(unbound_by_stage[name]) for name in present_stage_names
            ]
            if len(set(counts)) != 1:
                raise HierarchicalViewError(
                    "binding",
                    (
                        f"Stages {list(present_stage_names)} bound to one "
                        "instruction must have the same number of unbound "
                        f"ranks; found {counts}"
                    ),
                )

            for position in range(counts[0]):
                scoped_ranks = tuple(
                    (stage_name, unbound_by_stage[stage_name][position])
                    for stage_name in present_stage_names
                )
                identities = [
                    fusion_identity(rank) for rank in scoped_ranks
                ]
                if len(set(identities)) != 1:
                    displayed = [
                        f"{stage}.{rank}" for stage, rank in scoped_ranks
                    ]
                    raise HierarchicalViewError(
                        "binding",
                        (
                            f"Stages {list(present_stage_names)} bound to one "
                            "instruction require corresponding unbound ranks "
                            f"{displayed} to be fused"
                        ),
                    )

        instruction_stage_set = set(stage_names)
        for stage_name in present_stage_names:
            for rank_name in ranks_by_stage[stage_name]:
                identity = fusion_identity((stage_name, rank_name))
                foreign = sorted(
                    (other_stage, other_rank)
                    for other_stage, other_rank in identity
                    if other_stage not in instruction_stage_set
                )
                if foreign:
                    other_stage, other_rank = foreign[0]
                    raise HierarchicalViewError(
                        "binding",
                        (
                            f"Bound rank {stage_name}.{rank_name} is fused "
                            f"with {other_stage}.{other_rank}, but the stages "
                            "are assigned to different instructions"
                        ),
                    )


def _validate_pipelining_structure(
    binding: Dict[str, object],
    mapping: Dict[str, object],
    *,
    stage_views: Sequence[StageView],
) -> None:
    

    raw_pipelining = binding.get("pipelining", [])
    if raw_pipelining is None:
        return
    if not isinstance(raw_pipelining, list):
        raise HierarchicalViewError(
            "binding",
            "binding.pipelining must be a list when present",
        )
    if not raw_pipelining:
        return

    stage_by_name = {stage.stage_name: stage for stage in stage_views}
    engine_groups = _collect_engine_statement_groups(binding)
    fusion_name_by_rank, fusion_members_by_name = _collect_fusion_statement_groups(
        mapping
    )

    seen_loops: set[str] = set()
    for pipeline_index, raw_pipeline in enumerate(raw_pipelining):
        if not isinstance(raw_pipeline, dict):
            raise HierarchicalViewError(
                "binding",
                f"binding.pipelining[{pipeline_index}] must be a mapping",
            )

        loop_name_raw = raw_pipeline.get("loop")
        if not isinstance(loop_name_raw, str) or not loop_name_raw:
            raise HierarchicalViewError(
                "binding",
                f"binding.pipelining[{pipeline_index}].loop must be a nonempty string",
            )
        loop_name = loop_name_raw
        if loop_name in seen_loops:
            raise HierarchicalViewError(
                "binding",
                f"Duplicate pipelining entry for loop {loop_name!r}",
            )
        seen_loops.add(loop_name)

        members = fusion_members_by_name.get(loop_name)
        if members is None and loop_name.count(".") == 1:
            stage_name, rank_name = loop_name.split(".", 1)
            stage = stage_by_name.get(stage_name)
            if stage is not None and rank_name in set(stage.loop_order):
                members = ((stage_name, rank_name),)

                                                                           
                                                                               
                                                                            
        if members is None:
            continue

        expected = _expected_pipelining_statements(
            members,
            stage_by_name=stage_by_name,
            engine_groups=engine_groups,
            fusion_name_by_rank=fusion_name_by_rank,
        )

        raw_order = raw_pipeline.get("order", [])
        if not isinstance(raw_order, list):
            raise HierarchicalViewError(
                "binding",
                f"binding.pipelining[{pipeline_index}].order must be a list",
            )

        given: List[str] = []
        for order_index, raw_statement in enumerate(raw_order):
            if not isinstance(raw_statement, dict):
                raise HierarchicalViewError(
                    "binding",
                    (
                        f"binding.pipelining[{pipeline_index}].order"
                        f"[{order_index}] must be a mapping"
                    ),
                )
            statement = raw_statement.get("statement")
            if not isinstance(statement, str) or not statement:
                raise HierarchicalViewError(
                    "binding",
                    (
                        f"binding.pipelining[{pipeline_index}].order"
                        f"[{order_index}].statement must be a nonempty string"
                    ),
                )
            offset = raw_statement.get("offset", 0)
            if not isinstance(offset, int) or isinstance(offset, bool):
                raise HierarchicalViewError(
                    "binding",
                    (
                        f"binding.pipelining[{pipeline_index}].order"
                        f"[{order_index}].offset must be an integer"
                    ),
                )
            given.append(_normalize_pipelining_statement(statement))

        duplicate = _first_duplicate(given)
        location = f"{loop_name} loop pipelining"
        if duplicate is not None:
            raise HierarchicalViewError(
                "binding",
                f"Duplicate statement {duplicate!r} in {location}",
            )

        given_set = set(given)
        unknown = sorted(given_set - expected)
        if unknown:
            raise HierarchicalViewError(
                "binding",
                f"Unknown statement {unknown[0]!r} in {location}",
            )

        missing = sorted(expected - given_set)
        if missing:
            raise HierarchicalViewError(
                "binding",
                f"Missing statement {missing[0]!r} in {location}",
            )



def _normalize_pipelining_statement(statement: str) -> str:
    

    compact = "".join(statement.split())
    suffix = ".instruction"
    if compact.endswith(suffix):
        prefix = compact[: -len(suffix)]
        if prefix.startswith("(") and prefix.endswith(")"):
            names = [name for name in prefix[1:-1].split(",") if name]
            if len(names) == 1:
                return f"{names[0]}.instruction"
            return f"({', '.join(names)}).instruction"
        return f"{prefix}.instruction"
    return compact

def _collect_engine_statement_groups(
    binding: Dict[str, object],
) -> Dict[str, Tuple[str, ...]]:
    raw_engines = binding.get("engine", [])
    if not isinstance(raw_engines, list):
        return {}

    groups: Dict[str, Tuple[str, ...]] = {}
    for raw_engine in raw_engines:
        if not isinstance(raw_engine, dict):
            continue
        if "einsum" in raw_engine:
            stage_names = (str(raw_engine.get("einsum")),)
        elif "einsums" in raw_engine and isinstance(raw_engine.get("einsums"), list):
            stage_names = tuple(str(name) for name in raw_engine["einsums"])
        else:
            continue
        for stage_name in stage_names:
            groups[stage_name] = stage_names
    return groups


def _collect_fusion_statement_groups(
    mapping: Dict[str, object],
) -> Tuple[
    Dict[Tuple[str, str], str],
    Dict[str, Tuple[Tuple[str, str], ...]],
]:
    

    name_by_rank: Dict[Tuple[str, str], str] = {}
    members_by_name: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    raw_fusion = mapping.get("fusion", [])
    if not isinstance(raw_fusion, list):
        return name_by_rank, members_by_name

    for raw_entry in raw_fusion:
        if not isinstance(raw_entry, dict):
            continue
        raw_rank_groups = raw_entry.get("ranks", [])
        if not isinstance(raw_rank_groups, list):
            continue
        for raw_group in raw_rank_groups:
            if isinstance(raw_group, dict) and len(raw_group) == 1:
                group_name, raw_refs = next(iter(raw_group.items()))
                group_name = str(group_name)
            elif isinstance(raw_group, list):
                raw_refs = raw_group
                group_name = "-".join(str(ref) for ref in raw_refs)
            else:
                continue
            if not isinstance(raw_refs, list):
                continue

            members: List[Tuple[str, str]] = []
            for raw_ref in raw_refs:
                if not isinstance(raw_ref, str) or raw_ref.count(".") != 1:
                    continue
                stage_name, rank_name = raw_ref.split(".", 1)
                member = (stage_name, rank_name)
                members.append(member)
                name_by_rank[member] = group_name
            if members:
                members_by_name[group_name] = tuple(members)

    return name_by_rank, members_by_name


def _expected_pipelining_statements(
    members: Sequence[Tuple[str, str]],
    *,
    stage_by_name: Dict[str, StageView],
    engine_groups: Dict[str, Tuple[str, ...]],
    fusion_name_by_rank: Dict[Tuple[str, str], str],
) -> set[str]:
    expected: set[str] = set()

    for stage_name, rank_name in members:
        stage = stage_by_name.get(stage_name)
        if stage is None or rank_name not in set(stage.loop_order):
            continue

        rank_index = stage.loop_order.index(rank_name)
        is_instruction = rank_index == len(stage.loop_order) - 1
        if not is_instruction:
            next_rank = stage.loop_order[rank_index + 1]
            is_instruction = next_rank in set(stage.engine_binding.ranks)

        if is_instruction:
            group = engine_groups.get(stage_name, (stage_name,))
            if len(group) == 1:
                expected.add(f"{group[0]}.instruction")
            else:
                expected.add(f"({', '.join(group)}).instruction")
            continue

        next_rank = stage.loop_order[rank_index + 1]
        expected.add(
            fusion_name_by_rank.get(
                (stage_name, next_rank),
                f"{stage_name}.{next_rank}",
            )
        )

    return expected

def _first_duplicate(values: Sequence[str]) -> Optional[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _collect_architecture_templates(
    architecture: Dict[str, object],
) -> Dict[Tuple[str, str], Tuple[ArchitectureTemplateView, ...]]:
    

    templates: Dict[Tuple[str, str], List[ArchitectureTemplateView]] = {}

    def walk(nodes: Sequence[object]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_name = node.get("name")
            raw_templates = node.get("templates")
            if isinstance(raw_templates, list):
                for template in raw_templates:
                    if not isinstance(template, dict):
                        continue
                    _collect_one_architecture_template(
                        templates,
                        template,
                        node_name=None if node_name is None else str(node_name),
                    )

            raw_children = node.get("nodes")
            if isinstance(raw_children, list):
                walk(raw_children)

    root_nodes = _expect_list(architecture, "nodes", stage_name="architecture")
    walk(root_nodes)
    return {key: tuple(values) for key, values in templates.items()}


def _collect_one_architecture_template(
    templates: Dict[Tuple[str, str], List[ArchitectureTemplateView]],
    template: Dict[str, object],
    *,
    node_name: Optional[str],
) -> None:
    binding = template.get("binding", {})
    if not isinstance(binding, dict):
        return

    engines = binding.get("engine", [])
    if not isinstance(engines, list):
        return

    template_name = str(template.get("name", "unnamed-template"))
    parsed_stages = _parse_architecture_template_stages(template)
    mapping = template.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}
    loop_orders = mapping.get("loop-order", {})
    spacetime = mapping.get("spacetime", {})
    if not isinstance(loop_orders, dict):
        loop_orders = {}
    if not isinstance(spacetime, dict):
        spacetime = {}

    tensor_entries = binding.get("tensor", [])
    if not isinstance(tensor_entries, list):
        tensor_entries = []

    for engine_entry in engines:
        if not isinstance(engine_entry, dict):
            continue

        instruction_name = str(engine_entry.get("instruction"))
        raw_engine_component = component_name(engine_entry.get("component"))
        engine_component = (
            raw_engine_component
            if raw_engine_component not in (None, "None")
            else node_name
        )
        if engine_component is None:
            engine_component = "None"

        stage_names, ranks_by_stage = _template_engine_stage_ranks(engine_entry)
        if not stage_names:
            continue

        for template_stage_name in stage_names:
            parsed = parsed_stages.get(template_stage_name)
            if parsed is None:
                                                                              
                                                                             
                                                                           
                parsed = {
                    "op_kind": _infer_template_op_kind(template, instruction_name),
                    "output_tensor": None,
                    "input_tensors": tuple(),
                }

            engine_ranks = tuple(ranks_by_stage.get(template_stage_name, ()))
            template_loop_order = tuple(
                str(rank) for rank in loop_orders.get(template_stage_name, [])
            )
            stage_spacetime = spacetime.get(template_stage_name, {})
            if not isinstance(stage_spacetime, dict):
                stage_spacetime = {}
            template_space = tuple(
                str(rank) for rank in stage_spacetime.get("space", [])
            )
            template_time = tuple(
                str(rank) for rank in stage_spacetime.get("time", [])
            )

            tensor_components = _template_tensor_components_for_stage(
                tensor_entries,
                template_stage_name,
            )
            input_tensors = tuple(str(name) for name in parsed["input_tensors"])
            output_tensor = parsed["output_tensor"]
            input_components = tuple(
                tensor_components.get(tensor_name, tuple())
                for tensor_name in input_tensors
            )
            output_components = (
                tuple()
                if output_tensor is None
                else tensor_components.get(str(output_tensor), tuple())
            )

            key = (engine_component, instruction_name)
            templates.setdefault(key, []).append(
                ArchitectureTemplateView(
                    engine_component=engine_component,
                    instruction_name=instruction_name,
                    template_name=template_name,
                    template_stage_name=template_stage_name,
                    op_kind=str(parsed["op_kind"]),
                    engine_ranks=engine_ranks,
                    loop_order=template_loop_order,
                    space_ranks=template_space,
                    time_ranks=template_time,
                    input_components=input_components,
                    output_components=output_components,
                    input_count=len(input_tensors),
                    tensor_components=tensor_components,
                )
            )


def _template_engine_stage_ranks(
    engine_entry: Dict[str, object],
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    if "einsum" in engine_entry:
        stage_name = str(engine_entry.get("einsum"))
        ranks = engine_entry.get("ranks", [])
        if not isinstance(ranks, list):
            ranks = []
        return (stage_name,), {stage_name: tuple(str(rank) for rank in ranks)}

    raw_stage_names = engine_entry.get("einsums", [])
    if not isinstance(raw_stage_names, list):
        return tuple(), {}
    stage_names = tuple(str(stage) for stage in raw_stage_names)
    raw_ranks = engine_entry.get("ranks", {})
    ranks_by_stage: Dict[str, Tuple[str, ...]] = {}
    if isinstance(raw_ranks, dict):
        for stage_name in stage_names:
            ranks = raw_ranks.get(stage_name, [])
            if isinstance(ranks, list):
                ranks_by_stage[stage_name] = tuple(str(rank) for rank in ranks)
    elif isinstance(raw_ranks, list) and len(stage_names) == 1:
        ranks_by_stage[stage_names[0]] = tuple(str(rank) for rank in raw_ranks)
    return stage_names, ranks_by_stage


def _parse_architecture_template_stages(
    template: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    einsum = template.get("einsum", {})
    if not isinstance(einsum, dict):
        return {}
    expressions = einsum.get("expressions", [])
    if not isinstance(expressions, list):
        return {}

    parsed_stages: Dict[str, Dict[str, object]] = {}
    for position, raw_expression in enumerate(expressions):
        try:
            stage_name_override, expression = _normalize_expression_entry(
                raw_expression,
                position,
            )
            parsed = _parse_expression(expression, position)
            stage_name = stage_name_override or str(parsed["stage_name"])
            parsed_stages[stage_name] = parsed
        except HierarchicalViewError:
            fallback = _parse_template_expression_fallback(raw_expression, position)
            if fallback is not None:
                parsed_stages[str(fallback["stage_name"])] = fallback
    return parsed_stages


def _parse_template_expression_fallback(
    raw_expression: object,
    stage_position: int,
) -> Optional[Dict[str, object]]:
    try:
        stage_name_override, expression = _normalize_expression_entry(
            raw_expression,
            stage_position,
        )
    except HierarchicalViewError:
        return None

    body = expression.strip()
    label: Optional[str] = stage_name_override
    if ":" in body:
        prefix, remainder = body.split(":", 1)
        if "=" in remainder:
            label = label or prefix.strip()
            body = remainder.strip()
    if "=" not in body:
        return None

    lhs_text, rhs_text = [piece.strip() for piece in body.split("=", 1)]
    try:
        output_tensor, output_indices = _parse_tensor_ref(lhs_text, stage_position)
    except HierarchicalViewError:
        return None

    rhs_core = rhs_text.split(";;", 1)[0]
    refs = [
        match.group("tensor")
        for match in re.finditer(
            r"(?P<tensor>[A-Za-z_][A-Za-z0-9_]*)\[[^\]]+\]",
            rhs_core,
        )
    ]
    operators = list(_OP_TOKEN_RE.finditer(rhs_text))
    if not operators:
        op_kind = "rename" if refs else "init"
    else:
        has_map_unary = any(
            match.group("tag") == "M" and match.group("kind") == "unary"
            for match in operators
        )
        has_map_binary = any(
            match.group("tag") == "M" and match.group("kind") == "binary"
            for match in operators
        )
        has_reduce = any(
            match.group("tag") == "V" and match.group("kind") == "binary"
            for match in operators
        )
        if has_reduce and has_map_binary:
            op_kind = "binary_reduce"
        elif has_reduce and has_map_unary:
            op_kind = "unary_reduce"
        elif has_reduce:
            op_kind = "reduce"
        elif has_map_binary:
            op_kind = "binary"
        else:
            op_kind = "unary"

    return {
        "stage_name": label or output_tensor,
        "output_tensor": output_tensor,
        "input_tensors": tuple(refs),
        "output_indices": output_indices,
        "input_indices": tuple(),
        "op_kind": op_kind,
        "unary_operator": None,
        "reduction_operator": None,
        "binary_operator": None,
        "init_value": None,
    }


def _template_tensor_components_for_stage(
    tensor_entries: Sequence[object],
    template_stage_name: str,
) -> Dict[str, Tuple[str, ...]]:
    tensor_components: Dict[str, Tuple[str, ...]] = {}
    for tensor_entry in tensor_entries:
        if not isinstance(tensor_entry, dict):
            continue
        if str(tensor_entry.get("einsum")) != template_stage_name:
            continue
        raw_tensor_name = tensor_entry.get("name")
        if raw_tensor_name is None:
            continue
        tensor_name = str(raw_tensor_name)
        if "components" in tensor_entry:
            names = component_names(tensor_entry.get("components", []))
        else:
            name = component_name(tensor_entry.get("component"))
            names = tuple() if name is None else (name,)
        if names:
            tensor_components[tensor_name] = tuple(names)
    return tensor_components


def _infer_template_op_kind(template: Dict[str, object], instruction_name: str) -> str:
    instruction_lower = instruction_name.lower()
    if instruction_lower in {"load", "store", "renameonchip2d", "renameonchip3d", "renamehbm2d", "renamehbm3d"}:
        return "rename"
    if "transpose" in instruction_lower or "copy" in instruction_lower:
        return "rename"
    if "memset" in instruction_lower:
        return "init"

                                                                              
                                                                              
    if "scaledunary" in instruction_lower:
        return "map_reduce"
    if "biasedunaryreduce" in instruction_lower:
        return "map_reduce"
    if "tensorconstreduce" in instruction_lower or "tensorreducenegate" in instruction_lower:
        return "binary_reduce"
    if "unaryreduce" in instruction_lower:
        return "unary_reduce"

    if "tensortensor" in instruction_lower or "tensorscalar" in instruction_lower or "scalartensor" in instruction_lower:
        return "binary"
    if "reciprocal" in instruction_lower:
                                                                             
                                                                
                                                                       
                                                                             
                                                             
        return "binary"
    if "unary" in instruction_lower and "binary" not in instruction_lower:
        return "unary"
    if "binary" in instruction_lower or "matmul" in instruction_lower or "gemm" in instruction_lower:
        return "binary"

    expressions = template.get("einsum", {}).get("expressions", [])
    if isinstance(expressions, list) and expressions:
        expr_text = " ".join(str(expr) for expr in expressions)
        expr_lower = expr_text.lower()
        has_reduce = "v binary[" in expr_lower
        has_map_unary = "m unary[" in expr_lower
        has_map_binary = "m binary[" in expr_lower

        if "= 0" in expr_lower:
            return "init"
        if has_reduce and has_map_unary and has_map_binary:
            return "map_reduce"
        if has_reduce and has_map_binary:
            return "binary_reduce"
        if has_reduce and has_map_unary:
            return "unary_reduce"
        if has_reduce:
            return "reduce"
        if has_map_binary:
            return "binary"
        if has_map_unary or "unary[" in expr_lower:
            return "unary"
        if "." in expr_text.split("=")[-1]:
            return "binary"
        if _TENSOR_REF_RE.match(str(expressions[0]).split("=")[0].strip()) is not None:
            return "rename"

    return "unknown"


def _build_interface_views(stage_views: Sequence[StageView]) -> List[InterfaceView]:
    produced_by: Dict[str, StageView] = {}
    interfaces: List[InterfaceView] = []

    for stage in stage_views:
        for input_tensor, input_binding, input_format in zip(
            stage.input_tensors,
            stage.input_bindings,
            stage.input_formats,
        ):
            producer = produced_by.get(input_tensor)
            if producer is None:
                continue

            producer_order = producer.output_format.rank_orders.get(producer.stage_name, ())
            consumer_order = input_format.rank_orders.get(stage.stage_name, ())
            producer_normalized_order = _normalize_format_order(
                producer_order,
                producer.rank_roots,
            )
            consumer_normalized_order = _normalize_format_order(
                consumer_order,
                stage.rank_roots,
            )

            interfaces.append(
                InterfaceView(
                    obligation_id=f"iface:{input_tensor}:{producer.stage_name}->{stage.stage_name}",
                    tensor_name=input_tensor,
                    producer_stage_name=producer.stage_name,
                    consumer_stage_name=stage.stage_name,
                    producer_stage_position=producer.stage_position,
                    consumer_stage_position=stage.stage_position,
                    producer_binding=producer.output_binding,
                    consumer_binding=input_binding,
                    producer_format_order=tuple(producer_order),
                    consumer_format_order=tuple(consumer_order),
                    producer_normalized_format_order=producer_normalized_order,
                    consumer_normalized_format_order=consumer_normalized_order,
                    producer_bit_width=producer.output_format.bit_width,
                    consumer_bit_width=input_format.bit_width,
                    producer_data_type=producer.output_format.data_type,
                    consumer_data_type=input_format.data_type,
                )
            )

        produced_by[stage.output_tensor] = stage

    return interfaces


def parse_affine_token(token: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    text = token.strip()
    if _INT_TOKEN_RE.match(text):
        return None, int(text), None
    match = _AFFINE_TOKEN_RE.match(text)
    if match is None:
        return None, None, text
    base = match.group("base")
    offset_text = match.group("offset")
    if base and base[0].isupper() and offset_text is None:
        return None, None, base
    offset = 0 if offset_text is None else int(offset_text)
    return base, offset, None


def _split_indices(text: str) -> List[str]:
    return [piece.strip() for piece in str(text).split(",") if piece.strip()]


def _expect_mapping(mapping: Dict[str, object], key: str, stage_name: Optional[str] = None) -> Dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise HierarchicalViewError(stage_name or key, f"{key} must be a mapping")
    return value


def _expect_list(mapping: Dict[str, object], key: str, stage_name: Optional[str] = None) -> List[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise HierarchicalViewError(stage_name or key, f"{key} must be a list")
    return value


def _expect_string_list(value: object, path: str, stage_name: Optional[str] = None) -> List[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        raise HierarchicalViewError(stage_name or path, f"{path} must be a list of strings")

    result: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise HierarchicalViewError(stage_name or path, f"{path} must contain only strings")
        result.append(item)
    return result
