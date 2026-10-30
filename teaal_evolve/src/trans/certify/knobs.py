

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .candidate import Candidate
from .schema_utils import component_name


KNOB_SPACE_RESTRICTED = "restricted"
KNOB_SPACE_RICH = "rich"
SUPPORTED_KNOB_SPACES = {KNOB_SPACE_RESTRICTED, KNOB_SPACE_RICH}

_ALLOWED_ONCHIP_COMPONENTS = ("SBUF", "PSUM")
_ALLOWED_RICH_COMPONENTS = ("SBUF", "PSUM", "HBM")


@dataclass(frozen=True)
class CandidateKnob:
    

    knob_id: str
    kind: str
    current_value: str
    allowed_values: List[str]
    tensor_name: str
    binding_tensor_indices: List[int]
    detail: str
    metadata: Dict[str, str] = field(default_factory=dict)


def extract_candidate_knobs(
    candidate: Candidate,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> List[CandidateKnob]:
    if knob_space not in SUPPORTED_KNOB_SPACES:
        raise ValueError(f"Unsupported knob space: {knob_space}")

    if knob_space == KNOB_SPACE_RESTRICTED:
        return _extract_grouped_component_knobs(candidate, rich=False)

    knobs: List[CandidateKnob] = []
    knobs.extend(_extract_grouped_component_knobs(candidate, rich=True))
    knobs.extend(_extract_engine_instruction_knobs(candidate))
    knobs.extend(_extract_loop_order_knobs(candidate))
    knobs.extend(_extract_spacetime_knobs(candidate))
    knobs.extend(_extract_format_rank_order_knobs(candidate))
    return sorted(knobs, key=lambda knob: knob.knob_id)


def extract_knob_map(
    candidate: Candidate,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> Dict[str, CandidateKnob]:
    return {knob.knob_id: knob for knob in extract_candidate_knobs(candidate, knob_space=knob_space)}


def _extract_grouped_component_knobs(candidate: Candidate, *, rich: bool) -> List[CandidateKnob]:
    binding = candidate.spec_dict.get("binding", {})
    tensor_bindings = binding.get("tensor", [])

    groups: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}

    visible_components = (
        _ALLOWED_RICH_COMPONENTS if rich else _ALLOWED_ONCHIP_COMPONENTS
    )

    for index, entry in enumerate(tensor_bindings):
        component = component_name(entry.get("component"))
        name = entry.get("name")

        if component not in visible_components:
            continue
        if not isinstance(name, str):
            continue

        groups.setdefault(name, []).append((index, entry))

    knobs: List[CandidateKnob] = []
    allowed_components = list(_ALLOWED_RICH_COMPONENTS if rich else _ALLOWED_ONCHIP_COMPONENTS)

    for tensor_name, entries in sorted(groups.items()):
        indices = [index for index, _ in entries]
        current_values = {component_name(entry.get("component")) for _, entry in entries}

        if len(current_values) != 1:
            current_value = "mixed"
            detail = (
                "Grouped tensor component knob found inconsistent current values "
                "across binding.tensor entries"
            )
        else:
            current_value = next(iter(current_values))
            detail = (
                "Grouped tensor component knob. Mutating this knob updates all "
                "binding.tensor entries for the same logical tensor."
            )
            if rich:
                detail += " Rich mode also allows HBM to intentionally create illegal placements."

        allowed_values = _ordered_allowed_values(current_value, allowed_components)

        knobs.append(
            CandidateKnob(
                knob_id=f"binding.tensor.component::{tensor_name}",
                kind="tensor_component_group",
                current_value=current_value,
                allowed_values=allowed_values,
                tensor_name=tensor_name,
                binding_tensor_indices=indices,
                detail=detail,
                metadata={"tensor_name": tensor_name},
            )
        )

    return knobs


def _extract_engine_instruction_knobs(candidate: Candidate) -> List[CandidateKnob]:
    binding = candidate.spec_dict.get("binding", {})
    engine_bindings = binding.get("engine", [])
    if not isinstance(engine_bindings, list):
        return []

    architecture_values = _collect_architecture_engine_instruction_values(candidate)
    knobs: List[CandidateKnob] = []

    for index, entry in enumerate(engine_bindings):
        if not isinstance(entry, dict):
            continue

        if "einsum" in entry:
            stage_names = [str(entry.get("einsum"))]
        elif "einsums" in entry and isinstance(entry.get("einsums"), list):
            stage_names = [str(stage) for stage in entry.get("einsums", [])]
        else:
            continue

        component = component_name(entry.get("component"))
        instruction = entry.get("instruction")
        if component is None or instruction is None:
            continue

        stage_key = "+".join(stage_names)
        current_value = _serialize_engine_instruction(component, str(instruction))
        allowed_values = _unique_keep_order(
            [current_value] + [value for value in architecture_values if value != current_value]
        )

        knobs.append(
            CandidateKnob(
                knob_id=f"binding.engine.instruction::{stage_key}",
                kind="engine_instruction",
                current_value=current_value,
                allowed_values=allowed_values,
                tensor_name="",
                binding_tensor_indices=[index],
                detail=(
                    "Curated binding.engine instruction knob. Mutating this knob changes "
                    "the engine component/instruction pair for one binding.engine entry."
                ),
                metadata={"stage_name": stage_key, "engine_entry_index": str(index)},
            )
        )

    return knobs


def _collect_architecture_engine_instruction_values(candidate: Candidate) -> List[str]:
    architecture = candidate.spec_dict.get("architecture", {})
    root_nodes = architecture.get("nodes", [])
    values: List[str] = []

    def walk(nodes) -> None:
        if not isinstance(nodes, list):
            return

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_name = node.get("name")
            raw_templates = node.get("templates", [])
            if isinstance(raw_templates, list):
                for template in raw_templates:
                    if not isinstance(template, dict):
                        continue
                    binding = template.get("binding", {})
                    if not isinstance(binding, dict):
                        continue
                    engines = binding.get("engine", [])
                    if not isinstance(engines, list):
                        continue
                    for engine_entry in engines:
                        if not isinstance(engine_entry, dict):
                            continue
                        component = component_name(engine_entry.get("component"))
                        if component in (None, "None"):
                            if isinstance(node_name, str) and node_name != "None":
                                component = node_name
                            else:
                                component = None
                        instruction = engine_entry.get("instruction")
                        if component is None or instruction is None:
                            continue
                        values.append(_serialize_engine_instruction(component, str(instruction)))

            children = node.get("nodes", [])
            if isinstance(children, list):
                walk(children)

    walk(root_nodes)
    return _unique_keep_order(values)


def _serialize_engine_instruction(component: str, instruction: str) -> str:
    return f"{component}::{instruction}"


def parse_engine_instruction_value(value: str) -> tuple[str, str]:
    try:
        component, instruction = value.split("::", 1)
    except ValueError as exc:
        raise ValueError(
            f"Engine-instruction value must have form '<component>::<instruction>', got {value!r}"
        ) from exc

    component = component.strip()
    instruction = instruction.strip()
    if not component or not instruction:
        raise ValueError(
            f"Engine-instruction value must have non-empty component and instruction, got {value!r}"
        )
    return component, instruction


def _extract_loop_order_knobs(candidate: Candidate) -> List[CandidateKnob]:
    mapping = candidate.spec_dict.get("mapping", {})
    loop_order = mapping.get("loop-order", {})
    knobs: List[CandidateKnob] = []

    for stage_name, ranks in sorted(loop_order.items()):
        if not isinstance(stage_name, str) or not isinstance(ranks, list) or not ranks:
            continue

        current_value = _serialize_rank_list(ranks)
        allowed_values = _build_loop_order_alternatives([str(rank) for rank in ranks])
        knobs.append(
            CandidateKnob(
                knob_id=f"mapping.loop_order::{stage_name}",
                kind="mapping_loop_order",
                current_value=current_value,
                allowed_values=allowed_values,
                tensor_name="",
                binding_tensor_indices=[],
                detail=(
                    "Curated mapping.loop-order alternatives for richer search. "
                    "The first alternative is intended to stay plausible; later alternatives may be illegal."
                ),
                metadata={"stage_name": stage_name},
            )
        )

    return knobs


def _extract_spacetime_knobs(candidate: Candidate) -> List[CandidateKnob]:
    mapping = candidate.spec_dict.get("mapping", {})
    loop_order = mapping.get("loop-order", {})
    spacetime = mapping.get("spacetime", {})
    knobs: List[CandidateKnob] = []

    for stage_name, entry in sorted(spacetime.items()):
        if not isinstance(stage_name, str) or not isinstance(entry, dict):
            continue

        space = [str(rank) for rank in entry.get("space", [])]
        time = [str(rank) for rank in entry.get("time", [])]
        current_value = _serialize_spacetime(space, time)
        stage_loop_order = [str(rank) for rank in loop_order.get(stage_name, [])]
        allowed_values = _build_spacetime_alternatives(stage_loop_order, space, time)

        knobs.append(
            CandidateKnob(
                knob_id=f"mapping.spacetime::{stage_name}",
                kind="mapping_spacetime",
                current_value=current_value,
                allowed_values=allowed_values,
                tensor_name="",
                binding_tensor_indices=[],
                detail=(
                    "Curated mapping.spacetime alternatives for richer search. "
                    "Some alternatives preserve coverage while others intentionally break it."
                ),
                metadata={"stage_name": stage_name},
            )
        )

    return knobs


def _extract_format_rank_order_knobs(candidate: Candidate) -> List[CandidateKnob]:
    format_spec = candidate.spec_dict.get("format", {})
    knobs: List[CandidateKnob] = []

    for tensor_name, tensor_entry in sorted(format_spec.items()):
        if not isinstance(tensor_name, str) or not isinstance(tensor_entry, dict):
            continue

        rank_order = tensor_entry.get("rank-order", {})
        if not isinstance(rank_order, dict):
            continue

        for consumer_name, ranks in sorted(rank_order.items()):
            if not isinstance(consumer_name, str) or not isinstance(ranks, list) or not ranks:
                continue

            current_value = _serialize_rank_list(ranks)
            allowed_values = _build_format_rank_order_alternatives([str(rank) for rank in ranks])
            knobs.append(
                CandidateKnob(
                    knob_id=f"format.rank_order::{tensor_name}::{consumer_name}",
                    kind="format_rank_order",
                    current_value=current_value,
                    allowed_values=allowed_values,
                    tensor_name=tensor_name,
                    binding_tensor_indices=[],
                    detail=(
                        "Curated format.rank-order alternatives for richer search. "
                        "Later alternatives may intentionally omit a rank to create illegal candidates."
                    ),
                    metadata={"tensor_name": tensor_name, "consumer_name": consumer_name},
                )
            )

    return knobs


def _ordered_allowed_values(current_value: str, values: List[str]) -> List[str]:
                                                                     
                                               
    return list(values)


def _build_loop_order_alternatives(current: List[str]) -> List[str]:
    current_serialized = _serialize_rank_list(current)
    alternatives = [current_serialized]

    if len(current) > 1:
        reversed_value = _serialize_rank_list(list(reversed(current)))
        if reversed_value != current_serialized:
            alternatives.append(reversed_value)
        rotated = _serialize_rank_list(current[1:] + current[:1])
        if rotated not in alternatives:
            alternatives.append(rotated)
        shortened = _serialize_rank_list(current[:-1])
        if shortened not in alternatives:
            alternatives.append(shortened)

    return _unique_keep_order(alternatives)


def _build_spacetime_alternatives(
    loop_order: List[str],
    current_space: List[str],
    current_time: List[str],
) -> List[str]:
    current_serialized = _serialize_spacetime(current_space, current_time)
    alternatives = [current_serialized]

    if loop_order:
        alt_space = [loop_order[-1]]
        alt_time = [rank for rank in loop_order if rank not in alt_space]
        alternatives.append(_serialize_spacetime(alt_space, alt_time))

    if len(loop_order) > 1:
        invalid_missing = _serialize_spacetime(current_space, loop_order[:-1])
        alternatives.append(invalid_missing)

    return _unique_keep_order(alternatives)


def _build_format_rank_order_alternatives(current: List[str]) -> List[str]:
    current_serialized = _serialize_rank_list(current)
    alternatives = [current_serialized]

    if len(current) > 1:
        reversed_value = _serialize_rank_list(list(reversed(current)))
        if reversed_value != current_serialized:
            alternatives.append(reversed_value)
        shortened = _serialize_rank_list(current[:-1])
        if shortened not in alternatives:
            alternatives.append(shortened)

    return _unique_keep_order(alternatives)


def _serialize_rank_list(ranks: List[str]) -> str:
    return "[" + ",".join(ranks) + "]"


def parse_rank_list_value(value: str) -> List[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Rank-list value must have bracket form, got {value!r}")
    body = value[1:-1].strip()
    if not body:
        return []
    return [part.strip() for part in body.split(",") if part.strip()]


def _serialize_spacetime(space: List[str], time: List[str]) -> str:
    return f"space={_serialize_rank_list(space)};time={_serialize_rank_list(time)}"


def parse_spacetime_value(value: str) -> tuple[List[str], List[str]]:
    try:
        space_part, time_part = value.split(";", 1)
    except ValueError as exc:
        raise ValueError(f"Spacetime value must contain ';', got {value!r}") from exc

    if not space_part.startswith("space=") or not time_part.startswith("time="):
        raise ValueError(f"Malformed spacetime value: {value!r}")

    space = parse_rank_list_value(space_part[len("space="):])
    time = parse_rank_list_value(time_part[len("time="):])
    return space, time


def _unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
